"""A web browser an agent can use: open pages, read them, click, type, look.

    from agent_harness import Agent, Browser

    async with Browser() as browser:
        agent = Agent("shopper", "Find things on the web.", tools=browser.tools())
        await agent.run("What is the top story on Hacker News?")

The browser is Chrome, Chromium or Edge — whichever is installed — started
headless and driven over the DevTools protocol, so nothing is added to install.
`Browser(cdp_url="http://host:9222")` drives one that is already running
somewhere else instead: a container, a remote machine, a hosted browser.

The agent is not handed pixels to guess at. After every action it reads the
page as a list of the things that can be acted on, each with a number:

    Page: Sign in — https://shop.example/login
    [1] textbox "Email"
    [2] textbox "Password"
    [3] button "Sign in"
    [4] link "Forgot your password?" → /reset

and acts by number: `browser_type(ref=1, text=…)`, `browser_click(ref=3)`. Any
model can do that, vision or not. `browser_screenshot` is there for a model
that can see, and `Browser.computer()` gives one the mouse and keyboard.

Where it may go is yours to say: `allowed_domains`, `blocked_domains`, and no
private or loopback address unless `allow_private=True`. The policy is applied
to every page load — a typed address, a clicked link, a redirect, a frame.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import weakref
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from ..errors import ConfigurationError, ToolError
from ..toolkits._net import is_private_host
from ..toolkits.search import _domain, _matches
from ..types import ImageBlock
from .cdp import CDP, CDPError

__all__ = ["Browser", "find_browser"]

_Timeout = (TimeoutError, asyncio.TimeoutError)

_NAMES = ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser",
          "chrome", "microsoft-edge", "microsoft-edge-stable", "msedge", "brave-browser")
_PATHS = (
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
)
#: Environment variables that name the browser to start.
BROWSER_ENV = ("AGENT_HARNESS_BROWSER", "CHROME_PATH")

_MODIFIERS = {"alt": 1, "option": 1, "ctrl": 2, "control": 2, "meta": 4, "cmd": 4,
              "command": 4, "super": 4, "win": 4, "shift": 8}
#: name → (key, code, virtual key code, text)
_KEYS: dict[str, tuple[str, str, int, str]] = {
    "enter": ("Enter", "Enter", 13, "\r"), "return": ("Enter", "Enter", 13, "\r"),
    "tab": ("Tab", "Tab", 9, ""), "escape": ("Escape", "Escape", 27, ""),
    "esc": ("Escape", "Escape", 27, ""),
    "backspace": ("Backspace", "Backspace", 8, ""),
    "delete": ("Delete", "Delete", 46, ""), "del": ("Delete", "Delete", 46, ""),
    "space": (" ", "Space", 32, " "),
    "arrowup": ("ArrowUp", "ArrowUp", 38, ""), "up": ("ArrowUp", "ArrowUp", 38, ""),
    "arrowdown": ("ArrowDown", "ArrowDown", 40, ""),
    "down": ("ArrowDown", "ArrowDown", 40, ""),
    "arrowleft": ("ArrowLeft", "ArrowLeft", 37, ""),
    "left": ("ArrowLeft", "ArrowLeft", 37, ""),
    "arrowright": ("ArrowRight", "ArrowRight", 39, ""),
    "right": ("ArrowRight", "ArrowRight", 39, ""),
    "home": ("Home", "Home", 36, ""), "end": ("End", "End", 35, ""),
    "pageup": ("PageUp", "PageUp", 33, ""), "pgup": ("PageUp", "PageUp", 33, ""),
    "pagedown": ("PageDown", "PageDown", 34, ""), "pgdn": ("PageDown", "PageDown", 34, ""),
    "insert": ("Insert", "Insert", 45, ""),
    **{f"f{n}": (f"F{n}", f"F{n}", 111 + n, "") for n in range(1, 13)},
}
#: What a shortcut does to the text being edited; the browser needs telling.
_COMMANDS = {"a": "selectAll", "c": "copy", "v": "paste", "x": "cut", "z": "undo"}

# Everything the page is asked, in one object installed on first use. Numbers
# are kept for as long as the page lives, so a number read earlier still means
# the same element later.
_PAGE_JS = r"""
(() => {
  if (window.__ah) return;
  const refs = new Map(), ids = new WeakMap();
  let next = 1;
  const SEL = 'a[href],button,input,select,textarea,summary,iframe,[role=button],' +
    '[role=link],[role=tab],[role=menuitem],[role=menuitemcheckbox],[role=checkbox],' +
    '[role=radio],[role=switch],[role=option],[role=combobox],[role=textbox],' +
    '[role=searchbox],[role=slider],[contenteditable=""],[contenteditable=true],' +
    '[onclick],[tabindex]:not([tabindex="-1"])';
  const clean = (s, n) => {
    s = (s || '').replace(/\s+/g, ' ').trim();
    return s.length > n ? s.slice(0, n - 1) + '…' : s;
  };
  const visible = (el) => {
    const r = el.getBoundingClientRect();
    if (r.width < 1 || r.height < 1) return false;
    const s = getComputedStyle(el);
    return s.visibility !== 'hidden' && s.display !== 'none' && +s.opacity !== 0;
  };
  const role = (el) => {
    const given = el.getAttribute('role');
    if (given) return given;
    const tag = el.tagName.toLowerCase();
    if (tag === 'a') return 'link';
    if (tag === 'select') return 'combobox';
    if (tag === 'textarea') return 'textbox';
    if (tag === 'summary') return 'button';
    if (tag === 'iframe') return 'frame';
    if (tag === 'input') {
      const t = (el.type || 'text').toLowerCase();
      if (['button', 'submit', 'reset', 'image'].includes(t)) return 'button';
      if (['checkbox', 'radio', 'file', 'range'].includes(t)) return t === 'range' ? 'slider' : t;
      return t === 'search' ? 'searchbox' : 'textbox';
    }
    if (el.isContentEditable) return 'textbox';
    return tag === 'button' ? 'button' : 'clickable';
  };
  const name = (el) => {
    const by = el.getAttribute('aria-labelledby');
    if (by) {
      const t = by.split(/\s+/).map((i) => (document.getElementById(i) || {}).innerText || '').join(' ');
      if (t.trim()) return clean(t, 100);
    }
    const tag = el.tagName.toLowerCase();
    const img = el.querySelector && el.querySelector('img[alt]');
    const tries = [el.getAttribute('aria-label')];
    if (['input', 'select', 'textarea'].includes(tag)) {
      tries.push(el.labels && el.labels[0] && el.labels[0].innerText);
      if (['button', 'submit', 'reset'].includes(el.type)) tries.push(el.value);
      tries.push(el.placeholder, el.getAttribute('title'), el.name);
    } else {
      tries.push(el.innerText, img && img.alt, el.getAttribute('title'), el.getAttribute('alt'),
                 el.value, tag === 'iframe' ? el.name : '');
    }
    for (const t of tries) if (t && String(t).trim()) return clean(String(t), 100);
    return '';
  };
  const describe = (el) => {
    const tag = el.tagName.toLowerCase(), out = {role: role(el), name: name(el)};
    if (el.disabled || el.getAttribute('aria-disabled') === 'true') out.disabled = true;
    if (tag === 'a') {
      const href = el.getAttribute('href') || '';
      if (href && !href.startsWith('javascript:')) out.href = clean(href, 90);
    } else if (tag === 'iframe') {
      out.href = clean(el.src || '', 90);
    } else if (tag === 'select') {
      const picked = el.selectedOptions[0];
      out.value = clean(picked ? picked.text : '', 60);
      out.options = Array.from(el.options).slice(0, 15).map((o) => clean(o.text, 40));
      if (el.options.length > 15) out.options.push('… ' + (el.options.length - 15) + ' more');
    } else if (tag === 'input' || tag === 'textarea') {
      const t = (el.type || '').toLowerCase();
      if (t === 'checkbox' || t === 'radio') { if (el.checked) out.checked = true; }
      else if (t === 'password') { if (el.value) out.value = '•••'; }
      else if (!['button', 'submit', 'reset', 'image', 'file'].includes(t) && el.value)
        out.value = clean(el.value, 80);
    } else if (el.isContentEditable && el.innerText.trim()) {
      out.value = clean(el.innerText, 80);
    }
    const state = el.getAttribute('aria-checked') || el.getAttribute('aria-selected') ||
      el.getAttribute('aria-pressed');
    if (state === 'true') out.checked = true;
    if (el.getAttribute('aria-expanded')) out.expanded = el.getAttribute('aria-expanded') === 'true';
    return out;
  };
  const collect = (root, into) => {
    for (const el of root.querySelectorAll('*')) {
      if (el.matches(SEL)) into.push(el);
      if (el.shadowRoot) collect(el.shadowRoot, into);
    }
  };
  const parentOf = (n) => n.parentNode || n.host || null;
  const within = (a, b) => { for (let n = a; n; n = parentOf(n)) if (n === b) return true; return false; };
  const deepAt = (x, y) => {
    let el = document.elementFromPoint(x, y);
    while (el && el.shadowRoot) {
      const inner = el.shadowRoot.elementFromPoint(x, y);
      if (!inner || inner === el) break;
      el = inner;
    }
    return el;
  };
  const get = (ref) => {
    const held = refs.get(ref);
    const el = held && held.deref();
    return el && el.isConnected ? el : null;
  };
  window.__ah = {
    snapshot(maxElements, maxText) {
      const found = [];
      collect(document, found);
      // What is on screen comes first; a long page has more than can be listed.
      const here = [], below = [], above = [];
      const high = window.innerHeight, wide = window.innerWidth;
      for (const el of found) {
        if ((el.type || '') === 'hidden' || !visible(el)) continue;
        const r = el.getBoundingClientRect();
        if (r.bottom <= 0) above.push(el);
        else if (r.top >= high || r.right <= 0 || r.left >= wide) below.push(el);
        else here.push(el);
      }
      const elements = [];
      let hidden = 0;
      for (const [group, where] of [[here, ''], [below, 'below'], [above, 'above']]) {
        for (const el of group) {
          if (elements.length >= maxElements) { hidden += 1; continue; }
          const item = describe(el);
          if (!item.name && item.role === 'clickable') continue;
          let ref = ids.get(el);
          if (!ref) { ref = next++; ids.set(el, ref); refs.set(ref, new WeakRef(el)); }
          item.ref = ref;
          if (where) item.where = where;
          elements.push(item);
        }
      }
      // The words that are on screen, not the top of a page scrolled past.
      let text = '', total = 0;
      const walker = document.createTreeWalker(document.body || document, NodeFilter.SHOW_TEXT);
      const skip = new Set(['SCRIPT', 'STYLE', 'NOSCRIPT', 'TEMPLATE']);
      for (let node = walker.nextNode(); node; node = walker.nextNode()) {
        const words = node.nodeValue.replace(/\s+/g, ' ').trim();
        const parent = node.parentElement;
        if (!words || !parent || skip.has(parent.tagName)) continue;
        total += words.length + 1;
        if (text.length >= maxText) continue;
        const r = parent.getBoundingClientRect();
        if (r.bottom <= 0 || r.top >= high || r.width < 1 || r.height < 1) continue;
        if (!visible(parent)) continue;
        text += (text ? ' ' : '') + words;
      }
      const doc = document.documentElement, body = document.body;
      return {
        title: clean(document.title, 150), url: location.href, elements, more: hidden,
        text: clean(text, maxText), textLength: total,
        scrollY: Math.round(window.scrollY), viewHeight: high,
        pageHeight: Math.max(doc ? doc.scrollHeight : 0, body ? body.scrollHeight : 0),
      };
    },
    text() { return document.body ? document.body.innerText.replace(/\n{3,}/g, '\n\n').trim() : ''; },
    locate(ref) {
      const el = get(ref);
      if (!el) return {error: 'gone'};
      el.scrollIntoView({block: 'center', inline: 'center', behavior: 'instant'});
      const r = el.getBoundingClientRect();
      if (r.width < 1 || r.height < 1) return {error: 'hidden'};
      const x = Math.min(Math.max(r.left + r.width / 2, 1), window.innerWidth - 1);
      const y = Math.min(Math.max(r.top + r.height / 2, 1), window.innerHeight - 1);
      const hit = deepAt(x, y);
      const info = describe(el);
      return {x, y, covered: !(hit && (within(hit, el) || within(el, hit))),
              tag: el.tagName.toLowerCase(), type: (el.type || '').toLowerCase(),
              role: info.role, name: info.name, disabled: !!info.disabled,
              editable: el.isContentEditable || ['input', 'textarea'].includes(el.tagName.toLowerCase())};
    },
    click(ref) { const el = get(ref); if (!el) return false; el.click(); return true; },
    focus(ref, selectAll) {
      const el = get(ref);
      if (!el) return {error: 'gone'};
      el.focus();
      if (selectAll) {
        try {
          if (typeof el.select === 'function') el.select();
          else document.execCommand('selectAll', false, null);
        } catch (e) {
          try { el.value = ''; el.dispatchEvent(new Event('input', {bubbles: true})); } catch (e2) {}
        }
      }
      return {ok: document.activeElement === el || within(document.activeElement, el) ||
                  (el.getRootNode().activeElement === el)};
    },
    choose(ref, wanted) {
      const el = get(ref);
      if (!el) return {error: 'gone'};
      if (el.tagName.toLowerCase() !== 'select') return {error: 'notselect'};
      const want = String(wanted).trim().toLowerCase();
      const options = Array.from(el.options);
      const match = options.find((o) => o.text.trim().toLowerCase() === want || o.value.toLowerCase() === want)
        || options.find((o) => o.text.toLowerCase().includes(want));
      if (!match) return {error: 'nooption', options: options.slice(0, 30).map((o) => clean(o.text, 40))};
      if (match.disabled) return {error: 'disabled'};
      el.value = match.value;
      el.dispatchEvent(new Event('input', {bubbles: true}));
      el.dispatchEvent(new Event('change', {bubbles: true}));
      return {chosen: clean(match.text, 60)};
    },
  };
})();
"""


def find_browser(executable: str | None = None) -> str:
    """The path of a Chrome, Chromium or Edge on this machine."""
    for candidate in filter(None, [executable, *(os.environ.get(v) for v in BROWSER_ENV)]):
        found = shutil.which(candidate) or (candidate if Path(candidate).is_file() else "")
        if found:
            return found
        raise ConfigurationError(f"no browser at {candidate!r}")
    for name in _NAMES:
        found = shutil.which(name)
        if found:
            return found
    for path in _PATHS:
        if Path(path).is_file():
            return path
    raise ConfigurationError(
        "no Chrome, Chromium or Edge was found. Install one, name it with "
        f"executable= or ${BROWSER_ENV[0]}, or pass cdp_url= for a browser that is "
        "running somewhere else")


def _stop_process(proc: subprocess.Popen[bytes] | None, profile: str | None) -> None:
    """Kill the browser and remove its profile. Safe to call twice, and at exit."""
    if proc is not None:
        if proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=1)
            except Exception:  # noqa: S110 - it is killed just below
                pass
        # The whole group, and even when the browser itself has gone: it is a
        # dozen processes, and its helpers would go on writing to the profile.
        try:
            if hasattr(os, "killpg") and sys.platform != "win32":
                os.killpg(proc.pid, signal.SIGKILL)
            elif proc.poll() is None:
                proc.kill()
        except OSError:
            pass
        try:
            proc.wait(timeout=3)
        except Exception:  # noqa: S110 - nothing more can be done for it
            pass
    for _ in range(10 if profile else 0):
        shutil.rmtree(profile, ignore_errors=True)
        time.sleep(0.05)
        if not os.path.exists(profile):
            break


@dataclass
class _Page:
    target: str
    session: str
    url: str = ""
    title: str = ""
    #: The status of the last document this tab loaded.
    status: int = 0
    loaded: asyncio.Event = field(default_factory=asyncio.Event)
    crashed: bool = False


class Browser:
    """A browser for an agent: its tabs, its policy, and the tools to drive it.

    ``headless``          no window (the default). False shows the browser.
    ``executable``        the browser to start; found on its own when not given
    ``cdp_url``           drive a browser already running — `http://host:9222`
                          or its `ws://` address — instead of starting one
    ``viewport``          the size of the page, in pixels
    ``allowed_domains`` / ``blocked_domains``  where it may go. A plain name
                          covers its subdomains; globs work.
    ``allow_private``     let it open localhost and private addresses
    ``timeout``           seconds a page may take to load
    ``settle``            seconds to let a page react to an action before it is
                          read again
    ``accept_dialogs``    answer a page's confirm() and prompt() with yes
    ``user_data_dir``     a profile to keep — cookies and logins survive. By
                          default each browser has a fresh one, removed on close.
    ``sandbox``           Chrome's own process sandbox. Off automatically when
                          running as root, as in most containers.
    """

    def __init__(
        self,
        *,
        headless: bool = True,
        executable: str | None = None,
        cdp_url: str | None = None,
        viewport: tuple[int, int] = (1280, 800),
        allowed_domains: Iterable[str] | None = None,
        blocked_domains: Iterable[str] | None = None,
        allow_private: bool = False,
        timeout: float = 30.0,
        settle: float = 0.3,
        accept_dialogs: bool = False,
        user_data_dir: str | None = None,
        user_agent: str | None = None,
        proxy: str | None = None,
        sandbox: bool | None = None,
        args: Iterable[str] = (),
        max_elements: int = 120,
        text_chars: int = 1500,
        max_tabs: int = 8,
    ) -> None:
        if isinstance(allowed_domains, str) or isinstance(blocked_domains, str):
            raise ConfigurationError("allowed_domains and blocked_domains are lists")
        width, height = (int(v) for v in viewport)
        if not (200 <= width <= 4096 and 200 <= height <= 4096):
            raise ConfigurationError(f"a {width}x{height} viewport is not usable")
        self.headless = headless
        self.executable = executable
        self.cdp_url = cdp_url
        self.viewport = (width, height)
        self.allowed = [d for d in map(_domain, allowed_domains or ()) if d]
        self.blocked = [d for d in map(_domain, blocked_domains or ()) if d]
        self.allow_private = allow_private
        self.timeout = float(timeout)
        self.settle = max(0.0, float(settle))
        self.accept_dialogs = accept_dialogs
        self.user_data_dir = user_data_dir
        self.user_agent = user_agent
        self.proxy = proxy
        self.sandbox = sandbox
        self.args = list(args)
        self.max_elements = max(10, int(max_elements))
        self.text_chars = max(0, int(text_chars))
        self.max_tabs = max(1, int(max_tabs))

        self._cdp: CDP | None = None
        self._proc: subprocess.Popen[bytes] | None = None
        self._profile: str | None = None
        self._finalizer: Any = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._lock: asyncio.Lock | None = None
        self._pages: dict[str, _Page] = {}
        self._active = ""
        self._notes: list[str] = []
        self._tasks: set[asyncio.Task[Any]] = set()
        self._resolved: dict[str, bool] = {}
        self._starts = 0

    # ------------------------------------------------------------------
    # starting and stopping
    # ------------------------------------------------------------------
    @property
    def running(self) -> bool:
        return self._cdp is not None and not self._cdp.closed

    async def start(self) -> Browser:
        """Start the browser, or connect to the one named. Called on first use."""
        async with self._locked():
            await self._ensure()
        return self

    async def _ensure(self) -> None:
        if self.running and (self._proc is None or self._proc.poll() is None):
            return
        restarted = self._starts > 0
        self._discard()
        url = await (self._attach_url() if self.cdp_url else self._launch())
        try:
            self._cdp = await CDP.connect(url, timeout=15)
        except Exception as exc:
            self._discard()
            raise ToolError(f"could not connect to the browser: {exc}",
                            tool="browser") from None
        self._starts += 1
        self._cdp.on(self._event)
        try:
            if not self.cdp_url:
                await self._cdp.call("Browser.setDownloadBehavior", {"behavior": "deny"})
            found = await self._cdp.call("Target.getTargets")
            first = next((t for t in found.get("targetInfos", [])
                          if t.get("type") == "page"), None)
            target = first["targetId"] if first else (await self._cdp.call(
                "Target.createTarget", {"url": "about:blank"}))["targetId"]
            await self._adopt(target, activate=True)
            # From here on, a tab that one of ours opens is taken in as well.
            await self._cdp.call("Target.setDiscoverTargets", {"discover": True})
        except CDPError as exc:
            self._discard()
            raise ToolError(f"the browser could not be set up: {exc}",
                            tool="browser") from None
        if restarted:
            self._notes.append("the browser had stopped and was started again; the "
                               "pages that were open are gone")

    async def _attach_url(self) -> str:
        url = str(self.cdp_url)
        if url.startswith(("ws://", "wss://")):
            return url
        import httpx

        try:
            async with httpx.AsyncClient(timeout=10) as client:
                answer = await client.get(url.rstrip("/") + "/json/version")
            return answer.json()["webSocketDebuggerUrl"]
        except Exception as exc:
            raise ToolError(f"no browser is answering at {url} ({type(exc).__name__})",
                            tool="browser") from None

    async def _launch(self) -> str:
        binary = find_browser(self.executable)
        profile = self.user_data_dir or tempfile.mkdtemp(prefix="agent-harness-browser-")
        Path(profile).mkdir(parents=True, exist_ok=True)
        port_file = Path(profile) / "DevToolsActivePort"
        port_file.unlink(missing_ok=True)
        no_sandbox = (self.sandbox is False or (
            self.sandbox is None and hasattr(os, "geteuid") and os.geteuid() == 0))
        argv = [
            binary, "--remote-debugging-port=0", f"--user-data-dir={profile}",
            "--no-first-run", "--no-default-browser-check", "--disable-sync",
            "--disable-background-networking", "--disable-popup-blocking",
            "--disable-features=Translate,MediaRouter", "--mute-audio",
            "--password-store=basic", "--use-mock-keychain",
            f"--window-size={self.viewport[0]},{self.viewport[1]}",
        ]
        if self.headless:
            argv.append("--headless=new")
        if no_sandbox:
            argv += ["--no-sandbox", "--disable-dev-shm-usage"]
        if self.proxy:
            argv.append(f"--proxy-server={self.proxy}")
        argv += [*self.args, "about:blank"]
        log = Path(profile) / "agent-harness-browser.log"
        try:
            with log.open("wb") as errors:
                proc = subprocess.Popen(  # noqa: S603 - the browser the caller named
                    argv, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                    stderr=errors, start_new_session=sys.platform != "win32")
        except OSError as exc:
            raise ToolError(f"could not start {binary}: {exc}", tool="browser") from None
        self._proc = proc
        self._profile = None if self.user_data_dir else profile
        # Whatever happens to this object, the browser does not outlive it.
        self._finalizer = weakref.finalize(self, _stop_process, proc, self._profile)

        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                break
            try:
                lines = port_file.read_text().split("\n")
                if len(lines) >= 2 and lines[0].strip().isdigit() and lines[1].strip():
                    return f"ws://127.0.0.1:{lines[0].strip()}{lines[1].strip()}"
            except OSError:
                pass
            await asyncio.sleep(0.05)
        said = ""
        try:
            said = " ".join(log.read_text(errors="replace").split())[-400:]
        except OSError:
            pass
        code = proc.poll()
        self._discard()
        hint = (" — its sandbox cannot run here; pass sandbox=False"
                if "sandbox" in said.lower() else "")
        raise ToolError(
            f"the browser {'exited' if code is not None else 'did not come up'}"
            f"{f' with code {code}' if code is not None else ''}{hint}"
            + (f": {said}" if said else ""), tool="browser")

    def _discard(self) -> None:
        """Forget the browser there was. Kills the process if it was ours."""
        for task in self._tasks:
            task.cancel()
        self._tasks.clear()
        cdp, self._cdp = self._cdp, None
        if cdp is not None:
            cdp.closed = True
            cdp._reader.cancel()
            try:
                cdp._ws._writer.close()
            except Exception:  # noqa: S110 - the loop it belonged to may be gone
                pass
        if self._finalizer is not None:
            self._finalizer()
            self._finalizer = None
        self._proc = None
        self._profile = None
        self._pages.clear()
        self._active = ""

    async def aclose(self) -> None:
        """Close the browser. A browser that was attached to is left running."""
        cdp, proc = self._cdp, self._proc
        if cdp is not None and not cdp.closed and self._loop is asyncio.get_running_loop():
            if proc is not None:
                try:
                    await cdp.call("Browser.close", timeout=3)
                except CDPError:
                    pass
            await cdp.close()
            if proc is not None:
                try:
                    await asyncio.to_thread(proc.wait, 3)
                except Exception:  # noqa: S110 - it is killed just below
                    pass
        self._discard()

    async def __aenter__(self) -> Browser:
        return await self.start()

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    # ------------------------------------------------------------------
    # events
    # ------------------------------------------------------------------
    def _spawn(self, coro: Any) -> None:
        task = asyncio.get_running_loop().create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._finished)

    def _finished(self, task: asyncio.Task[Any]) -> None:
        self._tasks.discard(task)
        if not task.cancelled():
            task.exception()            # a background step that failed is not fatal

    def _by_session(self, session: str) -> _Page | None:
        return next((p for p in self._pages.values() if p.session == session), None)

    def _event(self, method: str, params: dict[str, Any], session: str) -> None:
        if method == "Fetch.requestPaused":
            self._spawn(self._gate(params, session))
        elif method == "Page.loadEventFired":
            page = self._by_session(session)
            if page:
                page.loaded.set()
        elif method == "Page.javascriptDialogOpening":
            self._spawn(self._dialog(params, session))
        elif method == "Target.targetCreated":
            info = params.get("targetInfo", {})
            # Only a tab one of ours opened: a browser that was attached to has
            # tabs of its own, and those are not this agent's to touch.
            if (info.get("type") == "page" and info.get("targetId") not in self._pages
                    and info.get("openerId") in self._pages):
                self._spawn(self._adopt(info["targetId"], activate=True, opened=True))
        elif method == "Target.targetInfoChanged":
            info = params.get("targetInfo", {})
            page = self._pages.get(info.get("targetId", ""))
            if page:
                page.url, page.title = info.get("url", page.url), info.get("title", page.title)
        elif method == "Target.targetDestroyed":
            target = params.get("targetId", "")
            if self._pages.pop(target, None) and self._active == target:
                self._active = next(reversed(self._pages), "")
        elif method in ("Inspector.targetCrashed", "Target.targetCrashed"):
            page = self._by_session(session) or self._pages.get(params.get("targetId", ""))
            if page:
                page.crashed = True
        elif method == "Browser.downloadWillBegin":
            self._notes.append(f"a download of {params.get('suggestedFilename', 'a file')} "
                               "was not allowed")

    async def _adopt(self, target: str, *, activate: bool, opened: bool = False) -> None:
        """Attach to a tab and set it up: its size, its policy, its events."""
        cdp = self._cdp
        if cdp is None or target in self._pages:
            return
        if opened and len(self._pages) >= self.max_tabs:
            await cdp.call("Target.closeTarget", {"targetId": target})
            self._notes.append(f"a new tab was closed: {self.max_tabs} are open already")
            return
        try:
            session = (await cdp.call("Target.attachToTarget",
                                      {"targetId": target, "flatten": True}))["sessionId"]
        except CDPError:
            return
        if target in self._pages:
            return
        page = _Page(target=target, session=session)
        self._pages[target] = page
        width, height = self.viewport
        await cdp.call("Page.enable", session=session)
        await cdp.call("Fetch.enable", {"patterns": [
            {"urlPattern": "*", "resourceType": "Document", "requestStage": "Request"},
            {"urlPattern": "*", "resourceType": "Document", "requestStage": "Response"},
        ]}, session=session)
        await cdp.call("Emulation.setDeviceMetricsOverride", {
            "width": width, "height": height, "deviceScaleFactor": 1, "mobile": False,
        }, session=session)
        if self.user_agent:
            await cdp.call("Emulation.setUserAgentOverride",
                           {"userAgent": self.user_agent}, session=session)
        # A tab opened by a page was waiting for someone to attach; let it go.
        try:
            await cdp.call("Runtime.runIfWaitingForDebugger", session=session)
        except CDPError:
            pass
        if activate or not self._active:
            self._active = target
            if opened:
                self._notes.append("a new tab opened and is now the one in use")

    async def _dialog(self, params: dict[str, Any], session: str) -> None:
        kind = params.get("type", "alert")
        accept = kind in ("alert", "beforeunload") or self.accept_dialogs
        message = " ".join(str(params.get("message", "")).split())[:300]
        self._notes.append(
            f"the page showed a {kind} dialog: {message!r} — "
            + ("it was accepted" if accept else "it was dismissed"))
        if self._cdp is not None:
            try:
                await self._cdp.call("Page.handleJavaScriptDialog", {"accept": accept},
                                     session=session)
            except CDPError:
                pass

    # ------------------------------------------------------------------
    # where it may go
    # ------------------------------------------------------------------
    async def refusal(self, url: str) -> str:
        """Why this address may not be opened; empty when it may."""
        try:
            parts = urlsplit(url)
        except ValueError:
            return "that is not a web address"
        scheme = parts.scheme.lower()
        if scheme in ("about", "data", "blob"):
            return ""
        if scheme not in ("http", "https"):
            return "only http and https pages can be opened"
        host = (parts.hostname or "").lower().rstrip(".")
        if not host:
            return "that is not a web address"
        if any(_matches(host, b) for b in self.blocked):
            return f"{host} is on the blocked list"
        if self.allowed and not any(_matches(host, a) for a in self.allowed):
            return f"{host} is not on the allowed list"
        if not self.allow_private and await is_private_host(host, self._resolved):
            return f"{host} is a private or loopback address"
        return ""

    async def _gate(self, params: dict[str, Any], session: str) -> None:
        """Every document a tab asks for stops here, and is let through or not."""
        cdp = self._cdp
        if cdp is None:
            return
        request = params.get("requestId")
        url = params.get("request", {}).get("url", "")
        page = self._by_session(session)
        try:
            if "responseStatusCode" in params or "responseErrorReason" in params:
                if page and params.get("frameId") == page.target:
                    page.status = int(params.get("responseStatusCode") or 0)
                await cdp.call("Fetch.continueRequest", {"requestId": request},
                               session=session)
                return
            reason = await self.refusal(url)
            if reason:
                self._notes.append(f"{url[:200]} was not opened: {reason}")
                await cdp.call("Fetch.failRequest", {
                    "requestId": request, "errorReason": "BlockedByClient"}, session=session)
            else:
                await cdp.call("Fetch.continueRequest", {"requestId": request},
                               session=session)
        except CDPError:
            pass
        except Exception:
            # Never leave a request hanging: a check that broke refuses the page.
            try:
                await cdp.call("Fetch.failRequest", {
                    "requestId": request, "errorReason": "Failed"}, session=session)
            except CDPError:
                pass

    # ------------------------------------------------------------------
    # talking to the page
    # ------------------------------------------------------------------
    def _page(self) -> _Page:
        page = self._pages.get(self._active)
        if page is None:
            raise ToolError("no tab is open — open a page first", tool="browser")
        return page

    async def _send(self, method: str, params: dict[str, Any] | None = None, *,
                    page: _Page | None = None, timeout: float = 30.0) -> dict[str, Any]:
        if self._cdp is None:
            raise ToolError("the browser is not running", tool="browser")
        page = page or self._page()
        try:
            return await self._cdp.call(method, params, session=page.session,
                                        timeout=timeout)
        except CDPError as exc:
            if exc.gone:
                raise ToolError("the browser stopped — try again and it will be "
                                "started afresh", tool="browser") from None
            raise

    async def _eval(self, expression: str, *, timeout: float = 15.0) -> Any:
        """Run JavaScript in the page in use and return its value."""
        answer = await self._send("Runtime.evaluate", {
            "expression": f"{_PAGE_JS}\n{expression}", "returnByValue": True,
            "awaitPromise": True, "userGesture": True}, timeout=timeout)
        problem = answer.get("exceptionDetails")
        if problem:
            text = (problem.get("exception", {}).get("description")
                    or problem.get("text") or "the page raised an error")
            raise ToolError(f"the page could not do that: {text.splitlines()[0][:200]}",
                            tool="browser")
        return answer.get("result", {}).get("value")

    async def _settle(self, *, limit: float = 8.0) -> None:
        """Wait for whatever the last action set off to finish loading."""
        await asyncio.sleep(self.settle)
        deadline = time.monotonic() + limit
        while time.monotonic() < deadline:
            try:
                if await self._eval("document.readyState", timeout=3) == "complete":
                    return
            except (ToolError, CDPError):
                pass                    # between two pages; ask again
            await asyncio.sleep(0.15)

    async def _guarded(self) -> None:
        """Before every action: a browser that is up, and a tab that is alive."""
        await self._ensure()
        page = self._pages.get(self._active)
        if page is not None and page.crashed:
            page.crashed = False
            self._notes.append("the tab had crashed and was reloaded")
            try:
                await self._send("Page.reload", page=page)
                await self._settle()
            except CDPError:
                pass

    def _locked(self) -> asyncio.Lock:
        loop = asyncio.get_running_loop()
        if self._loop is not loop or self._lock is None:
            # Left over from an event loop that has ended: nothing of it is usable.
            self._discard()
            self._loop, self._lock = loop, asyncio.Lock()
        return self._lock

    # ------------------------------------------------------------------
    # what the agent reads
    # ------------------------------------------------------------------
    async def _observe(self) -> str:
        try:
            snap = await self._eval(
                f"window.__ah.snapshot({self.max_elements}, {self.text_chars})")
        except (ToolError, CDPError) as exc:
            page = self._page()
            return f"Page: {page.title or '(untitled)'} — {page.url}\n(the page could not be read: {exc})"
        page = self._page()
        page.url, page.title = snap["url"], snap["title"]
        lines = [f"Page: {snap['title'] or '(untitled)'} — {snap['url']}"]
        if page.status >= 400:
            lines.append(f"Status: {page.status}")
        if len(self._pages) > 1:
            lines.append(f"Tabs: {len(self._pages)} open — browser_tabs lists them")
        top, view, total = snap["scrollY"], snap["viewHeight"], snap["pageHeight"]
        if total > view + 8:
            lines.append(f"Scrolled: {top}–{min(top + view, total)} of {total}px")
        for note in self._drain():
            lines.append(f"Note: {note}")
        lines.append("")
        section = ""
        for item in snap["elements"]:
            where = item.get("where", "")
            if where != section:
                section = where
                lines += [*([""] if lines[-1] else []),
                          f"Further {'down' if where == 'below' else 'up'} the page "
                          "(usable without scrolling to them):"]
            lines.append(_line(item))
        if not snap["elements"]:
            lines.append("(nothing on this page can be clicked or typed into)")
        if snap["more"]:
            lines.append(f"… and {snap['more']} more elements than are listed")
        if snap["text"]:
            more = snap["textLength"] - len(snap["text"])
            lines += ["", "Text on screen:", snap["text"]]
            if more > 200:
                lines.append(f"[the page has about {more} more characters of text — "
                             "browser_read has all of it]")
        return "\n".join(lines)

    def _drain(self) -> list[str]:
        notes, self._notes = list(dict.fromkeys(self._notes)), []
        return notes

    async def _after(self, did: str) -> str:
        await self._settle()
        return f"{did}\n\n{await self._observe()}"

    # ------------------------------------------------------------------
    # actions
    # ------------------------------------------------------------------
    async def goto(self, url: str) -> str:
        """Open a page in the tab in use."""
        url = str(url or "").strip()
        if not url:
            raise ToolError("no address was given", tool="browser")
        if "://" not in url and not url.startswith(("about:", "data:")):
            url = "https://" + url
        async with self._locked():
            await self._guarded()
            if not url.startswith(("http://", "https://")) and url != "about:blank":
                raise ToolError("only http and https pages can be opened", tool="browser")
            reason = await self.refusal(url)
            if reason:
                raise ToolError(f"{url} was not opened: {reason}", tool="browser")
            page = self._page()
            page.loaded.clear()
            page.status = 0
            answer = await self._send("Page.navigate", {"url": url}, timeout=self.timeout)
            error = answer.get("errorText")
            if error:
                notes = self._drain()
                why = next((n.split(": ", 1)[-1] for n in notes if "was not opened" in n), "")
                raise ToolError(f"{url} could not be opened: {why or error}", tool="browser")
            slow = ""
            try:
                await asyncio.wait_for(page.loaded.wait(), self.timeout)
            except _Timeout:
                slow = f" (still loading after {self.timeout:g}s)"
            return await self._after(f"Opened {url}{slow}")

    async def snapshot(self) -> str:
        """The page as it is now: what can be acted on, and what it says."""
        async with self._locked():
            await self._guarded()
            return await self._observe()

    async def _locate(self, ref: int) -> dict[str, Any]:
        try:
            number = int(ref)
        except (TypeError, ValueError):
            raise ToolError(f"ref must be a number from the page listing — got {ref!r}",
                            tool="browser") from None
        found = await self._eval(f"window.__ah.locate({number})")
        if not found or found.get("error") == "gone":
            raise ToolError(f"there is no element [{number}] on this page any more — "
                            "read the page again for the current numbers", tool="browser")
        if found.get("error") == "hidden":
            raise ToolError(f"element [{number}] is not visible", tool="browser")
        found["ref"] = number
        return found

    async def _mouse(self, kind: str, x: float, y: float, *, button: str = "none",
                     count: int = 0, buttons: int = 0, **extra: Any) -> None:
        await self._send("Input.dispatchMouseEvent", {
            "type": kind, "x": x, "y": y, "button": button, "buttons": buttons,
            "clickCount": count, **extra})

    async def click_at(self, x: float, y: float, *, button: str = "left",
                       count: int = 1) -> None:
        mask = {"left": 1, "right": 2, "middle": 4}.get(button, 1)
        await self._mouse("mouseMoved", x, y)
        for n in range(1, count + 1):
            await self._mouse("mousePressed", x, y, button=button, count=n, buttons=mask)
            await self._mouse("mouseReleased", x, y, button=button, count=n)

    async def click(self, ref: int) -> str:
        """Click the element with this number."""
        async with self._locked():
            await self._guarded()
            target = await self._locate(ref)
            label = _label(target)
            if target["disabled"]:
                raise ToolError(f"{label} is disabled", tool="browser")
            if target["tag"] == "select":
                raise ToolError(f"{label} is a list of options — choose one with "
                                "browser_select", tool="browser")
            if target["type"] == "file":
                raise ToolError(f"{label} takes a file, which this browser cannot "
                                "give it", tool="browser")
            if target["covered"]:
                # Something lies over it — a banner, a label. Click it directly.
                await self._eval(f"window.__ah.click({target['ref']})")
            else:
                await self.click_at(target["x"], target["y"])
            return await self._after(f"Clicked {label}")

    async def type(self, ref: int, text: str, *, submit: bool = False,
                   clear: bool = True) -> str:
        """Type into the element with this number, replacing what is there."""
        async with self._locked():
            await self._guarded()
            target = await self._locate(ref)
            label = _label(target)
            if target["disabled"]:
                raise ToolError(f"{label} is disabled", tool="browser")
            if not target["editable"] and target["role"] not in (
                    "textbox", "searchbox", "combobox"):
                raise ToolError(f"{label} is not something that can be typed into",
                                tool="browser")
            if not target["covered"]:
                await self.click_at(target["x"], target["y"])
            await self._eval(f"window.__ah.focus({target['ref']}, {json.dumps(clear)})")
            if text:
                await self._send("Input.insertText", {"text": str(text)})
            elif clear:
                await self._key("Delete")
            if submit:
                await self._key("Enter")
            shown = "•••" if target["type"] == "password" else repr(str(text)[:80])
            return await self._after(
                f"Typed {shown} into {label}" + (" and pressed Enter" if submit else ""))

    async def select(self, ref: int, option: str) -> str:
        """Choose an option of a drop-down list, by its text."""
        async with self._locked():
            await self._guarded()
            target = await self._locate(ref)
            answer = await self._eval(
                f"window.__ah.choose({target['ref']}, {json.dumps(str(option))})")
            error = answer.get("error")
            if error == "notselect":
                raise ToolError(f"{_label(target)} is not a list of options — click it "
                                "instead", tool="browser")
            if error == "nooption":
                raise ToolError(f"{_label(target)} has no option {option!r}; it has: "
                                + " | ".join(answer.get("options", [])), tool="browser")
            if error:
                raise ToolError(f"that option cannot be chosen ({error})", tool="browser")
            return await self._after(f"Chose {answer['chosen']!r} in {_label(target)}")

    async def _key(self, combo: str) -> str:
        combo = str(combo).strip()
        plus = combo == "+" or combo.endswith("++")
        parts = [p for p in combo.replace(" ", "").split("+") if p]
        *held, name = [*parts, "+"] if plus else (parts or [""])
        modifiers = 0
        for mod in held:
            if mod.lower() not in _MODIFIERS:
                raise ToolError(f"{mod!r} is not a modifier key; there are ctrl, shift, "
                                "alt and meta", tool="browser")
            modifiers |= _MODIFIERS[mod.lower()]
        known = _KEYS.get(name.lower())
        if known:
            key, code, number, text = known
        elif len(name) == 1:
            if modifiers & 8 and name.isalpha():
                name = name.upper()
            key, text = name, name
            number = ord(name.upper()) if name.isalnum() else 0
            code = (f"Key{name.upper()}" if name.isalpha() else
                    f"Digit{name}" if name.isdigit() else "")
        else:
            raise ToolError(f"{name!r} is not a key. Use a character, or one of Enter, "
                            "Tab, Escape, Backspace, Delete, Space, ArrowUp/Down/Left/"
                            "Right, Home, End, PageUp, PageDown, F1–F12", tool="browser")
        event: dict[str, Any] = {"key": key, "code": code, "modifiers": modifiers,
                                 "windowsVirtualKeyCode": number,
                                 "nativeVirtualKeyCode": number}
        chord = modifiers & ~8          # anything held besides shift
        if chord and name.lower() in _COMMANDS and modifiers & (2 | 4):
            event["commands"] = [_COMMANDS[name.lower()]]
        down = {**event, "type": "keyDown" if text and not chord else "rawKeyDown"}
        if text and not chord:
            down["text"] = text
        await self._send("Input.dispatchKeyEvent", down)
        await self._send("Input.dispatchKeyEvent", {**event, "type": "keyUp"})
        return "+".join([*held, key if key != " " else "Space"])

    async def press(self, key: str) -> str:
        """Press a key, or a combination such as `ctrl+a`."""
        async with self._locked():
            await self._guarded()
            return await self._after(f"Pressed {await self._key(key)}")

    async def type_text(self, text: str) -> None:
        """Type into whatever has the focus. For `Browser.computer()`."""
        await self._send("Input.insertText", {"text": str(text)})

    async def wheel(self, x: float, y: float, dx: float, dy: float) -> None:
        await self._mouse("mouseWheel", x, y, deltaX=dx, deltaY=dy)

    async def scroll(self, direction: str = "down", amount: float = 1.0) -> str:
        """Scroll the page: up, down, left, right, top or bottom."""
        direction = str(direction or "down").strip().lower()
        if direction not in ("up", "down", "left", "right", "top", "bottom"):
            raise ToolError("direction is one of up, down, left, right, top, bottom",
                            tool="browser")
        async with self._locked():
            await self._guarded()
            if direction in ("top", "bottom"):
                await self._eval("window.scrollTo(0, %s)" % (
                    "0" if direction == "top" else "document.documentElement.scrollHeight"))
            else:
                width, height = self.viewport
                pages = max(0.1, min(float(amount or 1), 20.0))
                step = (height if direction in ("up", "down") else width) * 0.85 * pages
                sign = -1 if direction in ("up", "left") else 1
                dx, dy = (0, sign * step) if direction in ("up", "down") else (sign * step, 0)
                await self.wheel(width / 2, height / 2, dx, dy)
            await asyncio.sleep(0.35)
            return await self._after(f"Scrolled {direction}")

    async def back(self) -> str:
        return await self._history(-1)

    async def forward(self) -> str:
        return await self._history(1)

    async def _history(self, step: int) -> str:
        async with self._locked():
            await self._guarded()
            history = await self._send("Page.getNavigationHistory")
            index = history.get("currentIndex", 0) + step
            entries = history.get("entries", [])
            if not 0 <= index < len(entries):
                raise ToolError("there is no page to go "
                                + ("back" if step < 0 else "forward") + " to",
                                tool="browser")
            await self._send("Page.navigateToHistoryEntry",
                             {"entryId": entries[index]["id"]})
            return await self._after("Went " + ("back" if step < 0 else "forward"))

    async def read(self, offset: int = 0, max_chars: int = 8000) -> str:
        """The text of the page, a stretch at a time."""
        async with self._locked():
            await self._guarded()
            text = await self._eval("window.__ah.text()") or ""
            offset = max(0, int(offset or 0))
            size = max(200, min(int(max_chars or 8000), 40_000))
            piece = text[offset:offset + size]
            left = len(text) - offset - len(piece)
            if not piece:
                return ("(the page has no text)" if not text else
                        f"(nothing past character {len(text)})")
            return piece + (f"\n… [{left} more characters — read on from "
                            f"offset={offset + len(piece)}]" if left > 0 else "")

    async def wait(self, seconds: float = 1.0, text: str = "") -> str:
        """Wait a moment — or, given `text`, until the page shows it."""
        limit = max(0.0, min(float(seconds or 0), 30.0))
        async with self._locked():
            await self._guarded()
            if not text:
                await asyncio.sleep(limit)
                return await self._after(f"Waited {limit:g}s")
            deadline = time.monotonic() + (limit or 10.0)
            wanted = json.dumps(str(text).lower())
            while True:
                try:
                    if await self._eval(
                            f"document.body.innerText.toLowerCase().includes({wanted})"):
                        return await self._after(f"The page now shows {text!r}")
                except (ToolError, CDPError):
                    pass
                if time.monotonic() >= deadline:
                    return await self._after(f"{text!r} did not appear within "
                                             f"{limit or 10:g}s")
                await asyncio.sleep(0.25)

    async def screenshot(self, *, format: str = "jpeg", quality: int = 70) -> ImageBlock:
        """A picture of the page as it is shown."""
        async with self._locked():
            await self._guarded()
            return await self._shot(format=format, quality=quality)

    async def _shot(self, *, format: str = "jpeg", quality: int = 70) -> ImageBlock:
        params: dict[str, Any] = {"format": "png" if format == "png" else "jpeg"}
        if params["format"] == "jpeg":
            params["quality"] = max(10, min(int(quality), 100))
        answer = await self._send("Page.captureScreenshot", params, timeout=20)
        return ImageBlock(data=answer["data"], media_type=f"image/{params['format']}",
                          name="screenshot")

    # ---- tabs ------------------------------------------------------------------
    async def tabs(self) -> str:
        async with self._locked():
            await self._guarded()
            return self._tab_list()

    def _tab_list(self) -> str:
        return "\n".join(
            f"[{n}] {p.title or '(untitled)'} — {p.url or 'about:blank'}"
            + ("  (in use)" if p.target == self._active else "")
            for n, p in enumerate(self._pages.values()))

    def _tab(self, index: int) -> _Page:
        pages = list(self._pages.values())
        try:
            return pages[int(index)]
        except (IndexError, TypeError, ValueError):
            raise ToolError(f"there is no tab {index!r}; the tabs are:\n"
                            f"{self._tab_list()}", tool="browser") from None

    async def switch_tab(self, index: int) -> str:
        async with self._locked():
            await self._guarded()
            page = self._tab(index)
            self._active = page.target
            assert self._cdp is not None
            await self._cdp.call("Target.activateTarget", {"targetId": page.target})
            return await self._after(f"Switched to tab {index}")

    async def new_tab(self, url: str = "") -> str:
        async with self._locked():
            await self._guarded()
            if len(self._pages) >= self.max_tabs:
                raise ToolError(f"{self.max_tabs} tabs are open already — close one first",
                                tool="browser")
            assert self._cdp is not None
            target = (await self._cdp.call("Target.createTarget",
                                           {"url": "about:blank"}))["targetId"]
            await self._adopt(target, activate=True)
            self._active = target
        return await self.goto(url) if url else "Opened a new tab"

    async def close_tab(self, index: int | None = None) -> str:
        async with self._locked():
            await self._guarded()
            page = self._tab(index) if index is not None else self._page()
            if len(self._pages) == 1:
                raise ToolError("that is the only tab — open another page in it instead",
                                tool="browser")
            assert self._cdp is not None
            await self._cdp.call("Target.closeTarget", {"targetId": page.target})
            self._pages.pop(page.target, None)
            if self._active == page.target:
                self._active = next(reversed(self._pages))
            return await self._after("Closed the tab")

    # ------------------------------------------------------------------
    # as tools
    # ------------------------------------------------------------------
    def tools(self, **options: Any) -> list[Any]:
        """The tools an agent drives this browser with. See `browser_tools`."""
        from .tools import browser_tools

        return browser_tools(self, **options)

    def computer(self) -> Any:
        """This browser as a screen with a mouse and a keyboard, for a model that sees."""
        from .computer import BrowserComputer

        return BrowserComputer(self)

    def __repr__(self) -> str:  # pragma: no cover - debugging affordance
        state = "running" if self.running else "not started"
        return f"<Browser {state}, {len(self._pages)} tabs>"


def _label(target: dict[str, Any]) -> str:
    name = f" {target['name']!r}" if target.get("name") else ""
    return f"[{target['ref']}] {target.get('role') or target.get('tag')}{name}"


def _line(item: dict[str, Any]) -> str:
    """One element of the page, as the model reads it."""
    name = json.dumps(item["name"], ensure_ascii=False) if item.get("name") else ""
    line = f"[{item['ref']}] {item['role']}" + (f" {name}" if name else "")
    if item.get("value"):
        line += f" value={json.dumps(item['value'], ensure_ascii=False)}"
    flags = [flag for flag, on in (("checked", item.get("checked")),
                                   ("disabled", item.get("disabled")),
                                   ("expanded", item.get("expanded"))) if on]
    if flags:
        line += f" ({', '.join(flags)})"
    if item.get("href"):
        line += f" → {item['href']}"
    if item.get("options"):
        line += "  options: " + " | ".join(item["options"])
    return line
