//! A web browser an agent can use: open pages, read them, click, type, look.
//!
//! ```no_run
//! # async fn demo() -> agent_harness::Result<()> {
//! use agent_harness::{Agent, browser::Browser};
//! let browser = Browser::default();
//! let agent = Agent::builder("shopper").instructions("Find things on the web.").tools(browser.tools()).build()?;
//! agent.run("What is the top story on Hacker News?").await?;
//! browser.close().await;
//! # Ok(()) }
//! ```
//!
//! The browser is Chrome, Chromium or Edge — whichever is installed — started
//! headless and driven over the DevTools protocol, so nothing is added to
//! install. `cdp_url: Some("http://host:9222")` drives one that is already
//! running somewhere else instead: a container, a remote machine, a hosted browser.
//!
//! The agent is not handed pixels to guess at. After every action it reads the
//! page as a list of the things that can be acted on, each with a number:
//!
//! ```text
//! Page: Sign in — https://shop.example/login
//! [1] textbox "Email"
//! [2] textbox "Password"
//! [3] button "Sign in"
//! [4] link "Forgot your password?" → /reset
//! ```
//!
//! and acts by number: `browser_type(ref=1, text=…)`, `browser_click(ref=3)`.
//! Any model can do that, vision or not. `browser_screenshot` is there for a
//! model that can see, and [`Browser::computer`] gives one the mouse and keyboard.
//!
//! Where it may go is yours to say: `allowed_domains`, `blocked_domains`, and
//! no private or loopback address unless `allow_private`. The policy is applied
//! to every page load — a typed address, a clicked link, a redirect, a frame.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, Weak};
use std::time::{Duration, Instant};

use serde_json::{Value, json};
use tokio::sync::watch;
use tokio::task::JoinHandle;

use crate::browser::cdp::Cdp;
use crate::errors::{Error, Result};
use crate::toolkits::net::is_private_host;
use crate::toolkits::openapi::py_repr;
use crate::toolkits::search::{domain, fmt_g, matches};
use crate::types::MediaBlock;

const NAMES: [&str; 9] = ["google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "chrome", "microsoft-edge", "microsoft-edge-stable", "msedge", "brave-browser"];
const PATHS: [&str; 7] = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
];
/// Environment variables that name the browser to start.
pub const BROWSER_ENV: [&str; 2] = ["AGENT_HARNESS_BROWSER", "CHROME_PATH"];

fn modifier(name: &str) -> Option<i64> {
    Some(match name {
        "alt" | "option" => 1,
        "ctrl" | "control" => 2,
        "meta" | "cmd" | "command" | "super" | "win" => 4,
        "shift" => 8,
        _ => return None,
    })
}

/// name → (key, code, virtual key code, text)
fn named_key(name: &str) -> Option<(String, String, i64, &'static str)> {
    let k = |key: &str, code: i64, text: &'static str| Some((key.to_string(), key.to_string(), code, text));
    match name {
        "enter" | "return" => Some(("Enter".into(), "Enter".into(), 13, "\r")),
        "tab" => k("Tab", 9, ""),
        "escape" | "esc" => k("Escape", 27, ""),
        "backspace" => k("Backspace", 8, ""),
        "delete" | "del" => k("Delete", 46, ""),
        "space" => Some((" ".into(), "Space".into(), 32, " ")),
        "arrowup" | "up" => k("ArrowUp", 38, ""),
        "arrowdown" | "down" => k("ArrowDown", 40, ""),
        "arrowleft" | "left" => k("ArrowLeft", 37, ""),
        "arrowright" | "right" => k("ArrowRight", 39, ""),
        "home" => k("Home", 36, ""),
        "end" => k("End", 35, ""),
        "pageup" | "pgup" => k("PageUp", 33, ""),
        "pagedown" | "pgdn" => k("PageDown", 34, ""),
        "insert" => k("Insert", 45, ""),
        _ => {
            let n: i64 = name.strip_prefix('f')?.parse().ok()?;
            (1..=12).contains(&n).then(|| (format!("F{n}"), format!("F{n}"), 111 + n, ""))
        }
    }
}

/// What a shortcut does to the text being edited; the browser needs telling.
fn command(name: &str) -> Option<&'static str> {
    Some(match name {
        "a" => "selectAll",
        "c" => "copy",
        "v" => "paste",
        "x" => "cut",
        "z" => "undo",
        _ => return None,
    })
}

// Everything the page is asked, in one object installed on first use. Numbers
// are kept for as long as the page lives, so a number read earlier still means
// the same element later.
const PAGE_JS: &str = r#"
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
"#;

fn tool_error(message: impl Into<String>) -> Error {
    Error::tool(message, "browser")
}

fn which(name: &str) -> Option<PathBuf> {
    let path = Path::new(name);
    if path.components().count() > 1 {
        return path.is_file().then(|| path.to_path_buf());
    }
    let exts: Vec<String> = if cfg!(windows) {
        std::env::var("PATHEXT").unwrap_or_else(|_| ".EXE;.BAT;.CMD".into()).split(';').map(str::to_lowercase).collect()
    } else {
        vec![String::new()]
    };
    for dir in std::env::split_paths(&std::env::var_os("PATH").unwrap_or_default()) {
        for ext in &exts {
            let candidate = dir.join(format!("{name}{ext}"));
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    None
}

/// The path of a Chrome, Chromium or Edge on this machine.
pub fn find_browser(executable: Option<&str>) -> Result<String> {
    let named: Vec<String> = executable.map(str::to_string).into_iter().chain(BROWSER_ENV.iter().filter_map(|v| std::env::var(v).ok())).filter(|s| !s.is_empty()).collect();
    if let Some(candidate) = named.first() {
        return match which(candidate).or_else(|| Path::new(candidate).is_file().then(|| PathBuf::from(candidate))) {
            Some(found) => Ok(found.to_string_lossy().into_owned()),
            None => Err(Error::config(format!("no browser at {}", py_repr(candidate)))),
        };
    }
    for name in NAMES {
        if let Some(found) = which(name) {
            return Ok(found.to_string_lossy().into_owned());
        }
    }
    for path in PATHS {
        if Path::new(path).is_file() {
            return Ok(path.into());
        }
    }
    Err(Error::config(format!(
        "no Chrome, Chromium or Edge was found. Install one, name it with executable or ${}, or pass cdp_url for a browser that is running somewhere else",
        BROWSER_ENV[0]
    )))
}

/// A browser this harness started: killed, and its profile removed, when dropped.
struct Launched {
    child: std::process::Child,
    profile: Option<PathBuf>,
}

impl Launched {
    fn alive(&mut self) -> bool {
        matches!(self.child.try_wait(), Ok(None))
    }
}

impl Drop for Launched {
    fn drop(&mut self) {
        let child = &mut self.child;
        // The whole group, and even when the browser itself has gone: it is a
        // dozen processes, and its helpers would go on writing to the profile.
        #[cfg(unix)]
        {
            let _ = std::process::Command::new("kill").args(["-KILL", &format!("-{}", child.id())]).stderr(std::process::Stdio::null()).status();
        }
        if matches!(child.try_wait(), Ok(None)) {
            let _ = child.kill();
        }
        let deadline = Instant::now() + Duration::from_secs(3);
        while matches!(child.try_wait(), Ok(None)) && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(20));
        }
        if let Some(profile) = &self.profile {
            for _ in 0..10 {
                let _ = std::fs::remove_dir_all(profile);
                if !profile.exists() {
                    break;
                }
                std::thread::sleep(Duration::from_millis(50));
            }
        }
    }
}

struct Page {
    target: String,
    session: String,
    url: String,
    title: String,
    /// The status of the last document this tab loaded.
    status: i64,
    loaded: watch::Sender<bool>,
    crashed: bool,
}

#[derive(Default)]
struct State {
    cdp: Option<Arc<Cdp>>,
    proc: Option<Launched>,
    /// In the order they were opened.
    pages: Vec<Page>,
    active: String,
    notes: Vec<String>,
    tasks: Vec<JoinHandle<()>>,
    starts: u32,
}

impl State {
    fn page(&self, target: &str) -> Option<&Page> {
        self.pages.iter().find(|p| p.target == target)
    }

    fn page_mut(&mut self, target: &str) -> Option<&mut Page> {
        self.pages.iter_mut().find(|p| p.target == target)
    }

    fn by_session(&mut self, session: &str) -> Option<&mut Page> {
        self.pages.iter_mut().find(|p| p.session == session)
    }

    fn remove(&mut self, target: &str) -> bool {
        let before = self.pages.len();
        self.pages.retain(|p| p.target != target);
        before != self.pages.len()
    }
}

/// How a [`Browser`] behaves.
///
/// `cdp_url` drives a browser already running — `http://host:9222` or its
/// `ws://` address — instead of starting one. `allowed_domains` and
/// `blocked_domains` say where it may go: a plain name covers its subdomains,
/// globs work. `settle` is seconds to let a page react to an action before it
/// is read again. `user_data_dir` is a profile to keep — cookies and logins
/// survive; by default each browser has a fresh one, removed on close.
/// `sandbox` is Chrome's own process sandbox, off automatically when running
/// as root, as in most containers.
#[derive(Debug, Clone)]
pub struct BrowserOptions {
    pub headless: bool,
    pub executable: Option<String>,
    pub cdp_url: Option<String>,
    pub viewport: (u32, u32),
    pub allowed_domains: Vec<String>,
    pub blocked_domains: Vec<String>,
    pub allow_private: bool,
    pub timeout: f64,
    pub settle: f64,
    pub accept_dialogs: bool,
    pub user_data_dir: Option<String>,
    pub user_agent: Option<String>,
    pub proxy: Option<String>,
    pub sandbox: Option<bool>,
    pub args: Vec<String>,
    pub max_elements: usize,
    pub text_chars: usize,
    pub max_tabs: usize,
}

impl Default for BrowserOptions {
    fn default() -> Self {
        Self {
            headless: true,
            executable: None,
            cdp_url: None,
            viewport: (1280, 800),
            allowed_domains: vec![],
            blocked_domains: vec![],
            allow_private: false,
            timeout: 30.0,
            settle: 0.3,
            accept_dialogs: false,
            user_data_dir: None,
            user_agent: None,
            proxy: None,
            sandbox: None,
            args: vec![],
            max_elements: 120,
            text_chars: 1500,
            max_tabs: 8,
        }
    }
}

struct Inner {
    opts: BrowserOptions,
    action: tokio::sync::Mutex<()>,
    state: Mutex<State>,
    resolved: Mutex<HashMap<String, bool>>,
}

/// A browser for an agent: its tabs, its policy, and the tools to drive it.
/// Cheap to clone; every clone is the same browser.
#[derive(Clone)]
pub struct Browser {
    inner: Arc<Inner>,
}

impl Default for Browser {
    fn default() -> Self {
        Self::new(BrowserOptions::default()).expect("the default options are valid")
    }
}

impl std::fmt::Debug for Browser {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let tabs = self.state().pages.len();
        write!(f, "<Browser {}, {tabs} tabs>", if self.running() { "running" } else { "not started" })
    }
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|p| p.into_inner())
}

fn s(v: &Value, key: &str) -> String {
    v.get(key).and_then(Value::as_str).unwrap_or("").to_string()
}

fn num(v: &Value, key: &str) -> i64 {
    v.get(key).and_then(|n| n.as_i64().or_else(|| n.as_f64().map(|f| f as i64))).unwrap_or(0)
}

impl Browser {
    pub fn new(mut opts: BrowserOptions) -> Result<Self> {
        let (width, height) = opts.viewport;
        if !((200..=4096).contains(&width) && (200..=4096).contains(&height)) {
            return Err(Error::config(format!("a {width}x{height} viewport is not usable")));
        }
        opts.allowed_domains = opts.allowed_domains.iter().map(|d| domain(d)).filter(|d| !d.is_empty()).collect();
        opts.blocked_domains = opts.blocked_domains.iter().map(|d| domain(d)).filter(|d| !d.is_empty()).collect();
        opts.settle = opts.settle.max(0.0);
        opts.max_elements = opts.max_elements.max(10);
        opts.max_tabs = opts.max_tabs.max(1);
        Ok(Self { inner: Arc::new(Inner { opts, action: tokio::sync::Mutex::new(()), state: Mutex::default(), resolved: Mutex::default() }) })
    }

    pub fn options(&self) -> &BrowserOptions {
        &self.inner.opts
    }

    pub fn viewport(&self) -> (u32, u32) {
        self.inner.opts.viewport
    }

    fn state(&self) -> std::sync::MutexGuard<'_, State> {
        lock(&self.inner.state)
    }

    pub fn running(&self) -> bool {
        self.state().cdp.as_ref().is_some_and(|c| !c.closed())
    }

    // ------------------------------------------------------------------
    // starting and stopping
    // ------------------------------------------------------------------

    /// Start the browser, or connect to the one named. Called on first use.
    pub async fn start(&self) -> Result<Self> {
        let _held = self.inner.action.lock().await;
        self.ensure().await?;
        Ok(self.clone())
    }

    async fn ensure(&self) -> Result<()> {
        {
            let mut st = self.state();
            let up = st.cdp.as_ref().is_some_and(|c| !c.closed());
            if up && st.proc.as_mut().is_none_or(Launched::alive) {
                return Ok(());
            }
        }
        let restarted = self.state().starts > 0;
        self.discard();
        let url = match &self.inner.opts.cdp_url {
            Some(u) => self.attach_url(u).await?,
            None => self.launch().await?,
        };
        let cdp = match Cdp::connect(&url, 15.0).await {
            Ok(c) => Arc::new(c),
            Err(e) => {
                self.discard();
                return Err(tool_error(format!("could not connect to the browser: {e}")));
            }
        };
        let weak: Weak<Inner> = Arc::downgrade(&self.inner);
        cdp.on(Arc::new(move |method, params, session| {
            if let Some(inner) = weak.upgrade() {
                Browser { inner }.event(method, params, session);
            }
        }));
        {
            let mut st = self.state();
            st.starts += 1;
            st.cdp = Some(cdp.clone());
        }
        let set_up = async {
            if self.inner.opts.cdp_url.is_none() {
                cdp.call("Browser.setDownloadBehavior", Some(json!({"behavior": "deny"})), "", 30.0).await?;
            }
            let found = cdp.call("Target.getTargets", None, "", 30.0).await?;
            let first = found["targetInfos"].as_array().into_iter().flatten().find(|t| t["type"] == "page").map(|t| s(t, "targetId"));
            let target = match first {
                Some(t) => t,
                None => s(&cdp.call("Target.createTarget", Some(json!({"url": "about:blank"})), "", 30.0).await?, "targetId"),
            };
            self.adopt(&target, true, false).await?;
            // From here on, a tab that one of ours opens is taken in as well.
            cdp.call("Target.setDiscoverTargets", Some(json!({"discover": true})), "", 30.0).await?;
            Ok::<_, Error>(())
        };
        if let Err(e) = set_up.await {
            self.discard();
            return Err(tool_error(format!("the browser could not be set up: {e}")));
        }
        if restarted {
            self.state().notes.push("the browser had stopped and was started again; the pages that were open are gone".into());
        }
        Ok(())
    }

    async fn attach_url(&self, url: &str) -> Result<String> {
        if url.starts_with("ws://") || url.starts_with("wss://") {
            return Ok(url.into());
        }
        let client = crate::sandboxes::command::http_client();
        let asked = async {
            let answer = client.get(format!("{}/json/version", url.trim_end_matches('/'))).timeout(Duration::from_secs(10)).send().await.map_err(|e| e.to_string())?;
            let body: Value = answer.json().await.map_err(|e| e.to_string())?;
            body["webSocketDebuggerUrl"].as_str().map(str::to_string).ok_or_else(|| "KeyError".to_string())
        };
        asked.await.map_err(|why| tool_error(format!("no browser is answering at {url} ({})", why.chars().take(120).collect::<String>())))
    }

    async fn launch(&self) -> Result<String> {
        let o = &self.inner.opts;
        let binary = find_browser(o.executable.as_deref())?;
        let profile = match &o.user_data_dir {
            Some(d) => PathBuf::from(d),
            None => std::env::temp_dir().join(format!("agent-harness-browser-{}", uuid::Uuid::new_v4().simple())),
        };
        std::fs::create_dir_all(&profile)?;
        let port_file = profile.join("DevToolsActivePort");
        let _ = std::fs::remove_file(&port_file);
        let no_sandbox = o.sandbox == Some(false) || (o.sandbox.is_none() && running_as_root());
        let (w, h) = o.viewport;
        let mut argv: Vec<String> = [
            "--remote-debugging-port=0".to_string(),
            format!("--user-data-dir={}", profile.display()),
            "--no-first-run".into(),
            "--no-default-browser-check".into(),
            "--disable-sync".into(),
            "--disable-background-networking".into(),
            "--disable-popup-blocking".into(),
            "--disable-features=Translate,MediaRouter".into(),
            "--mute-audio".into(),
            "--password-store=basic".into(),
            "--use-mock-keychain".into(),
            format!("--window-size={w},{h}"),
        ]
        .into();
        if o.headless {
            argv.push("--headless=new".into());
        }
        if no_sandbox {
            argv.extend(["--no-sandbox".into(), "--disable-dev-shm-usage".into()]);
        }
        if let Some(p) = &o.proxy {
            argv.push(format!("--proxy-server={p}"));
        }
        argv.extend(o.args.iter().cloned());
        argv.push("about:blank".into());
        let log = profile.join("agent-harness-browser.log");
        let errors = std::fs::File::create(&log)?;
        let mut command = std::process::Command::new(&binary);
        command.args(&argv).stdin(std::process::Stdio::null()).stdout(std::process::Stdio::null()).stderr(errors);
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            command.process_group(0);
        }
        let child = command.spawn().map_err(|e| tool_error(format!("could not start {binary}: {e}")))?;
        // Whatever happens to this browser object, the process does not outlive it.
        self.state().proc = Some(Launched { child, profile: o.user_data_dir.is_none().then(|| profile.clone()) });

        let deadline = Instant::now() + Duration::from_secs(20);
        while Instant::now() < deadline {
            if !self.state().proc.as_mut().is_some_and(Launched::alive) {
                break;
            }
            if let Ok(text) = std::fs::read_to_string(&port_file) {
                let lines: Vec<&str> = text.split('\n').collect();
                if lines.len() >= 2 && !lines[0].trim().is_empty() && lines[0].trim().chars().all(|c| c.is_ascii_digit()) && !lines[1].trim().is_empty() {
                    return Ok(format!("ws://127.0.0.1:{}{}", lines[0].trim(), lines[1].trim()));
                }
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        let said: String = std::fs::read(&log).map(|b| String::from_utf8_lossy(&b).split_whitespace().collect::<Vec<_>>().join(" ")).unwrap_or_default();
        let said: String = said.chars().rev().take(400).collect::<Vec<_>>().into_iter().rev().collect();
        let code = self.state().proc.as_mut().and_then(|p| p.child.try_wait().ok().flatten()).map(|s| s.code().unwrap_or(-1));
        self.discard();
        let hint = if said.to_lowercase().contains("sandbox") { " — its sandbox cannot run here; pass sandbox: Some(false)" } else { "" };
        Err(tool_error(format!(
            "the browser {}{}{hint}{}",
            if code.is_some() { "exited" } else { "did not come up" },
            code.map(|c| format!(" with code {c}")).unwrap_or_default(),
            if said.is_empty() { String::new() } else { format!(": {said}") }
        )))
    }

    /// Forget the browser there was. Kills the process if it was ours.
    fn discard(&self) {
        let (cdp, proc, tasks) = {
            let mut st = self.state();
            st.pages.clear();
            st.active.clear();
            (st.cdp.take(), st.proc.take(), std::mem::take(&mut st.tasks))
        };
        for t in tasks {
            t.abort();
        }
        if let Some(c) = cdp {
            c.abandon();
        }
        drop(proc);
    }

    /// Close the browser. A browser that was attached to is left running.
    pub async fn close(&self) {
        let (cdp, ours) = {
            let st = self.state();
            (st.cdp.clone(), st.proc.is_some())
        };
        if let Some(cdp) = cdp.filter(|c| !c.closed()) {
            if ours {
                let _ = cdp.call("Browser.close", None, "", 3.0).await;
            }
            cdp.close().await;
            if ours {
                let deadline = Instant::now() + Duration::from_secs(3);
                while self.state().proc.as_mut().is_some_and(Launched::alive) && Instant::now() < deadline {
                    tokio::time::sleep(Duration::from_millis(50)).await;
                }
            }
        }
        self.discard();
    }

    // ------------------------------------------------------------------
    // events
    // ------------------------------------------------------------------

    fn spawn(&self, f: impl std::future::Future<Output = ()> + Send + 'static) {
        let handle = tokio::spawn(f);
        let mut st = self.state();
        st.tasks.retain(|t| !t.is_finished());
        st.tasks.push(handle);
    }

    fn event(&self, method: &str, params: &Value, session: &str) {
        match method {
            "Fetch.requestPaused" => {
                let (me, params, session) = (self.clone(), params.clone(), session.to_string());
                self.spawn(async move { me.gate(&params, &session).await });
            }
            "Page.loadEventFired" => {
                if let Some(page) = self.state().by_session(session) {
                    page.loaded.send_replace(true);
                }
            }
            "Page.javascriptDialogOpening" => {
                let (me, params, session) = (self.clone(), params.clone(), session.to_string());
                self.spawn(async move { me.dialog(&params, &session).await });
            }
            "Target.targetCreated" => {
                let info = &params["targetInfo"];
                let target = s(info, "targetId");
                // Only a tab one of ours opened: a browser that was attached to has
                // tabs of its own, and those are not this agent's to touch.
                let ours = {
                    let st = self.state();
                    info["type"] == "page" && st.page(&target).is_none() && st.page(&s(info, "openerId")).is_some()
                };
                if ours {
                    let me = self.clone();
                    self.spawn(async move {
                        let _ = me.adopt(&target, true, true).await;
                    });
                }
            }
            "Target.targetInfoChanged" => {
                let info = &params["targetInfo"];
                if let Some(page) = self.state().page_mut(&s(info, "targetId")) {
                    if let Some(u) = info["url"].as_str() {
                        page.url = u.into();
                    }
                    if let Some(t) = info["title"].as_str() {
                        page.title = t.into();
                    }
                }
            }
            "Target.targetDestroyed" => {
                let target = s(params, "targetId");
                let mut st = self.state();
                if st.remove(&target) && st.active == target {
                    st.active = st.pages.last().map(|p| p.target.clone()).unwrap_or_default();
                }
            }
            "Inspector.targetCrashed" | "Target.targetCrashed" => {
                let mut st = self.state();
                let target = s(params, "targetId");
                let found = st.pages.iter().position(|p| p.session == session).or_else(|| st.pages.iter().position(|p| p.target == target));
                if let Some(i) = found {
                    st.pages[i].crashed = true;
                }
            }
            "Browser.downloadWillBegin" => {
                let file = params["suggestedFilename"].as_str().unwrap_or("a file");
                self.state().notes.push(format!("a download of {file} was not allowed"));
            }
            _ => {}
        }
    }

    /// Attach to a tab and set it up: its size, its policy, its events.
    async fn adopt(&self, target: &str, activate: bool, opened: bool) -> Result<()> {
        let (cdp, count) = {
            let st = self.state();
            match &st.cdp {
                Some(c) if st.page(target).is_none() => (c.clone(), st.pages.len()),
                _ => return Ok(()),
            }
        };
        let o = &self.inner.opts;
        if opened && count >= o.max_tabs {
            cdp.call("Target.closeTarget", Some(json!({"targetId": target})), "", 30.0).await?;
            self.state().notes.push(format!("a new tab was closed: {} are open already", o.max_tabs));
            return Ok(());
        }
        let Ok(attached) = cdp.call("Target.attachToTarget", Some(json!({"targetId": target, "flatten": true})), "", 30.0).await else { return Ok(()) };
        let session = s(&attached, "sessionId");
        {
            let mut st = self.state();
            if st.page(target).is_some() {
                return Ok(());
            }
            let (loaded, _) = watch::channel(false);
            st.pages.push(Page { target: target.into(), session: session.clone(), url: String::new(), title: String::new(), status: 0, loaded, crashed: false });
        }
        let (w, h) = o.viewport;
        cdp.call("Page.enable", None, &session, 30.0).await?;
        let patterns = json!({"patterns": [
            {"urlPattern": "*", "resourceType": "Document", "requestStage": "Request"},
            {"urlPattern": "*", "resourceType": "Document", "requestStage": "Response"},
        ]});
        cdp.call("Fetch.enable", Some(patterns), &session, 30.0).await?;
        cdp.call("Emulation.setDeviceMetricsOverride", Some(json!({"width": w, "height": h, "deviceScaleFactor": 1, "mobile": false})), &session, 30.0).await?;
        if let Some(ua) = &o.user_agent {
            cdp.call("Emulation.setUserAgentOverride", Some(json!({"userAgent": ua})), &session, 30.0).await?;
        }
        // A tab opened by a page was waiting for someone to attach; let it go.
        let _ = cdp.call("Runtime.runIfWaitingForDebugger", None, &session, 30.0).await;
        let mut st = self.state();
        if activate || st.active.is_empty() {
            st.active = target.into();
            if opened {
                st.notes.push("a new tab opened and is now the one in use".into());
            }
        }
        Ok(())
    }

    async fn dialog(&self, params: &Value, session: &str) {
        let kind = params["type"].as_str().unwrap_or("alert");
        let accept = matches!(kind, "alert" | "beforeunload") || self.inner.opts.accept_dialogs;
        let message: String = params["message"].as_str().map(|m| m.to_string()).unwrap_or_else(|| params["message"].to_string());
        let message: String = message.split_whitespace().collect::<Vec<_>>().join(" ").chars().take(300).collect();
        let cdp = {
            let mut st = self.state();
            st.notes.push(format!("the page showed a {kind} dialog: {} — {}", py_repr(&message), if accept { "it was accepted" } else { "it was dismissed" }));
            st.cdp.clone()
        };
        if let Some(cdp) = cdp {
            let _ = cdp.call("Page.handleJavaScriptDialog", Some(json!({"accept": accept})), session, 30.0).await;
        }
    }

    // ------------------------------------------------------------------
    // where it may go
    // ------------------------------------------------------------------

    /// Why this address may not be opened; empty when it may.
    pub async fn refusal(&self, url: &str) -> String {
        let Ok(parts) = reqwest::Url::parse(url) else {
            return if url.contains("://") { "that is not a web address".into() } else { "only http and https pages can be opened".into() };
        };
        let scheme = parts.scheme().to_lowercase();
        if matches!(scheme.as_str(), "about" | "data" | "blob") {
            return String::new();
        }
        if scheme != "http" && scheme != "https" {
            return "only http and https pages can be opened".into();
        }
        let host = parts.host_str().unwrap_or("").trim_matches(|c| c == '[' || c == ']').to_lowercase().trim_end_matches('.').to_string();
        if host.is_empty() {
            return "that is not a web address".into();
        }
        let o = &self.inner.opts;
        if o.blocked_domains.iter().any(|b| matches(&host, b)) {
            return format!("{host} is on the blocked list");
        }
        if !o.allowed_domains.is_empty() && !o.allowed_domains.iter().any(|a| matches(&host, a)) {
            return format!("{host} is not on the allowed list");
        }
        if !o.allow_private && is_private_host(&host, Some(&self.inner.resolved)).await {
            return format!("{host} is a private or loopback address");
        }
        String::new()
    }

    /// Every document a tab asks for stops here, and is let through or not.
    async fn gate(&self, params: &Value, session: &str) {
        let Some(cdp) = self.state().cdp.clone() else { return };
        let request = params["requestId"].clone();
        let url = s(&params["request"], "url");
        if params.get("responseStatusCode").is_some() || params.get("responseErrorReason").is_some() {
            {
                let mut st = self.state();
                let frame = s(params, "frameId");
                if let Some(page) = st.by_session(session).filter(|p| p.target == frame) {
                    page.status = num(params, "responseStatusCode");
                }
            }
            let _ = cdp.call("Fetch.continueRequest", Some(json!({"requestId": request})), session, 30.0).await;
            return;
        }
        let reason = self.refusal(&url).await;
        if reason.is_empty() {
            let _ = cdp.call("Fetch.continueRequest", Some(json!({"requestId": request})), session, 30.0).await;
        } else {
            let shown: String = url.chars().take(200).collect();
            self.state().notes.push(format!("{shown} was not opened: {reason}"));
            let _ = cdp.call("Fetch.failRequest", Some(json!({"requestId": request, "errorReason": "BlockedByClient"})), session, 30.0).await;
        }
    }

    // ------------------------------------------------------------------
    // talking to the page
    // ------------------------------------------------------------------

    fn active_session(&self) -> Result<String> {
        let st = self.state();
        st.page(&st.active).map(|p| p.session.clone()).ok_or_else(|| tool_error("no tab is open — open a page first"))
    }

    async fn send(&self, method: &str, params: Option<Value>, session: Option<&str>, timeout: f64) -> Result<Value> {
        let cdp = self.state().cdp.clone().ok_or_else(|| tool_error("the browser is not running"))?;
        let session = match session {
            Some(s) => s.to_string(),
            None => self.active_session()?,
        };
        match cdp.call(method, params, &session, timeout).await {
            Err(Error::Cdp { gone: true, .. }) => Err(tool_error("the browser stopped — try again and it will be started afresh")),
            other => other,
        }
    }

    /// Run JavaScript in the page in use and return its value.
    async fn eval(&self, expression: &str, timeout: f64) -> Result<Value> {
        let answer = self.send("Runtime.evaluate", Some(json!({"expression": format!("{PAGE_JS}\n{expression}"), "returnByValue": true, "awaitPromise": true, "userGesture": true})), None, timeout).await?;
        if let Some(problem) = answer.get("exceptionDetails").filter(|p| !p.is_null()) {
            let text = problem["exception"]["description"].as_str().or_else(|| problem["text"].as_str()).filter(|t| !t.is_empty()).unwrap_or("the page raised an error");
            let first: String = text.lines().next().unwrap_or("").chars().take(200).collect();
            return Err(tool_error(format!("the page could not do that: {first}")));
        }
        Ok(answer["result"]["value"].clone())
    }

    /// Wait for whatever the last action set off to finish loading.
    async fn settle(&self) {
        tokio::time::sleep(Duration::from_secs_f64(self.inner.opts.settle)).await;
        let deadline = Instant::now() + Duration::from_secs(8);
        while Instant::now() < deadline {
            if let Ok(v) = self.eval("document.readyState", 3.0).await
                && v == "complete"
            {
                return;
            }
            tokio::time::sleep(Duration::from_millis(150)).await;
        }
    }

    /// Before every action: a browser that is up, and a tab that is alive.
    pub(crate) async fn guarded(&self) -> Result<()> {
        self.ensure().await?;
        let crashed = {
            let mut st = self.state();
            let active = st.active.clone();
            match st.page_mut(&active) {
                Some(p) if p.crashed => {
                    p.crashed = false;
                    Some(p.session.clone())
                }
                _ => None,
            }
        };
        if let Some(session) = crashed {
            self.state().notes.push("the tab had crashed and was reloaded".into());
            if self.send("Page.reload", None, Some(&session), 30.0).await.is_ok() {
                self.settle().await;
            }
        }
        Ok(())
    }

    /// One action at a time: the lock every action holds.
    pub(crate) async fn locked(&self) -> tokio::sync::MutexGuard<'_, ()> {
        self.inner.action.lock().await
    }

    // ------------------------------------------------------------------
    // what the agent reads
    // ------------------------------------------------------------------

    async fn observe(&self) -> String {
        let o = &self.inner.opts;
        let snap = match self.eval(&format!("window.__ah.snapshot({}, {})", o.max_elements, o.text_chars), 15.0).await {
            Ok(v) if v.is_object() => v,
            Ok(_) => return self.unreadable("the page returned nothing"),
            Err(e) => return self.unreadable(&e.to_string()),
        };
        let (url, title) = (s(&snap, "url"), s(&snap, "title"));
        let mut lines = vec![format!("Page: {} — {url}", if title.is_empty() { "(untitled)" } else { &title })];
        {
            let mut st = self.state();
            let active = st.active.clone();
            let tabs = st.pages.len();
            if let Some(page) = st.page_mut(&active) {
                page.url = url.clone();
                page.title = title.clone();
                if page.status >= 400 {
                    lines.push(format!("Status: {}", page.status));
                }
            }
            if tabs > 1 {
                lines.push(format!("Tabs: {tabs} open — browser_tabs lists them"));
            }
        }
        let (top, view, total) = (num(&snap, "scrollY"), num(&snap, "viewHeight"), num(&snap, "pageHeight"));
        if total > view + 8 {
            lines.push(format!("Scrolled: {top}–{} of {total}px", (top + view).min(total)));
        }
        for note in self.drain() {
            lines.push(format!("Note: {note}"));
        }
        lines.push(String::new());
        let mut section = String::new();
        let elements = snap["elements"].as_array().cloned().unwrap_or_default();
        for item in &elements {
            let where_ = s(item, "where");
            if where_ != section {
                section = where_.clone();
                if lines.last().is_some_and(|l| !l.is_empty()) {
                    lines.push(String::new());
                }
                lines.push(format!("Further {} the page (usable without scrolling to them):", if where_ == "below" { "down" } else { "up" }));
            }
            lines.push(line(item));
        }
        if elements.is_empty() {
            lines.push("(nothing on this page can be clicked or typed into)".into());
        }
        let more = num(&snap, "more");
        if more > 0 {
            lines.push(format!("… and {more} more elements than are listed"));
        }
        let text = s(&snap, "text");
        if !text.is_empty() {
            let more = num(&snap, "textLength") - text.encode_utf16().count() as i64;
            lines.extend([String::new(), "Text on screen:".into(), text]);
            if more > 200 {
                lines.push(format!("[the page has about {more} more characters of text — browser_read has all of it]"));
            }
        }
        lines.join("\n")
    }

    fn unreadable(&self, why: &str) -> String {
        let st = self.state();
        let (title, url) = st.page(&st.active).map(|p| (p.title.clone(), p.url.clone())).unwrap_or_default();
        format!("Page: {} — {url}\n(the page could not be read: {why})", if title.is_empty() { "(untitled)" } else { &title })
    }

    fn drain(&self) -> Vec<String> {
        let notes = std::mem::take(&mut self.state().notes);
        let mut seen = std::collections::HashSet::new();
        notes.into_iter().filter(|n| seen.insert(n.clone())).collect()
    }

    async fn after(&self, did: &str) -> String {
        self.settle().await;
        format!("{did}\n\n{}", self.observe().await)
    }

    // ------------------------------------------------------------------
    // actions
    // ------------------------------------------------------------------

    /// Open a page in the tab in use.
    pub async fn goto(&self, url: &str) -> Result<String> {
        let mut url = url.trim().to_string();
        if url.is_empty() {
            return Err(tool_error("no address was given"));
        }
        if !url.contains("://") && !url.starts_with("about:") && !url.starts_with("data:") {
            url = format!("https://{url}");
        }
        let _held = self.locked().await;
        self.guarded().await?;
        if !url.starts_with("http://") && !url.starts_with("https://") && url != "about:blank" {
            return Err(tool_error("only http and https pages can be opened"));
        }
        let reason = self.refusal(&url).await;
        if !reason.is_empty() {
            return Err(tool_error(format!("{url} was not opened: {reason}")));
        }
        let mut loaded = {
            let mut st = self.state();
            let active = st.active.clone();
            let page = st.page_mut(&active).ok_or_else(|| tool_error("no tab is open — open a page first"))?;
            page.loaded.send_replace(false);
            page.status = 0;
            page.loaded.subscribe()
        };
        let timeout = self.inner.opts.timeout;
        let answer = self.send("Page.navigate", Some(json!({"url": url})), None, timeout).await?;
        if let Some(error) = answer["errorText"].as_str().filter(|e| !e.is_empty()) {
            let notes = self.drain();
            let why = notes.iter().find(|n| n.contains("was not opened")).map(|n| n.split_once(": ").map_or(n.as_str(), |(_, r)| r).to_string());
            return Err(tool_error(format!("{url} could not be opened: {}", why.as_deref().unwrap_or(error))));
        }
        let slow = match tokio::time::timeout(Duration::from_secs_f64(timeout), loaded.wait_for(|v| *v)).await {
            Ok(_) => String::new(),
            Err(_) => format!(" (still loading after {}s)", fmt_g(timeout)),
        };
        Ok(self.after(&format!("Opened {url}{slow}")).await)
    }

    /// The page as it is now: what can be acted on, and what it says.
    pub async fn snapshot(&self) -> Result<String> {
        let _held = self.locked().await;
        self.guarded().await?;
        Ok(self.observe().await)
    }

    async fn locate(&self, number: i64) -> Result<Value> {
        let mut found = self.eval(&format!("window.__ah.locate({number})"), 15.0).await?;
        if !found.is_object() || found["error"] == "gone" {
            return Err(tool_error(format!("there is no element [{number}] on this page any more — read the page again for the current numbers")));
        }
        if found["error"] == "hidden" {
            return Err(tool_error(format!("element [{number}] is not visible")));
        }
        found["ref"] = json!(number);
        Ok(found)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn mouse(&self, kind: &str, x: f64, y: f64, button: &str, count: i64, buttons: i64, extra: Option<Value>) -> Result<()> {
        let mut params = json!({"type": kind, "x": x, "y": y, "button": button, "buttons": buttons, "clickCount": count});
        if let Some(Value::Object(more)) = extra {
            params.as_object_mut().expect("an object").extend(more);
        }
        self.send("Input.dispatchMouseEvent", Some(params), None, 30.0).await?;
        Ok(())
    }

    /// Press and release a mouse button at a point. Holds no lock: for
    /// [`BrowserComputer`](crate::browser::BrowserComputer), which takes it.
    pub async fn click_at(&self, x: f64, y: f64, button: &str, count: i64) -> Result<()> {
        let mask = match button {
            "right" => 2,
            "middle" => 4,
            _ => 1,
        };
        self.mouse("mouseMoved", x, y, "none", 0, 0, None).await?;
        for n in 1..=count {
            self.mouse("mousePressed", x, y, button, n, mask, None).await?;
            self.mouse("mouseReleased", x, y, button, n, 0, None).await?;
        }
        Ok(())
    }

    /// Click the element with this number.
    pub async fn click(&self, reference: i64) -> Result<String> {
        let _held = self.locked().await;
        self.guarded().await?;
        let target = self.locate(reference).await?;
        let label = label(&target);
        if target["disabled"].as_bool().unwrap_or(false) {
            return Err(tool_error(format!("{label} is disabled")));
        }
        if target["tag"] == "select" {
            return Err(tool_error(format!("{label} is a list of options — choose one with browser_select")));
        }
        if target["type"] == "file" {
            return Err(tool_error(format!("{label} takes a file, which this browser cannot give it")));
        }
        if target["covered"].as_bool().unwrap_or(false) {
            // Something lies over it — a banner, a label. Click it directly.
            self.eval(&format!("window.__ah.click({reference})"), 15.0).await?;
        } else {
            self.click_at(target["x"].as_f64().unwrap_or(0.0), target["y"].as_f64().unwrap_or(0.0), "left", 1).await?;
        }
        Ok(self.after(&format!("Clicked {label}")).await)
    }

    /// Type into the element with this number, replacing what is there
    /// (`clear: false` adds to it).
    pub async fn type_into(&self, reference: i64, text: &str, submit: bool, clear: bool) -> Result<String> {
        let _held = self.locked().await;
        self.guarded().await?;
        let target = self.locate(reference).await?;
        let label = label(&target);
        if target["disabled"].as_bool().unwrap_or(false) {
            return Err(tool_error(format!("{label} is disabled")));
        }
        let role = s(&target, "role");
        if !target["editable"].as_bool().unwrap_or(false) && !matches!(role.as_str(), "textbox" | "searchbox" | "combobox") {
            return Err(tool_error(format!("{label} is not something that can be typed into")));
        }
        if !target["covered"].as_bool().unwrap_or(false) {
            self.click_at(target["x"].as_f64().unwrap_or(0.0), target["y"].as_f64().unwrap_or(0.0), "left", 1).await?;
        }
        self.eval(&format!("window.__ah.focus({reference}, {clear})"), 15.0).await?;
        if !text.is_empty() {
            self.send("Input.insertText", Some(json!({"text": text})), None, 30.0).await?;
        } else if clear {
            self.key("Delete").await?;
        }
        if submit {
            self.key("Enter").await?;
        }
        let shown = if target["type"] == "password" { "•••".to_string() } else { py_repr(&text.chars().take(80).collect::<String>()) };
        Ok(self.after(&format!("Typed {shown} into {label}{}", if submit { " and pressed Enter" } else { "" })).await)
    }

    /// Choose an option of a drop-down list, by its text.
    pub async fn select(&self, reference: i64, option: &str) -> Result<String> {
        let _held = self.locked().await;
        self.guarded().await?;
        let target = self.locate(reference).await?;
        let answer = self.eval(&format!("window.__ah.choose({reference}, {})", serde_json::to_string(option).unwrap_or_default()), 15.0).await?;
        let label = label(&target);
        match answer["error"].as_str() {
            Some("notselect") => return Err(tool_error(format!("{label} is not a list of options — click it instead"))),
            Some("nooption") => {
                let options: Vec<&str> = answer["options"].as_array().into_iter().flatten().filter_map(Value::as_str).collect();
                return Err(tool_error(format!("{label} has no option {}; it has: {}", py_repr(option), options.join(" | "))));
            }
            Some(other) => return Err(tool_error(format!("that option cannot be chosen ({other})"))),
            None => {}
        }
        Ok(self.after(&format!("Chose {} in {label}", py_repr(&s(&answer, "chosen")))).await)
    }

    /// Press a key or a combination. Holds no lock; returns what was pressed.
    pub async fn key(&self, combo: &str) -> Result<String> {
        let combo = combo.trim();
        let plus = combo == "+" || combo.ends_with("++");
        let mut parts: Vec<String> = combo.replace(' ', "").split('+').filter(|p| !p.is_empty()).map(str::to_string).collect();
        if plus {
            parts.push("+".into());
        }
        if parts.is_empty() {
            parts.push(String::new());
        }
        let name_given = parts.pop().unwrap_or_default();
        let held = parts;
        let mut modifiers = 0;
        for m in &held {
            modifiers |= modifier(&m.to_lowercase()).ok_or_else(|| tool_error(format!("{} is not a modifier key; there are ctrl, shift, alt and meta", py_repr(m))))?;
        }
        let mut name = name_given.clone();
        let (key, code, number, text) = if let Some((key, code, number, text)) = named_key(&name.to_lowercase()) {
            (key, code, number, text.to_string())
        } else if name.chars().count() == 1 {
            let c = name.chars().next().unwrap_or(' ');
            if modifiers & 8 != 0 && c.is_alphabetic() {
                name = name.to_uppercase();
            }
            let upper = name.to_uppercase();
            let number = if c.is_alphanumeric() { upper.chars().next().map_or(0, |u| u as i64) } else { 0 };
            let code = if c.is_alphabetic() {
                format!("Key{upper}")
            } else if c.is_ascii_digit() {
                format!("Digit{name}")
            } else {
                String::new()
            };
            (name.clone(), code, number, name.clone())
        } else {
            return Err(tool_error(format!(
                "{} is not a key. Use a character, or one of Enter, Tab, Escape, Backspace, Delete, Space, ArrowUp/Down/Left/Right, Home, End, PageUp, PageDown, F1–F12",
                py_repr(&name)
            )));
        };
        let mut event = json!({"key": key, "code": code, "modifiers": modifiers, "windowsVirtualKeyCode": number, "nativeVirtualKeyCode": number});
        let chord = modifiers & !8; // anything held besides shift
        if chord != 0
            && modifiers & (2 | 4) != 0
            && let Some(cmd) = command(&name.to_lowercase())
        {
            event["commands"] = json!([cmd]);
        }
        let mut down = event.clone();
        down["type"] = json!(if !text.is_empty() && chord == 0 { "keyDown" } else { "rawKeyDown" });
        if !text.is_empty() && chord == 0 {
            down["text"] = json!(text);
        }
        self.send("Input.dispatchKeyEvent", Some(down), None, 30.0).await?;
        let mut up = event;
        up["type"] = json!("keyUp");
        self.send("Input.dispatchKeyEvent", Some(up), None, 30.0).await?;
        let mut shown = held;
        shown.push(if key == " " { "Space".into() } else { key });
        Ok(shown.join("+"))
    }

    /// Press a key, or a combination such as `ctrl+a`.
    pub async fn press(&self, key: &str) -> Result<String> {
        let _held = self.locked().await;
        self.guarded().await?;
        let pressed = self.key(key).await?;
        Ok(self.after(&format!("Pressed {pressed}")).await)
    }

    /// Type into whatever has the focus. Holds no lock.
    pub async fn type_text(&self, text: &str) -> Result<()> {
        self.send("Input.insertText", Some(json!({"text": text})), None, 30.0).await?;
        Ok(())
    }

    /// Turn the mouse wheel at a point. Holds no lock.
    pub async fn wheel(&self, x: f64, y: f64, dx: f64, dy: f64) -> Result<()> {
        self.mouse("mouseWheel", x, y, "none", 0, 0, Some(json!({"deltaX": dx, "deltaY": dy}))).await
    }

    /// Scroll the page: up, down, left, right, top or bottom.
    pub async fn scroll(&self, direction: &str, amount: f64) -> Result<String> {
        let direction = direction.trim().to_lowercase();
        let direction = if direction.is_empty() { "down".to_string() } else { direction };
        if !matches!(direction.as_str(), "up" | "down" | "left" | "right" | "top" | "bottom") {
            return Err(tool_error("direction is one of up, down, left, right, top, bottom"));
        }
        let _held = self.locked().await;
        self.guarded().await?;
        if direction == "top" || direction == "bottom" {
            self.eval(&format!("window.scrollTo(0, {})", if direction == "top" { "0" } else { "document.documentElement.scrollHeight" }), 15.0).await?;
        } else {
            let (w, h) = self.inner.opts.viewport;
            let (w, h) = (f64::from(w), f64::from(h));
            let pages = (if amount == 0.0 { 1.0 } else { amount }).clamp(0.1, 20.0);
            let vertical = direction == "up" || direction == "down";
            let step = (if vertical { h } else { w }) * 0.85 * pages;
            let sign = if direction == "up" || direction == "left" { -1.0 } else { 1.0 };
            let (dx, dy) = if vertical { (0.0, sign * step) } else { (sign * step, 0.0) };
            self.wheel(w / 2.0, h / 2.0, dx, dy).await?;
        }
        tokio::time::sleep(Duration::from_millis(350)).await;
        Ok(self.after(&format!("Scrolled {direction}")).await)
    }

    pub async fn back(&self) -> Result<String> {
        self.history(-1).await
    }

    pub async fn forward(&self) -> Result<String> {
        self.history(1).await
    }

    async fn history(&self, step: i64) -> Result<String> {
        let _held = self.locked().await;
        self.guarded().await?;
        let history = self.send("Page.getNavigationHistory", None, None, 30.0).await?;
        let index = num(&history, "currentIndex") + step;
        let entries = history["entries"].as_array().cloned().unwrap_or_default();
        let way = if step < 0 { "back" } else { "forward" };
        if index < 0 || index as usize >= entries.len() {
            return Err(tool_error(format!("there is no page to go {way} to")));
        }
        self.send("Page.navigateToHistoryEntry", Some(json!({"entryId": entries[index as usize]["id"]})), None, 30.0).await?;
        Ok(self.after(&format!("Went {way}")).await)
    }

    /// The text of the page, a stretch at a time.
    pub async fn read(&self, offset: i64, max_chars: i64) -> Result<String> {
        let _held = self.locked().await;
        self.guarded().await?;
        let text: Vec<char> = self.eval("window.__ah.text()", 15.0).await?.as_str().unwrap_or("").chars().collect();
        let offset = offset.max(0) as usize;
        let size = (if max_chars == 0 { 8000 } else { max_chars }).clamp(200, 40_000) as usize;
        let piece: String = text.iter().skip(offset).take(size).collect();
        let taken = piece.chars().count();
        let left = text.len() as i64 - offset as i64 - taken as i64;
        if piece.is_empty() {
            return Ok(if text.is_empty() { "(the page has no text)".into() } else { format!("(nothing past character {})", text.len()) });
        }
        Ok(if left > 0 { format!("{piece}\n… [{left} more characters — read on from offset={}]", offset + taken) } else { piece })
    }

    /// Wait a moment — or, given `text`, until the page shows it.
    pub async fn wait(&self, seconds: f64, text: &str) -> Result<String> {
        let limit = seconds.clamp(0.0, 30.0);
        let _held = self.locked().await;
        self.guarded().await?;
        if text.is_empty() {
            tokio::time::sleep(Duration::from_secs_f64(limit)).await;
            return Ok(self.after(&format!("Waited {}s", fmt_g(limit))).await);
        }
        let span = if limit == 0.0 { 10.0 } else { limit };
        let deadline = Instant::now() + Duration::from_secs_f64(span);
        let wanted = serde_json::to_string(&text.to_lowercase()).unwrap_or_default();
        loop {
            if let Ok(Value::Bool(true)) = self.eval(&format!("document.body.innerText.toLowerCase().includes({wanted})"), 15.0).await {
                return Ok(self.after(&format!("The page now shows {}", py_repr(text))).await);
            }
            if Instant::now() >= deadline {
                return Ok(self.after(&format!("{} did not appear within {}s", py_repr(text), fmt_g(span))).await);
            }
            tokio::time::sleep(Duration::from_millis(250)).await;
        }
    }

    /// A picture of the page as it is shown: a JPEG at quality 70.
    pub async fn screenshot(&self) -> Result<MediaBlock> {
        self.screenshot_as("jpeg", 70).await
    }

    /// A picture of the page, as `png` or `jpeg` (at `quality`, 10–100).
    pub async fn screenshot_as(&self, format: &str, quality: i64) -> Result<MediaBlock> {
        let _held = self.locked().await;
        self.guarded().await?;
        self.shot(format, quality).await
    }

    async fn shot(&self, format: &str, quality: i64) -> Result<MediaBlock> {
        let format = if format == "png" { "png" } else { "jpeg" };
        let mut params = json!({"format": format});
        if format == "jpeg" {
            params["quality"] = json!(quality.clamp(10, 100));
        }
        let answer = self.send("Page.captureScreenshot", Some(params), None, 20.0).await?;
        Ok(MediaBlock { media_type: format!("image/{format}"), data: s(&answer, "data"), url: None, path: None, name: "screenshot".into() })
    }

    // ---- tabs ------------------------------------------------------------

    pub async fn tabs(&self) -> Result<String> {
        let _held = self.locked().await;
        self.guarded().await?;
        Ok(self.tab_list())
    }

    fn tab_list(&self) -> String {
        let st = self.state();
        st.pages
            .iter()
            .enumerate()
            .map(|(n, p)| {
                format!(
                    "[{n}] {} — {}{}",
                    if p.title.is_empty() { "(untitled)" } else { &p.title },
                    if p.url.is_empty() { "about:blank" } else { &p.url },
                    if p.target == st.active { "  (in use)" } else { "" }
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn tab(&self, index: i64) -> Result<String> {
        let found = {
            let st = self.state();
            usize::try_from(index).ok().and_then(|i| st.pages.get(i)).map(|p| p.target.clone())
        };
        found.ok_or_else(|| tool_error(format!("there is no tab {index}; the tabs are:\n{}", self.tab_list())))
    }

    pub async fn switch_tab(&self, index: i64) -> Result<String> {
        let _held = self.locked().await;
        self.guarded().await?;
        let target = self.tab(index)?;
        let cdp = {
            let mut st = self.state();
            st.active = target.clone();
            st.cdp.clone()
        };
        if let Some(cdp) = cdp {
            cdp.call("Target.activateTarget", Some(json!({"targetId": target})), "", 30.0).await?;
        }
        Ok(self.after(&format!("Switched to tab {index}")).await)
    }

    pub async fn new_tab(&self, url: &str) -> Result<String> {
        {
            let _held = self.locked().await;
            self.guarded().await?;
            let (cdp, count) = {
                let st = self.state();
                (st.cdp.clone(), st.pages.len())
            };
            if count >= self.inner.opts.max_tabs {
                return Err(tool_error(format!("{} tabs are open already — close one first", self.inner.opts.max_tabs)));
            }
            let cdp = cdp.ok_or_else(|| tool_error("the browser is not running"))?;
            let target = s(&cdp.call("Target.createTarget", Some(json!({"url": "about:blank"})), "", 30.0).await?, "targetId");
            self.adopt(&target, true, false).await?;
            self.state().active = target;
        }
        if url.is_empty() { Ok("Opened a new tab".into()) } else { self.goto(url).await }
    }

    /// Close a tab — the one in use when `index` is `None`.
    pub async fn close_tab(&self, index: Option<i64>) -> Result<String> {
        let _held = self.locked().await;
        self.guarded().await?;
        let target = match index {
            Some(i) => self.tab(i)?,
            None => {
                let st = self.state();
                st.page(&st.active).map(|p| p.target.clone()).ok_or_else(|| tool_error("no tab is open — open a page first"))?
            }
        };
        let cdp = {
            let st = self.state();
            if st.pages.len() == 1 {
                return Err(tool_error("that is the only tab — open another page in it instead"));
            }
            st.cdp.clone()
        };
        if let Some(cdp) = cdp {
            cdp.call("Target.closeTarget", Some(json!({"targetId": target})), "", 30.0).await?;
        }
        {
            let mut st = self.state();
            st.remove(&target);
            if st.active == target {
                st.active = st.pages.last().map(|p| p.target.clone()).unwrap_or_default();
            }
        }
        Ok(self.after("Closed the tab").await)
    }

    // ------------------------------------------------------------------
    // as tools
    // ------------------------------------------------------------------

    /// The tools an agent drives this browser with. See [`browser_tools`](crate::browser::browser_tools).
    pub fn tools(&self) -> Vec<crate::tools::Tool> {
        crate::browser::tools::browser_tools(self, &crate::browser::tools::BrowserToolsOptions::default())
    }

    /// This browser as a screen with a mouse and a keyboard, for a model that sees.
    pub fn computer(&self) -> crate::browser::computer::BrowserComputer {
        crate::browser::computer::BrowserComputer::new(self.clone())
    }
}

#[cfg(unix)]
fn running_as_root() -> bool {
    std::process::Command::new("id").arg("-u").output().ok().is_some_and(|o| String::from_utf8_lossy(&o.stdout).trim() == "0")
}

#[cfg(not(unix))]
fn running_as_root() -> bool {
    false
}

fn label(target: &Value) -> String {
    let name = s(target, "name");
    let kind = target["role"].as_str().filter(|r| !r.is_empty()).map_or_else(|| s(target, "tag"), str::to_string);
    format!("[{}] {kind}{}", num(target, "ref"), if name.is_empty() { String::new() } else { format!(" {}", py_repr(&name)) })
}

fn json_text(s: &str) -> String {
    serde_json::to_string(s).unwrap_or_default()
}

/// One element of the page, as the model reads it.
fn line(item: &Value) -> String {
    let name = s(item, "name");
    let mut out = format!("[{}] {}", num(item, "ref"), s(item, "role"));
    if !name.is_empty() {
        out.push_str(&format!(" {}", json_text(&name)));
    }
    let value = s(item, "value");
    if !value.is_empty() {
        out.push_str(&format!(" value={}", json_text(&value)));
    }
    let flags: Vec<&str> = [("checked", "checked"), ("disabled", "disabled"), ("expanded", "expanded")].iter().filter(|(k, _)| item[*k].as_bool().unwrap_or(false)).map(|(_, f)| *f).collect();
    if !flags.is_empty() {
        out.push_str(&format!(" ({})", flags.join(", ")));
    }
    let href = s(item, "href");
    if !href.is_empty() {
        out.push_str(&format!(" → {href}"));
    }
    let options: Vec<&str> = item["options"].as_array().into_iter().flatten().filter_map(Value::as_str).collect();
    if !options.is_empty() {
        out.push_str(&format!("  options: {}", options.join(" | ")));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lines_labels_and_options() {
        let item = json!({"ref": 4, "role": "link", "name": "Forgot your \"password\"?", "href": "/reset", "checked": true, "disabled": true});
        assert_eq!(line(&item), "[4] link \"Forgot your \\\"password\\\"?\" (checked, disabled) → /reset");
        let select = json!({"ref": 2, "role": "combobox", "value": "Red", "options": ["Red", "Blue"]});
        assert_eq!(line(&select), "[2] combobox value=\"Red\"  options: Red | Blue");
        assert_eq!(label(&json!({"ref": 3, "role": "button", "name": "Sign in"})), "[3] button 'Sign in'");
        assert_eq!(label(&json!({"ref": 3, "tag": "div"})), "[3] div");
        assert!(Browser::new(BrowserOptions { viewport: (100, 100), ..Default::default() }).is_err());
        assert_eq!(named_key("f5").unwrap().2, 116);
        assert!(named_key("f13").is_none());
    }

    #[tokio::test]
    async fn the_policy_is_applied_to_every_address() {
        let b = Browser::new(BrowserOptions { allowed_domains: vec!["example.com".into(), "*.test.org".into()], blocked_domains: vec!["ads.example.com".into()], ..Default::default() }).unwrap();
        assert_eq!(b.refusal("https://www.example.com/a").await, "");
        assert_eq!(b.refusal("https://ads.example.com/").await, "ads.example.com is on the blocked list");
        assert_eq!(b.refusal("https://evil.com/").await, "evil.com is not on the allowed list");
        assert_eq!(b.refusal("ftp://example.com/").await, "only http and https pages can be opened");
        assert_eq!(b.refusal("about:blank").await, "");
        let open = Browser::default();
        assert_eq!(open.refusal("http://127.0.0.1:8080/").await, "127.0.0.1 is a private or loopback address");
        assert_eq!(open.refusal("http://localhost/").await, "localhost is a private or loopback address");
    }

    /// A tiny web server for the live test.
    async fn site() -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            use tokio::io::{AsyncReadExt, AsyncWriteExt};
            loop {
                let Ok((mut sock, _)) = listener.accept().await else { return };
                tokio::spawn(async move {
                    let mut buf = vec![0u8; 4096];
                    let n = sock.read(&mut buf).await.unwrap_or(0);
                    let head = String::from_utf8_lossy(&buf[..n]).to_string();
                    let path = head.split_whitespace().nth(1).unwrap_or("/").to_string();
                    let body = if path.starts_with("/done") {
                        "<html><head><title>Done</title></head><body><h1>Thanks!</h1></body></html>".to_string()
                    } else {
                        "<html><head><title>Shop</title></head><body><h1>Welcome</h1>\
                         <form action=\"/done\"><input name=\"q\" placeholder=\"Search\"><button>Go</button></form>\
                         <select id=\"c\" aria-label=\"Colour\"><option>Red</option><option>Blue</option></select>\
                         <a href=\"/done\">Finish</a></body></html>"
                            .to_string()
                    };
                    let reply = format!("HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len());
                    let _ = sock.write_all(reply.as_bytes()).await;
                });
            }
        });
        format!("http://127.0.0.1:{}", addr.port())
    }

    /// Drives a real browser when one is installed; passes quietly when not.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_real_browser_reads_types_chooses_and_clicks() {
        if find_browser(None).is_err() || std::env::var("AGENT_HARNESS_SKIP_BROWSER").is_ok() {
            return;
        }
        let url = site().await;
        let b = Browser::new(BrowserOptions { allow_private: true, settle: 0.1, ..Default::default() }).unwrap();
        let page = b.goto(&url).await.unwrap();
        assert!(page.contains("Page: Shop —"), "{page}");
        assert!(page.contains("textbox \"Search\""), "{page}");
        let search = page.lines().find(|l| l.contains("textbox \"Search\"")).unwrap();
        let n: i64 = search[1..search.find(']').unwrap()].parse().unwrap();
        let typed = b.type_into(n, "shoes", false, true).await.unwrap();
        assert!(typed.contains("value=\"shoes\""), "{typed}");
        let colour = typed.lines().find(|l| l.contains("combobox")).unwrap();
        let c: i64 = colour[1..colour.find(']').unwrap()].parse().unwrap();
        assert!(b.select(c, "blue").await.unwrap().starts_with("Chose 'Blue'"));
        assert!(b.select(c, "green").await.unwrap_err().to_string().contains("has no option 'green'; it has: Red | Blue"));
        let finish = typed.lines().find(|l| l.contains("link \"Finish\"")).unwrap();
        let f: i64 = finish[1..finish.find(']').unwrap()].parse().unwrap();
        let done = b.click(f).await.unwrap();
        assert!(done.contains("Page: Done"), "{done}");
        assert!(b.read(0, 0).await.unwrap().contains("Thanks!"));
        assert!(b.back().await.unwrap().contains("Page: Shop"));
        let shot = b.screenshot().await.unwrap();
        assert_eq!(shot.media_type, "image/jpeg");
        assert!(shot.read().unwrap().starts_with(&[0xFF, 0xD8]));
        let strict = Browser::new(BrowserOptions { settle: 0.1, ..Default::default() }).unwrap();
        assert!(strict.goto(&url).await.unwrap_err().to_string().contains("private or loopback"));
        b.close().await;
        strict.close().await;
        assert!(!b.running());
    }
}
