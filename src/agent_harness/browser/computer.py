"""Computer use: a screen an agent looks at, and a mouse and keyboard it works.

    from agent_harness import Agent, Browser, computer_tool

    async with Browser() as browser:
        agent = Agent("operator", tools=[computer_tool(browser.computer())])
        await agent.run("Open example.com and tell me what the heading says.")

The agent calls one tool, `computer`, with an action — `screenshot`, `click`,
`type`, `key`, `scroll`, `drag` — and is shown the screen after each one. It
needs a model that can see. Any such model will do: the tool is an ordinary
one, not a vendor's.

Two screens ship. `Browser.computer()` is a browser page: the place for
anything on the web that the numbered listing of `browser.tools()` cannot
reach — a canvas, a map, a game. `DesktopComputer` is a Linux desktop driven
with `xdotool`, on this machine or — better — inside a sandbox:

    desktop = DesktopComputer(sandbox=DockerSandbox(image="my-desktop-image"))

Any other screen is a subclass of `Computer`.
"""

from __future__ import annotations

import asyncio
import base64
import shlex
from abc import ABC, abstractmethod
from typing import Any

from ..errors import ConfigurationError, ToolError
from ..tools import Permission, Tool
from ..types import ImageBlock
from .browser import Browser
from .cdp import CDPError

__all__ = ["Computer", "BrowserComputer", "DesktopComputer", "computer_tool"]

ACTIONS = ("screenshot", "click", "double_click", "right_click", "middle_click",
           "move", "drag", "type", "key", "scroll", "wait")


class Computer(ABC):
    """A screen with a pointer and a keyboard.

    Coordinates are pixels from the top-left corner of the screenshot. Subclass
    it for a screen that is not here — a VNC session, a phone, a remote desktop —
    and `computer_tool` gives an agent the use of it.
    """

    #: The screen, in pixels. Known once `start` has run.
    width: int = 0
    height: int = 0
    #: What the model is told this is.
    kind: str = "computer"
    #: Does `open(url)` do something here?
    can_open: bool = False

    async def start(self) -> None:  # noqa: B027 - optional, not abstract
        """Get ready: connect, measure the screen. Called before the first action."""

    async def aclose(self) -> None:  # noqa: B027 - optional, not abstract
        """Let go of the screen."""

    @abstractmethod
    async def screenshot(self) -> ImageBlock:
        """The screen as it is now."""

    @abstractmethod
    async def click(self, x: int, y: int, *, button: str = "left", count: int = 1) -> None:
        ...

    @abstractmethod
    async def move(self, x: int, y: int) -> None:
        ...

    @abstractmethod
    async def drag(self, x: int, y: int, to_x: int, to_y: int) -> None:
        ...

    @abstractmethod
    async def type(self, text: str) -> None:
        ...

    @abstractmethod
    async def key(self, keys: str) -> None:
        """Press a key or a combination: `Enter`, `ctrl+a`."""

    @abstractmethod
    async def scroll(self, x: int, y: int, direction: str, amount: int) -> None:
        """Turn the wheel `amount` notches at a point."""

    async def open(self, url: str) -> None:
        raise ToolError(f"this {self.kind} cannot be told an address — use its own "
                        "address bar", tool="computer")


class BrowserComputer(Computer):
    """A browser page as the screen. The mouse and keys are the real thing:
    the page cannot tell them from a person's."""

    kind = "browser"
    can_open = True

    def __init__(self, browser: Browser | None = None, **options: Any) -> None:
        self.browser = browser or Browser(**options)
        self.width, self.height = self.browser.viewport

    async def start(self) -> None:
        await self.browser.start()

    async def aclose(self) -> None:
        await self.browser.aclose()

    async def _do(self, action: Any) -> Any:
        b = self.browser
        async with b._locked():
            await b._guarded()
            return await action()

    async def screenshot(self) -> ImageBlock:
        return await self.browser.screenshot()

    async def click(self, x: int, y: int, *, button: str = "left", count: int = 1) -> None:
        await self._do(lambda: self.browser.click_at(x, y, button=button, count=count))

    async def move(self, x: int, y: int) -> None:
        await self._do(lambda: self.browser._mouse("mouseMoved", x, y))

    async def drag(self, x: int, y: int, to_x: int, to_y: int) -> None:
        b = self.browser

        async def go() -> None:
            await b._mouse("mouseMoved", x, y)
            await b._mouse("mousePressed", x, y, button="left", count=1, buttons=1)
            # In steps: a page that follows the pointer needs to see it travel.
            for step in range(1, 11):
                await b._mouse("mouseMoved", x + (to_x - x) * step / 10,
                               y + (to_y - y) * step / 10, button="left", buttons=1)
            await b._mouse("mouseReleased", to_x, to_y, button="left", count=1)

        await self._do(go)

    async def type(self, text: str) -> None:
        await self._do(lambda: self.browser.type_text(text))

    async def key(self, keys: str) -> None:
        await self._do(lambda: self.browser._key(keys))

    async def scroll(self, x: int, y: int, direction: str, amount: int) -> None:
        step = 100 * amount
        dx, dy = {"up": (0, -step), "down": (0, step),
                  "left": (-step, 0), "right": (step, 0)}[direction]
        await self._do(lambda: self.browser.wheel(x, y, dx, dy))

    async def open(self, url: str) -> None:
        await self.browser.goto(url)


#: The names a model uses, as xdotool knows them.
_XKEYS = {"enter": "Return", "return": "Return", "esc": "Escape", "escape": "Escape",
          "backspace": "BackSpace", "delete": "Delete", "del": "Delete", "tab": "Tab",
          "space": "space", "pageup": "Prior", "pgup": "Prior", "pagedown": "Next",
          "pgdn": "Next", "arrowup": "Up", "arrowdown": "Down", "arrowleft": "Left",
          "arrowright": "Right", "up": "Up", "down": "Down", "left": "Left",
          "right": "Right", "home": "Home", "end": "End", "insert": "Insert",
          "ctrl": "ctrl", "control": "ctrl", "alt": "alt", "option": "alt",
          "shift": "shift", "meta": "super", "cmd": "super", "command": "super",
          "super": "super", "win": "super"}
_SHOOTERS = (
    ("scrot", "scrot -o -z {path}"),
    ("import", "import -window root {path}"),
    ("gnome-screenshot", "gnome-screenshot -f {path}"),
    ("xfce4-screenshooter", "xfce4-screenshooter -f -s {path}"),
)


class DesktopComputer(Computer):
    """A Linux desktop (X11), driven with `xdotool`.

    ``sandbox``   the sandbox whose desktop this is — a container with a display,
                  `xdotool` and `scrot` in it. None drives *this* machine's
                  desktop, which is why the tool then asks before every action.
    ``display``   the X display, e.g. ":1". The default is the environment's.

    Wayland does not let one program drive another, so this needs X11 — which
    is what a virtual display (Xvfb) in a container is.
    """

    kind = "desktop"

    def __init__(self, sandbox: Any = None, *, display: str | None = None,
                 timeout: float = 20.0) -> None:
        self.sandbox = sandbox
        self.display = display
        self.timeout = float(timeout)
        self._shot = ""
        self._path = "/tmp/agent-harness-screen.png"  # noqa: S108 - inside the sandbox

    @property
    def local(self) -> bool:
        return self.sandbox is None

    async def _run(self, command: str) -> tuple[int, str, str]:
        if self.display:
            command = f"DISPLAY={shlex.quote(self.display)} {command}"
        if self.sandbox is not None:
            result = await self.sandbox.exec(command, timeout=self.timeout)
            return result.returncode, result.stdout, result.stderr
        from ..sandboxes.base import ProcessTimeout, run_process

        try:
            code, out, err = await run_process(["sh", "-c", command],
                                               timeout=self.timeout)
        except ProcessTimeout:
            return 124, "", f"timed out after {self.timeout:g}s"
        return code, out.decode(errors="replace"), err.decode(errors="replace")

    async def _x(self, arguments: str, what: str) -> str:
        code, out, err = await self._run(f"xdotool {arguments}")
        if code != 0:
            raise ToolError(f"could not {what}: {(err or out).strip()[:300] or code}",
                            tool="computer")
        return out

    async def start(self) -> None:
        if self._shot:
            return
        code, out, err = await self._run(
            "command -v xdotool >/dev/null 2>&1 || { echo NO-XDOTOOL; exit 0; }; "
            "xdotool getdisplaygeometry || { echo NO-DISPLAY; exit 0; }; "
            + "; ".join(f"command -v {name} >/dev/null 2>&1 && echo HAS-{name}"
                        for name, _ in _SHOOTERS) + "; true")
        where = "the sandbox" if self.sandbox is not None else "this machine"
        if "NO-XDOTOOL" in out:
            raise ConfigurationError(
                f"xdotool is not installed on {where} — a desktop is driven with it "
                "(apt install xdotool scrot)")
        if "NO-DISPLAY" in out or code != 0:
            raise ConfigurationError(
                f"no X display can be reached on {where}"
                f"{f' ({(err or out).strip()[:200]})' if (err or out).strip() else ''} — "
                "start one (Xvfb :1 -screen 0 1280x800x24) and pass display=':1'")
        size = out.split("\n", 1)[0].split()
        try:
            self.width, self.height = int(size[0]), int(size[1])
        except (IndexError, ValueError):
            raise ConfigurationError(
                f"the size of the display could not be read: {out.strip()[:200]!r}"
            ) from None
        for name, command in _SHOOTERS:
            if f"HAS-{name}" in out:
                self._shot = command.format(path=self._path)
                return
        raise ConfigurationError(
            f"nothing on {where} can take a screenshot — install one of: "
            + ", ".join(name for name, _ in _SHOOTERS))

    async def screenshot(self) -> ImageBlock:
        await self.start()
        code, out, err = await self._run(
            f"rm -f {self._path}; {self._shot} >/dev/null 2>&1; "
            f"base64 < {self._path} 2>/dev/null")
        try:
            data = base64.b64decode("".join(out.split()), validate=True)
        except ValueError:
            data = b""
        if not data.startswith(b"\x89PNG"):
            raise ToolError("the screenshot could not be taken"
                            f"{': ' + err.strip()[:200] if err.strip() else ''}",
                            tool="computer")
        return ImageBlock.from_bytes(data, "image/png", name="screenshot")

    async def click(self, x: int, y: int, *, button: str = "left", count: int = 1) -> None:
        number = {"left": 1, "middle": 2, "right": 3}.get(button, 1)
        repeat = f"--repeat {count} --delay 80 " if count > 1 else ""
        await self._x(f"mousemove {x} {y} click {repeat}{number}", "click")

    async def move(self, x: int, y: int) -> None:
        await self._x(f"mousemove {x} {y}", "move the pointer")

    async def drag(self, x: int, y: int, to_x: int, to_y: int) -> None:
        await self._x(f"mousemove {x} {y} mousedown 1 sleep 0.1 "
                      f"mousemove {to_x} {to_y} sleep 0.1 mouseup 1", "drag")

    async def type(self, text: str) -> None:
        await self._x(f"type --delay 12 -- {shlex.quote(text)}", "type")

    async def key(self, keys: str) -> None:
        parts = [p for p in str(keys).replace(" ", "").split("+") if p]
        if not parts:
            raise ToolError("no key was named", tool="computer")
        names = [_XKEYS.get(p.lower(), p) for p in parts]
        await self._x(f"key -- {shlex.quote('+'.join(names))}", f"press {keys}")

    async def scroll(self, x: int, y: int, direction: str, amount: int) -> None:
        button = {"up": 4, "down": 5, "left": 6, "right": 7}[direction]
        await self._x(f"mousemove {x} {y} click --repeat {amount} --delay 20 {button}",
                      "scroll")


def computer_tool(computer: Computer | None = None, *, name: str = "computer",
                  permission: Permission | None = None, settle: float = 0.8,
                  **options: Any) -> Tool:
    """The tool an agent uses a computer with.

    ``computer``    the screen; a headless browser made from `options` when not given
    ``permission``  "ask" to approve every action. The default asks only when the
                    screen is this machine's own desktop.
    ``settle``      seconds to let the screen react before it is looked at again
    """
    screen = computer or BrowserComputer(**options)
    if permission is None:
        permission = "ask" if getattr(screen, "local", False) else "allow"
    actions = [*ACTIONS, *(["open"] if screen.can_open else [])]
    started = False
    locks: dict[int, asyncio.Lock] = {}

    def lock() -> asyncio.Lock:
        # One per event loop: an agent asks for several actions at once, and a
        # screen can only do one thing at a time.
        loop = id(asyncio.get_running_loop())
        if loop not in locks:
            locks.clear()
            locks[loop] = asyncio.Lock()
        return locks[loop]

    def point(x: Any, y: Any, what: str) -> tuple[int, int]:
        if x is None or y is None:
            raise ToolError(f"{what} needs x and y — a point on the screenshot",
                            tool=name)
        px, py = int(round(float(x))), int(round(float(y)))
        if not (0 <= px < screen.width and 0 <= py < screen.height):
            raise ToolError(f"({px}, {py}) is off the screen, which is "
                            f"{screen.width}x{screen.height}", tool=name)
        return px, py

    async def computer_action(action: str, x: float | None = None, y: float | None = None,
                              text: str = "", to_x: float | None = None,
                              to_y: float | None = None, direction: str = "down",
                              amount: int = 3) -> Any:
        nonlocal started
        action = str(action or "").strip().lower().replace("-", "_")
        action = {"left_click": "click", "mouse_move": "move", "press": "key",
                  "keypress": "key", "left_click_drag": "drag", "navigate": "open",
                  "goto": "open"}.get(action, action)
        if action not in actions:
            raise ToolError(f"action is one of {', '.join(actions)} — got {action!r}",
                            tool=name)
        async with lock():
            try:
                if not started:
                    await screen.start()
                    started = True
                did = "Here is the screen."
                if action in ("click", "double_click", "right_click", "middle_click"):
                    px, py = point(x, y, action)
                    button = {"right_click": "right", "middle_click": "middle"}.get(
                        action, "left")
                    await screen.click(px, py, button=button,
                                       count=2 if action == "double_click" else 1)
                    did = f"{action.replace('_', ' ').capitalize()}ed at ({px}, {py})."
                elif action == "move":
                    px, py = point(x, y, action)
                    await screen.move(px, py)
                    did = f"Moved the pointer to ({px}, {py})."
                elif action == "drag":
                    px, py = point(x, y, "drag")
                    qx, qy = point(to_x, to_y, "drag (to_x, to_y)")
                    await screen.drag(px, py, qx, qy)
                    did = f"Dragged from ({px}, {py}) to ({qx}, {qy})."
                elif action == "type":
                    if not text:
                        raise ToolError("type needs text", tool=name)
                    await screen.type(str(text))
                    did = f"Typed {len(str(text))} characters."
                elif action == "key":
                    if not text:
                        raise ToolError("key needs text: the key to press, e.g. Enter "
                                        "or ctrl+a", tool=name)
                    await screen.key(str(text))
                    did = f"Pressed {text}."
                elif action == "scroll":
                    way = str(direction or "down").strip().lower()
                    if way not in ("up", "down", "left", "right"):
                        raise ToolError("direction is one of up, down, left, right",
                                        tool=name)
                    px, py = (point(x, y, "scroll") if x is not None and y is not None
                              else (screen.width // 2, screen.height // 2))
                    notches = max(1, min(int(amount or 3), 30))
                    await screen.scroll(px, py, way, notches)
                    did = f"Scrolled {way}."
                elif action == "wait":
                    await asyncio.sleep(max(0.0, min(float(amount or 1), 30.0)))
                    did = "Waited."
                elif action == "open":
                    if not text:
                        raise ToolError("open needs text: the address", tool=name)
                    await screen.open(str(text))
                    did = f"Opened {text}."
                if action not in ("screenshot", "wait", "open"):
                    await asyncio.sleep(settle)
                shot = await screen.screenshot()
            except CDPError as exc:
                raise ToolError(f"the {screen.kind} could not do that: {exc}",
                                tool=name) from None
        return [f"{did} The screen is {screen.width}x{screen.height}.", shot]

    computer_action.__name__ = name
    properties: dict[str, Any] = {
        "action": {"type": "string", "enum": actions, "description": (
            "screenshot: look at the screen. click, double_click, right_click, "
            "middle_click, move: at x, y. drag: from x, y to to_x, to_y. type: the "
            "text, into whatever has the focus. key: press the key or combination "
            "in text (Enter, Tab, Escape, ctrl+a). scroll: turn the wheel in "
            "direction, at x, y. wait: for amount seconds."
            + (" open: go to the address in text." if screen.can_open else ""))},
        "x": {"type": "number", "description": "pixels from the left edge of the screenshot."},
        "y": {"type": "number", "description": "pixels from the top edge of the screenshot."},
        "text": {"type": "string", "description": "what to type, the key to press"
                 + (", or the address to open." if screen.can_open else ".")},
        "to_x": {"type": "number", "description": "where a drag ends."},
        "to_y": {"type": "number", "description": "where a drag ends."},
        "direction": {"type": "string", "enum": ["up", "down", "left", "right"],
                      "description": "which way to scroll."},
        "amount": {"type": "integer", "description": "wheel notches to scroll, or "
                                                     "seconds to wait."},
    }
    size = (f" The screen is {screen.width}x{screen.height} pixels." if screen.width
            else "")
    return Tool(
        computer_action, name=name, permission=permission,
        tags=["builtin", "computer"],
        description=(
            f"Use a {screen.kind} the way a person does: look at the screen, move "
            "the mouse, click, type. Every action answers with a screenshot of what "
            "the screen shows afterwards — look at it before the next action, and "
            "take a screenshot first to see where things are. Click the middle of "
            f"what you mean to hit.{size}"),
        parameters={"type": "object", "properties": properties, "required": ["action"]})
