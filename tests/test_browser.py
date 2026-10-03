"""Browser and computer use — against a real headless browser and pages served here.

The browser tests are skipped on a machine with no Chrome, Chromium or Edge.
What does not need one — images in tool results, the desktop driven through a
sandbox — runs everywhere.
"""

from __future__ import annotations

import asyncio
import base64
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import httpx
import pytest

from agent_harness import (
    Agent,
    Browser,
    DesktopComputer,
    FakeProvider,
    Harness,
    ImageBlock,
    computer_tool,
    tool,
    tool_call,
)
from agent_harness.browser import find_browser
from agent_harness.browser.computer import Computer
from agent_harness.errors import ConfigurationError, ToolError
from agent_harness.llm_providers import AnthropicProvider, GeminiProvider, OpenAIProvider
from agent_harness.llm_providers.base import CompletionRequest
from agent_harness.sandboxes.base import ExecResult
from agent_harness.types import Message, TextBlock, ToolResultBlock

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==")

try:
    find_browser()
    HAVE_BROWSER = True
except ConfigurationError:
    HAVE_BROWSER = False

needs_browser = pytest.mark.skipif(not HAVE_BROWSER, reason="no Chrome or Chromium here")

PAGES = {
    "/": """<title>Shop</title><h1>Welcome</h1>
        <a href="/form">Sign in</a> <a href="/long">Terms</a>
        <a href="/popup" target="_blank">Help (new tab)</a>
        <a href="http://blocked.test/x">Partner</a>""",
    "/form": """<title>Sign in</title>
        <form onsubmit="event.preventDefault();
              document.getElementById('out').innerText = 'Hello ' + email.value + ' from '
                + country.value + (remember.checked ? ' (remembered)' : '');">
          <label for="email">Email</label><input id="email" value="old@example.com">
          <label>Password <input id="pw" type="password"></label>
          <select id="country" aria-label="Country">
            <option value="fr">France</option><option value="de">Germany</option></select>
          <label><input id="remember" type="checkbox"> Remember me</label>
          <input type="hidden" name="csrf" value="x">
          <button type="button" disabled>Not yet</button>
          <button id="go">Sign in</button>
        </form><p id="out"></p>""",
    "/long": "<title>Terms</title><p>" + "All of the terms. " * 600 + "</p><a href='/'>Home</a>",
    "/popup": "<title>Help</title><p>How can we help?</p>",
    "/dialog": """<title>Dialog</title>
        <button onclick="document.title = confirm('Delete everything?') ? 'deleted' : 'kept'">
        Delete</button>""",
    "/covered": """<title>Covered</title>
        <button id="b" onclick="document.title='clicked'">Accept</button>
        <div style="position:fixed;inset:0;background:rgba(0,0,0,.2)"></div>""",
    "/shadow": """<title>Shadow</title><div id="host"></div><script>
        const root = document.getElementById('host').attachShadow({mode: 'open'});
        root.innerHTML = '<button>Inside the shadow</button>';
        root.querySelector('button').onclick = () => document.title = 'shadow clicked';
        </script>""",
    "/canvas": """<title>Canvas</title><body style="margin:0">
        <canvas id="c" width="1280" height="800" style="display:block"></canvas>
        <input id="box" style="position:fixed;left:100px;top:700px;width:300px">
        <script>
          const log = [];
          document.addEventListener('mousedown', (e) => {
            log.push(e.button + '@' + e.clientX + ',' + e.clientY); document.title = log.join(' ');
          });
          document.addEventListener('mouseup', (e) => { window.up = e.clientX + ',' + e.clientY; });
          document.addEventListener('dblclick', () => { document.title += ' dbl'; });
        </script>""",
    "/slow": "<title>Slow</title><p id=p>wait</p><script>"
             "setTimeout(() => document.getElementById('p').innerText = 'ready now', 400)</script>",
    "/missing": None,
    "/redirect": "→http://blocked.test/secret",
}


class Site(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        page = PAGES.get(self.path.split("?")[0])
        if isinstance(page, str) and page.startswith("→"):
            self.send_response(302)
            self.send_header("location", page[1:])
            self.end_headers()
            return
        body = (page or "<title>Not found</title><h1>Nothing here</h1>").encode()
        self.send_response(200 if page else 404)
        self.send_header("content-type", "text/html; charset=utf-8")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: Any) -> None:
        return None


@pytest.fixture(scope="module")
def site():
    server = ThreadingHTTPServer(("127.0.0.1", 0), Site)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


@pytest.fixture
async def browser():
    # A CI runner has no user namespaces for Chrome's sandbox to use.
    made = Browser(allow_private=True, blocked_domains=["blocked.test"], timeout=10,
                   settle=0.1,
                   sandbox=False if os.environ.get("CI") else None)
    yield made
    await made.aclose()


def ref(listing: str, label: str) -> int:
    """The number of the element whose line mentions `label`."""
    line = next(ln for ln in listing.splitlines() if label in ln and ln.startswith("["))
    return int(line[1:line.index("]")])


# --- images in tool results (no browser needed) ------------------------------------

async def test_a_tool_can_return_an_image_and_the_model_is_shown_it():
    @tool
    def snap(label: str) -> list:
        """Take a picture."""
        return [f"picture {label}", ImageBlock.from_bytes(PNG, name="screenshot")]

    seen: list[list[Message]] = []

    def look(request):
        seen.append([m.model_copy(deep=True) for m in request.messages])
        return "done" if len(seen) > 5 else tool_call("snap", label=str(len(seen)))

    agent = Agent("a", tools=[snap], memory=False,
                  harness=Harness.testing(FakeProvider([look], loop=True)))
    await agent.run("look")

    last = seen[1][-1]
    assert [type(b) for b in last.content] == [ToolResultBlock, ImageBlock]
    assert last.content[0].content == "picture 1"
    # Only the newest three are kept; the earlier ones leave a note behind.
    images = [b for m in seen[-1] for b in m.content if isinstance(b, ImageBlock)]
    notes = [b.text for m in seen[-1] for b in m.content
             if isinstance(b, TextBlock) and "earlier screenshot" in b.text]
    assert len(images) == 3 and len(notes) == 2


async def test_every_provider_sends_a_tool_image_where_it_takes_images():
    request = CompletionRequest(model="m", messages=[
        Message.user("look"),
        Message(role="assistant", content=[tool_call("snap")]),
        Message(role="user", content=[
            ToolResultBlock(tool_use_id="c1", content="picture"),
            ImageBlock.from_bytes(PNG)])])
    request.messages[1].content[0].id = "c1"
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200)))

    anthropic, _ = AnthropicProvider(api_key="k", client=client)._encode_messages(
        request.messages)
    assert [b["type"] for b in anthropic[-1]["content"]] == ["tool_result", "image"]

    openai = OpenAIProvider(api_key="k", client=client)._encode_messages(request)
    assert [m["role"] for m in openai[-2:]] == ["tool", "user"]
    assert openai[-1]["content"][0]["type"] == "image_url"

    gemini, _ = GeminiProvider(api_key="k", client=client)._encode_contents(request)
    assert [next(iter(p)) for p in gemini[-1]["parts"]] == ["functionResponse", "inlineData"]
    await client.aclose()


# --- the browser ---------------------------------------------------------------------

@needs_browser
async def test_a_page_is_read_as_numbered_elements(browser, site):
    listing = await browser.goto(f"{site}/form")
    assert f"Page: Sign in — {site}/form" in listing
    assert '] textbox "Email" value="old@example.com"' in listing
    assert '] textbox "Password"' in listing
    assert '] combobox "Country" value="France"  options: France | Germany' in listing
    assert '] checkbox "Remember me"' in listing
    assert '] button "Not yet" (disabled)' in listing
    assert "csrf" not in listing                      # hidden inputs are not listed


@needs_browser
async def test_typing_choosing_and_clicking_fill_in_a_form(browser, site):
    listing = await browser.goto(f"{site}/form")
    await browser.type(ref(listing, '"Email"'), "ada@example.com")
    said = await browser.type(ref(listing, '"Password"'), "hunter2")
    assert "Typed ••• into" in said and "hunter2" not in said
    assert 'value="•••"' in said
    await browser.select(ref(listing, '"Country"'), "germany")
    await browser.click(ref(listing, '"Remember me"'))
    done = await browser.click(ref(listing, '"Sign in"'))
    assert "Hello ada@example.com from de (remembered)" in done
    assert '"Remember me" (checked)' in done

    with pytest.raises(ToolError, match="is disabled"):
        await browser.click(ref(listing, '"Not yet"'))
    with pytest.raises(ToolError, match="choose one with browser_select"):
        await browser.click(ref(listing, '"Country"'))
    with pytest.raises(ToolError, match="has no option 'Spain'; it has: France \\| Germany"):
        await browser.select(ref(listing, '"Country"'), "Spain")
    with pytest.raises(ToolError, match="not something that can be typed into"):
        await browser.type(ref(listing, '"Sign in"'), "x")
    with pytest.raises(ToolError, match="no element \\[999\\]"):
        await browser.click(999)


@needs_browser
async def test_enter_submits_and_keys_are_named(browser, site):
    listing = await browser.goto(f"{site}/form")
    done = await browser.type(ref(listing, '"Email"'), "bo@example.com", submit=True)
    assert "and pressed Enter" in done and "Hello bo@example.com" in done
    assert (await browser.press("ctrl+a")).startswith("Pressed ctrl+a")
    with pytest.raises(ToolError, match="is not a key"):
        await browser.press("Hyperdrive")
    with pytest.raises(ToolError, match="is not a modifier"):
        await browser.press("fn+a")


@needs_browser
async def test_links_history_and_a_missing_page(browser, site):
    home = await browser.goto(site)
    after = await browser.click(ref(home, '"Sign in"'))
    assert after.startswith("Clicked [") and "Page: Sign in" in after
    assert "Page: Shop" in await browser.back()
    assert "Page: Sign in" in await browser.forward()
    with pytest.raises(ToolError, match="no page to go forward"):
        await browser.forward()
    missing = await browser.goto(f"{site}/nowhere")
    assert "Status: 404" in missing and "Nothing here" in missing
    # A number from a page that has gone is refused, not guessed at.
    with pytest.raises(ToolError, match="read the page again"):
        await browser.click(ref(home, '"Sign in"'))


@needs_browser
async def test_where_it_may_go_holds_for_typed_clicked_and_redirected(browser, site):
    for url, why in [("http://blocked.test/", "blocked list"),
                     ("file:///etc/passwd", "only http and https"),
                     ("chrome://settings", "only http and https")]:
        with pytest.raises(ToolError, match=why):
            await browser.goto(url)
    with pytest.raises(ToolError, match="blocked.test is on the blocked list"):
        await browser.goto(f"{site}/redirect")

    home = await browser.goto(site)
    after = await browser.click(ref(home, '"Partner"'))
    assert "was not opened: blocked.test is on the blocked list" in after

    closed = Browser(allowed_domains=["example.com"])
    try:
        with pytest.raises(ToolError, match="127.0.0.1 is not on the allowed list"):
            await closed.goto(site)
        guarded = Browser()
        assert "private or loopback" in await guarded.refusal(site)
        assert "private or loopback" in await guarded.refusal("http://169.254.169.254/")
        assert "private or loopback" in await guarded.refusal("http://printer.local/")
        assert await guarded.refusal("https://93.184.216.34/") == ""
    finally:
        await closed.aclose()


@needs_browser
async def test_a_new_tab_is_followed_and_tabs_can_be_managed(browser, site):
    home = await browser.goto(site)
    after = await browser.click(ref(home, "Help (new tab)"))
    assert "Page: Help" in after and "a new tab opened" in after and "Tabs: 2 open" in after
    tabs = await browser.tabs()
    assert "[0] Shop" in tabs and "[1] Help" in tabs and tabs.count("(in use)") == 1
    assert "Page: Shop" in await browser.switch_tab(0)
    assert "Page: Terms" in await browser.new_tab(f"{site}/long")
    assert "Closed the tab" in await browser.close_tab(1)
    assert len(browser._pages) == 2
    with pytest.raises(ToolError, match="there is no tab 7"):
        await browser.switch_tab(7)


@needs_browser
async def test_dialogs_overlays_and_shadow_roots(browser, site):
    listing = await browser.goto(f"{site}/dialog")
    after = await browser.click(ref(listing, '"Delete"'))
    assert "Page: kept" in after
    assert "showed a confirm dialog: 'Delete everything?' — it was dismissed" in after

    listing = await browser.goto(f"{site}/covered")
    assert "Page: clicked" in await browser.click(ref(listing, '"Accept"'))

    listing = await browser.goto(f"{site}/shadow")
    assert "Page: shadow clicked" in await browser.click(ref(listing, "Inside the shadow"))


@needs_browser
async def test_long_pages_are_scrolled_and_read_in_stretches(browser, site):
    listing = await browser.goto(f"{site}/long")
    assert "Scrolled: 0–800 of" in listing
    assert "more characters of text — browser_read has all of it" in listing
    # What is off screen is listed after what is on it, and says so.
    assert listing.index("Further down the page") < listing.index('link "Home"')
    down = await browser.scroll("down", 2)
    assert "Scrolled: 0–800" not in down and "Further down the page" not in down
    assert "Scrolled: 0–800" in await browser.scroll("top")
    first = await browser.read(max_chars=500)
    assert first.startswith("All of the terms.") and "read on from offset=500" in first
    assert "All of the terms." in await browser.read(offset=500, max_chars=300)
    assert "nothing past character" in await browser.read(offset=10**7)
    with pytest.raises(ToolError, match="direction is one of"):
        await browser.scroll("sideways")

    await browser.goto(f"{site}/slow")
    assert "The page now shows 'ready now'" in await browser.wait(5, "ready now")
    assert "did not appear within 0.5s" in await browser.wait(0.5, "never")


@needs_browser
async def test_a_browser_that_dies_is_started_again(browser, site):
    await browser.goto(site)
    browser._proc.kill()
    browser._proc.wait()
    again = await browser.goto(f"{site}/form")       # the next call starts another
    assert "Page: Sign in" in again and "was started again" in again


@needs_browser
def test_a_browser_survives_one_event_loop_ending_and_another_starting(site):
    browser = Browser(allow_private=True, sandbox=False if os.environ.get("CI") else None)
    assert "Page: Shop" in asyncio.run(browser.goto(site))
    first = browser._proc
    assert "Page: Sign in" in asyncio.run(browser.goto(f"{site}/form"))
    assert first.poll() is not None and browser._proc.poll() is None
    asyncio.run(browser.aclose())
    assert browser._proc is None


def test_a_browser_that_cannot_be_found_or_set_up_says_so(monkeypatch, tmp_path):
    with pytest.raises(ConfigurationError, match="no browser at"):
        find_browser(str(tmp_path / "nope"))
    with pytest.raises(ConfigurationError, match="viewport"):
        Browser(viewport=(10, 10))
    with pytest.raises(ConfigurationError, match="are lists"):
        Browser(allowed_domains="example.com")
    broken = tmp_path / "chrome"
    broken.write_text("#!/bin/sh\necho 'cannot open display' >&2\nexit 3\n")
    broken.chmod(0o755)

    async def start():
        with pytest.raises(ToolError, match="exited with code 3: cannot open display"):
            await Browser(executable=str(broken)).start()
        with pytest.raises(ToolError, match="no browser is answering"):
            await Browser(cdp_url="http://127.0.0.1:9").start()

    asyncio.run(start())


@needs_browser
async def test_an_agent_browses_with_the_tools(browser, site):
    steps: list[Any] = []

    def model(request):
        last = request.messages[-1]
        text = "\n".join(b.content for b in last.content if isinstance(b, ToolResultBlock))
        steps.append(text)
        if len(steps) == 1:
            return tool_call("browser_navigate", url=f"{site}/form")
        if len(steps) == 2:
            return tool_call("browser_type", ref=ref(text, '"Email"'), text="ada@example.com")
        if len(steps) == 3:
            return [tool_call("browser_click", ref=ref(text, '"Sign in"')),
                    tool_call("browser_screenshot")]
        assert any(isinstance(b, ImageBlock) for b in last.content)
        return "Signed in as ada@example.com."

    agent = Agent("shopper", tools=browser.tools(), memory=False,
                  harness=Harness.testing(FakeProvider([model], loop=True)))
    names = {t.name for t in agent.tools}
    assert {"browser_navigate", "browser_click", "browser_type", "browser_select",
            "browser_snapshot", "browser_screenshot", "browser_tabs"} <= names
    result = await agent.run("Sign in as ada@example.com")
    assert result.output == "Signed in as ada@example.com."
    assert "Hello ada@example.com from fr" in steps[3]

    refused = await browser.tools()[0].run("c", {"url": "http://blocked.test/"})
    assert refused.is_error and "blocked list" in refused.content


# --- computer use --------------------------------------------------------------------

@needs_browser
async def test_the_computer_tool_works_a_browser_by_coordinates(browser, site):
    computer = computer_tool(browser.computer(), settle=0.05)
    assert computer.permission == "allow"
    assert "open" in computer.parameters["properties"]["action"]["enum"]
    assert "1280x800" in computer.description

    said, shot = await computer.invoke({"action": "open", "text": f"{site}/canvas"})
    assert said == f"Opened {site}/canvas. The screen is 1280x800."
    assert isinstance(shot, ImageBlock) and shot.size() > 1000

    await computer.invoke({"action": "click", "x": 640, "y": 300})
    await computer.invoke({"action": "right_click", "x": 20.4, "y": 30})
    await computer.invoke({"action": "double_click", "x": 500, "y": 500})
    title = await browser._eval("document.title")
    assert title == "0@640,300 2@20,30 0@500,500 0@500,500 dbl"

    await computer.invoke({"action": "drag", "x": 100, "y": 100, "to_x": 400, "to_y": 250})
    assert await browser._eval("window.up") == "400,250"

    await computer.invoke({"action": "click", "x": 200, "y": 710})
    await computer.invoke({"action": "type", "text": "hello wörld"})
    await computer.invoke({"action": "key", "text": "shift+a"})
    assert await browser._eval("document.getElementById('box').value") == "hello wörldA"
    await computer.invoke({"action": "key", "text": "ctrl+a"})
    await computer.invoke({"action": "key", "text": "Backspace"})
    assert await browser._eval("document.getElementById('box').value") == ""

    for bad, why in [({"action": "click", "x": 5000, "y": 5}, "off the screen"),
                     ({"action": "click"}, "needs x and y"),
                     ({"action": "type"}, "type needs text"),
                     ({"action": "teleport"}, "action is one of")]:
        outcome = await computer.run("c", bad)
        assert outcome.is_error and why in outcome.content


class FakeDesktop:
    """A sandbox that records what it was asked to run."""

    def __init__(self, probe: str = "1024 768\nHAS-scrot\n") -> None:
        self.probe, self.commands = probe, []

    async def exec(self, command: str, *, timeout: float | None = None) -> ExecResult:
        self.commands.append(command)
        if "getdisplaygeometry" in command:
            return ExecResult(0, self.probe)
        if "base64" in command:
            return ExecResult(0, base64.b64encode(PNG).decode())
        if "key -- Hyper" in command:
            return ExecResult(1, "", "No such key name 'Hyper'")
        return ExecResult(0, "")


async def test_a_desktop_is_driven_with_xdotool_through_its_sandbox():
    box = FakeDesktop()
    computer = computer_tool(DesktopComputer(box, display=":1"), settle=0)
    assert computer.permission == "allow"
    assert "open" not in computer.parameters["properties"]["action"]["enum"]

    said, shot = await computer.invoke({"action": "screenshot"})
    assert said == "Here is the screen. The screen is 1024x768." and shot.read() == PNG
    await computer.invoke({"action": "double_click", "x": 10, "y": 20})
    await computer.invoke({"action": "right_click", "x": 10, "y": 20})
    await computer.invoke({"action": "type", "text": "it's; rm -rf /"})
    await computer.invoke({"action": "key", "text": "ctrl+Enter"})
    await computer.invoke({"action": "scroll", "direction": "up", "amount": 4})
    await computer.invoke({"action": "drag", "x": 1, "y": 2, "to_x": 3, "to_y": 4})
    ran = [c for c in box.commands if "xdotool" in c and "geometry" not in c]
    assert ran == [
        "DISPLAY=:1 xdotool mousemove 10 20 click --repeat 2 --delay 80 1",
        "DISPLAY=:1 xdotool mousemove 10 20 click 3",
        "DISPLAY=:1 xdotool type --delay 12 -- 'it'\"'\"'s; rm -rf /'",
        "DISPLAY=:1 xdotool key -- ctrl+Return",
        "DISPLAY=:1 xdotool mousemove 512 384 click --repeat 4 --delay 20 4",
        "DISPLAY=:1 xdotool mousemove 1 2 mousedown 1 sleep 0.1 mousemove 3 4 sleep 0.1 mouseup 1",
    ]
    failed = await computer.run("c", {"action": "key", "text": "Hyper"})
    assert failed.is_error and "No such key name" in failed.content
    off = await computer.run("c", {"action": "click", "x": 1100, "y": 5})
    assert off.is_error and "off the screen, which is 1024x768" in off.content


async def test_a_desktop_that_is_not_ready_says_what_is_missing():
    for probe, why in [("NO-XDOTOOL\n", "xdotool is not installed on the sandbox"),
                       ("NO-DISPLAY\n", "no X display can be reached"),
                       ("1024 768\n", "nothing on the sandbox can take a screenshot")]:
        computer = computer_tool(DesktopComputer(FakeDesktop(probe)))
        outcome = await computer.run("c", {"action": "screenshot"})
        assert outcome.is_error and why in outcome.content
    # This machine's own desktop is never driven without being asked.
    assert computer_tool(DesktopComputer()).permission == "ask"

    class Odd(Computer):
        async def screenshot(self): ...
        async def click(self, x, y, *, button="left", count=1): ...
        async def move(self, x, y): ...
        async def drag(self, x, y, to_x, to_y): ...
        async def type(self, text): ...
        async def key(self, keys): ...
        async def scroll(self, x, y, direction, amount): ...

    with pytest.raises(ToolError, match="cannot be told an address"):
        await Odd().open("https://example.com")
