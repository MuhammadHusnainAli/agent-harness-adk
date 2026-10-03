"""Browser and computer use: an agent that opens pages, fills in a form, and looks.

A small shop runs here on a real socket. The agent is given a browser — Chrome,
Chromium or Edge, whichever is installed, started headless — and drives it two
ways: by the numbered elements of the page, which any model can do, and with a
mouse and a screenshot, for a model that sees.

On a machine with no browser installed the example says so and stops.
"""

from __future__ import annotations

import asyncio
import os
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from _common import pick_provider

from agent_harness import Agent, Browser, Harness, computer_tool, tool_call
from agent_harness.browser import find_browser
from agent_harness.errors import ConfigurationError

PAGES = {
    "/": """<title>Corner Shop</title><h1>Corner Shop</h1>
        <p>Order tracking for customers.</p><a href="/track">Track an order</a>""",
    "/track": """<title>Track an order</title><h1>Track an order</h1>
        <form onsubmit="event.preventDefault(); document.getElementById('out').innerText =
              'Order ' + order.value + ' shipped by ' + carrier.value + ' on 2 October.'">
          <label for="order">Order number</label><input id="order">
          <select id="carrier" aria-label="Carrier">
            <option>Post</option><option>Courier</option></select>
          <button>Look it up</button>
        </form><p id="out"></p>""",
}


class Shop(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        body = PAGES.get(self.path, "<title>Not found</title>").encode()
        self.send_response(200 if self.path in PAGES else 404)
        self.send_header("content-type", "text/html; charset=utf-8")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: Any) -> None:
        return None


def number(listing: str, label: str) -> int:
    """The number the page listing gives the element with this label."""
    return int(re.search(rf"\[(\d+)\] [^\n]*{re.escape(label)}", listing).group(1))


async def main() -> None:
    try:
        find_browser()
    except ConfigurationError as exc:
        print(f"(skipped: {exc})")
        return
    server = ThreadingHTTPServer(("127.0.0.1", 0), Shop)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}"

    # The shop is on this machine, so private addresses are allowed — and
    # nothing else is: the browser may go to the shop and nowhere besides.
    # (A CI runner cannot run Chrome's own sandbox; anywhere else it stays on.)
    async with Browser(allow_private=True, allowed_domains=["127.0.0.1"],
                       sandbox=False if os.environ.get("CI") else None) as browser:
        # --- drive it yourself ----------------------------------------------------
        print(await browser.goto(url), "\n")
        try:
            await browser.goto("https://example.com")
        except Exception as exc:
            print(f"refused    {exc}\n")

        # --- hand it to an agent -----------------------------------------------------
        # Scripted only when there is no API key: a model reads the listing and
        # picks the numbers itself.
        def scripted(request):
            said = "\n".join(getattr(b, "content", "") for b in request.messages[-1].content)
            turn = sum(1 for m in request.messages if m.role == "assistant")
            if turn == 0:
                return tool_call("browser_navigate", url=f"{url}/track")
            if turn == 1:
                return tool_call("browser_type", ref=number(said, "Order number"),
                                 text="4182")
            if turn == 2:
                return tool_call("browser_select", ref=number(said, "Carrier"),
                                 option="Courier")
            if turn == 3:
                return tool_call("browser_click", ref=number(said, "Look it up"))
            return "Order 4182 shipped by Courier on 2 October."

        provider, model = pick_provider([scripted])
        harness = Harness(provider=provider)
        agent = Agent("clerk", "Look things up on the shop's website.",
                      tools=browser.tools(), model=model, harness=harness, memory=False)
        async for event in agent.stream(f"When did order 4182 ship, and how? The shop "
                                        f"is at {url} and it went by courier."):
            if event.type == "tool_result":
                print(f"tool       {event.data['tool']} → {event.text.splitlines()[0]}")
            elif event.type == "run_end":
                print(f"\nclerk      {event.data['result'].output}\n")

        # --- computer use: the same browser, by mouse and screenshot ---------------
        computer = computer_tool(browser.computer())
        said, shot = await computer.invoke({"action": "open", "text": url})
        print(f"computer   {said}  ({shot.media_type}, {shot.size():,} bytes)")
        said, shot = await computer.invoke({"action": "click", "x": 60, "y": 110})
        print(f"computer   {said}  now on: {(await browser.snapshot()).splitlines()[0]}")
        await harness.aclose()
    server.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
