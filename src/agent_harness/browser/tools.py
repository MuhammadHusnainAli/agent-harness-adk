"""The browser, as the tools an agent calls."""

from __future__ import annotations

import inspect
from typing import Any

from ..errors import ToolError
from ..tools import Permission, Tool, tool
from .browser import Browser
from .cdp import CDPError

__all__ = ["browser_tools"]


def browser_tools(browser: Browser | None = None, *, prefix: str = "browser_",
                  screenshots: bool = True, vision: bool = False,
                  permission: Permission = "allow", **options: Any) -> list[Tool]:
    """The tools that drive a browser.

    ``browser``      the browser to drive; one is made from `options` when not given
    ``prefix``       what the tool names start with
    ``screenshots``  offer `browser_screenshot`, for a model that can see
    ``vision``       send a screenshot back with every action, not only on request
    ``permission``   "ask" to have every action approved first
    """
    b = browser or Browser(**options)
    tags = ["builtin", "browser", "web"]

    def made(fn: Any, name: str) -> Tool:
        async def run(**kwargs: Any) -> Any:
            try:
                said = await fn(**kwargs)
            except CDPError as exc:
                raise ToolError(f"the browser could not do that: {exc}",
                                tool=prefix + name) from None
            if vision and isinstance(said, str) and name not in ("read", "tabs"):
                try:
                    return [said, await b.screenshot()]
                except (CDPError, ToolError):
                    return said
            return said

        run.__name__ = prefix + name
        run.__doc__ = fn.__doc__
        run.__signature__ = inspect.signature(fn)  # type: ignore[attr-defined]
        run.__annotations__ = dict(fn.__annotations__)
        return tool(name=prefix + name, tags=tags, permission=permission)(run)

    async def navigate(url: str) -> str:
        """Open a web page, and read what is on it.

        Args:
            url: the address to open.
        """
        return await b.goto(url)

    async def snapshot() -> str:
        """Read the page again: its numbered elements and its text."""
        return await b.snapshot()

    async def click(ref: int) -> str:
        """Click an element of the page.

        Args:
            ref: the element's number in the page listing, e.g. 12 for [12].
        """
        return await b.click(ref)

    async def type_(ref: int, text: str, submit: bool = False) -> str:
        """Type into a field, replacing what was in it.

        Args:
            ref: the field's number in the page listing.
            text: what to type.
            submit: press Enter afterwards, to send a form or run a search.
        """
        return await b.type(ref, text, submit=submit)

    async def select(ref: int, option: str) -> str:
        """Choose an option from a drop-down list.

        Args:
            ref: the list's number in the page listing.
            option: the text of the option to choose.
        """
        return await b.select(ref, option)

    async def press(key: str) -> str:
        """Press a key: Enter, Tab, Escape, ArrowDown, PageDown, or a combination like ctrl+a.

        Args:
            key: the key, or keys joined with +.
        """
        return await b.press(key)

    async def scroll(direction: str = "down", amount: float = 1.0) -> str:
        """Scroll the page.

        Args:
            direction: up, down, left, right, top or bottom.
            amount: how many screens to move.
        """
        return await b.scroll(direction, amount)

    async def back() -> str:
        """Go back to the previous page."""
        return await b.back()

    async def read(offset: int = 0, max_chars: int = 8000) -> str:
        """Read the text of the page — all of it, where the listing shows only the start.

        Args:
            offset: the character to start from, to read on.
            max_chars: how much to return.
        """
        return await b.read(offset, max_chars)

    async def wait(seconds: float = 2.0, text: str = "") -> str:
        """Wait for the page: a number of seconds, or until some text appears.

        Args:
            seconds: how long to wait, at most 30.
            text: stop waiting as soon as the page shows this.
        """
        return await b.wait(seconds, text)

    async def tabs(action: str = "list", index: int = 0, url: str = "") -> str:
        """List the open tabs, switch to one, open a new one, or close one.

        Args:
            action: list, switch, new or close.
            index: the tab's number, for switch and close.
            url: the page to open, for new.
        """
        action = str(action or "list").strip().lower()
        if action == "list":
            return await b.tabs()
        if action == "switch":
            return await b.switch_tab(index)
        if action == "new":
            return await b.new_tab(url)
        if action == "close":
            return await b.close_tab(index)
        raise ToolError("action is one of list, switch, new, close", tool=prefix + "tabs")

    async def screenshot() -> Any:
        """Take a picture of the page as it is shown, to look at it."""
        return ["A screenshot of the page.", await b.screenshot()]

    out = [made(navigate, "navigate"), made(snapshot, "snapshot"), made(click, "click"),
           made(type_, "type"), made(select, "select"), made(press, "press"),
           made(scroll, "scroll"), made(back, "back"), made(read, "read"),
           made(wait, "wait"), made(tabs, "tabs")]
    if screenshots or vision:
        out.append(made(screenshot, "screenshot"))
    return out
