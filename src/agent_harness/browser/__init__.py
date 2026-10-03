"""Browser and computer use: a web browser, or a whole screen, for an agent.

    from agent_harness import Agent, Browser

    async with Browser() as browser:
        agent = Agent("researcher", tools=browser.tools())

See `Browser` for the browser, `computer_tool` for a mouse and keyboard.
"""

from .browser import Browser, find_browser
from .cdp import CDP, CDPError
from .computer import BrowserComputer, Computer, DesktopComputer, computer_tool
from .tools import browser_tools

__all__ = [
    "Browser",
    "browser_tools",
    "find_browser",
    "Computer",
    "BrowserComputer",
    "DesktopComputer",
    "computer_tool",
    "CDP",
    "CDPError",
]
