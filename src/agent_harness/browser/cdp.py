"""The Chrome DevTools Protocol, over the harness's own WebSocket client.

A browser is driven by sending it JSON commands and reading JSON events back.
This is the connection: commands answered by id, events handed to listeners,
and every waiting caller told at once when the browser goes away.
"""

from __future__ import annotations

import asyncio
import itertools
import json
from collections.abc import Callable
from typing import Any

from ..errors import HarnessError
from ..voice.ws import ConnectionClosed, WebSocket

__all__ = ["CDP", "CDPError"]

# On Python 3.10 asyncio.TimeoutError is a distinct class from the builtin.
_Timeout = (TimeoutError, asyncio.TimeoutError)

Listener = Callable[[str, dict[str, Any], str], None]


class CDPError(HarnessError):
    """The browser refused a command, did not answer it, or is gone."""

    def __init__(self, message: str, *, code: int | None = None,
                 gone: bool = False) -> None:
        super().__init__(message)
        self.code = code
        #: The connection is lost; nothing more can be asked of this browser.
        self.gone = gone


class CDP:
    """One connection to a browser. Build it with `CDP.connect`."""

    def __init__(self, ws: WebSocket) -> None:
        self._ws = ws
        self._ids = itertools.count(1)
        self._pending: dict[int, asyncio.Future[dict[str, Any]]] = {}
        self._listeners: list[Listener] = []
        self.closed = False
        self._reader = asyncio.get_running_loop().create_task(self._read())

    @classmethod
    async def connect(cls, url: str, *, timeout: float = 20.0) -> CDP:
        return cls(await WebSocket.connect(url, timeout=timeout,
                                           max_size=256 * 1024 * 1024))

    def on(self, listener: Listener) -> None:
        """Call `listener(method, params, session_id)` for every event."""
        self._listeners.append(listener)

    async def call(self, method: str, params: dict[str, Any] | None = None, *,
                   session: str = "", timeout: float = 30.0) -> dict[str, Any]:
        """Send one command and return its result."""
        if self.closed:
            raise CDPError("the browser is not connected", gone=True)
        number = next(self._ids)
        message: dict[str, Any] = {"id": number, "method": method,
                                   "params": params or {}}
        if session:
            message["sessionId"] = session
        future: asyncio.Future[dict[str, Any]] = (
            asyncio.get_running_loop().create_future())
        self._pending[number] = future
        try:
            await self._ws.send(json.dumps(message))
            return await asyncio.wait_for(future, timeout)
        except ConnectionClosed:
            self._lost()
            raise CDPError("the browser went away", gone=True) from None
        except _Timeout:
            raise CDPError(f"the browser did not answer {method} within "
                           f"{timeout:g}s") from None
        finally:
            self._pending.pop(number, None)

    async def _read(self) -> None:
        try:
            while True:
                raw = await self._ws.recv()
                try:
                    message = json.loads(raw)
                except ValueError:
                    continue
                if "id" in message:
                    future = self._pending.get(message["id"])
                    if future is None or future.done():
                        continue
                    error = message.get("error")
                    if error:
                        future.set_exception(CDPError(
                            str(error.get("message") or error), code=error.get("code")))
                    else:
                        future.set_result(message.get("result") or {})
                    continue
                method = message.get("method")
                if not method:
                    continue
                for listener in list(self._listeners):
                    try:
                        listener(method, message.get("params") or {},
                                 message.get("sessionId") or "")
                    except Exception:  # noqa: S110 - a listener must not end the reader
                        pass
        except (ConnectionClosed, asyncio.CancelledError):
            pass
        finally:
            self._lost()

    def _lost(self) -> None:
        self.closed = True
        for future in self._pending.values():
            if not future.done():
                future.set_exception(CDPError("the browser went away", gone=True))
                future.exception()      # read it, so nobody is warned it was not

    async def close(self) -> None:
        self.closed = True
        self._reader.cancel()
        try:
            await self._reader
        except (asyncio.CancelledError, Exception):  # noqa: S110
            pass
        try:
            await self._ws.close()
        except Exception:  # noqa: S110 - closing a dead socket
            pass
        self._lost()
