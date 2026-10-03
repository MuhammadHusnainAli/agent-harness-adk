"""A WebSocket client, small enough to read in one sitting.

Realtime speech models are reached over a WebSocket, and the harness's one HTTP
dependency does not speak it. Rather than add a dependency for a handshake and
a framing rule, this is RFC 6455's client side over `asyncio` streams: text and
binary messages, fragmentation, ping and pong, an orderly close. It does not
negotiate compression — audio is already compressed or does not compress — and
it does not go through an HTTP proxy.

    ws = await WebSocket.connect("wss://host/path", headers={"authorization": "…"})
    await ws.send('{"type": "hello"}')
    async for message in ws:
        ...
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import os
import ssl
import struct
from typing import Any
from urllib.parse import urlsplit

from ..errors import HarnessError

__all__ = ["WebSocket", "ConnectionClosed"]

_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
_TEXT, _BINARY, _CLOSE, _PING, _PONG, _CONTINUATION = 0x1, 0x2, 0x8, 0x9, 0xA, 0x0

# On Python 3.10 asyncio.TimeoutError is a distinct class from the builtin.
_Timeout = (TimeoutError, asyncio.TimeoutError)


class ConnectionClosed(HarnessError):
    """The other end closed the connection, or it was lost."""

    def __init__(self, code: int = 1006, reason: str = "") -> None:
        super().__init__(f"websocket closed ({code}){f': {reason}' if reason else ''}")
        self.code = code
        self.reason = reason


def _mask(data: bytes, key: bytes) -> bytes:
    """XOR with the repeating key — as two big integers, so it runs in C."""
    if not data:
        return data
    size = len(data)
    pad = (key * (size // 4 + 1))[:size]
    return (int.from_bytes(data, "big") ^ int.from_bytes(pad, "big")).to_bytes(
        size, "big")


class WebSocket:
    """One open connection. Build it with `WebSocket.connect`."""

    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter, *,
                 max_size: int = 32 * 1024 * 1024) -> None:
        self._reader = reader
        self._writer = writer
        self.max_size = max_size
        self.closed = False
        self._sending = asyncio.Lock()

    @classmethod
    async def connect(cls, url: str, *, headers: dict[str, str] | None = None,
                      timeout: float = 20.0, max_size: int = 32 * 1024 * 1024,
                      ssl_context: ssl.SSLContext | None = None) -> WebSocket:
        parts = urlsplit(url)
        if parts.scheme not in ("ws", "wss"):
            raise HarnessError(f"not a websocket URL: {url!r}")
        secure = parts.scheme == "wss"
        host = parts.hostname or ""
        port = parts.port or (443 if secure else 80)
        context = (ssl_context or ssl.create_default_context()) if secure else None
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port, ssl=context,
                                        server_hostname=host if secure else None),
                timeout)
        except _Timeout:
            raise HarnessError(f"timed out connecting to {host}:{port}") from None
        except OSError as exc:
            raise HarnessError(f"could not connect to {host}:{port}: {exc}") from None

        key = base64.b64encode(os.urandom(16)).decode()
        path = parts.path or "/"
        if parts.query:
            path += f"?{parts.query}"
        default_port = port in (80, 443)
        lines = [f"GET {path} HTTP/1.1",
                 f"Host: {host}" + ("" if default_port else f":{port}"),
                 "Upgrade: websocket", "Connection: Upgrade",
                 f"Sec-WebSocket-Key: {key}", "Sec-WebSocket-Version: 13"]
        taken = {"host", "upgrade", "connection", "sec-websocket-key",
                 "sec-websocket-version"}
        for name, value in (headers or {}).items():
            if name.lower() not in taken:
                lines.append(f"{name}: {value}")
        writer.write(("\r\n".join(lines) + "\r\n\r\n").encode())
        try:
            await writer.drain()
            head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout)
        except (_Timeout[0], _Timeout[1], asyncio.IncompleteReadError, OSError) as exc:
            writer.close()
            raise HarnessError(f"the websocket handshake with {host} failed: "
                               f"{type(exc).__name__}") from None

        status, *rest = head.decode("latin-1").split("\r\n")
        answer = {line.split(":", 1)[0].strip().lower(): line.split(":", 1)[1].strip()
                  for line in rest if ":" in line}
        if " 101" not in status:
            # The body usually says why: a bad key, an unknown model.
            body = b""
            try:
                body = await asyncio.wait_for(reader.read(2048), 2)
            except (_Timeout[0], _Timeout[1], OSError):
                pass
            writer.close()
            raise HarnessError(
                f"{host} refused the websocket: {status.strip()}"
                + (f" — {body.decode(errors='replace').strip()[:400]}" if body else ""))
        expected = base64.b64encode(
            hashlib.sha1((key + _GUID).encode()).digest()).decode()  # noqa: S324
        if answer.get("sec-websocket-accept") != expected:
            writer.close()
            raise HarnessError(f"{host} did not complete the websocket handshake")
        return cls(reader, writer, max_size=max_size)

    # ---- sending ----------------------------------------------------------
    async def _frame(self, opcode: int, payload: bytes) -> None:
        size = len(payload)
        head = bytearray([0x80 | opcode])
        if size < 126:
            head.append(0x80 | size)
        elif size < 65536:
            head.append(0x80 | 126)
            head += struct.pack("!H", size)
        else:
            head.append(0x80 | 127)
            head += struct.pack("!Q", size)
        key = os.urandom(4)
        async with self._sending:
            self._writer.write(bytes(head) + key + _mask(payload, key))
            await self._writer.drain()

    async def send(self, message: str | bytes) -> None:
        if self.closed:
            raise ConnectionClosed(1006, "the connection is closed")
        try:
            if isinstance(message, str):
                await self._frame(_TEXT, message.encode())
            else:
                await self._frame(_BINARY, bytes(message))
        except OSError as exc:
            self.closed = True
            raise ConnectionClosed(1006, str(exc)) from None

    # ---- receiving -----------------------------------------------------------
    async def _read(self) -> tuple[bool, int, bytes]:
        first, second = await self._reader.readexactly(2)
        size = second & 0x7F
        if size == 126:
            (size,) = struct.unpack("!H", await self._reader.readexactly(2))
        elif size == 127:
            (size,) = struct.unpack("!Q", await self._reader.readexactly(8))
        if size > self.max_size:
            raise ConnectionClosed(1009, f"a {size}-byte frame is more than this "
                                         "connection accepts")
        key = await self._reader.readexactly(4) if second & 0x80 else b""
        payload = await self._reader.readexactly(size) if size else b""
        return bool(first & 0x80), first & 0x0F, _mask(payload, key) if key else payload

    async def recv(self) -> str | bytes:
        """The next whole message. Raises `ConnectionClosed` when there is none."""
        if self.closed:
            raise ConnectionClosed(1006, "the connection is closed")
        kind, parts, total = 0, [], 0
        while True:
            try:
                final, opcode, payload = await self._read()
            except (asyncio.IncompleteReadError, OSError) as exc:
                self.closed = True
                raise ConnectionClosed(1006, type(exc).__name__) from None
            if opcode == _PING:
                await self._frame(_PONG, payload)
                continue
            if opcode == _PONG:
                continue
            if opcode == _CLOSE:
                code = struct.unpack("!H", payload[:2])[0] if len(payload) >= 2 else 1005
                reason = payload[2:].decode(errors="replace")
                await self._goodbye(code if code != 1005 else 1000)
                raise ConnectionClosed(code, reason)
            if opcode != _CONTINUATION:
                kind = opcode
            parts.append(payload)
            total += len(payload)
            if total > self.max_size:
                await self._goodbye(1009)
                raise ConnectionClosed(1009, "the message is more than this "
                                             "connection accepts")
            if final:
                data = b"".join(parts)
                return data.decode(errors="replace") if kind == _TEXT else data

    def __aiter__(self) -> WebSocket:
        return self

    async def __anext__(self) -> str | bytes:
        try:
            return await self.recv()
        except ConnectionClosed:
            raise StopAsyncIteration from None

    # ---- closing ---------------------------------------------------------------
    async def _goodbye(self, code: int) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            await self._frame(_CLOSE, struct.pack("!H", code))
        except OSError:
            pass
        self._writer.close()

    async def close(self, code: int = 1000) -> None:
        await self._goodbye(code)
        try:
            await asyncio.wait_for(self._writer.wait_closed(), 2)
        except (_Timeout[0], _Timeout[1], OSError):
            pass

    async def __aenter__(self) -> WebSocket:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()
