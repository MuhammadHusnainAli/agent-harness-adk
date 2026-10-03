"""Is this address on the public internet? Asked before the harness fetches
anything a model named — a page, a reference image."""

from __future__ import annotations

import asyncio
import ipaddress

__all__ = ["is_private_host"]

_Timeout = (TimeoutError, asyncio.TimeoutError)
_LOCAL_NAMES = (".localhost", ".local", ".internal", ".lan", ".home.arpa")


async def is_private_host(host: str, cache: dict[str, bool] | None = None) -> bool:
    """Does this name lead to this machine or a private network?

    The name is resolved, so one that merely points at a private address is
    caught too. A name that does not resolve is not private: it will not load.
    """
    host = host.lower().rstrip(".")
    if cache is not None and host in cache:
        return cache[host]
    private = host == "localhost" or host.endswith(_LOCAL_NAMES)
    if not private:
        addresses: list[str] = []
        try:
            ipaddress.ip_address(host.strip("[]"))
            addresses = [host.strip("[]")]
        except ValueError:
            try:
                found = await asyncio.wait_for(
                    asyncio.get_running_loop().getaddrinfo(host, None), 4)
                addresses = [str(info[4][0]) for info in found]
            except (OSError, *_Timeout):
                addresses = []
        for address in addresses:
            try:
                ip = ipaddress.ip_address(address.split("%")[0])
            except ValueError:
                continue
            if (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved
                    or ip.is_multicast or ip.is_unspecified):
                private = True
                break
    if cache is not None:
        cache[host] = private
    return private
