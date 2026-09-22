"""Result cache: an identical task with identical input is served, not re-run."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

__all__ = ["ResultCache"]


def cache_key(*parts: Any) -> str:
    blob = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.blake2b(blob.encode(), digest_size=16).hexdigest()


class ResultCache:
    """TTL cache with an optional on-disk tier. Never pay twice for the same call."""

    def __init__(self, *, ttl: float = 3600.0, max_entries: int = 2048,
                 path: str | Path | None = None, enabled: bool = True) -> None:
        self.ttl = ttl
        self.max_entries = max_entries
        self.enabled = enabled
        self.path = Path(path) if path else None
        self._data: dict[str, tuple[float, Any]] = {}
        self.hits = 0
        self.misses = 0
        if self.path:
            self.path.mkdir(parents=True, exist_ok=True)

    def key(self, *parts: Any) -> str:
        return cache_key(*parts)

    def get(self, key: str) -> Any | None:
        if not self.enabled:
            return None
        entry = self._data.get(key)
        if entry is None and self.path:
            entry = self._read_file(key)
        if entry is None:
            self.misses += 1
            return None
        expires, value = entry
        if expires and expires < time.time():
            self._data.pop(key, None)
            self.misses += 1
            return None
        self.hits += 1
        return value

    def set(self, key: str, value: Any, *, ttl: float | None = None) -> None:
        if not self.enabled:
            return
        expires = time.time() + (self.ttl if ttl is None else ttl)
        if len(self._data) >= self.max_entries:
            oldest = min(self._data, key=lambda k: self._data[k][0])
            self._data.pop(oldest, None)
        self._data[key] = (expires, value)
        if self.path:
            self._write_file(key, expires, value)

    def _file(self, key: str) -> Path:
        assert self.path is not None
        return self.path / f"{key}.json"

    def _read_file(self, key: str) -> tuple[float, Any] | None:
        file = self._file(key)
        if not file.exists():
            return None
        try:
            blob = json.loads(file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        return blob.get("expires", 0.0), blob.get("value")

    def _write_file(self, key: str, expires: float, value: Any) -> None:
        try:
            self._file(key).write_text(
                json.dumps({"expires": expires, "value": value}, default=str),
                encoding="utf-8",
            )
        except (OSError, TypeError, ValueError):
            pass  # a cache that cannot write is still a working cache

    def clear(self) -> None:
        self._data.clear()
        if self.path:
            for file in self.path.glob("*.json"):
                file.unlink(missing_ok=True)

    def stats(self) -> dict[str, int | float]:
        total = self.hits + self.misses
        return {"hits": self.hits, "misses": self.misses, "entries": len(self._data),
                "hit_rate": round(self.hits / total, 3) if total else 0.0}
