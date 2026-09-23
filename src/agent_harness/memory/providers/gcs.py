"""Google Cloud Storage memory."""

from __future__ import annotations

import asyncio
from typing import Any, ClassVar

from ...errors import ConfigurationError
from ._objectstore import ObjectStoreMemory

__all__ = ["GCSMemory"]


class GCSMemory(ObjectStoreMemory):
    """Memory in a Google Cloud Storage bucket.

        GCSMemory("my-bucket", project="my-project")

    Needs `google-cloud-storage`. The client is synchronous, so calls run in a
    worker thread.
    """

    driver_hint: ClassVar[str] = "pip install google-cloud-storage"

    def __init__(self, bucket: str, *, prefix: str = "agent-memory",
                 project: str | None = None, credentials: Any = None,
                 client: Any = None, **options: Any) -> None:
        super().__init__(bucket, prefix=prefix)
        self.project = project
        self.credentials = credentials
        self._client = client
        self._bucket: Any = None
        self._options = options
        self._lock = asyncio.Lock()

    async def _ensure(self) -> Any:
        if self._bucket is not None:
            return self._bucket
        async with self._lock:
            if self._bucket is None:
                if self._client is None:
                    try:
                        from google.cloud import storage
                    except ImportError as exc:
                        raise ConfigurationError(
                            "GCSMemory needs a driver — " + self.driver_hint
                        ) from exc
                    self._client = await asyncio.to_thread(
                        lambda: storage.Client(project=self.project,
                                               credentials=self.credentials,
                                               **self._options))
                self._bucket = await asyncio.to_thread(self._client.bucket,
                                                       self.bucket)
        return self._bucket

    async def _put(self, key: str, body: bytes, content_type: str) -> None:
        bucket = await self._ensure()
        blob = bucket.blob(key)
        await asyncio.to_thread(blob.upload_from_string, body,
                                content_type=content_type)

    async def _get(self, key: str) -> bytes | None:
        bucket = await self._ensure()
        blob = bucket.blob(key)

        def read() -> bytes | None:
            try:
                return blob.download_as_bytes()
            except Exception as exc:
                if type(exc).__name__ == "NotFound":
                    return None
                raise

        return await asyncio.to_thread(read)

    async def _list(self, prefix: str) -> list[str]:
        bucket = await self._ensure()
        return await asyncio.to_thread(
            lambda: [b.name for b in bucket.list_blobs(prefix=prefix)])

    async def _delete(self, keys: list[str]) -> None:
        bucket = await self._ensure()

        def delete() -> None:
            # A batch keeps this to one round trip per 100 objects.
            for start in range(0, len(keys), 100):
                with self._client.batch():
                    for key in keys[start:start + 100]:
                        bucket.blob(key).delete()

        await asyncio.to_thread(delete)
