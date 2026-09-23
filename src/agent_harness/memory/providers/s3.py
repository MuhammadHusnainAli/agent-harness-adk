"""Amazon S3 memory (and anything S3-compatible: MinIO, R2, Spaces)."""

from __future__ import annotations

import asyncio
from typing import Any, ClassVar

from ...errors import ConfigurationError
from ._objectstore import ObjectStoreMemory

__all__ = ["S3Memory"]


class S3Memory(ObjectStoreMemory):
    """Memory in an S3 bucket.

        S3Memory("my-bucket", region_name="eu-west-1")
        S3Memory("my-bucket", endpoint_url="http://localhost:9000")   # MinIO

    Needs `boto3`. Calls run in a worker thread, so a slow bucket never blocks
    the event loop.
    """

    driver_hint: ClassVar[str] = "pip install boto3"

    def __init__(self, bucket: str, *, prefix: str = "agent-memory",
                 client: Any = None, **options: Any) -> None:
        super().__init__(bucket, prefix=prefix)
        self._client = client
        self._client_options = options
        self._lock = asyncio.Lock()

    async def _ensure(self) -> Any:
        if self._client is not None:
            return self._client
        async with self._lock:
            if self._client is None:
                try:
                    import boto3
                except ImportError as exc:
                    raise ConfigurationError(
                        "S3Memory needs a driver — " + self.driver_hint) from exc
                self._client = await asyncio.to_thread(
                    lambda: boto3.client("s3", **self._client_options))
        return self._client

    async def _put(self, key: str, body: bytes, content_type: str) -> None:
        client = await self._ensure()
        await asyncio.to_thread(client.put_object, Bucket=self.bucket, Key=key,
                                Body=body, ContentType=content_type)

    async def _get(self, key: str) -> bytes | None:
        client = await self._ensure()

        def read() -> bytes | None:
            try:
                return client.get_object(Bucket=self.bucket, Key=key)["Body"].read()
            except Exception as exc:                 # the SDK's NoSuchKey
                if type(exc).__name__ in {"NoSuchKey", "ClientError", "404"}:
                    return None
                raise

        return await asyncio.to_thread(read)

    async def _list(self, prefix: str) -> list[str]:
        client = await self._ensure()

        def listing() -> list[str]:
            keys: list[str] = []
            token: str | None = None
            while True:
                kwargs: dict[str, Any] = {"Bucket": self.bucket, "Prefix": prefix,
                                          "MaxKeys": 1000}
                if token:
                    kwargs["ContinuationToken"] = token
                page = client.list_objects_v2(**kwargs)
                keys += [row["Key"] for row in page.get("Contents", [])]
                token = page.get("NextContinuationToken")
                if not page.get("IsTruncated"):
                    return keys

        return await asyncio.to_thread(listing)

    async def _delete(self, keys: list[str]) -> None:
        client = await self._ensure()

        def delete() -> None:
            for start in range(0, len(keys), 1000):   # the API's batch ceiling
                client.delete_objects(
                    Bucket=self.bucket,
                    Delete={"Objects": [{"Key": k} for k in keys[start:start + 1000]]})

        await asyncio.to_thread(delete)
