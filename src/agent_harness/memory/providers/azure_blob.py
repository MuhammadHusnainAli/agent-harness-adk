"""Azure Blob Storage memory."""

from __future__ import annotations

import asyncio
from typing import Any, ClassVar

from ...errors import ConfigurationError
from ._objectstore import ObjectStoreMemory

__all__ = ["AzureBlobMemory"]


class AzureBlobMemory(ObjectStoreMemory):
    """Memory in an Azure blob container.

        AzureBlobMemory("agents", connection_string="DefaultEndpointsProtocol=...")
        AzureBlobMemory("agents", account_url="https://acct.blob.core.windows.net",
                        credential=DefaultAzureCredential())

    Needs `azure-storage-blob`. The async client is used directly, so nothing
    is offloaded to a thread.
    """

    driver_hint: ClassVar[str] = "pip install azure-storage-blob aiohttp"

    def __init__(self, container: str, *, prefix: str = "agent-memory",
                 connection_string: str = "", account_url: str = "",
                 credential: Any = None, client: Any = None, **options: Any) -> None:
        super().__init__(container, prefix=prefix)
        self.connection_string = connection_string
        self.account_url = account_url
        self.credential = credential
        self._service = client
        self._options = options
        self._lock = asyncio.Lock()

    async def _ensure(self) -> Any:
        if self._service is not None:
            return self._service
        async with self._lock:
            if self._service is None:
                try:
                    from azure.storage.blob.aio import BlobServiceClient
                except ImportError as exc:
                    raise ConfigurationError(
                        "AzureBlobMemory needs a driver — " + self.driver_hint
                    ) from exc
                if self.connection_string:
                    self._service = BlobServiceClient.from_connection_string(
                        self.connection_string, **self._options)
                elif self.account_url:
                    self._service = BlobServiceClient(
                        self.account_url, credential=self.credential, **self._options)
                else:
                    raise ConfigurationError(
                        "AzureBlobMemory needs a connection_string or an account_url")
        return self._service

    async def _container(self) -> Any:
        service = await self._ensure()
        return service.get_container_client(self.bucket)

    async def _put(self, key: str, body: bytes, content_type: str) -> None:
        from azure.storage.blob import ContentSettings

        container = await self._container()
        await container.upload_blob(
            name=key, data=body, overwrite=True,
            content_settings=ContentSettings(content_type=content_type))

    async def _get(self, key: str) -> bytes | None:
        container = await self._container()
        try:
            stream = await container.download_blob(key)
            return await stream.readall()
        except Exception as exc:
            if type(exc).__name__ == "ResourceNotFoundError":
                return None
            raise

    async def _list(self, prefix: str) -> list[str]:
        container = await self._container()
        return [blob.name async for blob in
                container.list_blobs(name_starts_with=prefix)]

    async def _delete(self, keys: list[str]) -> None:
        container = await self._container()
        for start in range(0, len(keys), 256):       # the API's batch ceiling
            await container.delete_blobs(*keys[start:start + 256])

    async def aclose(self) -> None:
        if self._service is not None:
            await self._service.close()
            self._service = None
