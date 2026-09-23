"""DynamoDB memory. Serverless, and it scales without anyone watching it."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, ClassVar

from ...errors import ConfigurationError
from ..base import MemoryRecord, MemoryStore
from ..trace import Trace

__all__ = ["DynamoDBMemory"]


class DynamoDBMemory(MemoryStore):
    """One table, partitioned by namespace and sorted by time.

        DynamoDBMemory("agent-memory", region_name="eu-west-1")

    Partition key `pk` is `{namespace}#{scope}`, sort key `sk` is
    `{ts}#{id}` — so reading one user's memory is a single query, newest first,
    and `limit` is served by DynamoDB. Needs `boto3`.
    """

    driver_hint: ClassVar[str] = "pip install boto3"

    def __init__(self, table: str = "agent_memory", *, client: Any = None,
                 create: bool = False, **options: Any) -> None:
        self.table_name = table
        self.create = create
        self._client = client
        self._options = options
        self._ready = client is not None
        self._lock = asyncio.Lock()

    async def _ensure(self) -> Any:
        if self._client is None:
            async with self._lock:
                if self._client is None:
                    try:
                        import boto3
                    except ImportError as exc:
                        raise ConfigurationError(
                            "DynamoDBMemory needs a driver — " + self.driver_hint
                        ) from exc
                    self._client = await asyncio.to_thread(
                        lambda: boto3.client("dynamodb", **self._options))
        if self.create and not self._ready:
            await self._create_table()
            self._ready = True
        return self._client

    async def _create_table(self) -> None:
        def make() -> None:
            try:
                self._client.create_table(
                    TableName=self.table_name,
                    KeySchema=[{"AttributeName": "pk", "KeyType": "HASH"},
                               {"AttributeName": "sk", "KeyType": "RANGE"}],
                    AttributeDefinitions=[
                        {"AttributeName": "pk", "AttributeType": "S"},
                        {"AttributeName": "sk", "AttributeType": "S"}],
                    BillingMode="PAY_PER_REQUEST")
            except Exception as exc:
                if type(exc).__name__ != "ResourceInUseException":
                    raise

        await asyncio.to_thread(make)

    @staticmethod
    def _namespace(trace: Trace | None, record: MemoryRecord | None = None) -> str:
        if trace is not None:
            return trace.slug
        if record is not None:
            return Trace(user_id=record.user_id, session_id=record.session_id,
                         tenant_id=record.tenant_id).slug
        return "_shared"

    # ---- records ---------------------------------------------------------------
    async def append(self, record: MemoryRecord) -> MemoryRecord:
        client = await self._ensure()
        item = {
            "pk": {"S": f"{self._namespace(None, record)}#{record.scope}"},
            "sk": {"S": f"{record.ts:.6f}#{record.id}"},
            "body": {"S": record.model_dump_json()},
            "kind": {"S": record.kind},
        }
        await asyncio.to_thread(client.put_item, TableName=self.table_name,
                                Item=item)
        return record

    async def all(self, scope: str | None = None, *, limit: int | None = None,
                  kind: str | None = None,
                  trace: Trace | None = None) -> list[MemoryRecord]:
        client = await self._ensure()
        namespace = self._namespace(trace)
        scopes = [scope] if scope else ["user", "session", "orchestrator", "job"]

        def query(one: str) -> list[dict[str, Any]]:
            kwargs: dict[str, Any] = {
                "TableName": self.table_name,
                "KeyConditionExpression": "pk = :pk",
                "ExpressionAttributeValues": {":pk": {"S": f"{namespace}#{one}"}},
                "ScanIndexForward": False,          # newest first
            }
            if limit and not kind:
                kwargs["Limit"] = int(limit)
            return client.query(**kwargs).get("Items", [])

        rows: list[MemoryRecord] = []
        for items in await asyncio.gather(
                *(asyncio.to_thread(query, one) for one in scopes)):
            for item in items:
                try:
                    rows.append(MemoryRecord(**json.loads(item["body"]["S"])))
                except (json.JSONDecodeError, KeyError, ValueError):
                    continue
        if kind:
            rows = [r for r in rows if r.kind == kind]
        if trace is not None:
            rows = [r for r in rows if trace.matches(r)]
        rows.sort(key=lambda r: r.ts)
        return rows[-limit:] if limit else rows

    async def clear(self, scope: str | None = None, *,
                    trace: Trace | None = None) -> None:
        client = await self._ensure()
        rows = await self.all(scope, trace=trace)
        namespace = self._namespace(trace)

        def delete() -> None:
            for record in rows:
                client.delete_item(
                    TableName=self.table_name,
                    Key={"pk": {"S": f"{namespace}#{record.scope}"},
                         "sk": {"S": f"{record.ts:.6f}#{record.id}"}})

        await asyncio.to_thread(delete)

    # ---- documents ---------------------------------------------------------------
    async def read_doc(self, name: str, *, trace: Trace | None = None) -> str:
        client = await self._ensure()
        found = await asyncio.to_thread(
            client.get_item, TableName=self.table_name,
            Key={"pk": {"S": "_docs"}, "sk": {"S": self.doc_key(name, trace)}})
        return found.get("Item", {}).get("body", {}).get("S", "")

    async def write_doc(self, name: str, text: str, *,
                        trace: Trace | None = None) -> None:
        client = await self._ensure()
        await asyncio.to_thread(
            client.put_item, TableName=self.table_name,
            Item={"pk": {"S": "_docs"},
                  "sk": {"S": self.doc_key(name, trace)},
                  "body": {"S": text},
                  "updated": {"N": str(time.time())}})
