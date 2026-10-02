"""Sessions in DynamoDB.

An item may not exceed 400 KB, and a conversation easily does. So a session is
a small *head* item — who owns it, its version, how many pieces — and its body,
compressed and cut into pieces of its own. The head is written last and only if
the version still matches: until that one conditional write succeeds, the new
pieces are invisible, and the old ones are removed only after it has.

    pk = _session#{id}    sk = head | body#{rev}#{n}
    pk = _sessions#{tenant}#{user}  (and #_all)    sk = {id}      ← for listing
"""

from __future__ import annotations

import asyncio
import json
import zlib
from typing import Any

from ..runtime.session import DurableSessionStore, Session
from ..types import new_id

__all__ = ["DynamoDBSessionStore"]

_PIECE = 350_000


class DynamoDBSessionStore(DurableSessionStore):
    """Sessions in the DynamoDB memory backend's table (`pk` / `sk` keys).

        DynamoDBSessionStore(DynamoDBMemory("agents", region_name="eu-west-1"))
    """

    def __init__(self, dynamo: Any) -> None:
        self.dynamo = dynamo

    @property
    def _table(self) -> str:
        return self.dynamo.table_name

    @staticmethod
    def _pk(session_id: str) -> dict[str, str]:
        return {"S": f"_session#{session_id}"}

    @staticmethod
    def _owner(tenant_id: str | None, user_id: str | None) -> str:
        return f"_sessions#{tenant_id or '_'}#{user_id or '_'}"

    async def _write(self, session: Session, expect: int) -> bool:
        client = await self.dynamo._ensure()
        rev = new_id("rev")
        body = zlib.compress(session.model_dump_json().encode())
        pieces = [body[i:i + _PIECE] for i in range(0, len(body), _PIECE)] or [b""]
        owner = self._owner(session.tenant_id, session.user_id)

        def write() -> bool:
            for n, piece in enumerate(pieces):
                client.put_item(TableName=self._table, Item={
                    "pk": self._pk(session.id), "sk": {"S": f"body#{rev}#{n:05d}"},
                    "body": {"B": piece}})
            head = {
                "pk": self._pk(session.id), "sk": {"S": "head"},
                "version": {"N": str(session.version)}, "rev": {"S": rev},
                "pieces": {"N": str(len(pieces))}, "owner": {"S": owner},
            }
            condition: dict[str, Any] = (
                {"ConditionExpression": "attribute_not_exists(pk)"} if expect == 0
                else {"ConditionExpression": "#v = :expect",
                      "ExpressionAttributeNames": {"#v": "version"},
                      "ExpressionAttributeValues": {":expect": {"N": str(expect)}}})
            try:
                old = client.put_item(TableName=self._table, Item=head,
                                      ReturnValues="ALL_OLD", **condition)
            except Exception as exc:
                if type(exc).__name__ != "ConditionalCheckFailedException":
                    raise
                self._drop(client, session.id, rev, len(pieces))
                return False
            was = old.get("Attributes") or {}
            if was:
                self._drop(client, session.id, was["rev"]["S"],
                           int(was["pieces"]["N"]))
                if was.get("owner", {}).get("S") not in (None, owner):
                    client.delete_item(TableName=self._table, Key={
                        "pk": {"S": was["owner"]["S"]}, "sk": {"S": session.id}})
            row = {"sk": {"S": session.id}, "updated": {"N": repr(session.updated)},
                   "agent": {"S": session.agent or "_"}}
            for pk in (owner, "_sessions#_all"):
                client.put_item(TableName=self._table, Item={"pk": {"S": pk}, **row})
            return True

        return await asyncio.to_thread(write)

    def _drop(self, client: Any, session_id: str, rev: str, pieces: int) -> None:
        for n in range(pieces):
            client.delete_item(TableName=self._table, Key={
                "pk": self._pk(session_id), "sk": {"S": f"body#{rev}#{n:05d}"}})

    def _load(self, client: Any, session_id: str) -> Session | None:
        head = client.get_item(TableName=self._table, ConsistentRead=True, Key={
            "pk": self._pk(session_id), "sk": {"S": "head"}}).get("Item")
        if not head:
            return None
        body = b""
        for n in range(int(head["pieces"]["N"])):
            piece = client.get_item(TableName=self._table, ConsistentRead=True, Key={
                "pk": self._pk(session_id),
                "sk": {"S": f"body#{head['rev']['S']}#{n:05d}"}}).get("Item")
            if not piece:
                return None           # replaced under us: the caller reads again
            body += bytes(piece["body"]["B"])
        try:
            found = Session(**json.loads(zlib.decompress(body)))
        except (zlib.error, json.JSONDecodeError, ValueError, TypeError):
            return None
        found.version = int(head["version"]["N"])
        return found

    async def _read(self, session_id: str) -> Session | None:
        client = await self.dynamo._ensure()

        def read() -> Session | None:
            # A save can swap the pieces between the two reads; once more is
            # enough, because the head it then finds is the new one.
            return self._load(client, session_id) or self._load(client, session_id)

        return await asyncio.to_thread(read)

    async def _list(self, *, limit: int, user_id: str | None,
                    tenant_id: str | None, agent: str | None) -> list[Session]:
        client = await self.dynamo._ensure()
        exact = user_id is not None and tenant_id is not None
        pk = self._owner(tenant_id, user_id) if exact else "_sessions#_all"

        def listing() -> list[Session]:
            rows: list[dict[str, Any]] = []
            kwargs: dict[str, Any] = {
                "TableName": self._table, "KeyConditionExpression": "pk = :pk",
                "ExpressionAttributeValues": {":pk": {"S": pk}}}
            while True:
                page = client.query(**kwargs)
                rows += page.get("Items", [])
                if not page.get("LastEvaluatedKey"):
                    break
                kwargs["ExclusiveStartKey"] = page["LastEvaluatedKey"]
            rows.sort(key=lambda r: -float(r["updated"]["N"]))
            out: list[Session] = []
            for row in rows:
                if agent is not None and row.get("agent", {}).get("S") != agent:
                    continue
                found = self._load(client, row["sk"]["S"])
                if (found is not None
                        and (user_id is None or found.user_id == user_id)
                        and (tenant_id is None or found.tenant_id == tenant_id)):
                    out.append(found)
                    if len(out) >= limit:
                        break
            return out

        return await asyncio.to_thread(listing)

    async def _delete(self, session_id: str) -> None:
        client = await self.dynamo._ensure()

        def delete() -> None:
            head = client.get_item(TableName=self._table, ConsistentRead=True, Key={
                "pk": self._pk(session_id), "sk": {"S": "head"}}).get("Item")
            if not head:
                return
            client.delete_item(TableName=self._table, Key={
                "pk": self._pk(session_id), "sk": {"S": "head"}})
            self._drop(client, session_id, head["rev"]["S"], int(head["pieces"]["N"]))
            for pk in (head["owner"]["S"], "_sessions#_all"):
                client.delete_item(TableName=self._table, Key={
                    "pk": {"S": pk}, "sk": {"S": session_id}})

        await asyncio.to_thread(delete)

    async def aclose(self) -> None:
        await self.dynamo.aclose()
