"""Amazon S3 Vectors, and retrieval from Amazon Bedrock Knowledge Bases.

Amazon's other vector stores are databases this library already speaks to:
OpenSearch Service and OpenSearch Serverless (`OpenSearchStore`, signed with
SigV4), Aurora and RDS for PostgreSQL (`PgVectorStore`), and MemoryDB
(`RedisVectorStore`). SageMaker hosts models; it is not a vector store.
"""

from __future__ import annotations

import os
from typing import Any

from ..base import (
    HTTPVectorStore,
    VectorHit,
    VectorRecord,
    VectorStoreError,
    conditions,
    holds,
)

__all__ = ["S3VectorsStore", "BedrockKnowledgeBase"]


class _Signed(HTTPVectorStore):
    service = ""

    def __init__(self, *, region: str | None = None, access_key: str | None = None,
                 secret_key: str | None = None, session_token: str | None = None,
                 **kw: Any) -> None:
        super().__init__(**kw)
        self.region = region or os.environ.get("AWS_REGION") or os.environ.get(
            "AWS_DEFAULT_REGION") or "us-east-1"
        self._aws = (access_key, secret_key, session_token)

    def _secrets(self) -> list[str]:
        return [self._aws[1] or "", os.environ.get("AWS_SECRET_ACCESS_KEY", "")]

    def _sign(self, method: str, url: str, body: bytes,
              headers: dict[str, str]) -> dict[str, str]:
        from ...llm_providers._sigv4 import resolve_credentials, sign

        credentials = resolve_credentials(*self._aws)
        if not credentials:
            raise VectorStoreError(
                f"no AWS credentials were found for {self.name} — pass them, or set "
                "AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY", store=self.name)
        signed = sign(method=method, url=url, region=self.region, service=self.service,
                      body=body, credentials=credentials,
                      headers={k: v for k, v in headers.items()
                               if k.lower() == "content-type"})
        return {**headers, **signed}


class S3VectorsStore(_Signed):
    """
        S3VectorsStore(bucket="my-vector-bucket", index="handbook", region="eu-west-1")

    The vector bucket is created in AWS; the index is created here if it is not
    there. S3 Vectors answers with at most 30 results, and deletes by key — a
    delete by filter lists the index to find the keys.
    """

    name = "s3-vectors"
    service = "s3vectors"
    batch_size = 500
    max_k = 30

    def __init__(self, *, bucket: str, index: str = "knowledge", url: str = "",
                 **kw: Any) -> None:
        super().__init__(**kw)
        self.bucket, self.index = bucket, index
        self.url = (url or f"https://s3vectors.{self.region}.api.aws").rstrip("/")

    async def _call(self, operation: str, ok: tuple[int, ...] = (), **body: Any) -> Any:
        return await self._request(
            "POST", f"{self.url}/{operation}", ok=ok,
            json_body={"vectorBucketName": self.bucket, "indexName": self.index, **body})

    async def ensure(self, dimension: int) -> None:
        found = await self._call("GetIndex", ok=(404,))
        if found == 404:
            await self._call("CreateIndex", ok=(409,), dataType="float32",
                             dimension=dimension, distanceMetric="cosine",
                             metadataConfiguration={
                                 "nonFilterableMetadataKeys": ["_text"]})
            return
        held = (found.get("index") or {}).get("dimension")
        if held and int(held) != dimension:
            raise VectorStoreError(
                f"the S3 Vectors index holds {held}-dimension vectors; {dimension} "
                "was asked for", store=self.name)

    @staticmethod
    def _filter(wanted: list[tuple[str, str, Any]]) -> dict[str, Any] | None:
        parts = [{name: {op: value}} for name, op, value in wanted]
        if not parts:
            return None
        return parts[0] if len(parts) == 1 else {"$and": parts}

    async def _upsert(self, records: list[VectorRecord]) -> None:
        await self._call("PutVectors", vectors=[
            {"key": r.id, "data": {"float32": [float(x) for x in r.vector]},
             "metadata": {**{k: v for k, v in r.metadata.items() if v is not None},
                          "_text": r.text}} for r in records])

    async def _query(self, vector: list[float], k: int,
                     wanted: list[tuple[str, str, Any]]) -> list[VectorHit]:
        body: dict[str, Any] = {"queryVector": {"float32": [float(x) for x in vector]},
                                "topK": k, "returnMetadata": True, "returnDistance": True}
        found = self._filter(wanted)
        if found:
            body["filter"] = found
        answer = await self._call("QueryVectors", **body)
        out = []
        for row in answer.get("vectors") or []:
            meta = dict(row.get("metadata") or {})
            text = meta.pop("_text", "")
            out.append(VectorHit(row["key"], 1.0 - float(row.get("distance") or 0.0),
                                 text, meta))
        return out

    async def _listed(self, metadata: bool) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        token = None
        for _ in range(100_000):
            body: dict[str, Any] = {"maxResults": 500, "returnMetadata": metadata}
            if token:
                body["nextToken"] = token
            page = await self._call("ListVectors", **body)
            rows += page.get("vectors") or []
            token = page.get("nextToken")
            if not token:
                break
        return rows

    async def _delete(self, ids: list[str] | None,
                      wanted: list[tuple[str, str, Any]]) -> None:
        if wanted:
            ids = [row["key"] for row in await self._listed(True)
                   if (ids is None or row["key"] in ids)
                   and holds(row.get("metadata") or {}, wanted)]
        for start in range(0, len(ids or []), 500):
            await self._call("DeleteVectors", keys=(ids or [])[start:start + 500])

    async def count(self) -> int:
        return len(await self._listed(False))


class BedrockKnowledgeBase(_Signed):
    """Retrieval from a knowledge base that Amazon Bedrock manages.

        kb = BedrockKnowledgeBase("ABCDEFGHIJ", region="us-east-1")
        passages = await kb.retrieve("how long do refunds take?", k=5)
        agent = Agent("support", tools=[KnowledgeBase(retriever=kb).as_tool()])

    Bedrock does the chunking, the embedding and the storage; this asks it a
    question. It is a retriever, not a store: there is nothing to upsert.
    """

    name = "bedrock-knowledge-base"
    service = "bedrock"

    def __init__(self, knowledge_base_id: str, *, url: str = "", **kw: Any) -> None:
        super().__init__(**kw)
        self.knowledge_base_id = knowledge_base_id
        self.url = (url or f"https://bedrock-agent-runtime.{self.region}.amazonaws.com"
                    ).rstrip("/")

    @staticmethod
    def _filter(wanted: list[tuple[str, str, Any]]) -> dict[str, Any] | None:
        names = {"$eq": "equals", "$ne": "notEquals", "$in": "in", "$gt": "greaterThan",
                 "$gte": "greaterThanOrEquals", "$lt": "lessThan",
                 "$lte": "lessThanOrEquals"}
        parts = [{names[op]: {"key": name, "value": value}} for name, op, value in wanted]
        if not parts:
            return None
        return parts[0] if len(parts) == 1 else {"andAll": parts}

    async def retrieve(self, query: str, *, k: int = 5,
                       filter: dict[str, Any] | None = None) -> list[VectorHit]:
        search: dict[str, Any] = {"numberOfResults": max(1, min(int(k), 100))}
        found = self._filter(conditions(filter))
        if found:
            search["filter"] = found
        answer = await self._request(
            "POST", f"{self.url}/knowledgebases/{self.knowledge_base_id}/retrieve",
            json_body={"retrievalQuery": {"text": query},
                       "retrievalConfiguration": {"vectorSearchConfiguration": search}})
        out = []
        for n, row in enumerate(answer.get("retrievalResults") or []):
            location = row.get("location") or {}
            where = next((v.get("uri") or v.get("url") for v in location.values()
                          if isinstance(v, dict) and (v.get("uri") or v.get("url"))), "")
            meta = {**(row.get("metadata") or {}), **({"source": where} if where else {})}
            out.append(VectorHit(str(meta.get("x-amz-bedrock-kb-chunk-id") or n),
                                 float(row.get("score") or 0.0),
                                 (row.get("content") or {}).get("text") or "", meta))
        return out

    # A retriever has no records of its own to manage.
    async def ensure(self, dimension: int) -> None:
        return None

    async def _upsert(self, records: list[VectorRecord]) -> None:
        raise VectorStoreError("a Bedrock knowledge base is filled from its data "
                               "sources in AWS, not from here", store=self.name)

    async def _query(self, vector: list[float], k: int,
                     wanted: list[tuple[str, str, Any]]) -> list[VectorHit]:
        raise VectorStoreError("a Bedrock knowledge base is asked in words — use "
                               "retrieve(query)", store=self.name)

    async def _delete(self, ids: list[str] | None,
                      wanted: list[tuple[str, str, Any]]) -> None:
        raise VectorStoreError("a Bedrock knowledge base is managed in AWS",
                               store=self.name)

    async def count(self) -> int:
        raise VectorStoreError("a Bedrock knowledge base does not say how much it holds",
                               store=self.name)
