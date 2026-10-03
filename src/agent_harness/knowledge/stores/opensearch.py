"""OpenSearch and Elasticsearch — self-hosted, Elastic Cloud, Amazon OpenSearch
Service, and Amazon OpenSearch Serverless."""

from __future__ import annotations

import base64
import json
import os
from typing import Any

from ..base import HTTPVectorStore, VectorHit, VectorRecord, VectorStoreError, normalise

__all__ = ["OpenSearchStore", "ElasticsearchStore"]

_RANGE = {"$gt": "gt", "$gte": "gte", "$lt": "lt", "$lte": "lte"}


class OpenSearchStore(HTTPVectorStore):
    """
        OpenSearchStore("http://localhost:9200", index="handbook")
        OpenSearchStore("https://host:9200", index="handbook", username="…", password="…")
        OpenSearchStore("https://search-x.eu-west-1.es.amazonaws.com", index="handbook",
                        aws_region="eu-west-1")                    # signed with SigV4
        OpenSearchStore("https://x.eu-west-1.aoss.amazonaws.com", index="handbook",
                        aws_region="eu-west-1", serverless=True)   # OpenSearch Serverless

    AWS credentials come from the arguments, then the environment, then
    botocore if it is installed.

    OpenSearch Serverless does not let a document choose its own id and takes a
    few seconds to show what it was given, so there an upsert first removes the
    record it replaces.
    """

    name = "opensearch"
    batch_size = 500
    flavour = "opensearch"

    def __init__(self, url: str = "http://localhost:9200", *, index: str = "knowledge",
                 username: str | None = None, password: str | None = None,
                 api_key: str | None = None, aws_region: str | None = None,
                 aws_access_key: str | None = None, aws_secret_key: str | None = None,
                 aws_session_token: str | None = None, serverless: bool | None = None,
                 engine: str | None = None, refresh: bool = True, **kw: Any) -> None:
        super().__init__(**kw)
        self.url = url.rstrip("/")
        self.index = index
        self.username = username or os.environ.get("OPENSEARCH_USERNAME", "")
        self.password = password or os.environ.get("OPENSEARCH_PASSWORD", "")
        self.api_key = api_key or ""
        self.aws_region = aws_region
        self._aws = (aws_access_key, aws_secret_key, aws_session_token)
        self.serverless = (".aoss." in url) if serverless is None else serverless
        # Serverless runs faiss, and scores an inner product; vectors are sent
        # at unit length so that is a cosine.
        self.engine = engine or ("faiss" if self.serverless else "lucene")
        # Make a write visible to the next search. Serverless does not take it.
        self.refresh = refresh and not self.serverless

    def _secrets(self) -> list[str]:
        return [self.password, self.api_key, self._aws[1] or ""]

    async def _headers(self) -> dict[str, str]:
        if self.api_key:
            return {"authorization": f"ApiKey {self.api_key}"}
        if self.username and not self.aws_region:
            token = base64.b64encode(f"{self.username}:{self.password}".encode()).decode()
            return {"authorization": f"Basic {token}"}
        return {}

    def _sign(self, method: str, url: str, body: bytes,
              headers: dict[str, str]) -> dict[str, str]:
        if not self.aws_region:
            return headers
        from ...llm_providers._sigv4 import resolve_credentials, sign

        credentials = resolve_credentials(*self._aws)
        if not credentials:
            raise VectorStoreError(
                "no AWS credentials were found for OpenSearch — pass them, or set "
                "AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY", store=self.name)
        signed = sign(method=method, url=url, region=self.aws_region,
                      service="aoss" if self.serverless else "es", body=body,
                      headers={k: v for k, v in headers.items()
                               if k.lower() == "content-type"},
                      credentials=credentials)
        return {**headers, **signed}

    # ---- the shape of the index -----------------------------------------------
    def _mapping(self, dimension: int) -> dict[str, Any]:
        space = "innerproduct" if self.engine == "faiss" else "cosinesimil"
        return {
            "settings": {"index": {"knn": True}},
            "mappings": {
                # Strings in metadata are exact values to filter on, not prose.
                "dynamic_templates": [{"metadata_strings": {
                    "path_match": "meta.*", "match_mapping_type": "string",
                    "mapping": {"type": "keyword"}}}],
                "properties": {
                    "vector": {"type": "knn_vector", "dimension": dimension,
                               "method": {"name": "hnsw", "space_type": space,
                                          "engine": self.engine}},
                    "ref": {"type": "keyword"},
                    "text": {"type": "text"}}}}

    def _search(self, vector: list[float], k: int,
                found: list[dict[str, Any]]) -> dict[str, Any]:
        knn: dict[str, Any] = {"vector": vector, "k": k}
        if found:
            knn["filter"] = {"bool": found[0]}
        return {"size": k, "_source": ["ref", "text", "meta"],
                "query": {"knn": {"vector": knn}}}

    def _similarity(self, score: float) -> float:
        if self.engine == "faiss":
            # An inner product p is scored p + 1 when it is not negative, and
            # 1 / (1 - p) when it is.
            return score - 1.0 if score >= 1.0 else 1.0 - 1.0 / score if score else 0.0
        return 2.0 * score - 1.0          # lucene: (1 + cosine) / 2

    def _prepare(self, vector: list[float]) -> list[float]:
        return normalise(vector) if self.engine == "faiss" else vector

    # ---- the store ---------------------------------------------------------------
    async def ensure(self, dimension: int) -> None:
        if await self._request("HEAD", f"{self.url}/{self.index}", ok=(404,)) != 404:
            return
        try:
            await self._request("PUT", f"{self.url}/{self.index}",
                                json_body=self._mapping(dimension))
        except VectorStoreError as exc:
            if "resource_already_exists" not in str(exc):     # another replica won
                raise

    @staticmethod
    def _bool(wanted: list[tuple[str, str, Any]]) -> list[dict[str, Any]]:
        must: list[dict[str, Any]] = []
        must_not: list[dict[str, Any]] = []
        for name, op, value in wanted:
            key = f"meta.{name}"
            if op == "$eq":
                must.append({"term": {key: value}})
            elif op == "$ne":
                must_not.append({"term": {key: value}})
            elif op == "$in":
                must.append({"terms": {key: value}})
            else:
                must.append({"range": {key: {_RANGE[op]: value}}})
        if not must and not must_not:
            return []
        return [{**({"filter": must} if must else {}),
                 **({"must_not": must_not} if must_not else {})}]

    async def _bulk(self, lines: list[dict[str, Any]]) -> None:
        body = ("\n".join(json.dumps(line, ensure_ascii=False) for line in lines)
                + "\n").encode()
        answer = await self._request(
            "POST", f"{self.url}/_bulk", content=body,
            params={"refresh": "true"} if self.refresh else None,
            headers={"content-type": "application/x-ndjson"})
        if answer.get("errors"):
            for item in answer.get("items") or []:
                result = next(iter(item.values()), {})
                error = result.get("error")
                if error and result.get("status") != 404:
                    raise VectorStoreError(
                        f"{self.name} refused a record: {json.dumps(error)[:300]}",
                        store=self.name)

    async def _upsert(self, records: list[VectorRecord]) -> None:
        if self.serverless:
            await self._delete([r.id for r in records], [])
        lines: list[dict[str, Any]] = []
        for r in records:
            action: dict[str, Any] = {"_index": self.index}
            if not self.serverless:
                action["_id"] = r.id
            lines += [{"index": action},
                      {"vector": self._prepare(r.vector), "ref": r.id, "text": r.text,
                       "meta": r.metadata}]
        await self._bulk(lines)

    async def _query(self, vector: list[float], k: int,
                     wanted: list[tuple[str, str, Any]]) -> list[VectorHit]:
        answer = await self._request(
            "POST", f"{self.url}/{self.index}/_search",
            json_body=self._search(self._prepare(vector), k, self._bool(wanted)))
        out = []
        for hit in (answer.get("hits") or {}).get("hits") or []:
            source = hit.get("_source") or {}
            out.append(VectorHit(source.get("ref") or hit.get("_id", ""),
                                 self._similarity(float(hit.get("_score") or 0.0)),
                                 source.get("text") or "", source.get("meta") or {}))
        return out

    async def _delete(self, ids: list[str] | None,
                      wanted: list[tuple[str, str, Any]]) -> None:
        if ids is not None and not wanted and not self.serverless:
            await self._bulk([{"delete": {"_index": self.index, "_id": i}} for i in ids])
            return
        found = self._bool(wanted)
        clause = dict(found[0]) if found else {}
        if ids is not None:
            clause.setdefault("filter", []).append({"terms": {"ref": ids}})
        query = {"bool": clause} if clause else {"match_all": {}}
        if self.serverless:
            # No delete-by-query there: find what matches, then remove it by id.
            for _ in range(1000):
                answer = await self._request(
                    "POST", f"{self.url}/{self.index}/_search",
                    json_body={"size": 500, "_source": False, "query": query}, ok=(404,))
                hits = [] if answer == 404 else (answer.get("hits") or {}).get("hits") or []
                if not hits:
                    return
                await self._bulk([{"delete": {"_index": self.index, "_id": h["_id"]}}
                                  for h in hits])
                if len(hits) < 500:
                    return
            return
        await self._request(
            "POST", f"{self.url}/{self.index}/_delete_by_query",
            params={"refresh": "true", "conflicts": "proceed"} if self.refresh
            else {"conflicts": "proceed"}, json_body={"query": query})

    async def count(self) -> int:
        answer = await self._request("GET", f"{self.url}/{self.index}/_count", ok=(404,))
        return 0 if answer == 404 else int(answer.get("count") or 0)

    async def clear(self) -> None:
        await self._delete(None, [])


class ElasticsearchStore(OpenSearchStore):
    """
        ElasticsearchStore("http://localhost:9200", index="handbook")
        ElasticsearchStore("https://x.es.europe-west1.gcp.cloud.es.io", api_key="…",
                           index="handbook")
    """

    name = "elasticsearch"
    flavour = "elasticsearch"

    def __init__(self, url: str = "http://localhost:9200", *, index: str = "knowledge",
                 api_key: str | None = None, **kw: Any) -> None:
        super().__init__(url, index=index, serverless=False, engine="lucene",
                         api_key=api_key or os.environ.get("ELASTICSEARCH_API_KEY"),
                         **kw)
        self.username = kw.get("username") or os.environ.get("ELASTICSEARCH_USERNAME", "")
        self.password = kw.get("password") or os.environ.get("ELASTICSEARCH_PASSWORD", "")

    def _mapping(self, dimension: int) -> dict[str, Any]:
        return {"mappings": {
            "dynamic_templates": [{"metadata_strings": {
                "path_match": "meta.*", "match_mapping_type": "string",
                "mapping": {"type": "keyword"}}}],
            "properties": {
                "vector": {"type": "dense_vector", "dims": dimension, "index": True,
                           "similarity": "cosine"},
                "ref": {"type": "keyword"}, "text": {"type": "text"}}}}

    def _search(self, vector: list[float], k: int,
                found: list[dict[str, Any]]) -> dict[str, Any]:
        knn: dict[str, Any] = {"field": "vector", "query_vector": vector, "k": k,
                               "num_candidates": max(100, k * 10)}
        if found:
            knn["filter"] = {"bool": found[0]}
        return {"size": k, "_source": ["ref", "text", "meta"], "knn": knn}

    def _similarity(self, score: float) -> float:
        return 2.0 * score - 1.0
