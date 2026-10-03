"""Knowledge bases and vector stores — without a network.

The stores that speak HTTP are checked here for what they send and how they
read what comes back. `tests/test_knowledge_live.py` runs the same stores
against real databases when it is pointed at them.
"""

from __future__ import annotations

import base64
import json

import httpx
import pytest

from agent_harness import (
    Agent,
    FakeProvider,
    Harness,
    InMemoryStore,
    KnowledgeBase,
    SemanticMemory,
    Trace,
    VectorHit,
    VectorRecord,
    VectorStoreError,
    tool_call,
    vector_store,
)
from agent_harness.errors import ConfigurationError, ToolError
from agent_harness.knowledge import chunk_text, conditions, vector_stores
from agent_harness.knowledge.stores.aws import BedrockKnowledgeBase, S3VectorsStore
from agent_harness.knowledge.stores.azure import AzureAISearchStore
from agent_harness.knowledge.stores.chroma import ChromaStore
from agent_harness.knowledge.stores.cloudflare import VectorizeStore
from agent_harness.knowledge.stores.local import MemoryVectorStore, SQLiteVectorStore
from agent_harness.knowledge.stores.milvus import MilvusStore
from agent_harness.knowledge.stores.opensearch import ElasticsearchStore, OpenSearchStore
from agent_harness.knowledge.stores.pinecone import PineconeStore
from agent_harness.knowledge.stores.qdrant import QdrantStore
from agent_harness.knowledge.stores.upstash import UpstashStore
from agent_harness.knowledge.stores.vertex import SQLitePayloads, VertexVectorStore
from agent_harness.knowledge.stores.weaviate import WeaviateStore
from agent_harness.llm_providers.resilience import RetryPolicy
from agent_harness.memory.base import MemoryRecord

FAST = RetryPolicy(max_retries=2, initial_delay=0.0, max_delay=0.0, jitter=0.0)
HANDBOOK = (
    "# Refunds\n\nRefunds are issued within 30 days of purchase. They go back to the "
    "original card.\n\n## How long\n\nA refund takes five working days to arrive. "
    + "Banks vary in how fast they post it. " * 40
    + "\n\n# Shipping\n\nOrders ship within two working days. The code XK-42 means "
    "express delivery.")
RECORD = VectorRecord("doc#0", [0.6, 0.8], "Refunds take five days.",
                      {"source": "faq.md", "year": 2026})
WANTED = {"source": "faq.md", "year": {"$gte": 2025}, "lang": {"$in": ["en", "fr"]}}


def served(make, answers):
    """A store answered by `answers` — `(method, path-ending) → body` — and what it sent."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        for (method, ending), body in answers.items():
            if request.method == method and request.url.path.endswith(ending):
                if isinstance(body, httpx.Response):
                    return body
                return httpx.Response(200, json=body)
        return httpx.Response(200, json={})

    return make(transport=httpx.MockTransport(handler), retry=FAST), seen


def sent(request: httpx.Request):
    return json.loads(request.content)


# --- filters and chunks -----------------------------------------------------------

def test_a_filter_is_one_language_for_every_store():
    assert conditions(None) == [] and conditions({}) == []
    assert conditions(WANTED) == [("source", "$eq", "faq.md"), ("year", "$gte", 2025),
                                  ("lang", "$in", ["en", "fr"])]
    assert conditions({"$and": [{"a": 1}, {"b": {"$ne": 2}}]}) == [
        ("a", "$eq", 1), ("b", "$ne", 2)]
    for bad, why in [({"a": {"$regex": "x"}}, "unknown filter operator"),
                     ({"$or": []}, "filters take \\$and"), ({"a": {"$in": "x"}}, "takes a list")]:
        with pytest.raises(VectorStoreError, match=why):
            conditions(bad)


def test_text_is_cut_at_paragraphs_with_overlap_and_its_headings():
    chunks = chunk_text(HANDBOOK, size=400, overlap=60)
    assert all(len(chunk) <= 460 for chunk in chunks) and len(chunks) > 4
    assert chunks[0].startswith("# Refunds") and "30 days" in chunks[0]
    # A passage from the middle of a section still says which section it is.
    middle = next(c for c in chunks[2:] if "Banks vary" in c)
    assert middle.startswith("[Refunds > How long]")
    assert chunks[-1].startswith("# Shipping") and "XK-42" in chunks[-1]
    # Nothing is lost between passages.
    assert HANDBOOK.count("Banks vary") <= sum(c.count("Banks vary") for c in chunks)
    assert chunk_text("") == [] and chunk_text("One line.") == ["One line."]
    long_word = chunk_text("x" * 3000, size=500)
    assert len(long_word) == 6 and all(len(c) <= 500 for c in long_word)


# --- local stores -------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["memory", "sqlite"])
async def test_the_local_stores_pass_the_check_every_store_is_held_to(kind, tmp_path):
    store = MemoryVectorStore() if kind == "memory" else SQLiteVectorStore(tmp_path / "k.db")
    report = await store.check()
    assert report["ok"], report["steps"]
    assert [s["step"] for s in report["steps"]] == [
        "ensure", "upsert", "query", "filter", "replace", "delete by id", "delete by filter"]
    assert await store.count() == 0

    await store.upsert([RECORD, VectorRecord("doc#1", [0.8, 0.6], "Shipping.", {"tags": ["a"]})])
    hits = await store.query([0.6, 0.8], k=5)
    assert hits[0].id == "doc#0" and hits[0].score == pytest.approx(1.0)
    assert hits[0].metadata == {"source": "faq.md", "year": 2026}
    assert [h.id for h in await store.query([1, 0], filter={"tags": "a"})] == ["doc#1"]
    with pytest.raises(VectorStoreError, match="takes ids or a filter"):
        await store.delete()
    with pytest.raises(VectorStoreError, match="no usable vector"):
        await store.upsert([VectorRecord("bad", [float("nan"), 1.0])])
    with pytest.raises(VectorStoreError, match="needs an id"):
        await store.upsert([VectorRecord("", [1.0, 0.0])])
    with pytest.raises(VectorStoreError, match="2-dimension vectors; 3 was asked"):
        await store.ensure(3)


async def test_a_sqlite_store_is_still_there_after_a_restart(tmp_path):
    first = SQLiteVectorStore(tmp_path / "k.db", table="handbook")
    await first.upsert([RECORD])
    await first.aclose()
    again = vector_store(f"sqlite:///{tmp_path / 'k.db'}?table=handbook")
    assert [(h.id, h.text) for h in await again.query([0.6, 0.8])] == [
        ("doc#0", "Refunds take five days.")]
    await again.clear()
    assert await again.count() == 0


def test_stores_are_named_or_given_as_urls():
    assert {s["name"] for s in vector_stores()} >= {
        "qdrant", "chroma", "weaviate", "milvus", "pinecone", "opensearch", "elasticsearch",
        "pgvector", "redis", "mongodb", "azure-ai-search", "cloudflare-vectorize",
        "upstash", "vertex-vector-search", "s3-vectors", "sqlite", "memory"}
    q = vector_store("qdrant+https://xyz.cloud.qdrant.io:6333/handbook?api_key=k")
    assert (q.url, q.collection, q.api_key) == ("https://xyz.cloud.qdrant.io:6333", "handbook", "k")
    assert vector_store("opensearch://localhost:9200/docs").index == "docs"
    assert vector_store("es://localhost:9200/docs").name == "elasticsearch"
    assert vector_store("zilliz+https://in03.zillizcloud.com/handbook").collection == "handbook"
    assert vector_store("weaviate://localhost:8080/handbook").collection == "Handbook"
    pg = vector_store("supabase://user:pw@db.example:5432/postgres?table=docs")
    assert pg.dsn == "postgresql://user:pw@db.example:5432/postgres" and '"docs"' in pg.table
    assert vector_store("azure-search://svc.search.windows.net/docs", api_key="k").index == "docs"
    assert vector_store("vectorize://acct123/docs", api_token="t").index == "docs"
    assert vector_store("s3vectors://bucket/docs?region=eu-west-1").url == (
        "https://s3vectors.eu-west-1.api.aws")
    assert vector_store("pinecone://handbook", api_key="k").index == "handbook"
    assert vector_store(q) is q
    with pytest.raises(ConfigurationError, match="no vector store named 'faiss'"):
        vector_store("faiss")
    with pytest.raises(ConfigurationError, match="a name, a URL, or a VectorStore"):
        vector_store(42)


# --- what each database is sent ---------------------------------------------------

async def test_qdrant():
    store, seen = served(lambda **kw: QdrantStore("http://q:6333", collection="kb",
                                                  api_key="secret-key", **kw), {
        ("GET", "/collections/kb"): httpx.Response(404, json={}),
        ("POST", "/points/query"): {"result": {"points": [
            {"id": "u", "score": 0.91, "payload": {"ref": "doc#0", "text": "T",
                                                   "meta": {"year": 2026}}}]}}})
    await store.ensure(2)
    assert sent(seen[1]) == {"vectors": {"size": 2, "distance": "Cosine"}}
    assert seen[0].headers["api-key"] == "secret-key"
    await store.upsert([RECORD])
    point = sent(seen[2])["points"][0]
    assert point["payload"] == {"ref": "doc#0", "text": "Refunds take five days.",
                                "meta": {"source": "faq.md", "year": 2026}}
    assert len(point["id"]) == 36 and seen[2].url.params["wait"] == "true"
    hits = await store.query([0.6, 0.8], k=3, filter=WANTED)
    assert sent(seen[3])["filter"] == {"must": [
        {"key": "meta.source", "match": {"value": "faq.md"}},
        {"key": "meta.year", "range": {"gte": 2025}},
        {"key": "meta.lang", "match": {"any": ["en", "fr"]}}]}
    assert hits == [VectorHit("doc#0", 0.91, "T", {"year": 2026})]
    await store.delete(["doc#0"], filter={"draft": {"$ne": True}})
    body = sent(seen[4])["filter"]
    assert body["must_not"] == [{"key": "meta.draft", "match": {"value": True}}]
    assert body["must"][0]["has_id"] == [point["id"]]


async def test_chroma_and_milvus():
    store, seen = served(lambda **kw: ChromaStore("http://c:8000", collection="kb", **kw), {
        ("POST", "/collections"): {"id": "col-1"},
        ("POST", "/query"): {"ids": [["doc#0"]], "distances": [[0.2]], "documents": [["T"]],
                             "metadatas": [[{"year": 2026, "tags__json": '["a"]'}]]}})
    await store.upsert([VectorRecord("doc#0", [1, 0], "T", {"year": 2026, "tags": ["a"]})])
    assert sent(seen[0])["get_or_create"] is True
    assert seen[1].url.path.endswith("/collections/col-1/upsert")
    assert sent(seen[1])["metadatas"] == [{"year": 2026, "tags__json": '["a"]'}]
    hits = await store.query([1, 0], filter=WANTED)
    assert sent(seen[2])["where"] == {"$and": [
        {"source": {"$eq": "faq.md"}}, {"year": {"$gte": 2025}}, {"lang": {"$in": ["en", "fr"]}}]}
    assert hits[0].score == pytest.approx(0.8) and hits[0].metadata == {"year": 2026, "tags": ["a"]}

    store, seen = served(lambda **kw: MilvusStore("http://m:19530", collection="kb",
                                                  token="root:Milvus", **kw), {
        ("POST", "/collections/has"): {"code": 0, "data": {"has": False}},
        ("POST", "/entities/search"): {"code": 0, "data": [
            {"id": "doc#0", "distance": 0.9, "text": "T", "meta": {"year": 2026}}]},
        ("POST", "/entities/delete"): {"code": 1100, "message": "bad filter root:Milvus"}})
    await store.ensure(2)
    schema = sent(seen[1])
    assert schema["indexParams"][0]["metricType"] == "COSINE"
    assert seen[1].headers["authorization"] == "Bearer root:Milvus"
    hits = await store.query([1, 0], filter=WANTED)
    assert sent(seen[2])["filter"] == ('meta["source"] == "faq.md" and meta["year"] >= 2025 '
                                       'and meta["lang"] in ["en", "fr"]')
    assert hits == [VectorHit("doc#0", 0.9, "T", {"year": 2026})]
    # Milvus says 200 and reports the failure in the body; it is still a failure.
    with pytest.raises(VectorStoreError, match="1100") as caught:
        await store.delete(["a"])
    assert "root:Milvus" not in str(caught.value)


async def test_weaviate():
    store, seen = served(lambda **kw: WeaviateStore("http://w:8080", collection="handbook", **kw), {
        ("POST", "/v1/batch/objects"): [{"result": {}}],
        ("POST", "/v1/graphql"): {"data": {"Get": {"Handbook": [
            {"ref": "doc#0", "text": "T", "meta": '{"year": 2026}',
             "_additional": {"id": "u", "distance": 0.25}}]}}}})
    await store.upsert([RECORD])
    item = sent(seen[0])["objects"][0]
    assert item["class"] == "Handbook" and item["properties"]["m_source"] == "faq.md"
    assert json.loads(item["properties"]["meta"]) == {"source": "faq.md", "year": 2026}
    hits = await store.query([1, 0], filter=WANTED)
    query = sent(seen[1])["query"]
    assert 'path: ["m_source"], operator: Equal, valueText: "faq.md"' in query
    assert 'path: ["m_year"], operator: GreaterThanEqual, valueNumber: 2025' in query
    assert 'operator: ContainsAny, valueText: ["en", "fr"]' in query and "operator: And" in query
    assert hits == [VectorHit("doc#0", 0.75, "T", {"year": 2026})]


async def test_opensearch_and_elasticsearch():
    answer = {"hits": {"hits": [{"_id": "doc#0", "_score": 0.95,
                                 "_source": {"ref": "doc#0", "text": "T", "meta": {"y": 1}}}]}}
    store, seen = served(lambda **kw: OpenSearchStore(
        "https://os:9200", index="kb", username="admin", password="pw-secret", **kw), {
        ("HEAD", "/kb"): httpx.Response(404), ("POST", "/_bulk"): {"errors": False},
        ("POST", "/_search"): answer})
    await store.ensure(2)
    mapping = sent(seen[1])
    assert mapping["settings"] == {"index": {"knn": True}}
    assert mapping["mappings"]["properties"]["vector"]["method"] == {
        "name": "hnsw", "space_type": "cosinesimil", "engine": "lucene"}
    assert seen[0].headers["authorization"] == "Basic " + base64.b64encode(
        b"admin:pw-secret").decode()
    await store.upsert([RECORD])
    lines = [json.loads(line) for line in seen[2].content.decode().splitlines()]
    assert lines[0] == {"index": {"_index": "kb", "_id": "doc#0"}}
    assert lines[1]["meta"] == {"source": "faq.md", "year": 2026}
    assert seen[2].url.params["refresh"] == "true"
    hits = await store.query([0.6, 0.8], filter=WANTED)
    knn = sent(seen[3])["query"]["knn"]["vector"]
    assert knn["filter"] == {"bool": {"filter": [
        {"term": {"meta.source": "faq.md"}}, {"range": {"meta.year": {"gte": 2025}}},
        {"terms": {"meta.lang": ["en", "fr"]}}]}}
    assert hits[0].score == pytest.approx(0.9)          # (1 + cosine) / 2, undone

    store, seen = served(lambda **kw: ElasticsearchStore(
        "http://es:9200", index="kb", api_key="es-key", **kw), {("POST", "/_search"): answer})
    await store.query([0.6, 0.8], k=4)
    assert sent(seen[0])["knn"] == {"field": "vector", "query_vector": [0.6, 0.8], "k": 4,
                                    "num_candidates": 100}
    assert seen[0].headers["authorization"] == "ApiKey es-key"


async def test_opensearch_on_aws_is_signed_and_serverless_keeps_its_own_ids(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAEXAMPLE")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "aws-secret-value")
    store, seen = served(lambda **kw: OpenSearchStore(
        "https://x.eu-west-1.aoss.amazonaws.com", index="kb", aws_region="eu-west-1", **kw), {
        ("POST", "/_search"): {"hits": {"hits": [{"_id": "auto-1", "_score": 1.9, "_source": {
            "ref": "doc#0", "text": "T", "meta": {}}}]}},
        ("POST", "/_bulk"): {"errors": False}})
    assert store.serverless and store.engine == "faiss" and not store.refresh
    await store.upsert([VectorRecord("doc#0", [3.0, 4.0], "T")])
    # First what it replaces is found and removed, by the id the service chose…
    assert sent(seen[0])["query"] == {"bool": {"filter": [{"terms": {"ref": ["doc#0"]}}]}}
    assert json.loads(seen[1].content.decode().splitlines()[0]) == {
        "delete": {"_index": "kb", "_id": "auto-1"}}
    # …then the record goes in with no id of its own, at unit length.
    lines = [json.loads(line) for line in seen[2].content.decode().splitlines()]
    assert lines[0] == {"index": {"_index": "kb"}} and lines[1]["vector"] == [0.6, 0.8]
    assert "refresh" not in seen[2].url.params
    auth = seen[2].headers["authorization"]
    assert auth.startswith("AWS4-HMAC-SHA256 Credential=AKIAEXAMPLE/")
    assert "/eu-west-1/aoss/aws4_request" in auth and "aws-secret-value" not in auth
    hits = await store.query([3.0, 4.0])
    assert hits[0].id == "doc#0" and hits[0].score == pytest.approx(0.9)   # p + 1, undone

    monkeypatch.delenv("AWS_ACCESS_KEY_ID")
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY")
    with pytest.raises(VectorStoreError, match="no AWS credentials"):
        await store.count()


async def test_pinecone_finds_its_index_and_creates_it_when_there_is_none():
    store, seen = served(lambda **kw: PineconeStore("handbook", api_key="pc-secret",
                                                    namespace="acme", **kw), {
        ("GET", "/indexes/handbook"): httpx.Response(404, json={}),
        ("POST", "/indexes"): {"host": "handbook-abc.svc.pinecone.io", "dimension": 2},
        ("POST", "/query"): {"matches": [{"id": "doc#0", "score": 0.88,
                                          "metadata": {"year": 2026, "_text": "T"}}]}})
    await store.ensure(2)
    assert sent(seen[1]) == {"name": "handbook", "dimension": 2, "metric": "cosine",
                             "spec": {"serverless": {"cloud": "aws", "region": "us-east-1"}}}
    assert seen[0].headers["api-key"] == "pc-secret"
    await store.upsert([RECORD])
    assert str(seen[2].url) == "https://handbook-abc.svc.pinecone.io/vectors/upsert"
    body = sent(seen[2])
    assert body["namespace"] == "acme" and body["vectors"][0]["metadata"]["_text"].startswith("Refunds")
    hits = await store.query([1, 0], filter=WANTED)
    assert sent(seen[3])["filter"] == {"$and": [
        {"source": {"$eq": "faq.md"}}, {"year": {"$gte": 2025}}, {"lang": {"$in": ["en", "fr"]}}]}
    assert hits == [VectorHit("doc#0", 0.88, "T", {"year": 2026})]


async def test_pinecone_deletes_by_filter_even_where_the_index_cannot():
    matches = [{"matches": [{"id": "a"}, {"id": "b"}]}, {"matches": []}]

    def delete(request):
        return httpx.Response(400, json={"message": "Serverless indexes do not support "
                                                    "deleting with metadata filtering"}
                              ) if "filter" in sent(request) else httpx.Response(200, json={})

    seen: list[httpx.Request] = []

    def handler(request):
        seen.append(request)
        if request.url.path.endswith("/vectors/delete"):
            return delete(request)
        if request.url.path.endswith("/describe_index_stats"):
            return httpx.Response(200, json={"dimension": 3})
        return httpx.Response(200, json=matches.pop(0))

    store = PineconeStore(host="https://h.pinecone.io", api_key="k", retry=FAST,
                          transport=httpx.MockTransport(handler))
    await store.delete(filter={"doc": "handbook"})
    deletes = [sent(r) for r in seen if r.url.path.endswith("/vectors/delete")]
    assert deletes == [{"filter": {"doc": {"$eq": "handbook"}}}, {"ids": ["a", "b"]}]
    assert sent(seen[2])["vector"] == [1.0, 1.0, 1.0]


async def test_azure_ai_search():
    store, seen = served(lambda **kw: AzureAISearchStore(
        "https://svc.search.windows.net", index="kb", api_key="az-secret",
        filterable={"source": "string", "year": "int", "lang": "strings"}, **kw), {
        ("GET", "/indexes/kb"): httpx.Response(404, json={}),
        ("POST", "/docs/index"): {"value": [{"key": "x", "status": True, "statusCode": 200}]},
        ("POST", "/docs/search"): {"value": [
            {"@search.score": 0.8, "ref": "doc#0", "text": "T", "meta": '{"year": 2026}'}]}})
    await store.ensure(2)
    fields = {f["name"]: f for f in sent(seen[1])["fields"]}
    assert fields["vector"]["dimensions"] == 2 and fields["m_year"]["type"] == "Edm.Int64"
    assert fields["m_lang"]["type"] == "Collection(Edm.String)" and fields["m_doc"]["filterable"]
    assert seen[1].url.params["api-version"] == "2024-07-01"
    assert seen[1].headers["api-key"] == "az-secret"
    await store.upsert([RECORD])
    row = sent(seen[2])["value"][0]
    assert row["@search.action"] == "mergeOrUpload" and row["ref"] == "doc#0"
    assert base64.urlsafe_b64decode(row["id"]).decode() == "doc#0"      # '#' is not a key
    assert row["m_source"] == "faq.md" and row["m_year"] == 2026 and row["m_lang"] == []
    hits = await store.query([1, 0], k=3, filter=WANTED)
    body = sent(seen[3])
    assert body["filter"] == ("m_source eq 'faq.md' and m_year ge 2025 and "
                              "m_lang/any(t: t eq 'en' or t eq 'fr')")
    assert body["vectorQueries"][0]["k"] == 3
    assert hits[0].score == pytest.approx(0.75)         # 1 / (2 - cosine), undone
    with pytest.raises(VectorStoreError, match="'colour' is not filterable in this Azure"):
        await store.query([1, 0], filter={"colour": "red"})
    with pytest.raises(VectorStoreError, match="a filterable field is one of"):
        AzureAISearchStore("https://x", filterable={"a": "date"})


async def test_cloudflare_vectorize_and_upstash():
    store, seen = served(lambda **kw: VectorizeStore(
        account_id="acct", index="kb", api_token="cf-secret",
        filterable={"source": "string", "year": "number", "lang": "string"}, **kw), {
        ("GET", "/indexes/kb"): httpx.Response(404, json={}),
        ("POST", "/query"): {"success": True, "result": {"matches": [
            {"id": "doc#0", "score": 0.9, "metadata": {"year": 2026, "_text": "T"}}]}}})
    await store.ensure(2)
    assert sent(seen[1]) == {"name": "kb", "config": {"dimensions": 2, "metric": "cosine"}}
    created = [sent(r)["propertyName"] for r in seen if r.url.path.endswith("metadata_index/create")]
    assert created == ["doc", "namespace", "source", "year", "lang"]
    await store.upsert([RECORD])
    upsert = seen[-1]
    assert upsert.headers["content-type"] == "application/x-ndjson"
    assert json.loads(upsert.content.decode().splitlines()[0])["metadata"]["_text"].startswith("Refunds")
    hits = await store.query([1, 0], k=500, filter=WANTED)
    body = sent(seen[-1])
    assert body["topK"] == 20                                    # its ceiling with metadata
    assert body["filter"] == {"source": {"$eq": "faq.md"}, "year": {"$gte": 2025},
                              "lang": {"$in": ["en", "fr"]}}
    assert hits == [VectorHit("doc#0", 0.9, "T", {"year": 2026})]
    with pytest.raises(VectorStoreError, match="deletes by id only"):
        await store.delete(filter={"source": "x"})

    store, seen = served(lambda **kw: UpstashStore("https://u.upstash.io", token="up-secret",
                                                   namespace="acme", **kw), {
        ("POST", "/query/acme"): {"result": [
            {"id": "doc#0", "score": 0.95, "data": "T", "metadata": {"year": 2026}}]}})
    await store.upsert([RECORD])
    assert sent(seen[0]) == [{"id": "doc#0", "vector": [0.6, 0.8], "data": RECORD.text,
                              "metadata": {"source": "faq.md", "year": 2026}}]
    hits = await store.query([1, 0], filter=WANTED)
    assert sent(seen[1])["filter"] == "source = 'faq.md' AND year >= 2025 AND lang IN ('en', 'fr')"
    assert hits[0].score == pytest.approx(0.9)


async def test_vertex_keeps_text_beside_the_index(tmp_path):
    class Token:
        async def token(self):
            return "ya29.token"

    payloads = SQLitePayloads(tmp_path / "payloads.db")
    store, seen = served(lambda **kw: VertexVectorStore(
        project="p", region="us-central1", index="111", endpoint="222",
        deployed_index="kb_v1", endpoint_domain="1.us-central1-9.vdb.vertexai.goog",
        payloads=payloads, auth=Token(), **kw), {
        ("POST", ":findNeighbors"): {"nearestNeighbors": [{"neighbors": [
            {"distance": 0.93, "datapoint": {"datapointId": "doc#0"}}]}]}})
    await store.upsert([VectorRecord("doc#0", [3.0, 4.0], "Refunds take five days.",
                                     {"source": "faq.md", "year": 2026, "lang": ["en"]})])
    point = sent(seen[0])["datapoints"][0]
    assert seen[0].url.path.endswith("/indexes/111:upsertDatapoints")
    assert seen[0].headers["authorization"] == "Bearer ya29.token"
    assert point["featureVector"] == [0.6, 0.8]                   # unit length
    assert point["restricts"] == [{"namespace": "source", "allowList": ["faq.md"]},
                                  {"namespace": "lang", "allowList": ["en"]}]
    assert point["numericRestricts"] == [{"namespace": "year", "valueInt": "2026"}]
    hits = await store.query([3.0, 4.0], filter=WANTED)
    asked = sent(seen[1])
    assert seen[1].url.host == "1.us-central1-9.vdb.vertexai.goog"
    assert asked["deployedIndexId"] == "kb_v1"
    assert asked["queries"][0]["datapoint"]["numericRestricts"] == [
        {"namespace": "year", "op": "GREATER_EQUAL", "valueInt": "2025"}]
    # The text and metadata come from the payloads, not from Vertex.
    assert hits == [VectorHit("doc#0", 0.93, "Refunds take five days.",
                              {"source": "faq.md", "year": 2026, "lang": ["en"]})]
    await store.delete(filter={"source": "faq.md"})
    assert sent(seen[2]) == {"datapointIds": ["doc#0"]} and await payloads.all() == {}


async def test_s3_vectors_and_a_bedrock_knowledge_base(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAEXAMPLE")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "aws-secret-value")
    store, seen = served(lambda **kw: S3VectorsStore(bucket="vb", index="kb",
                                                     region="eu-west-1", **kw), {
        ("POST", "/GetIndex"): httpx.Response(404, json={}),
        ("POST", "/QueryVectors"): {"vectors": [
            {"key": "doc#0", "distance": 0.1, "metadata": {"year": 2026, "_text": "T"}}]},
        ("POST", "/ListVectors"): {"vectors": [
            {"key": "doc#0", "metadata": {"source": "faq.md"}},
            {"key": "doc#1", "metadata": {"source": "other.md"}}]}})
    await store.ensure(2)
    assert sent(seen[1]) == {
        "vectorBucketName": "vb", "indexName": "kb", "dataType": "float32", "dimension": 2,
        "distanceMetric": "cosine",
        "metadataConfiguration": {"nonFilterableMetadataKeys": ["_text"]}}
    assert "/eu-west-1/s3vectors/aws4_request" in seen[1].headers["authorization"]
    await store.upsert([RECORD])
    assert sent(seen[2])["vectors"][0]["data"] == {"float32": [0.6, 0.8]}
    hits = await store.query([1, 0], k=200, filter={"source": "faq.md"})
    assert sent(seen[3])["topK"] == 30 and sent(seen[3])["filter"] == {"source": {"$eq": "faq.md"}}
    assert hits == [VectorHit("doc#0", 0.9, "T", {"year": 2026})]
    await store.delete(filter={"source": "faq.md"})
    assert sent(seen[-1])["keys"] == ["doc#0"]

    kb, seen = served(lambda **kw: BedrockKnowledgeBase("KB123", region="us-east-1", **kw), {
        ("POST", "/retrieve"): {"retrievalResults": [
            {"content": {"text": "Refunds take five days."}, "score": 0.71,
             "location": {"s3Location": {"uri": "s3://docs/faq.md"}},
             "metadata": {"x-amz-bedrock-kb-chunk-id": "c1"}}]}})
    found = await KnowledgeBase(retriever=kb).search("refunds?", k=3, filter={"lang": "en"})
    asked = sent(seen[0])
    assert seen[0].url.path == "/knowledgebases/KB123/retrieve"
    assert asked["retrievalQuery"] == {"text": "refunds?"}
    assert asked["retrievalConfiguration"]["vectorSearchConfiguration"]["filter"] == {
        "equals": {"key": "lang", "value": "en"}}
    assert (found[0].text, found[0].source, found[0].score) == (
        "Refunds take five days.", "s3://docs/faq.md", 0.71)
    with pytest.raises(VectorStoreError, match="reads from a retriever"):
        await KnowledgeBase(retriever=kb).add("x")


async def test_a_call_that_may_succeed_later_is_retried_and_keys_stay_out_of_errors():
    answers = [httpx.Response(503), httpx.Response(429, headers={"retry-after": "0"}),
               httpx.Response(200, json={"result": {"count": 7}})]
    seen: list[httpx.Request] = []

    def flaky(request):
        seen.append(request)
        return answers.pop(0)

    store = QdrantStore("http://q:6333", api_key="k", retry=FAST,
                        transport=httpx.MockTransport(flaky))
    assert await store.count() == 7 and len(seen) == 3

    def refuse(request):
        return httpx.Response(403, json={"status": {"error": "bad key qd-secret-key"}})

    store = QdrantStore("http://q:6333", api_key="qd-secret-key", retry=FAST,
                        transport=httpx.MockTransport(refuse))
    with pytest.raises(VectorStoreError) as caught:
        await store.count()
    assert caught.value.status == 403 and "qd-secret-key" not in str(caught.value)

    def down(request):
        raise httpx.ConnectError("no route to http://q:6333/?api-key=qd-secret-key")

    store = QdrantStore("http://q:6333", api_key="qd-secret-key", retry=FAST,
                        transport=httpx.MockTransport(down))
    with pytest.raises(VectorStoreError, match="could not be reached \\(ConnectError\\)") as caught:
        await store.count()
    assert "qd-secret-key" not in str(caught.value)


# --- the knowledge base -------------------------------------------------------------

async def test_documents_go_in_are_found_replaced_and_removed():
    kb = KnowledgeBase(chunk_size=400, chunk_overlap=60)
    added = await kb.add(HANDBOOK, id="handbook", title="Handbook", source="handbook.md",
                         metadata={"lang": "en"})
    assert added.chunks == await kb.count() and added.chunks > 4 and not added.unchanged
    await kb.add("Le remboursement prend cinq jours ouvrés.", id="fr", metadata={"lang": "fr"})

    found = await kb.search("what does the code XK-42 mean?", k=2)
    assert "XK-42" in found[0].text
    assert (found[0].document, found[0].title, found[0].source) == (
        "handbook", "Handbook", "handbook.md")
    assert found[0].metadata == {"lang": "en"}                 # only the document's own
    assert [p.document for p in await kb.search("remboursement", filter={"lang": "fr"})] == ["fr"]

    # The same again writes nothing — and embeds only enough to find that out.
    embedded: list[int] = []
    inner = kb.embedder

    class Counting:
        async def embed(self, texts):
            embedded.append(len(texts))
            return await inner.embed(texts)

    kb.embedder = Counting()
    same = await kb.add(HANDBOOK, id="handbook", title="Handbook", source="handbook.md",
                        metadata={"lang": "en"})
    assert same.unchanged and embedded == [1]

    # A shorter version replaces it; the passages past its new end are gone.
    shorter = await kb.add("Refunds now take two days.", id="handbook")
    assert shorter.chunks == 1 and await kb.count() == 2
    await kb.delete("handbook")
    await kb.delete("never-there")
    assert await kb.count() == 1 and await kb.search("refunds") != []

    for bad, why in [({"text": "  "}, "no text"), ({}, "takes one of"),
                     ({"text": "x", "metadata": {"doc": "y"}}, "may not use the names doc")]:
        with pytest.raises(ToolError, match=why):
            await kb.add(**bad)
    with pytest.raises(ToolError, match="query is empty"):
        await kb.search(" ")
    with pytest.raises(VectorStoreError, match="unknown filter operator"):
        await kb.search("x", filter={"a": {"$like": "b"}})


async def test_files_are_read_and_one_store_holds_several_knowledge_bases(tmp_path):
    (tmp_path / "refunds.md").write_text("# Refunds\n\nA refund takes five working days.")
    (tmp_path / "prices.csv").write_text("item,price\nwidget,9.99\ngadget,24.50\n")
    store = MemoryVectorStore()
    acme = KnowledgeBase(store, namespace="acme")
    docs = await acme.add_many([tmp_path / "refunds.md", {"path": tmp_path / "prices.csv",
                                                          "id": "prices"}])
    assert [d.title for d in docs] == ["refunds.md", "prices.csv"]
    assert docs[0].id == str(tmp_path / "refunds.md") and docs[1].id == "prices"
    assert "gadget" in (await acme.search("how much is a gadget", k=1))[0].text

    other = KnowledgeBase(store, namespace="other")
    await other.add("Refunds are not offered.", id="policy")
    assert [p.document for p in await other.search("refund", k=5)] == ["policy"]
    assert "policy" not in [p.document for p in await acme.search("refund", k=5)]
    await other.delete("policy")
    assert await store.count() == 2 and await other.search("refund") == []


async def test_results_are_ordered_by_meaning_and_by_shared_words_and_can_be_reranked():
    class Flat:
        """Every text looks the same to this embedder: only the words can tell."""
        async def embed(self, texts):
            return [[1.0, 0.0] for _ in texts]

    kb = KnowledgeBase(embedder=Flat())
    await kb.add("General notes about delivery and couriers.", id="a")
    await kb.add("The part number ZX-9000 is discontinued.", id="b")
    await kb.add("More general notes about returns.", id="c")
    assert (await kb.search("ZX-9000", k=1))[0].document == "b"
    assert (await KnowledgeBase(kb.store, embedder=Flat(), hybrid=False)
            .search("ZX-9000", k=1))[0].document == "a"

    async def by_length(query, passages):
        return [len(p.text) for p in passages]

    longest = KnowledgeBase(kb.store, embedder=Flat(), reranker=by_length)
    assert (await longest.search("anything", k=1))[0].document == "a"
    with pytest.raises(VectorStoreError, match="returned 1 scores for 3"):
        await KnowledgeBase(kb.store, embedder=Flat(),
                            reranker=lambda q, p: [1.0]).search("x")

    class Wrong:
        async def embed(self, texts):
            return [[1.0, 0.0]]

    with pytest.raises(VectorStoreError, match="returned 1 vectors for 2 texts"):
        await KnowledgeBase(embedder=Wrong(), batch=8)._embed(["a", "b"])
    with pytest.raises(ConfigurationError, match="a store or a retriever, not both"):
        KnowledgeBase("memory", retriever=object())


async def test_a_store_that_deletes_only_by_id_still_loses_whole_documents():
    class ByIdOnly(MemoryVectorStore):
        async def _delete(self, ids, wanted):
            if ids is None or wanted:
                raise VectorStoreError("deletes by id only")
            await super()._delete(ids, wanted)

    kb = KnowledgeBase(ByIdOnly(), chunk_size=400, chunk_overlap=60)
    await kb.add(HANDBOOK, id="handbook")
    await kb.add("Something else.", id="other")
    await kb.delete("handbook")
    assert await kb.count() == 1


async def test_an_agent_searches_the_knowledge_base_it_was_given():
    kb = KnowledgeBase(chunk_size=400, chunk_overlap=60)
    await kb.add(HANDBOOK, id="handbook", title="Handbook", source="handbook.md",
                 metadata={"tenant": "acme"})
    await kb.add("Internal: the refund budget is 1M.", id="secret", metadata={"tenant": "hq"})
    search = kb.as_tool(filter={"tenant": "acme"}, k=3)
    assert search.name == "search_knowledge" and {"knowledge", "search"} <= search.tags
    assert set(search.parameters["properties"]) == {"query", "limit"}

    read: list[str] = []
    harness = Harness.testing(FakeProvider([
        tool_call("search_knowledge", query="what does XK-42 mean"),
        "XK-42 means express delivery (Handbook)."]))
    agent = Agent("support", tools=[search], harness=harness, memory=False)
    async for event in agent.stream("What is XK-42?"):
        if event.type == "tool_result":
            read.append(event.text)
    assert '"n": 1' in read[0] and "XK-42" in read[0]
    found = await search.invoke({"query": "what does XK-42 mean"})
    assert found[0]["title"] == "Handbook" and found[0]["source"] == "handbook.md"
    # The tool's filter is not the model's to remove.
    budget = await search.invoke({"query": "refund budget", "limit": 50})
    assert all("1M" not in row["text"] for row in budget) and len(budget) <= 10
    empty = await KnowledgeBase().as_tool().invoke({"query": "anything"})
    assert empty.startswith("Nothing in the knowledge base matches")


async def test_semantic_memory_can_keep_its_index_in_a_vector_database():
    memory = SemanticMemory(InMemoryStore(), index=MemoryVectorStore())
    assert memory.remote
    await memory.append(MemoryRecord(text="Ada prefers invoices by email", scope="user",
                                     user_id="ada", tenant_id="acme"))
    await memory.append(MemoryRecord(text="Bob prefers invoices by post", scope="user",
                                     user_id="bob", tenant_id="acme"))
    await memory.append(MemoryRecord(text="Invoices are sent on the 1st", scope="user"))
    await memory.append(MemoryRecord(text="Invoices for the project team", scope="project"))

    ada = await memory.search("how are invoices sent", scope="user",
                              trace=Trace(user_id="ada", tenant_id="acme"), limit=10)
    assert sorted(r.text for r in ada) == ["Ada prefers invoices by email",
                                           "Invoices are sent on the 1st"]
    # Held to its owner in the database, not filtered after the fact.
    stored = await memory.index.store.query([0.0] * 255 + [1.0], k=10,
                                            filter={"user_id": {"$in": ["bob", ""]}})
    assert "Ada prefers invoices by email" not in [h.text for h in stored]
    scored = await memory.search_scored("invoices by post", limit=1)
    assert scored[0][1].text == "Bob prefers invoices by post" and scored[0][0] > 0.3
    await memory.clear()
    assert await memory.index.store.count() == 0
