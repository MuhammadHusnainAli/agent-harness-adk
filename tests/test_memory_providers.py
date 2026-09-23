"""The storage backends: key layout, query construction and driver handling.

SQLite and the file store are exercised for real in `test_memory_trace.py`.
Here the backends that need a server are driven through an injected fake client,
so the code that builds keys and queries is genuinely run — only the network is
absent.
"""

from __future__ import annotations

import json

import httpx
import pytest

from agent_harness.errors import ConfigurationError
from agent_harness.memory import memory_provider
from agent_harness.memory.base import MemoryRecord
from agent_harness.memory.providers import (
    BACKENDS,
    DRIVERS,
    SCHEMES,
    available,
    missing_driver,
    register_backend,
)
from agent_harness.memory.trace import Trace


def record(text: str = "a fact", **kw) -> MemoryRecord:
    return MemoryRecord(scope="user", text=text, **kw)


# --- the registry ---------------------------------------------------------------

def test_every_backend_the_readme_promises_is_registered():
    assert set(BACKENDS) >= {
        "memory", "file", "sqlite", "postgres", "mysql", "mongo", "redis",
        "dynamodb", "elasticsearch", "s3", "azure", "gcs", "http",
    }


def test_nothing_is_imported_until_it_is_asked_for():
    import subprocess
    import sys

    # A fresh interpreter: importing the registry must not pull in any driver.
    code = (
        "import sys;"
        "import agent_harness.memory.providers as p;"
        "drivers={'boto3','pymongo','motor','redis','asyncpg','aiomysql',"
        "'elasticsearch','azure','google'};"
        "print(sorted(m for m in sys.modules if m.split('.')[0] in drivers))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, check=True).stdout.strip()
    assert out == "[]", f"a driver was imported eagerly: {out}"


def test_a_backend_that_needs_a_driver_says_which_one():
    assert available()["sqlite"] is True        # standard library
    assert available()["http"] is True          # httpx ships with the harness
    assert missing_driver("sqlite") == ""
    for name in ("postgres", "mysql", "mongo", "redis", "s3", "azure", "gcs"):
        if not available()[name]:
            assert "pip install" in missing_driver(name)
    assert "azure-storage-blob" in missing_driver("azure") or available()["azure"]


def test_urls_route_to_the_right_backend():
    for url, expected in [
        ("sqlite:///./m.db", "sqlite"), ("postgresql://u@h/db", "postgres"),
        ("postgres://u@h/db", "postgres"), ("mysql://u@h/db", "mysql"),
        ("mariadb://u@h/db", "mysql"), ("mongodb://h/agents", "mongo"),
        ("mongodb+srv://h/agents", "mongo"), ("redis://h:6379/0", "redis"),
        ("rediss://h:6379/0", "redis"), ("dynamodb://table", "dynamodb"),
        ("es://h:9200", "elasticsearch"), ("opensearch://h:9200", "elasticsearch"),
        ("s3://bucket/prefix", "s3"), ("minio://bucket", "s3"),
        ("gs://bucket", "gcs"), ("azure://container", "azure"),
        ("https://api.test/memory", "http"), ("memory://", "memory"),
    ]:
        assert SCHEMES[url.split("://")[0]] == expected, url


def test_an_unknown_scheme_is_refused_with_the_known_ones():
    with pytest.raises(ConfigurationError, match="no memory backend"):
        memory_provider("cassandra://host/keyspace")


def test_a_url_builds_a_working_store(tmp_path):
    store = memory_provider(f"sqlite://{tmp_path / 'm.db'}")
    assert type(store).__name__ == "SQLiteMemory"
    assert memory_provider("memory://").__class__.__name__ == "InMemoryStore"


def test_you_can_register_your_own_backend():
    register_backend("cassandra", "agent_harness.memory.base", "InMemoryStore",
                     schemes=["cassandra"], drivers=["cassandra_driver"])
    try:
        assert SCHEMES["cassandra"] == "cassandra"
        assert DRIVERS["cassandra"] == ["cassandra_driver"]
        assert available()["cassandra"] is False
    finally:
        for registry, key in ((BACKENDS, "cassandra"), (DRIVERS, "cassandra"),
                              (SCHEMES, "cassandra")):
            registry.pop(key, None)


# --- the object-store family, through a fake bucket --------------------------------

class FakeS3:
    """Enough of the S3 client to drive ObjectStoreMemory for real."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.calls: list[str] = []

    def put_object(self, *, Bucket, Key, Body, ContentType):  # noqa: N803
        self.calls.append("put")
        self.objects[Key] = Body

    def get_object(self, *, Bucket, Key):  # noqa: N803
        self.calls.append("get")
        if Key not in self.objects:
            raise type("NoSuchKey", (Exception,), {})()
        return {"Body": type("B", (), {"read": lambda s: self.objects[Key]})()}

    def list_objects_v2(self, *, Bucket, Prefix, MaxKeys=1000, **kw):  # noqa: N803
        self.calls.append("list")
        keys = sorted(k for k in self.objects if k.startswith(Prefix))
        return {"Contents": [{"Key": k} for k in keys], "IsTruncated": False}

    def delete_objects(self, *, Bucket, Delete):  # noqa: N803
        self.calls.append("delete")
        for row in Delete["Objects"]:
            self.objects.pop(row["Key"], None)


async def test_an_object_store_lays_keys_out_so_a_trace_is_a_prefix():
    from agent_harness.memory.providers.s3 import S3Memory

    fake = FakeS3()
    store = S3Memory("bucket", client=fake, prefix="mem")
    alice, bob = Trace(user_id="alice"), Trace(user_id="bob")

    await store.append(alice.stamp(record("alice's fact")))
    await store.append(bob.stamp(record("bob's fact")))

    keys = sorted(fake.objects)
    assert keys[0].startswith("mem/records/u-alice/user/")
    assert keys[1].startswith("mem/records/u-bob/user/")
    assert keys[0].endswith(".json")

    assert [r.text for r in await store.all(trace=alice)] == ["alice's fact"]
    assert [r.text for r in await store.all(trace=bob)] == ["bob's fact"]
    assert len(await store.all()) == 2


async def test_an_object_store_keeps_documents_per_trace():
    from agent_harness.memory.providers.s3 import S3Memory

    store = S3Memory("bucket", client=FakeS3(), prefix="mem")
    alice, bob = Trace(user_id="alice"), Trace(user_id="bob")

    await store.write_doc("user.md", "- alice", trace=alice)
    await store.write_doc("user.md", "- bob", trace=bob)

    assert await store.read_doc("user.md", trace=alice) == "- alice"
    assert await store.read_doc("user.md", trace=bob) == "- bob"
    assert await store.read_doc("nothing.md", trace=alice) == ""


async def test_clearing_one_trace_leaves_the_others_alone():
    from agent_harness.memory.providers.s3 import S3Memory

    fake = FakeS3()
    store = S3Memory("bucket", client=fake, prefix="mem")
    alice, bob = Trace(user_id="alice"), Trace(user_id="bob")

    await store.append(alice.stamp(record("alice")))
    await store.append(bob.stamp(record("bob")))
    await store.write_doc("user.md", "- alice", trace=alice)

    await store.clear(trace=alice)
    assert await store.all(trace=alice) == []
    assert await store.read_doc("user.md", trace=alice) == ""
    assert [r.text for r in await store.all(trace=bob)] == ["bob"]


async def test_records_are_fetched_concurrently_not_one_at_a_time():
    from agent_harness.memory.providers.s3 import S3Memory

    fake = FakeS3()
    store = S3Memory("bucket", client=fake, prefix="mem", concurrency=8)
    alice = Trace(user_id="alice")
    for index in range(12):
        await store.append(alice.stamp(record(f"fact {index}", ts=1000.0 + index)))

    fake.calls.clear()
    rows = await store.all(trace=alice)
    assert [r.text for r in rows] == [f"fact {i}" for i in range(12)]
    assert fake.calls.count("list") == 1          # one listing, then parallel gets


async def test_a_limit_narrows_before_the_objects_are_fetched():
    from agent_harness.memory.providers.s3 import S3Memory

    fake = FakeS3()
    store = S3Memory("bucket", client=fake, prefix="mem")
    alice = Trace(user_id="alice")
    for index in range(10):
        await store.append(alice.stamp(record(f"fact {index}", ts=1000.0 + index)))

    fake.calls.clear()
    rows = await store.all(trace=alice, limit=3)
    assert [r.text for r in rows] == ["fact 7", "fact 8", "fact 9"]
    assert fake.calls.count("get") == 3           # only the three we wanted


# --- the SQL family ---------------------------------------------------------------

def test_each_dialect_writes_its_own_placeholders():
    from agent_harness.memory.providers.mysql import MySQLMemory
    from agent_harness.memory.providers.postgres import PostgresMemory
    from agent_harness.memory.providers.sqlite import SQLiteMemory

    sql = "SELECT * FROM t WHERE a = ? AND b = ?"
    assert SQLiteMemory(":memory:")._q(sql).endswith("a = ? AND b = ?")
    assert MySQLMemory("mysql://h/db")._q(sql).endswith("a = %s AND b = %s")

    postgres = PostgresMemory("postgresql://h/db")
    postgres.paramstyle = "numeric"
    assert postgres._q(sql).endswith("a = $1 AND b = $2")


def test_the_filter_treats_an_untagged_record_as_shared():
    from agent_harness.memory.providers.sqlite import SQLiteMemory

    where, params = SQLiteMemory._where("user", None, Trace(user_id="alice"))
    assert "scope = ?" in where
    assert "(user_id = ? OR user_id IS NULL)" in where
    assert params == ["user", "alice"]


def test_mysql_reads_its_connection_details_from_a_url():
    from agent_harness.memory.providers.mysql import MySQLMemory

    store = MySQLMemory("mysql://bob:s3cret@db.internal:3307/agents")
    assert store.connect_args == {"host": "db.internal", "port": 3307,
                                  "user": "bob", "password": "s3cret",
                                  "db": "agents"}


def test_mysql_keys_stay_indexable():
    from agent_harness.memory.providers.mysql import MySQLMemory

    # utf8mb4 indexes cap at 191 characters on older MySQL.
    assert "191" in MySQLMemory.key_type
    assert any("191" in statement for statement in
               MySQLMemory("mysql://h/db")._schema())


def test_the_schema_indexes_the_way_the_harness_reads():
    from agent_harness.memory.providers.sqlite import SQLiteMemory

    schema = " ".join(SQLiteMemory(":memory:")._schema())
    assert "user_id, session_id, tenant_id, scope, ts" in schema


# --- document stores, through their query builders ----------------------------------

def test_mongo_builds_a_query_that_keeps_shared_records():
    from agent_harness.memory.providers.mongo import MongoMemory

    query = MongoMemory._query("user", "fact", Trace(user_id="alice",
                                                     tenant_id="acme"))
    assert query["scope"] == "user" and query["kind"] == "fact"
    assert query["user_id"] == {"$in": ["alice", None]}
    assert query["tenant_id"] == {"$in": ["acme", None]}
    assert MongoMemory._query(None, None, None) == {}


def test_elasticsearch_filters_on_the_trace_and_scores_the_text():
    from agent_harness.memory.providers.elasticsearch import ElasticsearchMemory

    query = ElasticsearchMemory._query("user", None, Trace(user_id="alice"),
                                       "refund window")
    assert query["bool"]["must"] == [{"match": {"text": "refund window"}}]
    assert {"term": {"scope": "user"}} in query["bool"]["filter"]
    trace_filter = query["bool"]["filter"][-1]["bool"]["should"]
    assert {"term": {"user_id": "alice"}} in trace_filter


def test_redis_keys_carry_the_namespace_and_the_scope():
    from agent_harness.memory.providers.redis import RedisMemory

    store = RedisMemory(prefix="mem")
    assert store._records_key("u-alice", "user") == "mem:rec:u-alice:user"
    assert store._doc_key("user.md", Trace(user_id="alice")) == \
        "mem:doc:u-alice/user.md"
    assert store._namespace(None, record(user_id="alice")) == "u-alice"


def test_dynamodb_partitions_by_namespace_and_sorts_by_time():
    from agent_harness.memory.providers.dynamodb import DynamoDBMemory

    assert DynamoDBMemory._namespace(Trace(user_id="alice")) == "u-alice"
    assert DynamoDBMemory._namespace(None, record(user_id="bob")) == "u-bob"
    assert DynamoDBMemory._namespace(None) == "_shared"


# --- the HTTP backend, over a real transport ------------------------------------------

def http_store(**kw):
    from agent_harness.memory.providers.http import HTTPMemory

    state: dict[str, object] = {"records": [], "docs": {}, "seen": []}

    def handler(request: httpx.Request) -> httpx.Response:
        state["seen"].append(f"{request.method} {request.url.path}")
        if request.url.path.endswith("/records"):
            if request.method == "POST":
                state["records"].append(json.loads(request.content))
                return httpx.Response(201, json={})
            if request.method == "DELETE":
                state["records"] = []
                return httpx.Response(204)
            wanted = dict(request.url.params)
            rows = [r for r in state["records"]
                    if all(r.get(k) in (v, None) for k, v in wanted.items()
                           if k not in {"limit"})]
            return httpx.Response(200, json=rows)
        key = request.url.path.rsplit("/", 1)[-1]
        if request.method == "PUT":
            state["docs"][key] = json.loads(request.content)["text"]
            return httpx.Response(204)
        if key not in state["docs"]:
            return httpx.Response(404)
        return httpx.Response(200, json={"text": state["docs"][key]})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return HTTPMemory("https://api.test/memory", client=client, **kw), state


async def test_the_http_backend_speaks_the_documented_contract():
    store, state = http_store()
    alice = Trace(user_id="alice")

    await store.append(alice.stamp(record("alice's fact")))
    assert state["records"][0]["user_id"] == "alice"

    rows = await store.all("user", trace=alice)
    assert [r.text for r in rows] == ["alice's fact"]
    assert "GET /memory/records" in state["seen"]

    await store.write_doc("user.md", "- alice", trace=alice)
    assert await store.read_doc("user.md", trace=alice) == "- alice"
    assert await store.read_doc("missing.md", trace=alice) == ""

    await store.clear(trace=alice)
    assert await store.all(trace=alice) == []


async def test_the_http_backend_reports_a_server_error_clearly():
    from agent_harness.errors import ProviderError
    from agent_harness.memory.providers.http import HTTPMemory

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(500, text="boom")))
    store = HTTPMemory("https://api.test", client=client)
    with pytest.raises(ProviderError, match="returned 500"):
        await store.append(record())


# --- a missing driver is a clear message, never an ImportError ----------------------

@pytest.mark.parametrize("name,expect", [
    ("postgres", "asyncpg"),
    ("mysql", "aiomysql"),
    ("mongo", "motor"),
    ("redis", "redis"),
    ("s3", "boto3"),
    ("azure", "azure-storage-blob"),
    ("gcs", "google-cloud-storage"),
    ("elasticsearch", "elasticsearch"),
])
async def test_a_missing_driver_names_the_install(name, expect):
    if available()[name]:
        pytest.skip(f"{name}'s driver is installed here")

    from agent_harness.memory.providers import get_backend

    cls = get_backend(name)
    store = (cls("bucket") if name in {"s3", "gcs"}
             else cls("container", connection_string="x") if name == "azure"
             else cls("scheme://host/db"))
    with pytest.raises(ConfigurationError, match=expect):
        await store.all()
