"""Where chats are kept: every store holds a session under the same promises.

One contract — `exercise()` — is run against each store. In-process, files and
SQLite run for real everywhere. MongoDB, DynamoDB and object storage also run
here against small stand-ins for their clients. And every database runs for real
when its URL is in the environment:

    AH_TEST_POSTGRES_URL=postgresql://postgres:pw@127.0.0.1:55432/agents
    AH_TEST_MYSQL_URL=mysql://root:pw@127.0.0.1:53306/agents
    AH_TEST_MONGO_URL=mongodb://127.0.0.1:57017/agents
    AH_TEST_REDIS_URL=redis://127.0.0.1:56379/0
    AH_TEST_DYNAMODB_ENDPOINT=http://127.0.0.1:58000
    AH_TEST_S3_ENDPOINT=http://127.0.0.1:59000        (MinIO: minioadmin/minioadmin)
    AH_TEST_AZURE_CONNECTION=<an Azurite or storage-account connection string>
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

import pytest

from agent_harness import (
    Agent,
    Blueprint,
    ConfigurationError,
    FakeProvider,
    FileSessionStore,
    Harness,
    InMemorySessionStore,
    Message,
    Session,
    SessionConflict,
    SessionStore,
    session_backends,
    session_provider,
    tool,
    tool_call,
)
from agent_harness.memory.providers._objectstore import ObjectStoreMemory
from agent_harness.memory.providers.dynamodb import DynamoDBMemory
from agent_harness.memory.providers.mongo import MongoMemory
from agent_harness.sessions.dynamodb import DynamoDBSessionStore
from agent_harness.sessions.mongo import MongoSessionStore
from agent_harness.sessions.objects import ObjectSessionStore
from agent_harness.sessions.sql import SQLSessionStore
from agent_harness.types import ToolResultBlock, ToolUseBlock, Usage

# --- the contract --------------------------------------------------------------------

def conversation(**fields: Any) -> Session:
    call = ToolUseBlock(name="lookup", input={"q": "π ≈ 3.14", "$weird.key": [1, {"a": None}]})
    return Session(**{
        "agent": "support", "user_id": "alice", "tenant_id": "acme",
        "messages": [Message.user("Where is order 4182?"),
                     Message(role="assistant", content=[call]),
                     Message.tool_results([ToolResultBlock(tool_use_id=call.id,
                                                           content="shipped ✓")]),
                     Message.assistant("It shipped.")],
        "usage": Usage(input_tokens=120, output_tokens=8, cost_usd=0.0021, calls=2),
        "metadata": {"sandbox": {"sandbox": "docker", "id": "abc"}},
        **fields})


async def exercise(store: SessionStore, *, atomic: bool = True) -> None:
    """Everything a chat needs of a store. Leaves nothing behind."""
    made: list[str] = []
    try:
        # --- it comes back as it went in, and counts its saves
        first = conversation(title="Order 4182")
        await store.save(first)
        made.append(first.id)
        assert first.version == 1
        back = await store.load(first.id)
        assert back.model_dump(exclude={"updated"}) == first.model_dump(
            exclude={"updated"})
        assert back.messages[1].tool_uses[0].input["$weird.key"] == [1, {"a": None}]

        back.messages.append(Message.user("Thanks."))
        await store.save(back)
        assert (await store.load(first.id)).version == 2
        assert len((await store.load(first.id)).messages) == 5

        # --- a stale save is refused, and leaves both sides as they were
        stale = (await store.load(first.id)).model_copy(deep=True)
        fresh = (await store.load(first.id)).model_copy(deep=True)
        fresh.title = "saved first"
        await store.save(fresh)
        stale.title = "saved second"
        with pytest.raises(SessionConflict):
            await store.save(stale)
        assert stale.version == 2
        held = await store.load(first.id)
        assert (held.title, held.version) == ("saved first", 3)

        # --- two that both think they are creating it: one does
        twin = Session(id=first.id, agent="support", user_id="alice", tenant_id="acme")
        with pytest.raises(SessionConflict):
            await store.save(twin)

        # --- listing: newest first, and narrowed to whose it is
        await asyncio.sleep(0.01)
        second = conversation(title="Bob's", user_id="bob")
        third = conversation(title="Another tenant", tenant_id="globex")
        fourth = conversation(title="Alice again", agent="billing")
        for session in (second, third, fourth):
            await asyncio.sleep(0.01)
            await store.save(session)
            made.append(session.id)

        mine = await store.list(user_id="alice", tenant_id="acme")
        assert [s.title for s in mine] == ["Alice again", "saved first"]
        assert [s.title for s in await store.list(user_id="bob", tenant_id="acme")] == [
            "Bob's"]
        assert {s.id for s in await store.list(tenant_id="acme", limit=50)} >= {
            first.id, second.id, fourth.id}
        assert third.id not in {s.id for s in await store.list(tenant_id="acme",
                                                               limit=50)}
        assert [s.title for s in await store.list(
            user_id="alice", tenant_id="acme", agent="billing")] == ["Alice again"]
        assert [s.title for s in await store.list(
            user_id="alice", tenant_id="acme", limit=1)] == ["Alice again"]
        assert await store.list(user_id="nobody", tenant_id="acme") == []

        # --- a long conversation, larger than any one row or item is allowed
        long = conversation(title="long")
        long.messages += [Message.user(f"turn {n} " + os.urandom(600).hex())
                          for n in range(700)]
        await store.save(long)
        made.append(long.id)
        assert len((await store.load(long.id)).messages) == 704
        long = await store.load(long.id)
        long.messages = long.messages[:10]          # compacted: it shrinks, too
        await store.save(long)
        assert len((await store.load(long.id)).messages) == 10

        # --- a fork is its own session, owned by the same person
        fork = await store.fork(first.id, at=2)
        made.append(fork.id)
        assert fork.id != first.id and fork.parent_id == first.id
        assert (fork.user_id, fork.tenant_id, len(fork.messages)) == ("alice", "acme", 2)

        # --- of many racing from one version, exactly one wins
        if atomic:
            copies = [(await store.load(first.id)).model_copy(deep=True)
                      for _ in range(8)]
            outcomes = await asyncio.gather(*(store.save(c) for c in copies),
                                            return_exceptions=True)
            won = [o for o in outcomes if isinstance(o, Session)]
            assert len(won) == 1, outcomes
            assert all(isinstance(o, (Session, SessionConflict)) for o in outcomes)

        # --- delete, and deleting what is not there is not an error
        await store.delete(first.id)
        with pytest.raises(ConfigurationError, match="no session"):
            await store.load(first.id)
        assert first.id not in {s.id for s in await store.list(
            user_id="alice", tenant_id="acme")}
        await store.delete(first.id)
        with pytest.raises(ConfigurationError, match="no session"):
            await store.load("ses_never_was")
    finally:
        for session_id in made:
            await store.delete(session_id)


# --- the stores that need nothing ------------------------------------------------------

async def test_in_process_holds_the_contract():
    await exercise(InMemorySessionStore())


async def test_files_hold_the_contract(tmp_path):
    store = FileSessionStore(tmp_path)
    await exercise(store)
    assert list(tmp_path.glob("*.tmp")) == []          # nothing half-written left


async def test_sqlite_holds_the_contract(tmp_path):
    store = session_provider(f"sqlite:///{tmp_path}/chats.db", table="chats")
    assert isinstance(store, SQLSessionStore) and store.table == "chats"
    await exercise(store)
    assert (await store.check())["ok"]
    await store.aclose()


async def test_a_file_written_before_sessions_had_versions_still_loads(tmp_path):
    (tmp_path / "ses_old.json").write_text(json.dumps({
        "id": "ses_old", "agent": "support",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]}))
    store = FileSessionStore(tmp_path)
    old = await store.load("ses_old")
    assert (old.version, old.user_id, old.messages[0].text) == (0, None, "hi")
    await store.save(old)
    assert (await store.load("ses_old")).version == 1


async def test_every_store_can_prove_itself(tmp_path):
    for store in (InMemorySessionStore(), FileSessionStore(tmp_path / "f"),
                  session_provider(f"sqlite:///{tmp_path}/c.db")):
        report = await store.check()
        assert report["ok"], report
        assert [s["step"] for s in report["steps"]] == [
            "save", "load", "list by owner", "refuse a stale save", "delete"]


async def test_a_store_that_does_not_hold_the_line_fails_its_own_check():
    class LastWriteWins(InMemorySessionStore):
        async def save(self, session):
            session.version += 1
            self._sessions[session.id] = session
            return session

    report = await LastWriteWins().check()
    assert not report["ok"]
    failed = next(s for s in report["steps"] if not s["ok"])
    assert failed["step"] == "refuse a stale save"
    assert "overwrite each other" in failed["detail"]


# --- stand-ins for clients that need a server ------------------------------------------

class _DuplicateKeyError(Exception):
    pass


_DuplicateKeyError.__name__ = "DuplicateKeyError"


class _Cursor:
    def __init__(self, rows):
        self.rows = rows

    def sort(self, key, direction):
        self.rows.sort(key=lambda r: r[key], reverse=direction < 0)
        return self

    def limit(self, n):
        self.rows = self.rows[:n]
        return self

    def __iter__(self):
        return iter(self.rows)


class _Collection:
    def __init__(self):
        self.docs: dict[str, dict] = {}
        self.indexes: list[Any] = []

    def create_index(self, keys, **kw):
        self.indexes.append(keys)

    def insert_one(self, doc):
        if doc["_id"] in self.docs:
            raise _DuplicateKeyError(doc["_id"])
        self.docs[doc["_id"]] = dict(doc)

    def replace_one(self, query, doc):
        held = self.docs.get(query["_id"])
        matched = held is not None and held["version"] == query["version"]
        if matched:
            self.docs[query["_id"]] = dict(doc)
        return type("Result", (), {"matched_count": int(matched)})()

    def find_one(self, query):
        return self.docs.get(query["_id"])

    def find(self, query):
        return _Cursor([dict(d) for d in self.docs.values()
                        if all(d.get(k) == v for k, v in query.items())])

    def delete_one(self, query):
        self.docs.pop(query["_id"], None)


async def test_mongo_holds_the_contract_against_a_stand_in():
    collections: dict[str, _Collection] = {}

    class Database(dict):
        def __missing__(self, name):
            return collections.setdefault(name, _Collection())

    client = {"agents": Database()}
    mongo = MongoMemory(database="agents", client=client)
    mongo._async_driver, mongo._ready = False, True
    store = session_provider(mongo, collection="chats")
    assert isinstance(store, MongoSessionStore)
    await exercise(store)
    assert collections["chats"].indexes == [
        [("tenant_id", 1), ("user_id", 1), ("updated", -1)]]
    assert collections["chats"].docs == {}


class _ConditionalCheckFailedException(Exception):
    pass


_ConditionalCheckFailedException.__name__ = "ConditionalCheckFailedException"


class _Dynamo:
    """Enough of DynamoDB to keep it honest: conditional puts, the 400 KB item
    limit, consistent reads and paged queries."""

    def __init__(self):
        self.items: dict[tuple[str, str], dict] = {}

    @staticmethod
    def _size(item):
        return sum(len(k) + len(next(iter(v.values()))) for k, v in item.items())

    def put_item(self, TableName, Item, ConditionExpression=None,
                 ExpressionAttributeValues=None, ReturnValues=None, **_):
        assert self._size(Item) <= 400_000, "an item may not exceed 400 KB"
        key = (Item["pk"]["S"], Item["sk"]["S"])
        held = self.items.get(key)
        if ConditionExpression == "attribute_not_exists(pk)" and held is not None:
            raise _ConditionalCheckFailedException()
        if ConditionExpression == "#v = :expect" and (
                held is None or held["version"] != ExpressionAttributeValues[":expect"]):
            raise _ConditionalCheckFailedException()
        self.items[key] = Item
        return {"Attributes": held} if ReturnValues == "ALL_OLD" and held else {}

    def get_item(self, TableName, Key, **_):
        item = self.items.get((Key["pk"]["S"], Key["sk"]["S"]))
        return {"Item": item} if item else {}

    def delete_item(self, TableName, Key, **_):
        self.items.pop((Key["pk"]["S"], Key["sk"]["S"]), None)

    def query(self, TableName, ExpressionAttributeValues, ExclusiveStartKey=None, **_):
        pk = ExpressionAttributeValues[":pk"]["S"]
        rows = sorted((v for (p, _), v in self.items.items() if p == pk),
                      key=lambda r: r["sk"]["S"])
        start = ExclusiveStartKey or 0
        page = {"Items": rows[start:start + 2]}          # tiny pages, to page
        if start + 2 < len(rows):
            page["LastEvaluatedKey"] = start + 2
        return page


async def test_dynamodb_holds_the_contract_against_a_stand_in():
    client = _Dynamo()
    store = session_provider(DynamoDBMemory("agents", client=client))
    assert isinstance(store, DynamoDBSessionStore)
    await exercise(store)
    # Every piece of every replaced body was cleared away behind it.
    assert client.items == {}


class _Bucket(ObjectStoreMemory):
    def __init__(self):
        super().__init__("bucket", prefix="chats")
        self.blobs: dict[str, bytes] = {}

    async def append(self, record):  # pragma: no cover - not what is under test
        return record

    async def _put(self, key, body, content_type):
        self.blobs[key] = body

    async def _get(self, key):
        return self.blobs.get(key)

    async def _list(self, prefix):
        return [k for k in self.blobs if k.startswith(prefix)]

    async def _delete(self, keys):
        for key in keys:
            self.blobs.pop(key, None)


async def test_object_storage_holds_the_contract_against_a_stand_in():
    bucket = _Bucket()
    store = session_provider(bucket)
    assert isinstance(store, ObjectSessionStore)
    # The check before a write is not atomic on an object store, and says so.
    await exercise(store, atomic=False)
    assert bucket.blobs == {}


# --- the real databases, when they are there -----------------------------------------------

def _live() -> list[Any]:
    cases = []
    for name, variable, driver in (
            ("postgres", "AH_TEST_POSTGRES_URL", "asyncpg"),
            ("mysql", "AH_TEST_MYSQL_URL", "aiomysql"),
            ("mongo", "AH_TEST_MONGO_URL", "pymongo"),
            ("redis", "AH_TEST_REDIS_URL", "redis")):
        cases.append(pytest.param(name, variable, driver, id=name))
    return cases


@pytest.mark.parametrize("name, variable, driver", _live())
async def test_a_real_database_holds_the_contract(name, variable, driver):
    url = os.environ.get(variable)
    if not url:
        pytest.skip(f"set {variable} to run against a real {name}")
    pytest.importorskip(driver)
    store = session_provider(url)
    try:
        await exercise(store)
        assert (await store.check())["ok"]
    finally:
        await store.aclose()


async def test_a_real_dynamodb_holds_the_contract():
    endpoint = os.environ.get("AH_TEST_DYNAMODB_ENDPOINT")
    if not endpoint:
        pytest.skip("set AH_TEST_DYNAMODB_ENDPOINT to run against DynamoDB")
    pytest.importorskip("boto3")
    store = session_provider(DynamoDBMemory(
        "ah_sessions_test", create=True, endpoint_url=endpoint,
        region_name="us-east-1", aws_access_key_id="test",
        aws_secret_access_key="test"))
    await exercise(store)


async def test_a_real_s3_bucket_holds_the_contract():
    endpoint = os.environ.get("AH_TEST_S3_ENDPOINT")
    if not endpoint:
        pytest.skip("set AH_TEST_S3_ENDPOINT to run against S3 or MinIO")
    boto3 = pytest.importorskip("boto3")
    keys = {"endpoint_url": endpoint, "region_name": "us-east-1",
            "aws_access_key_id": os.environ.get("AH_TEST_S3_KEY", "minioadmin"),
            "aws_secret_access_key": os.environ.get("AH_TEST_S3_SECRET", "minioadmin")}
    client = boto3.client("s3", **keys)
    try:
        client.create_bucket(Bucket="ah-sessions-test")
    except Exception as exc:
        if "BucketAlready" not in type(exc).__name__ + str(exc):
            raise
    await exercise(session_provider("s3://ah-sessions-test/chats", **keys),
                   atomic=False)


async def test_a_real_azure_storage_account_holds_the_contract():
    connection = os.environ.get("AH_TEST_AZURE_CONNECTION")
    if not connection:
        pytest.skip("set AH_TEST_AZURE_CONNECTION to run against Azure Blob or Azurite")
    pytest.importorskip("azure.storage.blob")
    from azure.storage.blob.aio import BlobServiceClient

    service = BlobServiceClient.from_connection_string(connection)
    try:
        await service.create_container("ah-sessions-test")
    except Exception as exc:
        if type(exc).__name__ != "ResourceExistsError":
            raise
    finally:
        await service.close()
    store = session_provider("azure://ah-sessions-test/chats",
                             connection_string=connection)
    try:
        await exercise(store, atomic=False)
    finally:
        await store.aclose()


# --- naming a store ----------------------------------------------------------------------------

def test_a_store_is_named_by_url_or_built_on_a_memory_backend(tmp_path):
    assert isinstance(session_provider("memory://"), InMemorySessionStore)
    assert isinstance(session_provider(str(tmp_path / "s")), FileSessionStore)
    assert isinstance(session_provider(f"file://{tmp_path}/f"), FileSessionStore)
    held = InMemorySessionStore()
    assert session_provider(held) is held

    for url, kind in (("postgresql://u:p@h/db", "SQLSessionStore"),
                      ("mysql://u:p@h/db", "SQLSessionStore"),
                      ("mongodb://h:27017/db", "MongoSessionStore"),
                      ("redis://h:6379/0", "RedisSessionStore"),
                      ("dynamodb://table", "DynamoDBSessionStore"),
                      ("s3://bucket/prefix", "ObjectSessionStore"),
                      ("azure://container", "ObjectSessionStore"),
                      ("gs://bucket", "ObjectSessionStore")):
        # Nothing connects, and no driver is imported, until it is used.
        assert type(session_provider(url)).__name__ == kind, url

    assert session_provider("redis://h/0", ttl=60).ttl == 60
    assert session_provider("s3://bucket/a/b").objects.prefix == "a/b"
    with pytest.raises(ConfigurationError, match="no session store for 'elasticsearch'"):
        session_provider("elasticsearch://h:9200")
    with pytest.raises(ConfigurationError, match="cannot hold sessions"):
        session_provider(object())
    with pytest.raises(ValueError, match="not a table name"):
        session_provider("sqlite:///x.db", table="chats; drop table x")
    assert session_backends()["sqlite"] and session_backends()["memory"]


def test_a_sqlite_url_names_the_file_you_would_expect(tmp_path):
    from agent_harness.memory import memory_provider

    for url, path in (("sqlite://:memory:", ":memory:"),
                      ("sqlite:///:memory:", ":memory:"),
                      ("sqlite:///chats.db", "chats.db"),
                      ("sqlite:///./data/chats.db", "./data/chats.db"),
                      ("sqlite:////var/lib/chats.db", "/var/lib/chats.db"),
                      ("sqlite:///var/lib/chats.db", "/var/lib/chats.db"),
                      (f"sqlite:///{tmp_path}/c.db", f"{tmp_path}/c.db"),
                      (f"sqlite://{tmp_path}/c.db", f"{tmp_path}/c.db")):
        assert memory_provider(url).path == path, url
        assert session_provider(url).db.path == path, url


async def test_a_harness_takes_a_url_and_can_share_one_database(tmp_path):
    by_url = Harness(sessions=f"sqlite:///{tmp_path}/a.db",
                     memory_store=f"sqlite:///{tmp_path}/a.db")
    assert isinstance(by_url.sessions, SQLSessionStore)

    one = Harness.on(f"sqlite:///{tmp_path}/one.db")
    # Memory and chats: one database, one connection.
    assert one.sessions.db is one.memory_store
    await one.sessions.save(Session(agent="a"))
    assert len(await one.sessions.list()) == 1
    await one.aclose()

    mixed = Harness.local(tmp_path / "state", trace=False,
                          sessions=f"sqlite:///{tmp_path}/chats.db")
    assert isinstance(mixed.sessions, SQLSessionStore)
    assert (tmp_path / "state" / "memory").is_dir()
    assert not (tmp_path / "state" / "sessions").exists()


def test_a_blueprint_declares_where_chats_are_kept(tmp_path):
    blueprint = Blueprint.from_text(f"""
sessions: {{url: "sqlite:///{tmp_path}/chats.db", table: chats}}
agents:
  support: {{instructions: Help.}}
""")
    store = blueprint.session_store()
    assert isinstance(store, SQLSessionStore) and store.table == "chats"
    assert Blueprint.from_text("agents: {a: {}}").session_store() is None


# --- the agent and its sessions ------------------------------------------------------------------

def replica(tmp_path, script: list, **agent: Any) -> Agent:
    """One of several processes serving the same chats from one database."""
    harness = Harness(sessions=f"sqlite:///{tmp_path}/chats.db",
                      provider=FakeProvider(script))
    return Agent("support", mode="chat", harness=harness, memory=False, **agent)


async def test_a_chat_is_continued_by_another_replica(tmp_path):
    first = replica(tmp_path, ["Hello Ada."], trace={"user_id": "ada", "tenant_id": "acme"})
    started = await first.run("I am Ada, order 4182.")
    await first.harness.aclose()

    second = replica(tmp_path, ["Order 4182, Ada."],
                     trace={"user_id": "ada", "tenant_id": "acme"})
    result = await second.run("Which order was it?", session=started.session_id)

    assert result.ok and result.warnings == []
    assert [m.text for m in second.harness.provider.requests[0].messages] == [
        "I am Ada, order 4182.", "Hello Ada.", "Which order was it?"]
    saved = await second.harness.sessions.load(started.session_id)
    assert (saved.user_id, saved.tenant_id, saved.version) == ("ada", "acme", 2)
    assert saved.title == "I am Ada, order 4182."
    mine = await second.harness.sessions.list(user_id="ada", tenant_id="acme")
    assert [s.id for s in mine] == [started.session_id]


async def test_someone_elses_chat_does_not_exist_as_far_as_you_can_tell(tmp_path):
    ada = replica(tmp_path, ["Hello Ada."], trace={"user_id": "ada", "tenant_id": "acme"})
    hers = await ada.run("My card ends 4242.")

    for trace in ({"user_id": "bob", "tenant_id": "acme"},
                  {"user_id": "ada", "tenant_id": "globex"}):
        other = replica(tmp_path, ["never asked"], trace=trace)
        with pytest.raises(ConfigurationError, match="no session") as refused:
            await other.run("What card?", session=hers.session_id)
        # The same words as for an id that was never there: nothing to probe.
        with pytest.raises(ConfigurationError, match="no session") as missing:
            await other.run("What card?", session="ses_never_was")
        assert str(refused.value).replace(hers.session_id, "X") == str(
            missing.value).replace("ses_never_was", "X").split(" in ")[0]
        with pytest.raises(ConfigurationError, match="no session"):
            await other.resume(hers.session_id)
        assert other.harness.provider.requests == []
        assert any(e.action == "session_denied" for e in other.harness.audit.entries)

    # An agent acting for nobody in particular is a single-user setup, and trusted.
    admin = replica(tmp_path, ["It ends 4242."])
    assert (await admin.run("What card?", session=hers.session_id)).ok


async def test_two_requests_on_one_chat_both_keep_their_turn(tmp_path):
    """The second to finish finds the session has moved on, and adds its own
    turn to what is there rather than overwriting the other's."""
    opened = replica(tmp_path, ["Hi."])
    chat = (await opened.run("Hello.")).session_id
    fast = replica(tmp_path, ["The fast answer."])

    @tool
    async def slow_lookup() -> str:
        """Takes long enough for another request to finish first."""
        await fast.run("A second question.", session=chat)
        return "found it"

    slow = replica(tmp_path, [tool_call("slow_lookup"), "The slow answer."],
                   tools=[slow_lookup])
    result = await slow.run("A first question.", session=chat)

    assert result.ok and result.session_id == chat and result.warnings == []
    saved = await slow.harness.sessions.load(chat)
    assert [m.text for m in saved.messages if m.text] == [
        "Hello.", "Hi.", "A second question.", "The fast answer.",
        "A first question.", "The slow answer."]
    # The tool call and its result travelled together.
    calls = [b.id for m in saved.messages for b in m.content
             if isinstance(b, ToolUseBlock)]
    results = [b.tool_use_id for m in saved.messages for b in m.content
               if isinstance(b, ToolResultBlock)]
    assert calls == results and len(calls) == 1
    assert saved.usage.calls == 4
    assert any(e.action == "session_merged" for e in slow.harness.audit.entries)


async def test_a_run_that_cannot_be_merged_is_kept_as_a_fork(tmp_path):
    opened = replica(tmp_path, ["Hi."])
    chat = (await opened.run("Hello.")).session_id
    fast = replica(tmp_path, ["The fast answer."])

    @tool
    async def slow_lookup() -> str:
        """Another request finishes while this one is still working."""
        await fast.run("A second question.", session=chat)
        return "x" * 4000

    def script(request):
        if request.messages[-1].text.startswith("Summarise"):
            return "an earlier exchange"
        return next(steps)

    steps = iter([tool_call("slow_lookup"), "The slow answer."])
    # Its history is compacted along the way, so "this run's messages" no longer
    # lines up with what the other request saved.
    slow = replica(tmp_path, [script], tools=[slow_lookup], compact_at=200,
                   compact_keep_last=1)
    slow.harness.provider.loop = True
    result = await slow.run("A first question. " + "pad " * 300, session=chat)

    assert result.ok and result.output == "The slow answer."
    assert result.session_id != chat
    assert "is kept as" in result.warnings[0]
    original = await slow.harness.sessions.load(chat)
    assert [m.text for m in original.messages if m.text][-1] == "The fast answer."
    fork = await slow.harness.sessions.load(result.session_id)
    assert fork.parent_id == chat and fork.messages[-1].text == "The slow answer."
    # The agent carries on in the fork.
    assert slow._thread.id == result.session_id


async def test_a_store_that_is_down_loses_the_record_not_the_answer():
    class Down(InMemorySessionStore):
        async def save(self, session):
            raise ConnectionError("the database is unreachable")

    harness = Harness(sessions=Down(), provider=FakeProvider(["The answer."]))
    result = await Agent("support", harness=harness).run("A question.")

    assert result.ok and result.output == "The answer."
    assert result.warnings == [
        "the session could not be saved: ConnectionError: the database is unreachable"]
    assert any(e.action == "session_save" and e.decision == "error"
               for e in harness.audit.entries)


async def test_a_plain_agent_starts_each_run_clean_but_keeps_the_whole_record():
    provider = FakeProvider(["one", "two"])
    harness = Harness.testing(provider)
    agent = Agent("plain", harness=harness)

    first = await agent.run("first")
    second = await agent.run("second")

    assert [m.text for m in provider.requests[1].messages] == ["second"]
    assert second.session_id == first.session_id
    saved = await harness.sessions.load(first.session_id)
    assert [m.text for m in saved.messages] == ["first", "one", "second", "two"]
    assert (saved.version, saved.usage.calls) == (2, 2)


async def test_a_one_off_run_is_not_kept_as_a_chat():
    harness = Harness.testing(FakeProvider(["a side answer"]))
    agent = Agent("helper", harness=harness)
    result = await agent.as_tool().invoke({"task": "a side job"})
    assert result == "a side answer"
    assert await harness.sessions.list() == []


# --- the command line ------------------------------------------------------------------------------

def test_the_cli_keeps_chats_in_a_database_and_lists_them_by_owner(
        tmp_path, capsys, monkeypatch):
    from agent_harness import cli
    from agent_harness.llm_providers import register_provider

    class Scripted(FakeProvider):
        name = "scripted"

        def __init__(self, **kw):
            super().__init__(["Noted.", "Order 4182."], **kw)

    register_provider("scripted", Scripted)
    url = f"sqlite:///{tmp_path}/chats.db"
    common = ["--provider", "scripted", "--model", "fake-1", "--no-memory",
              "--mode", "chat", "--sessions", url, "--tenant", "acme"]

    assert cli.main(["sessions", "--store", url, "--check"]) == 0
    assert "SQLSessionStore works" in capsys.readouterr().out

    assert cli.main(["run", "My order is 4182.", "--user", "ada", *common]) == 0
    cli.main(["run", "Something else.", "--user", "bob", *common])
    capsys.readouterr()

    assert cli.main(["sessions", "--store", url, "--user", "ada", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert [(r["user_id"], r["tenant_id"], r["title"]) for r in rows] == [
        ("ada", "acme", "My order is 4182.")]

    # Bob cannot continue Ada's chat; Ada can.
    assert cli.main(["run", "Which order?", "--user", "bob", "--session",
                     rows[0]["id"], *common]) == 1
    assert "no session" in capsys.readouterr().err
    assert cli.main(["run", "Which order?", "--user", "ada", "--session",
                     rows[0]["id"], *common]) == 0

    assert cli.main(["sessions", "--store", url, "--show", rows[0]["id"]]) == 0
    shown = capsys.readouterr().out
    assert "user: My order is 4182." in shown and "user: Which order?" in shown

    assert cli.main(["sessions", "--store", url, "--delete", rows[0]["id"]]) == 0
    assert cli.main(["sessions", "--store", url, "--user", "ada"]) == 0
    assert "no sessions yet" in capsys.readouterr().out
    assert cli.main(["sessions", "--backends"]) == 0
    assert "sqlite     ready" in capsys.readouterr().out
