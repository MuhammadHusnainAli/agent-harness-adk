"""Traces: whose memory this is, and keeping one user out of another's."""

from __future__ import annotations

import pytest

from agent_harness import Agent, FakeProvider, Harness, MemoryManager, Trace, tool_call
from agent_harness.memory import InMemoryStore
from agent_harness.memory.base import FileStore, MemoryRecord

MODEL = "claude-sonnet-5"


def record(text: str = "a fact", **kw) -> MemoryRecord:
    return MemoryRecord(scope="user", text=text, **kw)


# --- the trace itself ---------------------------------------------------------

def test_a_trace_reads_as_a_namespace():
    trace = Trace(tenant_id="acme", user_id="alice", session_id="s-42")
    assert trace.namespace == "t=acme/u=alice"      # scope defaults to user
    assert trace.at("session").namespace == "t=acme/u=alice/s=s-42"
    assert trace.at("tenant").namespace == "t=acme"
    assert trace.at("global").namespace == ""


def test_the_slug_is_safe_for_a_filename_or_an_object_key():
    trace = Trace(user_id="alice@example.com", tenant_id="ACME Ltd/EU")
    assert "/" not in trace.slug and "@" not in trace.slug
    assert trace.slug == "t-ACME_Ltd_EU__u-alice_example.com"
    assert Trace().slug == "_shared"


def test_an_empty_trace_is_falsy_and_shared():
    assert not Trace()
    assert Trace().is_empty
    assert Trace(user_id="alice")


def test_of_accepts_a_bare_user_id_a_dict_or_nothing():
    assert Trace.of("alice").user_id == "alice"
    assert Trace.of({"user_id": "bob", "tenant_id": "acme"}).tenant_id == "acme"
    assert Trace.of(None).is_empty
    assert Trace.of(Trace(user_id="carol"), session_id="s1").session_id == "s1"


def test_stamping_puts_the_identity_on_a_record():
    trace = Trace(user_id="alice", tenant_id="acme", agent="support")
    stamped = trace.stamp(record())
    assert stamped.user_id == "alice" and stamped.tenant_id == "acme"
    assert stamped.source == "support"


def test_stamping_never_overwrites_an_identity_already_there():
    trace = Trace(user_id="alice")
    already = record(user_id="bob")
    assert trace.stamp(already).user_id == "bob"


def test_matching_isolates_users_but_keeps_shared_records():
    alice = Trace(user_id="alice")
    assert alice.matches(record(user_id="alice"))
    assert not alice.matches(record(user_id="bob"))
    # A record written before traces existed belongs to everyone, not nobody.
    assert alice.matches(record())


def test_a_child_narrows_without_touching_the_parent():
    parent = Trace(tenant_id="acme", user_id="alice")
    child = parent.child(session_id="s-1")
    assert child.session_id == "s-1" and parent.session_id is None
    assert child.user_id == "alice"


# --- stores honour it -----------------------------------------------------------

@pytest.mark.parametrize("make_store", [
    lambda tmp: InMemoryStore(),
    lambda tmp: FileStore(tmp / "file"),
    lambda tmp: __import__("agent_harness.memory.providers.sqlite",
                           fromlist=["SQLiteMemory"]).SQLiteMemory(":memory:"),
])
async def test_every_store_keeps_users_apart(make_store, tmp_path):
    store = make_store(tmp_path)
    alice, bob = Trace(user_id="alice"), Trace(user_id="bob")

    await store.append(alice.stamp(record("alice bills in EUR")))
    await store.append(bob.stamp(record("bob bills in USD")))

    assert [r.text for r in await store.all("user", trace=alice)] == \
        ["alice bills in EUR"]
    assert [r.text for r in await store.all("user", trace=bob)] == \
        ["bob bills in USD"]
    assert len(await store.all("user")) == 2        # unscoped sees both

    await store.write_doc("user.md", "- alice: EUR", trace=alice)
    await store.write_doc("user.md", "- bob: USD", trace=bob)
    assert await store.read_doc("user.md", trace=alice) == "- alice: EUR"
    assert await store.read_doc("user.md", trace=bob) == "- bob: USD"

    await store.clear(trace=alice)
    assert await store.all("user", trace=alice) == []
    assert [r.text for r in await store.all("user", trace=bob)] == \
        ["bob bills in USD"]
    assert await store.read_doc("user.md", trace=alice) == ""
    assert await store.read_doc("user.md", trace=bob) == "- bob: USD"
    await store.aclose()


async def test_a_limit_returns_the_newest_in_chronological_order(tmp_path):
    from agent_harness.memory.providers.sqlite import SQLiteMemory

    store = SQLiteMemory(":memory:")
    for index in range(10):
        await store.append(MemoryRecord(scope="user", text=f"fact {index}",
                                        ts=1000.0 + index))
    rows = await store.all("user", limit=3)
    assert [r.text for r in rows] == ["fact 7", "fact 8", "fact 9"]
    await store.aclose()


async def test_sqlite_survives_a_restart(tmp_path):
    from agent_harness.memory.providers.sqlite import SQLiteMemory

    path = tmp_path / "memory.db"
    alice = Trace(user_id="alice")

    store = SQLiteMemory(path)
    await store.append(alice.stamp(record("remember this")))
    await store.write_doc("user.md", "- remembered", trace=alice)
    await store.aclose()

    reopened = SQLiteMemory(path)
    assert [r.text for r in await reopened.all(trace=alice)] == ["remember this"]
    assert await reopened.read_doc("user.md", trace=alice) == "- remembered"
    await reopened.aclose()


# --- the manager and the agent -----------------------------------------------------

async def test_the_manager_scopes_everything_to_its_trace():
    store = InMemoryStore()
    alice = MemoryManager(store, semantic=False, trace="alice")
    bob = MemoryManager(store, semantic=False, trace=Trace(user_id="bob"))

    await alice.user.remember("bills in EUR")
    await bob.user.remember("bills in USD")

    assert "EUR" in await alice.user.load()
    assert "EUR" not in await bob.user.load()
    assert "USD" in await bob.user.load()


async def test_one_store_serves_many_users():
    store = InMemoryStore()
    shared = MemoryManager(store, semantic=False)

    await shared.for_trace("alice").user.remember("alice: EUR")
    await shared.for_trace("bob").user.remember("bob: USD")

    assert "alice: EUR" in await shared.for_trace("alice").user.load()
    assert "alice" not in await shared.for_trace("bob").user.load()


async def test_the_session_id_lands_on_the_trace():
    manager = MemoryManager(InMemoryStore(), semantic=False, trace="alice")
    assert manager.trace.session_id == manager.session.id

    await manager.remember("something", scope="job")
    stored = await manager.store.all("job")
    assert stored[0].user_id == "alice"
    assert stored[0].session_id == manager.session.id


async def test_recall_only_reaches_the_right_users_memory():
    store = InMemoryStore()
    alice = MemoryManager(store, trace="alice")
    bob = MemoryManager(store, trace="bob")

    await alice.remember("the refund window is 30 days", scope="job")
    assert await bob.recall("refund window") == []
    assert await alice.recall("refund window")


async def test_an_agent_takes_a_trace_and_keeps_users_apart():
    harness = Harness.testing(FakeProvider([
        tool_call("remember", fact="prefers metric units"), "noted",
    ]))
    alice = Agent("support", provider=harness.provider, model=MODEL,
                  harness=harness, trace="alice")
    await alice.run("I prefer metric")

    bob = Agent("support", provider=harness.provider, model=MODEL,
                harness=harness, trace="bob")
    assert "metric" in await alice.memory.user.load()
    assert "metric" not in await bob.memory.user.load()
    assert alice.trace.user_id == "alice"
    assert alice.trace.agent == "support"


async def test_a_trace_can_be_a_session_instead_of_a_user():
    store = InMemoryStore()
    first = MemoryManager(store, semantic=False,
                          trace=Trace(session_id="s-1", scope="session"))
    second = MemoryManager(store, semantic=False,
                           trace=Trace(session_id="s-2", scope="session"))

    await first.user.remember("this conversation is about refunds")
    assert "refunds" in await first.user.load()
    assert "refunds" not in await second.user.load()


async def test_tenants_are_isolated_even_with_the_same_user_id():
    store = InMemoryStore()
    acme = MemoryManager(store, semantic=False,
                         trace=Trace(tenant_id="acme", user_id="alice"))
    globex = MemoryManager(store, semantic=False,
                           trace=Trace(tenant_id="globex", user_id="alice"))

    await acme.user.remember("acme's alice prefers EUR")
    assert "EUR" in await acme.user.load()
    assert "EUR" not in await globex.user.load()


async def test_for_trace_reuses_the_store_rather_than_rebuilding_it():
    """Serving a request per user must cost a small object, not a new index."""
    from agent_harness.memory import SemanticMemory

    shared = MemoryManager(InMemoryStore())
    assert isinstance(shared.store, SemanticMemory)

    per_user = [shared.for_trace(f"user{index}") for index in range(5)]
    assert all(m.store is shared.store for m in per_user)
    assert all(m.store.index is shared.store.index for m in per_user)

    plain = MemoryManager(InMemoryStore(), semantic=False)
    assert plain.for_trace("alice").store is plain.store
