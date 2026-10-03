"""A2A: serving an agent to other frameworks, and calling theirs."""

from __future__ import annotations

import asyncio
import base64
import json
import threading
import time
from typing import Any

import httpx
import pytest

from agent_harness import (
    Agent,
    Blueprint,
    ConfigurationError,
    FakeProvider,
    Harness,
    Workflow,
    tool,
    tool_call,
)
from agent_harness.a2a import (
    A2AClient,
    A2AError,
    A2AServer,
    MemoryTaskStore,
    RemoteAgent,
    SessionTaskStore,
)
from agent_harness.a2a import protocol as wire
from agent_harness.cli import main

BASE = "http://agents.test"
GATE: dict[str, asyncio.Event] = {}


@tool
async def slow(label: str = "") -> str:
    """Wait until the test lets it go.

    Args:
        label: which gate to wait at.
    """
    GATE.setdefault("started", asyncio.Event()).set()
    await GATE.setdefault(label or "go", asyncio.Event()).wait()
    return "waited"


@pytest.fixture(autouse=True)
def _gates():
    GATE.clear()
    yield
    GATE.clear()


def served(*script: Any, harness: Harness | None = None, tools: Any = (),
           agent_kw: dict[str, Any] | None = None,
           **kw: Any) -> tuple[A2AServer, Agent]:
    harness = harness or Harness.testing()
    agent = Agent("desk", "Answer the customer.", description="A support desk.",
                  provider=FakeProvider(list(script), stream_words=True),
                  harness=harness, memory=False, tools=list(tools), **(agent_kw or {}))
    kw.setdefault("poll_interval", 0.01)
    return A2AServer(agent, **kw), agent


def calling(server: A2AServer, path: str = "", **kw: Any) -> A2AClient:
    kw.setdefault("retries", 0)
    return A2AClient(f"{BASE}{path}", transport=httpx.ASGITransport(app=server), **kw)


async def rpc(server: A2AServer, method: str, params: dict[str, Any] | None = None,
              **kw: Any) -> dict[str, Any]:
    return await server.handle({"jsonrpc": "2.0", "id": 1, "method": method,
                                "params": params or {}}, **kw)


def said(text: str, **fields: Any) -> dict[str, Any]:
    return {"message": {"role": "user", "parts": [{"kind": "text", "text": text}],
                        "messageId": wire.new_uuid(), **fields}}


# ----------------------------------------------------------------------
# the card
# ----------------------------------------------------------------------
async def test_the_card_says_what_the_agent_is_and_where():
    server, _ = served(version="2.1.0", provider={"organization": "Acme",
                                                  "url": "https://acme.test"})
    async with calling(server) as client:
        card = await client.card()
        legacy = (await client.http.get(f"{BASE}/.well-known/agent.json")).json()

    assert card == legacy
    assert card["protocolVersion"] == "0.3.0" and card["preferredTransport"] == "JSONRPC"
    assert (card["name"], card["description"], card["version"]) == (
        "desk", "A support desk.", "2.1.0")
    assert card["url"] == f"{BASE}/"                    # read from the request
    assert card["capabilities"] == {"streaming": True, "pushNotifications": False,
                                    "stateTransitionHistory": False}
    assert card["skills"] == [{"id": "desk", "name": "desk",
                               "description": "A support desk.", "tags": ["agent"]}]
    assert "securitySchemes" not in card and card["provider"]["organization"] == "Acme"

    locked, _ = served(auth="sk-1", url="https://public.example/a2a/")
    card = locked.card()
    assert card["url"] == "https://public.example/a2a/"
    assert card["security"] == [{"bearer": []}]


# ----------------------------------------------------------------------
# sending a message
# ----------------------------------------------------------------------
async def test_a_message_becomes_a_task_that_completes():
    server, agent = served("Your order shipped on Thursday.")
    async with calling(server) as client:
        task = await client.send("Where is order 4182?")

    assert task.ok and task.done and task.text == "Your order shipped on Thursday."
    raw = task.raw
    assert raw["kind"] == "task" and raw["status"]["state"] == "completed"
    assert raw["artifacts"][0]["name"] == "response"
    assert raw["artifacts"][0]["parts"] == [
        {"kind": "text", "text": "Your order shipped on Thursday."}]
    assert [m["role"] for m in raw["history"]] == ["user", "agent"]
    assert raw["history"][0]["taskId"] == task.id == raw["id"]
    assert raw["contextId"] == task.context_id
    assert agent.provider.requests[0].messages[0].text == "Where is order 4182?"
    assert server.health()["completed"] == 1


async def test_a_context_id_carries_the_conversation_on():
    server, agent = served("Noted: order 4182.", "It was 4182.")
    async with calling(server) as client:
        first = await client.send("My order is 4182.")
        second = await client.send("Which order?", context_id=first.context_id)
        other = await client.send("Which order?")

    assert second.context_id == first.context_id and second.id != first.id
    assert second.text == "It was 4182."
    seen = [[m.text for m in r.messages] for r in agent.provider.requests]
    assert seen[1] == ["My order is 4182.", "Noted: order 4182.", "Which order?"]
    assert seen[2] == ["Which order?"] and other.context_id != first.context_id


async def test_a_stream_is_the_task_as_it_happens():
    server, _ = served("Your order shipped on Thursday.")
    async with calling(server) as client:
        events = [e async for e in client.stream("Where is my order?")]

    kinds = [e.kind for e in events]
    assert kinds[0] == "task" and events[0].state == "submitted"
    assert kinds[1] == "status-update" and events[1].state == "working"
    assert kinds[-1] == "status-update" and events[-1].final
    assert events[-1].state == "completed"
    chunks = [e for e in events if e.kind == "artifact-update"]
    assert "".join(e.text for e in chunks) == "Your order shipped on Thursday."
    assert len(chunks) > 3 and chunks[0].raw["append"] is False
    assert all(c.raw["append"] for c in chunks[1:]) and chunks[-1].raw["lastChunk"]
    assert len({c.raw["artifact"]["artifactId"] for c in chunks}) == 1
    # The task that was streamed is the task that is kept.
    kept = await rpc(server, "tasks/get", {"id": events[0].task_id})
    assert wire.answer_of(kept["result"]) == "Your order shipped on Thursday."


async def test_what_was_kept_replaces_what_was_streamed_when_they_differ():
    harness = Harness.testing()

    @harness.hooks.on("run_end")
    def withhold(ctx):
        ctx.data["result"].output = "[withheld by policy]"

    server, _ = served("Write to ada@example.com today.", harness=harness)
    async with calling(server) as client:
        events = [e async for e in client.stream("Who do I write to?")]
    last = [e for e in events if e.kind == "artifact-update"][-1]
    assert last.raw["append"] is False and last.raw["lastChunk"] is True
    assert last.text == "[withheld by policy]"


async def test_a_message_can_be_sent_without_waiting_for_it():
    server, _ = served(tool_call("slow"), "Done.", tools=[slow])
    async with calling(server) as client:
        task = await client.send("Take your time.", blocking=False)
        assert task.state in ("submitted", "working") and not task.done
        await GATE.setdefault("started", asyncio.Event()).wait()
        assert (await client.get(task.id)).state == "working"
        GATE.setdefault("go", asyncio.Event()).set()
        finished = await client.wait(task.id, interval=0.01)
    assert finished.ok and finished.text == "Done."
    assert (await rpc(server, "tasks/get", {"id": task.id, "historyLength": 1})
            )["result"]["history"][0]["role"] == "agent"


async def test_the_same_message_sent_twice_is_one_task():
    server, agent = served("Refunded.", "never said")
    async with calling(server) as client:
        first = await client.send("Refund order 4182.", message_id="msg-1")
        again = await client.send("Refund order 4182.", message_id="msg-1")
        events = [e async for e in client.stream("Refund order 4182.",
                                                 message_id="msg-1")]
    assert again.id == first.id and again.text == "Refunded."
    assert len(agent.provider.requests) == 1
    assert events[0].kind == "task" and events[0].final
    assert server.health()["deduplicated"] == 2


async def test_files_and_data_reach_the_agent():
    server, agent = served("It says hello.")
    note = base64.b64encode(b"hello from the file").decode()
    answer = await rpc(server, "message/send", {"message": {
        "role": "user", "messageId": "m1", "parts": [
            {"kind": "text", "text": "Read these."},
            {"kind": "data", "data": {"order": 4182}},
            {"kind": "file", "file": {"name": "note.txt", "mimeType": "text/plain",
                                      "bytes": note}}]}})
    task = answer["result"]
    assert task["status"]["state"] == "completed"
    sent = agent.provider.requests[0].messages[0]
    assert "Read these." in sent.text and '{"order": 4182}' in sent.text
    assert "hello from the file" in "".join(
        getattr(b, "text", "") for b in sent.content)
    # The record keeps that a file came, not the file.
    assert task["history"][0]["parts"][2] == {
        "kind": "file", "file": {"name": "note.txt", "mimeType": "text/plain"}}


async def test_an_agent_that_fails_is_a_failed_task():
    server, _ = served(tool_call("nope"), tool_call("nope"),
                       agent_kw={"max_steps": 2})
    async with calling(server) as client:
        task = await client.send("Go.")
    assert task.state == "failed" and "MaxStepsExceeded" in task.error
    assert task.raw["artifacts"] == [] and server.health()["failed"] == 1


# ----------------------------------------------------------------------
# what is refused
# ----------------------------------------------------------------------
async def test_requests_that_are_wrong_are_answered_as_json_rpc_says():
    server, _ = served("Hello.")
    assert (await server.handle([1]))["error"]["code"] == -32600
    assert (await rpc(server, "tasks/nope"))["error"]["code"] == -32601
    assert (await rpc(server, "message/send", {}))["error"]["code"] == -32602
    assert (await rpc(server, "message/send", {"message": {"parts": []}}))[
        "error"]["code"] == -32602
    assert (await rpc(server, "message/send", {"message": {
        "parts": [{"kind": "hologram"}]}}))["error"]["code"] == -32005
    assert (await rpc(server, "tasks/get", {"id": "ghost"}))["error"]["code"] == -32001
    assert (await rpc(server, "agent/getAuthenticatedExtendedCard"))[
        "error"]["code"] == -32007
    assert (await rpc(server, "tasks/pushNotificationConfig/set", {
        "taskId": "x"}))["error"]["code"] == -32001

    async with calling(server) as client:
        bad = await client.http.post(f"{BASE}/", content=b"{not json")
        assert bad.json()["error"]["code"] == -32700
        assert (await client.http.get(f"{BASE}/")).status_code == 405
        assert (await client.http.get(f"{BASE}/nowhere")).status_code == 404
        done = await client.send("Hi.")
        with pytest.raises(A2AError, match="takes no more messages") as refused:
            await client.send("And?", task_id=done.id)
        assert refused.value.code == -32602


async def test_a_body_that_is_too_large_is_refused():
    server, _ = served("Hello.", max_body_bytes=200)
    async with calling(server) as client:
        with pytest.raises(A2AError, match="larger than 200 bytes") as refused:
            await client.send("x" * 500)
    assert refused.value.status == 413


# ----------------------------------------------------------------------
# who is calling
# ----------------------------------------------------------------------
async def test_a_caller_must_be_known_and_has_only_what_is_theirs():
    harness = Harness.testing()
    server, agent = served(
        "Noted.", "Yours is 4182.", "I do not know.", harness=harness,
        auth={"sk-ada": {"user_id": "ada", "tenant_id": "acme"}, "sk-bob": "bob"})

    async with calling(server) as nobody:
        with pytest.raises(A2AError, match="not authorised") as refused:
            await nobody.send("Hello?")
        assert refused.value.status == 401
        assert (await nobody.card())["name"] == "desk"         # the card is public
    async with calling(server, token="sk-wrong") as wrong:
        with pytest.raises(A2AError, match="not authorised"):
            await wrong.send("Hello?")

    async with calling(server, token="sk-ada") as ada, \
            calling(server, token="sk-bob") as bob:
        mine = await ada.send("My order is 4182.")
        assert (await ada.send("Which?", context_id=mine.context_id)).text == (
            "Yours is 4182.")
        # Bob is told Ada's task and Ada's conversation do not exist.
        with pytest.raises(A2AError, match="no task") as hidden:
            await bob.get(mine.id)
        assert hidden.value.code == -32001
        with pytest.raises(A2AError, match="no task"):
            await bob.cancel(mine.id)
        with pytest.raises(A2AError, match="no conversation"):
            await bob.send("Which order was it?", context_id=mine.context_id)

    # The conversation is a session of the agent's, and it is Ada's.
    chats = await harness.sessions.list(user_id="ada", tenant_id="acme", agent="desk")
    assert len(chats) == 1 and len(chats[0].messages) == 4
    assert await harness.sessions.list(user_id="bob", agent="desk") == []


async def test_a_function_can_decide_who_is_calling():
    async def gate(headers: dict[str, str]) -> Any:
        key = headers.get("x-api-key")
        return {"user_id": key, "plan": "gold"} if key in ("k1", "k2") else None

    server, _ = served("Hello.", auth=gate,
                       security_schemes={"key": {"type": "apiKey", "in": "header",
                                                 "name": "X-API-Key"}})
    assert server.card()["security"] == [{"key": []}]
    async with calling(server, headers={"X-API-Key": "k1"}) as client:
        assert (await client.send("Hi.")).ok
    async with calling(server, headers={"X-API-Key": "zz"}) as client:
        with pytest.raises(A2AError, match="not authorised"):
            await client.send("Hi.")


async def test_every_conversation_gets_an_agent_of_its_own():
    built: list[str | None] = []

    def factory(principal: Any) -> Agent:
        built.append(principal.user_id)
        return Agent("desk", provider=FakeProvider(["Hello."]), memory=False,
                     harness=harness, trace={"user_id": principal.user_id})

    harness = Harness.testing()
    server = A2AServer(factory, auth={"t1": "ada", "t2": "bob"},
                       store=SessionTaskStore(harness.sessions))
    async with calling(server, token="t1") as ada, calling(server, token="t2") as bob:
        first = await ada.send("Hi.")
        await ada.send("Again.", context_id=first.context_id)     # the same agent
        await bob.send("Hi.")
    assert built == ["ada", "bob"]

    with pytest.raises(ConfigurationError, match="needs `store=`"):
        A2AServer(factory)
    with pytest.raises(ConfigurationError, match="serves an Agent"):
        A2AServer({"x": 5})


# ----------------------------------------------------------------------
# cancelling
# ----------------------------------------------------------------------
async def test_a_running_task_can_be_cancelled():
    server, agent = served(tool_call("slow"), "never said", tools=[slow])
    async with calling(server) as client:
        task = await client.send("Wait.", blocking=False)
        await GATE.setdefault("started", asyncio.Event()).wait()
        cancelled = await client.cancel(task.id)
        assert cancelled.state == "canceled" and cancelled.error == "Canceled."
        with pytest.raises(A2AError, match="already canceled") as twice:
            await client.cancel(task.id)
    assert twice.value.code == -32002
    assert len(agent.provider.requests) == 1 and server.health()["running"] == 0


async def test_a_task_that_runs_too_long_is_stopped():
    server, _ = served(tool_call("slow"), "never", tools=[slow], task_timeout=0.05)
    async with calling(server) as client:
        task = await client.send("Wait.")
    assert task.state == "failed" and "ran longer than 0.05s" in task.error


# ----------------------------------------------------------------------
# at scale: more than one replica
# ----------------------------------------------------------------------
def replicas(*script: Any, tools: Any = (), **kw: Any) -> tuple[A2AServer, A2AServer]:
    """Two servers as two processes would be: their own agents and harnesses,
    one store of sessions and tasks between them."""
    shared = Harness.testing().sessions
    provider = FakeProvider(list(script), stream_words=True)

    def one() -> A2AServer:
        harness = Harness.testing(sessions=shared)
        agent = Agent("desk", provider=provider, harness=harness, memory=False,
                      tools=list(tools))
        return A2AServer(agent, poll_interval=0.01, **kw)

    return one(), one()


async def test_any_replica_answers_for_a_task_and_carries_on_a_conversation():
    a, b = replicas("Noted: 4182.", "It was 4182.")
    async with calling(a) as on_a, calling(b) as on_b:
        first = await on_a.send("My order is 4182.")
        assert (await on_b.get(first.id)).text == "Noted: 4182."
        second = await on_b.send("Which order?", context_id=first.context_id)
    assert second.text == "It was 4182."


async def test_a_retry_that_lands_on_another_replica_waits_for_the_first():
    a, b = replicas(tool_call("slow"), "Refunded.", tools=[slow])
    async with calling(a) as on_a, calling(b) as on_b:
        started = await on_a.send("Refund it.", message_id="m-1", blocking=False)
        await GATE.setdefault("started", asyncio.Event()).wait()
        retry = asyncio.create_task(on_b.send("Refund it.", message_id="m-1"))
        await asyncio.sleep(0.03)
        assert not retry.done()
        GATE.setdefault("go", asyncio.Event()).set()
        finished = await retry
    assert finished.id == started.id and finished.text == "Refunded."
    assert b.health()["accepted"] == 0 and b.health()["deduplicated"] == 1


async def test_a_task_is_cancelled_from_a_replica_that_is_not_running_it():
    a, b = replicas(tool_call("slow"), "never said", tools=[slow])
    async with calling(a) as on_a, calling(b) as on_b:
        task = await on_a.send("Wait.", blocking=False)
        await GATE.setdefault("started", asyncio.Event()).wait()
        asked = await on_b.cancel(task.id)
        assert asked.state == "working"            # marked; not b's to stop
        GATE.setdefault("go", asyncio.Event()).set()
        finished = await on_a.wait(task.id, interval=0.01)
    assert finished.state == "canceled"


async def test_a_task_is_followed_from_a_replica_that_is_not_running_it():
    a, b = replicas(tool_call("slow"), "All done.", tools=[slow])
    async with calling(a) as on_a, calling(b) as on_b:
        task = await on_a.send("Wait.", blocking=False)
        await GATE.setdefault("started", asyncio.Event()).wait()

        async def follow(client: A2AClient) -> list[Any]:
            return [e async for e in client.resubscribe(task.id)]

        here, there = asyncio.create_task(follow(on_a)), asyncio.create_task(follow(on_b))
        await asyncio.sleep(0.03)
        GATE.setdefault("go", asyncio.Event()).set()
        local, remote = await here, await there
    for events in (local, remote):
        assert events[0].kind == "task" and events[-1].final
        assert events[-1].state == "completed"
        assert "All done." in "".join(e.text for e in events)


async def test_a_task_whose_worker_died_is_reported_failed():
    a, b = replicas("unused", task_timeout=0.05)
    accepted, *_ = await a._accept(a.mounts["desk"], said("Hello."), a_principal())
    # Replica a took the task and died before working on it.
    async with calling(b) as on_b:
        assert (await on_b.get(accepted.id)).state == "submitted"
        record = await b.store.load(accepted.id)
        record.updated = time.time() - 3600
        await b.store.save(record)
        lost = await on_b.get(accepted.id)
    assert lost.state == "failed" and "worker handling this task was lost" in lost.error


def a_principal() -> Any:
    from agent_harness.a2a import Principal

    return Principal()


async def test_a_full_replica_says_so_and_the_client_comes_back():
    server, _ = served(tool_call("slow"), "One.", "Two.", tools=[slow],
                       max_concurrency=1, max_queue=0)
    async with calling(server) as client:
        first = await client.send("One?", blocking=False)
        with pytest.raises(A2AError, match="at capacity") as full:
            await client.send("Two?")
        assert (full.value.status, full.value.code) == (429, -32000)
        raw = await client.http.post(f"{BASE}/", json={
            "jsonrpc": "2.0", "id": 1, "method": "message/send", "params": said("x")})
        assert raw.status_code == 429 and raw.headers["retry-after"] == "2"
        assert server.health()["rejected"] == 2

        patient = calling(server, retries=3)
        waiting = asyncio.create_task(patient.send("Two?"))
        await asyncio.sleep(0.05)
        GATE.setdefault("go", asyncio.Event()).set()
        await client.wait(first.id, interval=0.01)
        assert (await waiting).ok
        await patient.aclose()


async def test_a_server_that_is_closing_finishes_its_books():
    server, _ = served(tool_call("slow"), "never", tools=[slow])
    async with calling(server) as client:
        task = await client.send("Wait.", blocking=False)
        await GATE.setdefault("started", asyncio.Event()).wait()
        await server.aclose()
        after = await client.get(task.id)
        assert after.state == "canceled" and "shut down" in after.error
        health = await client.http.get(f"{BASE}/healthz")
        assert health.status_code == 503 and health.json()["status"] == "draining"
        with pytest.raises(A2AError, match="shutting down"):
            await client.send("More?")


async def test_tasks_can_be_kept_somewhere_of_your_own():
    store = MemoryTaskStore()
    server, _ = served("Hello.", store=store)
    async with calling(server) as client:
        task = await client.send("Hi.")
    held = await store.load(task.id)
    assert held.state == "completed" and held.agent == "desk"
    # The default keeps them with the chats, but never lists them among them.
    harness = Harness.testing()
    other, _ = served("Hello.", harness=harness)
    await rpc(other, "message/send", said("Hi."))
    assert sorted(s.agent for s in await harness.sessions.list()) == [
        "desk", "desk#a2a"]
    assert len(await harness.sessions.list(agent="desk")) == 1


# ----------------------------------------------------------------------
# push notifications
# ----------------------------------------------------------------------
async def test_a_webhook_is_told_when_the_task_is_over():
    received: list[tuple[dict[str, str], dict[str, Any]]] = []

    def webhook(request: httpx.Request) -> httpx.Response:
        received.append((dict(request.headers), json.loads(request.content)))
        return httpx.Response(200)

    server, _ = served("Refunded.", push_notifications=True)
    server._http = httpx.AsyncClient(transport=httpx.MockTransport(webhook))
    assert server.card()["capabilities"]["pushNotifications"] is True
    async with calling(server) as client:
        task = await client.send("Refund it.", push={
            "url": "https://hooks.example.com/a2a", "token": "t-9"})
        listed = await client.call("tasks/pushNotificationConfig/list", {"id": task.id})
        await asyncio.gather(*server._pushes)
    headers, body = received[0]
    assert headers["x-a2a-notification-token"] == "t-9"
    assert body["id"] == task.id and body["status"]["state"] == "completed"
    assert listed[0]["pushNotificationConfig"]["url"] == "https://hooks.example.com/a2a"
    await server.aclose()


async def test_webhooks_are_off_unless_allowed_and_never_point_inwards():
    closed, _ = served("Hi.")
    refused = await rpc(closed, "message/send", {
        **said("Hi."), "configuration": {"pushNotificationConfig": {
            "url": "https://hooks.example.com"}}})
    assert refused["error"]["code"] == -32003

    server, _ = served("Hi.", push_notifications=True)
    for url in ("http://hooks.example.com", "https://localhost/x",
                "https://10.0.0.8/x", "https://127.0.0.1/x", "https://db.internal/x"):
        answer = await rpc(server, "message/send", {
            **said("Hi."), "configuration": {"pushNotificationConfig": {"url": url}}})
        assert answer["error"]["code"] == -32602, url

    own, _ = served("Hi.", push_notifications=lambda url: url.startswith("http://ci."))
    task = (await rpc(own, "message/send", said("Hi.")))["result"]
    done = await rpc(own, "tasks/pushNotificationConfig/set", {
        "taskId": task["id"], "pushNotificationConfig": {"url": "http://ci.test/hook"}})
    config = done["result"]["pushNotificationConfig"]
    assert (await rpc(own, "tasks/pushNotificationConfig/get", {"id": task["id"]}))[
        "result"]["pushNotificationConfig"] == config
    await rpc(own, "tasks/pushNotificationConfig/delete", {
        "id": task["id"], "pushNotificationConfigId": config["id"]})
    assert (await rpc(own, "tasks/pushNotificationConfig/list", {"id": task["id"]}))[
        "result"] == []


# ----------------------------------------------------------------------
# several agents on one server
# ----------------------------------------------------------------------
async def test_several_agents_are_served_under_their_names():
    harness = Harness.testing()
    billing = Agent("billing", provider=FakeProvider(["Refunded."]), harness=harness,
                    memory=False)
    orders = Agent("orders", provider=FakeProvider(["Shipped."]), harness=harness,
                   memory=False)
    server = A2AServer({"billing": billing, "orders": orders})
    async with calling(server, "/billing") as to_billing, \
            calling(server, "/orders") as to_orders:
        assert (await to_orders.card())["url"] == f"{BASE}/orders/"
        assert (await to_billing.send("Refund.")).text == "Refunded."
        assert (await to_orders.send("Where?")).text == "Shipped."
        index = (await to_orders.http.get(f"{BASE}/")).json()
    assert index["agents"]["billing"] == f"{BASE}/billing/.well-known/agent-card.json"


# ----------------------------------------------------------------------
# a remote agent, used as one of your own
# ----------------------------------------------------------------------
async def remote(server: A2AServer, **kw: Any) -> RemoteAgent:
    return await RemoteAgent.connect(
        BASE, transport=httpx.ASGITransport(app=server), **kw)


async def test_a_remote_agent_is_a_sub_agent_a_tool_and_a_workflow_step():
    server, far = served("The gold plan is 40 EUR.", "Still 40 EUR.", "40 EUR, flat.")
    pricing = await remote(server)
    assert (pricing.name, pricing.description) == ("desk", "A support desk.")

    harness = Harness.testing()
    manager = Agent(
        "manager", harness=harness, memory=False, subagents=[pricing],
        tools=[pricing.as_tool("ask_pricing")],
        provider=FakeProvider([
            tool_call("delegate", agent_name="desk", task="Price of gold?"),
            tool_call("ask_pricing", task="And now?"), "Gold is 40 EUR."]))
    result = await manager.run("What does gold cost?")

    assert result.output == "Gold is 40 EUR."
    assert result.children[0].agent == "desk"
    assert result.children[0].output == "The gold plan is 40 EUR."
    handed = [b.content for m in result.messages for b in m.content
              if getattr(b, "type", "") == "tool_result"]
    assert handed == ["The gold plan is 40 EUR.", "Still 40 EUR."]

    flow = Workflow.from_text("""
steps:
  - {id: ask, agent: desk, input: "Price of {{ inputs.plan }}?"}
""", agents=[pricing], harness=harness)
    assert (await flow.run({"plan": "gold"})).output == "40 EUR, flat."
    assert far.provider.requests[-1].messages[0].text == "Price of gold?"
    await pricing.aclose()


async def test_a_remote_agent_run_directly_keeps_its_conversation():
    server, far = served("Noted.", "It was 4182.", "Which one?")
    pricing = await remote(server, name="pricing")
    first = await pricing.run("My order is 4182.")
    second = await pricing.run("Which order?")
    assert second.output == "It was 4182." and second.session_id == first.session_id
    pricing.new_session()
    assert (await pricing.run("Which order?")).session_id != first.session_id
    assert len(far.provider.requests[1].messages) == 3
    await pricing.aclose()


async def test_a_remote_failure_is_a_result_not_an_exception():
    server, _ = served(tool_call("nope"), tool_call("nope"), agent_kw={"max_steps": 2})
    harness = Harness.testing()
    far = await remote(server, harness=harness)
    result = await far.run("Go.")
    assert not result.ok and "MaxStepsExceeded" in result.error
    assert harness.audit.entries[-1].action == "a2a_call"

    gone = RemoteAgent("http://nowhere.test", name="gone", transport=httpx.MockTransport(
        lambda request: httpx.Response(502, text="bad gateway")))
    gone.client.retries = 0
    result = await gone.run("Hello?")
    assert not result.ok and "A2AError" in result.error

    with pytest.raises(A2AError, match="no A2A agent card"):
        await RemoteAgent.connect("http://nowhere.test", transport=httpx.MockTransport(
            lambda request: httpx.Response(404)))


async def test_governance_must_know_a_remote_agent_before_it_is_delegated_to():
    from agent_harness.governance import Governance

    server, _ = served("40 EUR.", "40 EUR.")
    harness = Harness.testing(governance=Governance())
    script = [tool_call("delegate", agent_name="desk", task="Price?"), "Done."]

    stranger = await remote(server)
    boss = Agent("boss", harness=harness, memory=False, subagents=[stranger],
                 provider=FakeProvider(list(script)))
    refused = await boss.run("Ask.")
    assert "[desk not permitted]" in refused.messages[2].content[0].content

    known = await remote(server, harness=harness,
                         identity={"owner": "pricing-team", "purpose": "pricing"})
    boss = Agent("boss", harness=harness, memory=False, subagents=[known],
                 provider=FakeProvider(list(script)))
    assert (await boss.run("Ask.")).children[0].output == "40 EUR."
    assert harness.governance.inventory.agents["desk"]["provider"] == "a2a"


async def test_a_blueprint_can_name_an_agent_by_where_it_is(monkeypatch):
    monkeypatch.setenv("PRICING_TOKEN", "sk-9")
    blueprint = Blueprint.from_text("""
agents:
  pricing:
    description: Knows the price list.
    a2a: {url: "https://agents.example.com/pricing", token_env: PRICING_TOKEN}
  other: {a2a: "https://agents.example.com/other"}
""")
    pricing = blueprint.build("pricing")
    assert isinstance(pricing, RemoteAgent)
    assert (pricing.name, pricing.description) == ("pricing", "Knows the price list.")
    assert pricing.client.headers["Authorization"] == "Bearer sk-9"
    monkeypatch.delenv("PRICING_TOKEN")
    with pytest.raises(ConfigurationError, match="PRICING_TOKEN is not set"):
        blueprint.build("pricing")


async def test_the_client_reads_an_agent_that_answers_with_a_message():
    def plain(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json={"name": "Echo Agent", "url": "http://e.test/rpc"})
        assert str(request.url) == "http://e.test/rpc"
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": 1, "result": {
            "kind": "message", "role": "agent", "messageId": "r1", "contextId": "c1",
            "parts": [{"kind": "text", "text": "echo"}]}})

    echo = await RemoteAgent.connect("http://e.test", transport=httpx.MockTransport(plain))
    assert echo.name == "Echo_Agent"
    result = await echo.run("hi")
    assert result.ok and result.output == "echo" and result.session_id == "c1"


# ----------------------------------------------------------------------
# the built-in server, and the command line
# ----------------------------------------------------------------------
class Running:
    """The development server, in a thread of its own, as a real process would
    have it: a socket, and nothing shared with the caller but the wire."""

    def __init__(self, *script: Any, **kw: Any) -> None:
        self.script, self.kw = script, kw
        self.url = ""
        self._ready = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._task: asyncio.Task[Any] | None = None
        self._thread = threading.Thread(target=self._main, daemon=True)

    def _main(self) -> None:
        async def go() -> None:
            server, _ = served(*self.script, **self.kw)
            self._loop = asyncio.get_running_loop()
            self._task = asyncio.current_task()

            def ready(url: str) -> None:
                self.url = url
                self._ready.set()

            try:
                await server.serve("127.0.0.1", 0, ready=ready)
            except asyncio.CancelledError:
                pass

        asyncio.run(go())

    def __enter__(self) -> Running:
        self._thread.start()
        assert self._ready.wait(5), "the server did not start"
        return self

    def __exit__(self, *exc: object) -> None:
        self._loop.call_soon_threadsafe(self._task.cancel)
        self._thread.join(5)


async def test_the_built_in_server_speaks_http():
    with Running("Your order shipped on Thursday.", "It arrives Monday.") as running:
        async with A2AClient(running.url) as client:
            card = await client.card()
            assert card["url"] == f"{running.url}/"
            task = await client.send("Where is my order?")
            events = [e async for e in client.stream("When does it arrive?",
                                                     context_id=task.context_id)]
    assert task.text == "Your order shipped on Thursday."
    assert "".join(e.text for e in events) == "It arrives Monday."
    assert events[-1].final and events[-1].state == "completed"


def test_the_command_line_reads_a_card_and_sends_a_message(capsys):
    with Running("Shipped on Thursday.", "Arrives Monday.", auth="sk-1") as running:
        assert main(["a2a", "card", running.url]) == 0
        assert json.loads(capsys.readouterr().out)["name"] == "desk"

        assert main(["a2a", "send", running.url, "Where is it?", "--token", "sk-1"]) == 0
        printed = capsys.readouterr()
        assert printed.out.strip() == "Shipped on Thursday."
        assert "[completed · task " in printed.err

        assert main(["a2a", "send", running.url, "When?", "--token", "sk-1",
                     "--stream"]) == 0
        assert capsys.readouterr().out.strip() == "Arrives Monday."

        assert main(["a2a", "send", running.url, "Hello?"]) == 1
        assert "not authorised" in capsys.readouterr().err
