"""Audit trail (immutable who-did-what) and service health (is the platform ok)."""

from __future__ import annotations

from agent_harness import (
    Agent,
    AuditTrail,
    FakeProvider,
    Harness,
    PolicyGate,
    ServiceHealth,
    tool,
    tool_call,
)
from agent_harness.errors import ProviderError

MODEL = "claude-sonnet-5"


@tool
def read_file(path: str) -> str:
    """Read a file.

    Args:
        path: which file
    """
    return f"contents of {path}"


@tool
def wire_money(amount: float) -> str:
    """Send money.

    Args:
        amount: how much
    """
    return f"sent {amount}"


# --- audit trail ---------------------------------------------------------------

def test_the_chain_links_every_entry_to_the_one_before():
    trail = AuditTrail()
    first = trail.record("agent-a", "tool_call", target="read_file", decision="allow")
    second = trail.record("agent-a", "tool_call", target="wire_money", decision="deny")

    assert first.seq == 1 and second.seq == 2
    assert first.prev_hash == "0" * 64
    assert second.prev_hash == first.hash
    ok, reason = trail.verify()
    assert ok, reason


def test_editing_a_record_breaks_the_chain_and_says_where():
    trail = AuditTrail()
    for i in range(5):
        trail.record("agent", "tool_call", target=f"tool{i}", decision="allow")
    assert trail.verify()[0]

    trail.entries[2].decision = "deny"          # someone edits the record
    ok, reason = trail.verify()
    assert not ok
    assert "entry 3" in reason


def test_deleting_a_record_breaks_the_chain():
    trail = AuditTrail()
    for i in range(4):
        trail.record("agent", "run_start", target=f"task{i}")
    del trail.entries[1]
    ok, reason = trail.verify()
    assert not ok
    assert "sequence" in reason or "follow" in reason


def test_secrets_are_kept_out_of_the_record():
    trail = AuditTrail()
    entry = trail.record("agent", "tool_call", target="api",
                         api_key="sk-ant-do-not-store-me",
                         authorization="Bearer abc", note="fine")
    assert entry.detail["api_key"] == "[redacted]"
    assert entry.detail["authorization"] == "[redacted]"
    assert entry.detail["note"] == "fine"


def test_long_values_are_trimmed_rather_than_stored_whole():
    trail = AuditTrail()
    entry = trail.record("agent", "tool_call", target="x", blob="y" * 5000)
    assert len(entry.detail["blob"]) < 600
    assert "5000 chars" in entry.detail["blob"]


def test_the_trail_survives_a_restart_and_still_verifies(tmp_path):
    path = tmp_path / "audit.jsonl"
    trail = AuditTrail(path)
    trail.record("agent", "run_start", target="job", run_id="r1")
    trail.record("agent", "run_end", target="job", run_id="r1", decision="ok")

    reopened = AuditTrail.load(path)
    assert len(reopened) == 2
    assert reopened.verify()[0]
    assert [e.action for e in reopened.for_run("r1")] == ["run_start", "run_end"]

    # Appending to a reloaded trail keeps the chain intact.
    reopened.record("agent", "run_start", target="job2")
    assert reopened.verify()[0]


async def test_a_run_writes_start_end_and_every_permission_decision():
    provider = FakeProvider([tool_call("read_file", path="/tmp/a"),
                             tool_call("wire_money", amount=100.0),
                             "I was not allowed to send money."])
    harness = Harness.testing(provider)
    harness.policy = PolicyGate("allow", deny=["wire_money"])
    agent = Agent("clerk", provider=provider, model=MODEL, harness=harness,
                  tools=[read_file, wire_money], memory=False)

    result = await agent.run("read the file then wire 100")

    actions = [(e.action, e.target, e.decision) for e in harness.audit.entries]
    assert ("run_start", "read the file then wire 100", "ok") in actions
    assert ("tool_call", "read_file", "allow") in actions
    assert ("tool_call", "wire_money", "deny") in actions
    assert any(a == "run_end" for a, _, _ in actions)

    assert harness.audit.verify()[0]
    assert [e.target for e in harness.audit.denials()] == ["wire_money"]
    assert harness.audit.for_run(result.run_id)


def test_the_render_is_readable():
    trail = AuditTrail()
    trail.record("manager", "tool_call", target="delegate", decision="allow")
    line = trail.render()
    assert "manager" in line and "delegate" in line and "allow" in line


# --- service health ------------------------------------------------------------

def test_health_tracks_latency_and_failure_rate():
    health = ServiceHealth()
    for _ in range(8):
        health.record("claude-sonnet-5", 100.0, kind="model")
    health.record("claude-sonnet-5", 900.0, kind="model", ok=False, error="timeout")
    health.record("claude-sonnet-5", 120.0, kind="model")

    entry = health.component("claude-sonnet-5", "model")
    assert entry.calls == 10 and entry.failures == 1
    assert entry.failure_rate == 0.1
    assert entry.p50 <= entry.p95
    assert entry.last_error == "timeout"


def test_status_moves_from_healthy_to_degraded_to_unhealthy():
    health = ServiceHealth()
    assert health.component("nothing-yet", "tool").status() == "unknown"

    for _ in range(10):
        health.record("flaky", 10.0, kind="tool")
    assert health.component("flaky", "tool").status() == "healthy"

    for _ in range(2):
        health.record("flaky", 10.0, kind="tool", ok=False, error="nope")
    assert health.component("flaky", "tool").status() == "degraded"

    for _ in range(12):
        health.record("flaky", 10.0, kind="tool", ok=False, error="nope")
    assert health.component("flaky", "tool").status() == "unhealthy"
    assert health.status() == "unhealthy"      # the worst component sets the tone


def test_saturation_reads_the_scheduler():
    from agent_harness import ConcurrencyScheduler

    health = ServiceHealth()
    scheduler = ConcurrencyScheduler(max_concurrency=4)
    assert health.saturation(scheduler) == 0.0
    scheduler.running = 4
    assert health.saturation(scheduler) == 1.0
    assert health.snapshot(scheduler)["saturated"] is True


def test_inflight_is_tracked_across_begin_and_end():
    health = ServiceHealth()
    health.begin("slow-tool", "tool")
    assert health.component("slow-tool", "tool").inflight == 1
    health.end("slow-tool", 50.0, kind="tool")
    assert health.component("slow-tool", "tool").inflight == 0
    assert health.component("slow-tool", "tool").calls == 1


async def test_a_run_records_the_health_of_the_model_and_every_tool():
    provider = FakeProvider([tool_call("read_file", path="/x"), "done"])
    harness = Harness.testing(provider)
    agent = Agent("worker", provider=provider, model=MODEL, harness=harness,
                  tools=[read_file], memory=False)
    await agent.run("read it")

    snapshot = harness.health.snapshot(harness.scheduler)
    names = {(c["kind"], c["name"]) for c in snapshot["components"]}
    assert ("model", MODEL) in names
    assert ("tool", "read_file") in names
    assert snapshot["status"] == "healthy"


async def test_a_failing_provider_shows_up_as_unhealthy():
    class Broken(FakeProvider):
        async def complete(self, req):
            raise ProviderError("upstream is down", provider="fake")

    provider = Broken()
    harness = Harness.testing(provider)
    agent = Agent("worker", provider=provider, model=MODEL, harness=harness,
                  memory=False)
    result = await agent.run("try")

    assert not result.ok and "upstream is down" in result.error
    entry = harness.health.component(MODEL, "model")
    assert entry.failures == 1 and entry.failure_rate == 1.0
    assert harness.health.status() == "unhealthy"
    assert "upstream is down" in entry.last_error


async def test_a_failing_tool_is_attributed_to_the_tool_not_the_model():
    @tool
    def broken() -> str:
        """Always fails."""
        raise RuntimeError("the vendor is down")

    provider = FakeProvider([tool_call("broken"), "I could not."])
    harness = Harness.testing(provider)
    agent = Agent("worker", provider=provider, model=MODEL, harness=harness,
                  tools=[broken], memory=False)
    await agent.run("try it")

    assert harness.health.component("broken", "tool").failure_rate == 1.0
    assert harness.health.component(MODEL, "model").failure_rate == 0.0
