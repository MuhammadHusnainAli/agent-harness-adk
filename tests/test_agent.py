from __future__ import annotations

import asyncio
import time

import pytest
from pydantic import BaseModel

from agent_harness import (
    Agent,
    Budget,
    FakeProvider,
    Harness,
    HookContext,
    HookEngine,
    Message,
    PolicyGate,
    tool,
    tool_call,
)
from agent_harness.runtime.checkpoints import Checkpointer

MODEL = "claude-sonnet-5"


@tool
def add(a: int, b: int) -> int:
    """Add two numbers.

    Args:
        a: the first number
        b: the second number
    """
    return a + b


@tool
def fail_tool() -> str:
    """Always fails."""
    raise RuntimeError("the vendor is down")


def build(script, **kw) -> Agent:
    provider = FakeProvider(script)
    harness = kw.pop("harness", None) or Harness.testing(provider)
    return Agent("tester", "You test things.", provider=provider, model=MODEL,
                 harness=harness, memory=kw.pop("memory", False), **kw)


async def test_a_plain_answer_comes_straight_back():
    agent = build(["The answer is 4."])
    result = await agent.run("2+2?")
    assert result.output == "The answer is 4."
    assert result.ok and result.steps == 1
    assert result.usage.calls == 1


async def test_the_loop_calls_a_tool_then_answers():
    agent = build([tool_call("add", a=2, b=3), "It is 5."], tools=[add])
    result = await agent.run("add 2 and 3")
    assert result.output == "It is 5."
    assert [c.name for c in result.tool_calls] == ["add"]
    # The tool result is in the transcript the model saw.
    assert any("5" in b.content for m in result.messages for b in m.content
               if getattr(b, "type", "") == "tool_result")


async def test_tool_failures_are_reported_to_the_model_not_raised():
    agent = build([tool_call("fail_tool"), "I could not reach the vendor."],
                  tools=[fail_tool])
    result = await agent.run("try it")
    assert result.ok
    assert "could not reach" in result.output


async def test_an_unknown_tool_is_answered_not_fatal():
    agent = build([tool_call("nonexistent"), "I do not have that tool."], tools=[add])
    result = await agent.run("do something")
    assert result.ok and "do not have" in result.output


async def test_parallel_tool_calls_run_together():
    order: list[str] = []

    @tool
    async def slow(label: str) -> str:
        """Wait a moment.

        Args:
            label: which call this is
        """
        await asyncio.sleep(0.02)
        order.append(label)
        return label

    from agent_harness.types import Message as M
    from agent_harness.types import ToolUseBlock

    both = M(role="assistant", content=[
        ToolUseBlock(id="a", name="slow", input={"label": "first"}),
        ToolUseBlock(id="b", name="slow", input={"label": "second"}),
    ])
    agent = build([both, "both done"], tools=[slow])
    started = time.perf_counter()
    result = await agent.run("run both")
    elapsed = time.perf_counter() - started
    assert result.output == "both done"
    assert len(order) == 2
    assert elapsed < 0.035  # they overlapped rather than queued


async def test_max_steps_stops_a_runaway_loop():
    provider = FakeProvider([tool_call("add", a=1, b=1)], loop=True)
    agent = Agent("looper", provider=provider, model=MODEL, tools=[add], max_steps=3,
                  harness=Harness.testing(provider), memory=False)
    result = await agent.run("go")
    assert not result.ok
    assert "MaxStepsExceeded" in result.error
    assert result.steps == 3


async def test_the_budget_ceiling_stops_the_run():
    harness = Harness.testing(FakeProvider(["x"], loop=True))
    harness.reset_budget(Budget(max_steps=2))
    agent = Agent("spender", provider=harness.provider, model=MODEL, harness=harness,
                  memory=False, tools=[add], max_steps=10)
    provider = harness.provider
    provider.responses = [tool_call("add", a=1, b=1)]
    provider.loop = True
    result = await agent.run("go")
    assert not result.ok and "BudgetExceeded" in result.error


async def test_output_contracts_are_validated_and_retried():
    class Ticket(BaseModel):
        id: str
        priority: int

    agent = build(["not json at all", '{"id": "T-1", "priority": 2}'],
                  output_type=Ticket)
    result = await agent.run("classify this")
    assert isinstance(result.data, Ticket)
    assert result.data.id == "T-1"
    assert result.steps == 2  # it was told to fix the first reply


async def test_an_unfixable_output_contract_fails_loudly():
    class Ticket(BaseModel):
        id: str

    agent = build(["nope", "still nope", "nope again"], output_type=Ticket,
                  contract_retries=1)
    result = await agent.run("classify this")
    assert not result.ok and "OutputContract" in result.error


async def test_policy_denies_a_tool_and_the_agent_carries_on():
    harness = Harness.testing(FakeProvider([tool_call("add", a=1, b=2),
                                            "I am not allowed to add."]))
    harness.policy = PolicyGate("allow", deny=["add"])
    agent = Agent("gated", provider=harness.provider, model=MODEL, harness=harness,
                  tools=[add], memory=False)
    result = await agent.run("add them")
    assert result.ok and "not allowed" in result.output
    denied = [e for e in harness.journal.entries if e.kind == "denied"]
    assert denied and "add" in denied[0].text


async def test_ask_without_an_approver_is_denied():
    harness = Harness.testing(FakeProvider([tool_call("add", a=1, b=2), "blocked"]))
    harness.policy = PolicyGate("allow", ask=["add"])
    agent = Agent("gated", provider=harness.provider, model=MODEL, harness=harness,
                  tools=[add], memory=False)
    result = await agent.run("add them")
    assert result.ok
    assert any("Not permitted" in b.content for m in result.messages for b in m.content
               if getattr(b, "type", "") == "tool_result")


async def test_an_approver_can_let_it_through():
    async def approve(tool_name, args, reason):
        return True

    harness = Harness.testing(FakeProvider([tool_call("add", a=1, b=2), "it is 3"]))
    harness.policy = PolicyGate("allow", ask=["add"], approver=approve)
    agent = Agent("gated", provider=harness.provider, model=MODEL, harness=harness,
                  tools=[add], memory=False)
    result = await agent.run("add them")
    assert result.output == "it is 3"


async def test_hooks_can_block_a_tool_and_rewrite_its_result():
    hooks = HookEngine()

    @hooks.on("pre_tool")
    def guard(ctx: HookContext) -> None:
        if ctx.data["args"].get("a", 0) > 100:
            ctx.block("that number is too big")

    @hooks.on("post_tool")
    def shout(ctx: HookContext) -> None:
        ctx.replace(str(ctx.data["outcome"].content).upper() + "!")

    agent = build([tool_call("add", a=1, b=2), "done"], tools=[add], hooks=hooks)
    result = await agent.run("add")
    contents = [b.content for m in result.messages for b in m.content
                if getattr(b, "type", "") == "tool_result"]
    assert contents == ["3!"]

    agent2 = build([tool_call("add", a=500, b=2), "done"], tools=[add], hooks=hooks)
    result2 = await agent2.run("add")
    blocked = [b.content for m in result2.messages for b in m.content
               if getattr(b, "type", "") == "tool_result"]
    assert "too big" in blocked[0]


async def test_guardrails_redact_secrets_leaving_the_agent():
    agent = build(["the key is sk-ant-abcdefghijklmnopqrstuv"])
    result = await agent.run("what is the key")
    assert "sk-ant-" not in result.output
    assert "[redacted]" in result.output


async def test_guardrails_block_secrets_coming_back_from_a_tool():
    @tool
    def leak() -> str:
        """Returns a private key."""
        return "-----BEGIN RSA PRIVATE KEY-----\nabc"

    agent = build([tool_call("leak"), "I cannot show that."], tools=[leak])
    result = await agent.run("show me")
    blocked = [b.content for m in result.messages for b in m.content
               if getattr(b, "type", "") == "tool_result"]
    assert "Blocked" in blocked[0]
    assert "BEGIN RSA" not in blocked[0]


async def test_streaming_emits_events_and_ends_with_the_result():
    agent = build([tool_call("add", a=1, b=1), "two"], tools=[add])
    kinds = []
    final = None
    async for event in agent.stream("add"):
        kinds.append(event.type)
        if event.type == "run_end":
            final = event.data["result"]
    assert kinds[0] == "run_start" and kinds[-1] == "run_end"
    assert "tool_result" in kinds
    assert "text" in kinds          # the answer arrives as it is generated
    assert final.output == "two"


async def test_streaming_passes_the_model_tokens_through():
    from agent_harness.providers.base import CompletionRequest, Provider
    from agent_harness.types import Message as M
    from agent_harness.types import ModelResponse, StreamEvent, Usage

    class Chunked(Provider):
        name = "chunked"
        BASE_URL = "http://none"

        async def complete(self, req):
            return ModelResponse(message=M.assistant("hello there"), usage=Usage(calls=1))

        async def stream(self, req: CompletionRequest):
            for piece in ("hel", "lo ", "there"):
                yield StreamEvent(type="text", text=piece)
            response = ModelResponse(message=M.assistant("hello there"),
                                     usage=Usage(calls=1))
            yield StreamEvent(type="step_end",
                              data={"response": response.model_dump(mode="json")})

    provider = Chunked(api_key="x")
    agent = Agent("streamer", provider=provider, model=MODEL,
                  harness=Harness.testing(provider), memory=False)
    pieces = [e.text async for e in agent.stream("hi") if e.type == "text"]
    assert pieces == ["hel", "lo ", "there"]


async def test_sessions_resume_and_fork():
    harness = Harness.testing(FakeProvider(["first", "second"], loop=True))
    agent = Agent("chat", provider=harness.provider, model=MODEL, harness=harness,
                  memory=False)
    first = await agent.run("hello")
    session_id = first.session_id

    resumed = await agent.run("and again", session=session_id)
    assert len(resumed.messages) > len(first.messages)

    forked = await harness.sessions.fork(session_id)
    assert forked.parent_id == session_id
    assert len(forked.messages) == len(resumed.messages)


async def test_checkpoints_capture_every_step(tmp_path):
    harness = Harness.testing(FakeProvider([tool_call("add", a=1, b=1), "two"]))
    harness.checkpoints = Checkpointer(tmp_path, every=1)
    agent = Agent("cp", provider=harness.provider, model=MODEL, harness=harness,
                  tools=[add], memory=False)
    result = await agent.run("add")
    history = await harness.checkpoints.history(result.run_id)
    assert [c.step for c in history] == [1, 2]
    restored = await harness.checkpoints.load(result.run_id, step=1)
    assert restored.messages[-1].tool_uses[0].name == "add"


def test_run_sync_from_plain_python():
    agent = build(["sync answer"])
    assert agent.run_sync("hi").output == "sync answer"


async def test_run_sync_refuses_inside_a_loop():
    agent = build(["x"])
    with pytest.raises(RuntimeError, match="event loop"):
        agent.run_sync("hi")


async def test_an_agent_can_be_used_as_a_tool():
    inner = build(["the inner answer"])
    entry = inner.as_tool(name="researcher", description="Looks things up")
    assert entry.name == "researcher"
    assert await entry.invoke({"task": "look it up"}) == "the inner answer"


async def test_memory_persists_across_runs_of_one_agent():
    provider = FakeProvider([
        tool_call("remember", fact="The user bills in EUR"), "noted",
        tool_call("recall", query="how does the user bill"), "You bill in EUR.",
    ])
    harness = Harness.testing(provider)
    agent = Agent("assistant", provider=provider, model=MODEL, harness=harness,
                  memory=True)
    await agent.run("I bill in euros")
    second = await agent.run("how do I bill?", messages=[])
    assert "EUR" in second.output
    assert "EUR" in await agent.memory.user.load()


async def test_the_system_prompt_carries_identity_instructions_and_tools():
    provider = FakeProvider(["ok"])
    agent = Agent("support", "Answer billing questions.", provider=provider,
                  model=MODEL, harness=Harness.testing(provider), tools=[add],
                  memory=False)
    await agent.run("hi")
    system = provider.requests[0].system
    assert "You are support" in system
    assert "Answer billing questions." in system
    assert "add" in system


async def test_input_guardrails_stop_a_run_before_the_model_is_called():
    from agent_harness import Guardrails
    from agent_harness.runtime.guardrails import Rule

    harness = Harness.testing(FakeProvider(["should not be reached"]))
    harness.guardrails = Guardrails([Rule("no_pii", r"\b\d{3}-\d{2}-\d{4}\b", "block")])
    agent = Agent("gated", provider=harness.provider, model=MODEL, harness=harness,
                  memory=False)
    result = await agent.run("my ssn is 123-45-6789")
    assert not result.ok and "GuardrailTripped" in result.error
    assert harness.provider.requests == []


async def test_messages_can_be_supplied_directly_for_a_clean_run():
    agent = build(["clean"])
    result = await agent.run("go", messages=[Message.user("earlier turn")])
    assert result.messages[0].text == "earlier turn"
