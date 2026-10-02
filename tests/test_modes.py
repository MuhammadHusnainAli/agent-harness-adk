"""Modes: chat, research and cowork — and the depth each one works at."""

from __future__ import annotations

import pytest

from agent_harness import (
    Agent,
    AgentGuardrails,
    Blueprint,
    Budget,
    ConfigurationError,
    FakeProvider,
    Harness,
    Message,
    Mode,
    SubAgentSpec,
    Workspace,
    modes,
    tool,
    tool_call,
)
from agent_harness.context import close_open_tool_calls
from agent_harness.modes import Notebook, citations, resolve_mode
from agent_harness.types import ToolResultBlock, ToolUseBlock

PAGES = {
    "https://example.com/q3": "Q3 revenue was 4.2M, up 12% on the year.",
    "https://example.com/costs": "Q3 costs were 3.1M.",
    "https://example.com/outlook": "Guidance for Q4 is flat.",
}


@tool
def fetch(url: str) -> str:
    """Fetch a page.

    Args:
        url: the page to read.
    """
    return PAGES.get(url, "not found")


def harness_with(*script, **kw) -> tuple[Harness, FakeProvider]:
    provider = FakeProvider(list(script), **kw)
    return Harness.testing(provider), provider


def todos(*items: tuple[str, str], **notes: str):
    return tool_call("todo_write", todos=[
        {"content": c, "status": s, **({"note": notes[c]} if c in notes else {})}
        for c, s in items])


# --- resolving a mode ---------------------------------------------------------

def test_an_agent_without_a_mode_is_the_plain_loop(harness):
    agent = Agent("plain", harness=harness)
    assert agent.mode is None and agent.depth == ""
    assert agent.max_steps == 20 and agent.workspace is None
    assert not agent.runtime_agents and not agent.conversational
    assert "todo_write" not in agent.tools


@pytest.mark.parametrize("name", ["chat", "research", "cowork"])
def test_every_mode_resolves_at_every_depth(name):
    steps = [resolve_mode(name, depth).max_steps for depth in modes.DEPTHS]
    assert steps == sorted(steps) and len(set(steps)) == 3
    assert resolve_mode(name).depth == "balanced"


def test_depth_names_a_tier_only_when_it_is_asked_for(harness):
    assert Agent("a", mode="chat", harness=harness).model == harness.router.default
    assert (Agent("a", mode="chat", depth="deep", harness=harness).model
            == harness.router.tiers["deep"])
    assert (Agent("a", mode="chat", depth="fast", harness=harness).model
            == harness.router.tiers["fast"])
    # An explicit model is kept; the depth still sets how hard it thinks.
    pinned = Agent("a", mode="chat", depth="deep", model="gpt-4.1", harness=harness)
    assert pinned.model == "gpt-4.1" and pinned.effort == "high"


def test_depth_and_mode_have_the_names_people_actually_use(harness):
    assert resolve_mode("deep-research", "slow").depth == "deep"
    assert resolve_mode("Deep_Research", "quick").name == "research"
    assert resolve_mode({"name": "research", "depth": "fast", "min_sources": 5}
                        ).min_sources == 5


def test_what_you_say_explicitly_beats_the_mode(harness):
    agent = Agent("a", mode="cowork", depth="deep", max_steps=7, workspace=False,
                  runtime_agents=False, tier="fast", harness=harness)
    assert agent.max_steps == 7 and agent.workspace is None
    assert not agent.runtime_agents and "spawn_agent" not in agent.tools
    assert agent.model == harness.router.tiers["fast"]


@pytest.mark.parametrize("kwargs, needle", [
    ({"mode": "telepathy"}, "unknown mode"),
    ({"mode": "chat", "depth": "ludicrous"}, "depth must be one of"),
    ({"depth": "deep"}, "needs a mode"),
    ({"mode": modes.research("fast"), "depth": "deep"}, "pass the depth to the mode"),
    ({"mode": {"name": "research", "nonsense": 1}}, "research mode"),
    ({"mode": 3}, "mode must be"),
])
def test_a_bad_mode_fails_when_the_agent_is_built(harness, kwargs, needle):
    with pytest.raises(ConfigurationError, match=needle):
        Agent("a", harness=harness, **kwargs)


def test_the_mode_tells_the_model_how_to_work_only_about_tools_it_has(harness):
    cowork = Agent("c", "Be brief.", mode="cowork", harness=harness)
    assert "`todo_write`" in cowork.assembler.mode
    assert "You cannot ask questions" in cowork.assembler.mode
    assert "`spawn_agent`" in cowork.assembler.mode

    alone = Agent("c", mode=modes.cowork("fast", ask=lambda q, o: "yes"),
                  workspace=False, harness=harness)
    assert "You cannot ask questions" not in alone.assembler.mode
    assert "`ask_user` is for" in alone.assembler.mode
    assert "fs_write" not in alone.assembler.mode
    assert "spawn_agent" not in alone.assembler.mode


async def test_the_mode_section_sits_before_your_own_instructions(harness):
    agent = Agent("c", "Always answer in French.", mode="chat", harness=harness)
    system = await agent.assembler.build()
    assert system.index("## How you work") < system.index("Always answer in French.")


# --- chat ------------------------------------------------------------------------

async def test_chat_keeps_the_thread_between_runs():
    harness, provider = harness_with("Hello Ada.", "You are Ada.")
    agent = Agent("pal", mode="chat", harness=harness)

    first = await agent.run("My name is Ada.")
    second = await agent.run("What is my name?")

    assert second.session_id == first.session_id
    assert [m.text for m in provider.requests[1].messages] == [
        "My name is Ada.", "Hello Ada.", "What is my name?"]
    assert second.mode == "chat" and second.depth == "balanced"


async def test_a_plain_agent_still_starts_every_run_clean():
    harness, provider = harness_with("one", "two")
    agent = Agent("plain", harness=harness)
    await agent.run("first")
    await agent.run("second")
    assert [m.text for m in provider.requests[1].messages] == ["second"]


async def test_new_session_drops_the_thread_and_takes_the_trace_with_it():
    harness, provider = harness_with("a", "b")
    agent = Agent("pal", mode="chat", harness=harness)
    first = await agent.run("remember 42")
    old_trace = agent.trace.session_id

    agent.new_session()
    second = await agent.run("what number?")

    assert second.session_id != first.session_id
    assert agent.trace.session_id != old_trace
    assert [m.text for m in provider.requests[1].messages] == ["what number?"]
    # The first conversation is still in the store, untouched.
    kept = await harness.sessions.load(first.session_id)
    assert kept.messages[0].text == "remember 42"


async def test_an_explicit_session_becomes_the_thread():
    harness, provider = harness_with("a", "b", "c")
    agent = Agent("pal", mode="chat", harness=harness)
    first = await agent.run("one")
    agent.new_session()
    await agent.run("two")
    await agent.run("three", session=first.session_id)
    assert [m.text for m in provider.requests[2].messages] == ["one", "a", "three"]


async def test_a_one_off_run_does_not_touch_the_conversation():
    harness, provider = harness_with("hi", "tool answer", "still here")
    agent = Agent("pal", mode="chat", harness=harness)
    await agent.run("hello")
    await agent.as_tool().invoke({"task": "a side job"})
    await agent.run("are you there?")
    assert [m.text for m in provider.requests[2].messages] == [
        "hello", "hi", "are you there?"]


async def test_a_conversation_survives_a_run_that_was_cut_off_mid_step():
    """A budget stop leaves a tool call with no result; the next turn must
    still be a conversation a provider would accept."""
    harness, provider = harness_with(tool_call("fetch", url="https://example.com/q3"),
                                     "picked back up")
    agent = Agent("pal", mode="chat", tools=[fetch], harness=harness,
                  budget=Budget(max_tool_calls=0))
    stopped = await agent.run("read the page")
    assert stopped.stop_reason == "budget"

    agent.budget = None
    resumed = await agent.run("carry on")

    assert resumed.output == "picked back up"
    sent = provider.requests[1].messages
    asked = [b.id for m in sent for b in m.content if isinstance(b, ToolUseBlock)]
    answered = [b.tool_use_id for m in sent for b in m.content
                if isinstance(b, ToolResultBlock)]
    assert asked and asked == answered


def test_open_tool_calls_are_closed_and_finished_ones_left_alone():
    call = ToolUseBlock(name="fetch", input={})
    done = [Message.user("go"), Message(role="assistant", content=[call]),
            Message.tool_results([ToolResultBlock(tool_use_id=call.id, content="ok")])]
    assert close_open_tool_calls(list(done)) == done

    cut = close_open_tool_calls(done[:2])
    assert len(cut) == 3
    block = cut[2].content[0]
    assert block.tool_use_id == call.id and block.is_error


# --- research ----------------------------------------------------------------------

def research_script(report: str):
    return [
        todos(("What was Q3 revenue?", "in_progress"), ("What were Q3 costs?", "pending")),
        [tool_call("fetch", url="https://example.com/q3"),
         tool_call("fetch", url="https://example.com/costs")],
        [tool_call("record_source", ref="https://example.com/q3",
                   finding="Q3 revenue 4.2M, up 12%", title="Q3 results"),
         tool_call("record_source", ref="https://example.com/costs",
                   finding="Q3 costs 3.1M")],
        todos(("What was Q3 revenue?", "done"), ("What were Q3 costs?", "done")),
        report,
    ]


async def test_research_plans_reads_records_and_cites():
    harness, _ = harness_with(*research_script(
        "Revenue was 4.2M [1] against costs of 3.1M [2]."))
    agent = Agent("analyst", mode="research", depth="fast", tools=[fetch],
                  harness=harness)

    result = await agent.run("How did Q3 go?")

    assert result.ok and result.violations == []
    assert result.mode == "research" and result.depth == "fast"
    assert [(s.id, s.ref) for s in result.sources] == [
        (1, "https://example.com/q3"), (2, "https://example.com/costs")]
    assert all(t.status == "done" for t in result.todos)
    # The source list is appended from the ledger, not written by the model.
    assert result.output.endswith(
        "## Sources\n- [1] Q3 results — https://example.com/q3\n"
        "- [2] https://example.com/costs")
    report = next(a for a in result.artifacts if a.name == "report.md")
    assert report.content == result.output
    assert "report.md" in harness.deliverables


async def test_a_citation_to_nothing_is_sent_back():
    harness, provider = harness_with(
        tool_call("fetch", url="https://example.com/q3"),
        tool_call("record_source", ref="https://example.com/q3", finding="4.2M"),
        "Revenue was 4.2M [1] and margins doubled [7].",
        "Revenue was 4.2M [1].",
    )
    agent = Agent("analyst", mode=modes.research("fast", min_sources=1),
                  tools=[fetch], harness=harness)

    result = await agent.run("How did Q3 go?")

    assert result.ok and "[7]" not in result.output and result.violations == []
    pushback = provider.requests[3].messages[-1].text
    assert "you cite [7]" in pushback and "Put it right" in pushback


async def test_a_source_nobody_read_cannot_be_recorded():
    harness, provider = harness_with(
        tool_call("record_source", ref="https://made-up.example/paper",
                  finding="a convenient number"),
        "I could not find a source.",
    )
    agent = Agent("analyst", mode=modes.research("fast", min_sources=0),
                  tools=[fetch], harness=harness)

    result = await agent.run("How did Q3 go?")

    assert result.sources == []
    refusal = provider.requests[1].messages[-1].content[0]
    assert refusal.is_error and "nothing you have read" in refusal.content


async def test_verification_can_be_turned_off():
    harness, _ = harness_with(
        tool_call("record_source", ref="the 2019 annual report", finding="x"),
        "As reported [1].",
    )
    agent = Agent("analyst", tools=[fetch], harness=harness,
                  mode=modes.research("fast", min_sources=1, verify_sources=False))
    result = await agent.run("q")
    assert [s.ref for s in result.sources] == ["the 2019 annual report"]


async def test_a_cached_read_still_counts_as_evidence():
    @tool(cacheable=True)
    def cached_fetch(url: str) -> str:
        """Fetch a page.

        Args:
            url: the page.
        """
        return PAGES[url]

    harness, _ = harness_with(
        tool_call("cached_fetch", url="https://example.com/q3"), "warm",
        tool_call("cached_fetch", url="https://example.com/q3"),
        tool_call("record_source", ref="https://example.com/q3", finding="4.2M"),
        "It was 4.2M [1].",
    )
    warm = Agent("warm", tools=[cached_fetch], harness=harness)
    await warm.run("warm the cache")
    agent = Agent("analyst", mode=modes.research("fast", min_sources=1),
                  tools=[cached_fetch], harness=harness)
    result = await agent.run("q")
    assert len(result.sources) == 1 and result.violations == []


async def test_recording_a_source_twice_gives_it_one_number():
    harness, _ = harness_with(
        tool_call("fetch", url="https://example.com/q3"),
        tool_call("record_source", ref="https://example.com/q3", finding="4.2M"),
        tool_call("record_source", ref="HTTPS://example.com/q3/", finding="up 12%"),
        "4.2M, up 12% [1].",
    )
    agent = Agent("analyst", mode=modes.research("fast", min_sources=1),
                  tools=[fetch], harness=harness)
    result = await agent.run("q")
    assert len(result.sources) == 1
    assert result.sources[0].finding == "4.2M\nup 12%"


async def test_too_few_sources_is_retried_then_delivered_with_what_is_missing():
    harness, provider = harness_with("Probably fine.", "Still fine.", "Fine, I say.")
    agent = Agent("analyst", mode="research", depth="fast", tools=[fetch],
                  harness=harness)

    result = await agent.run("How did Q3 go?")

    # Not an error: work that fell short is delivered, and says so.
    assert result.ok and result.output == "Fine, I say."
    assert result.steps == 3
    assert result.violations and "0 of the 2 sources" in result.violations[0]
    assert any(a.name == "report.md" for a in result.artifacts)
    assert any(e.action == "mode" for e in harness.audit.entries)


async def test_research_with_nothing_to_read_fails_before_it_costs_anything():
    harness, provider = harness_with("an answer from thin air")
    agent = Agent("analyst", mode="research", harness=harness)

    result = await agent.run("How did Q3 go?")

    assert "nothing to read with" in (result.error or "")
    assert provider.requests == []


async def test_a_tool_attached_after_construction_is_enough(harness):
    agent = Agent("analyst", mode="research", depth="fast", harness=harness)
    agent.add_tool(fetch)
    agent._check_mode()


async def test_a_report_that_writes_its_own_source_list_is_left_alone():
    script = research_script("Revenue 4.2M [1], costs 3.1M [2].\n\n## Sources\n"
                             "[1] Q3 results\n[2] Costs")
    harness, _ = harness_with(*script)
    agent = Agent("analyst", mode="research", depth="fast", tools=[fetch],
                  harness=harness)
    result = await agent.run("q")
    assert result.output.count("## Sources") == 1


async def test_a_helper_writes_in_the_leads_ledger():
    def script(request):
        last = request.messages[-1]
        text = last.text
        results = [b for b in last.content if isinstance(b, ToolResultBlock)]
        if "Research Q3 revenue" in text:
            return tool_call("fetch", url="https://example.com/q3")
        if results and "4.2M" in results[0].content:
            return tool_call("record_source", ref="https://example.com/q3",
                             finding="Q3 revenue 4.2M")
        if results and "Recorded as [2]" in results[0].content:
            return "Revenue was 4.2M [2]."
        if results and "Revenue was 4.2M [2]" in results[0].content:
            return "Costs were 3.1M [1] and revenue 4.2M [2]."
        if results and "Recorded as [1]" in results[0].content:
            return tool_call("delegate", agent_name="reader",
                             task="Research Q3 revenue")
        if results:
            return tool_call("record_source", ref="https://example.com/costs",
                             finding="Q3 costs 3.1M")
        return tool_call("fetch", url="https://example.com/costs")

    harness = Harness.testing(FakeProvider([script], loop=True))
    lead = Agent("lead", mode=modes.research("fast", helpers=0), tools=[fetch],
                 subagents=[SubAgentSpec(name="reader", description="reads pages")],
                 harness=harness)

    result = await lead.run("How did Q3 go?")

    assert result.violations == []
    assert [(s.id, s.agent) for s in result.sources] == [(1, "lead"), (2, "reader")]
    helper = lead.subagents["reader"]
    assert "record_source" in helper.tools and "todo_write" not in helper.tools
    assert "## Sources" in result.children[0].messages[0].text


def test_citations_are_read_the_way_people_write_them():
    assert citations("A [1], B [2, 3], C [4-6] and D [7][8].") == set(range(1, 9))
    assert citations("rows[0] and `x`\n```\ndata[3]\n```\nbut see [2]") == {2}
    assert citations("nothing here") == set()


def test_the_notebook_checks_a_reference_against_what_was_read():
    notebook = Notebook(tracking=True)
    notebook.saw("fetch", '{"url": "https://www.example.com/a/"}', "The Annual Report")
    assert notebook.mentions("https://www.example.com/a")
    assert notebook.mentions("example.com/a")
    assert notebook.mentions("the annual report")
    assert notebook.mentions("fetch")
    assert not notebook.mentions("https://other.example/b")
    assert not notebook.mentions("a")
    # A helper shares the ledger and the evidence, not the todo list.
    child = notebook.child()
    child.record("https://www.example.com/a", "x")
    assert notebook.sources and child.todos is not notebook.todos


# --- cowork --------------------------------------------------------------------------

async def test_cowork_plans_works_in_the_workspace_and_hands_back_files(tmp_path):
    harness, _ = harness_with(
        todos(("Draft the summary", "in_progress"), ("Check it", "pending")),
        tool_call("fs_write", path="out/summary.md", content="# Q3\nRevenue 4.2M."),
        todos(("Draft the summary", "done"), ("Check it", "done")),
        "Done. The summary is in out/summary.md.",
    )
    workspace = Workspace(tmp_path)
    workspace.write("brief.txt", "already here")
    agent = Agent("colleague", mode="cowork", depth="fast", workspace=workspace,
                  harness=harness)

    result = await agent.run("Write up Q3.")

    assert result.ok and result.violations == []
    assert [a.name for a in result.artifacts] == ["out/summary.md"]
    produced = result.artifacts[0]
    assert produced.content == "# Q3\nRevenue 4.2M."
    assert produced.media_type == "text/markdown"
    assert produced.path == str(tmp_path / "out" / "summary.md")
    assert [t.status for t in result.todos] == ["done", "done"]
    assert "parse_document" in agent.tools and "run_python" not in agent.tools


async def test_cowork_may_not_finish_with_items_still_open():
    harness, provider = harness_with(
        todos(("Draft it", "done"), ("Check it", "pending")),
        "All done!",
        todos(("Draft it", "done"), ("Check it", "skipped"),
              **{"Check it": "nothing to run it against"}),
        "Drafted. I skipped the check: nothing to run it against.",
    )
    agent = Agent("colleague", mode="cowork", depth="fast", harness=harness)

    result = await agent.run("Draft the note.")

    assert result.ok and result.violations == []
    assert result.output.startswith("Drafted.")
    assert "still open: Check it" in provider.requests[2].messages[-1].text
    assert result.todos[1].note == "nothing to run it against"


async def test_a_todo_list_is_validated_where_it_is_written():
    harness, provider = harness_with(
        tool_call("todo_write", todos=[{"content": "x", "status": "skipped"}]),
        tool_call("todo_write", todos=[{"content": "", "status": "pending"}]),
        tool_call("todo_write", todos=[{"content": "x", "status": "someday"}]),
        tool_call("todo_write", todos=[{"content": "x", "status": "completed"}]),
        "done",
    )
    agent = Agent("colleague", mode="cowork", depth="fast", harness=harness)
    result = await agent.run("go")
    replies = [r.messages[-1].content[0].content for r in provider.requests[1:5]]
    assert "skipped without a `note`" in replies[0]
    assert "has no `content`" in replies[1]
    assert "status must be" in replies[2]
    assert replies[3].startswith("1 items · 1 done")
    assert [t.status for t in result.todos] == ["done"]


async def test_cowork_can_ask_the_person_it_works_for():
    asked: list[tuple[str, list[str]]] = []

    async def answer(question: str, options: list[str]) -> str:
        asked.append((question, options))
        return "PDF"

    harness, provider = harness_with(
        tool_call("ask_user", question="Which format?", options=["PDF", "Word"]),
        tool_call("ask_user", question="Really?"),
        "Went with PDF.",
    )
    agent = Agent("colleague", harness=harness,
                  mode=modes.cowork("fast", ask=answer, max_questions=1))

    result = await agent.run("Export the report.")

    assert asked == [("Which format?", ["PDF", "Word"])]
    assert provider.requests[1].messages[-1].content[0].content == "PDF"
    assert "limit for one run" in provider.requests[2].messages[-1].content[0].content
    assert result.output == "Went with PDF."


def test_without_someone_to_ask_there_is_no_ask_tool(harness):
    agent = Agent("colleague", mode="cowork", harness=harness)
    assert "ask_user" not in agent.tools


async def test_cowork_keeps_the_thread_and_only_hands_back_what_each_run_changed(
        tmp_path):
    harness, provider = harness_with(
        tool_call("fs_write", path="a.txt", content="one"), "wrote a",
        tool_call("fs_write", path="b.txt", content="two"), "wrote b",
    )
    agent = Agent("colleague", mode="cowork", depth="fast",
                  workspace=Workspace(tmp_path), harness=harness, memory=False)

    first = await agent.run("write a")
    second = await agent.run("now b")

    assert [a.name for a in first.artifacts] == ["a.txt"]
    assert [a.name for a in second.artifacts] == ["b.txt"]
    assert provider.requests[2].messages[0].text == "write a"


async def test_files_are_handed_back_even_when_the_run_fails(tmp_path):
    harness, _ = harness_with(
        tool_call("fs_write", path="partial.md", content="half"),
        tool_call("fs_list"), tool_call("fs_list"),
    )
    agent = Agent("colleague", mode="cowork", workspace=Workspace(tmp_path),
                  max_steps=3, harness=harness)
    result = await agent.run("go")
    assert "MaxStepsExceeded" in (result.error or "")
    assert [a.name for a in result.artifacts] == ["partial.md"]


async def test_deep_cowork_can_spin_up_helpers_and_they_do_not_get_its_list(tmp_path):
    harness = Harness.testing(FakeProvider(["ok"], loop=True))
    agent = Agent("colleague", mode="cowork", depth="deep",
                  workspace=Workspace(tmp_path), harness=harness)
    assert agent.runtime_agents and agent.max_runtime_agents == 8
    assert agent.max_steps == 120
    child = agent.add_subagent(SubAgentSpec(name="helper", description="helps"))
    assert "fs_write" in child.tools
    assert "todo_write" not in child.tools and "ask_user" not in child.tools


async def test_a_file_that_is_not_text_is_handed_back_by_its_path(tmp_path):
    workspace = Workspace(tmp_path / "ws")
    harness = Harness.local(tmp_path / "state", trace=False,
                            provider=FakeProvider(["made the chart"]))

    @tool
    def draw() -> str:
        """Draw the chart."""
        (workspace.root / "chart.png").write_bytes(b"\x89PNG\xff\xfe\x00binary")
        return "drawn"

    harness.provider.responses.insert(0, tool_call("draw"))
    agent = Agent("colleague", mode="cowork", depth="fast", workspace=workspace,
                  tools=[draw], harness=harness)

    result = await agent.run("chart it")

    chart = result.artifacts[0]
    assert (chart.name, chart.content, chart.media_type) == ("chart.png", "", "image/png")
    assert chart.path == str(workspace.root / "chart.png")
    # The deliverable store keeps the bytes, not an empty file.
    kept = harness.deliverables.get("chart.png")
    assert kept.path != chart.path
    assert open(kept.path, "rb").read() == b"\x89PNG\xff\xfe\x00binary"


def test_python_is_opt_in_and_asks_for_approval(tmp_path, harness):
    agent = Agent("colleague", mode=modes.cowork("fast", python=True, documents=False),
                  workspace=Workspace(tmp_path), harness=harness)
    assert agent.tools.get("run_python").permission == "ask"
    assert "parse_document" not in agent.tools


def test_a_workspace_knows_what_changed(tmp_path):
    workspace = Workspace(tmp_path)
    workspace.write("keep.txt", "same")
    workspace.write("edit.txt", "before")
    before = workspace.snapshot()
    workspace.write("edit.txt", "after, and longer")
    workspace.write("new/file.txt", "hi")
    workspace.write(".hidden/secret", "no")
    workspace.write("_snippet.py", "scratch")
    assert workspace.changed(before) == ["edit.txt", "new/file.txt"]


# --- the last step, compaction, streaming -------------------------------------------

async def test_the_last_step_is_for_handing_over():
    harness, provider = harness_with(
        tool_call("fetch", url="https://example.com/q3"),
        tool_call("fetch", url="https://example.com/costs"),
        "Here is what I have.",
    )
    agent = Agent("pal", mode="chat", tools=[fetch], max_steps=3, harness=harness)

    result = await agent.run("dig")

    assert result.ok and result.output == "Here is what I have."
    assert provider.requests[1].tool_choice is None
    assert "last step" not in provider.requests[1].system
    assert provider.requests[2].tool_choice == "none"
    assert provider.requests[2].system.endswith(agent.mode.wrap_up)


async def test_a_plain_agents_last_step_is_an_ordinary_step():
    harness, provider = harness_with(tool_call("fetch", url="x"), "done")
    agent = Agent("plain", tools=[fetch], max_steps=2, harness=harness)
    await agent.run("go")
    assert provider.requests[1].tool_choice is None


async def test_the_todo_list_and_the_ledger_survive_compaction():
    steps = iter([
        todos(("Read the long page", "in_progress")),
        tool_call("fetch", url="https://example.com/q3"),
        tool_call("record_source", ref="https://example.com/q3", finding="4.2M"),
        todos(("Read the long page", "done")),
        "It was 4.2M [1].",
    ])

    def script(request):
        # The compactor's own summarising call shares the provider.
        if request.messages[-1].text.startswith("Summarise this conversation"):
            return "an earlier exchange"
        return next(steps)

    provider = FakeProvider([script], loop=True)
    agent = Agent("analyst", mode=modes.research("fast", min_sources=1),
                  tools=[fetch], harness=Harness.testing(provider),
                  compact_at=60, compact_keep_last=1)

    result = await agent.run("q " + "padding " * 200)

    assert result.violations == []
    compacted = [r.messages[0].text for r in provider.requests
                 if "compacted" in r.messages[0].text]
    assert compacted and "Your todo list: [~] Read the long page" in compacted[0]
    assert any("Sources recorded so far: [1] https://example.com/q3" in text
               for text in compacted)


async def test_progress_is_streamed_as_the_list_changes():
    harness, _ = harness_with(
        todos(("One thing", "in_progress")),
        todos(("One thing", "done")),
        "done",
    )
    agent = Agent("colleague", mode="cowork", depth="fast", harness=harness)

    events = [e async for e in agent.stream("go")]

    progress = [e for e in events if e.type == "progress"]
    assert [e.text for e in progress] == ["1 items · 1 in progress", "1 items · 1 done"]
    assert progress[1].data["todos"] == [
        {"content": "One thing", "status": "done", "note": ""}]
    assert any(e.kind == "progress" for e in harness.journal.entries)


async def test_a_budget_stop_keeps_the_sources_with_the_partial_work():
    harness, _ = harness_with(
        tool_call("fetch", url="https://example.com/q3"),
        tool_call("record_source", ref="https://example.com/q3", finding="4.2M"),
        "So far: revenue 4.2M [1].",
    )
    agent = Agent("analyst", mode=modes.research("fast", min_sources=1),
                  tools=[fetch], harness=harness, budget=Budget(max_steps=2))
    result = await agent.run("q")
    assert result.stop_reason == "budget"
    assert [s.id for s in result.sources] == [1]


# --- alongside everything else ---------------------------------------------------------

async def test_your_own_guardrails_still_apply_on_top_of_the_mode():
    harness, provider = harness_with(
        tool_call("fetch", url="https://example.com/q3"),
        tool_call("record_source", ref="https://example.com/q3", finding="4.2M"),
        "It was 4.2M [1].",
        "Revenue was 4.2M [1].",
    )
    agent = Agent("analyst", mode=modes.research("fast", min_sources=1),
                  tools=[fetch], harness=harness,
                  guardrails=AgentGuardrails(must_include=["revenue"]))
    result = await agent.run("q")
    assert result.ok and result.output.startswith("Revenue was 4.2M [1].")
    assert result.steps == 4


async def test_a_structured_answer_is_not_given_a_source_list():
    from pydantic import BaseModel

    class Finding(BaseModel):
        revenue: str

    harness, _ = harness_with(
        tool_call("fetch", url="https://example.com/q3"),
        tool_call("record_source", ref="https://example.com/q3", finding="4.2M"),
        '{"revenue": "4.2M [1]"}',
    )
    agent = Agent("analyst", mode=modes.research("fast", min_sources=1),
                  tools=[fetch], harness=harness, output_type=Finding)
    result = await agent.run("q")
    assert result.data.revenue == "4.2M [1]" and "## Sources" not in result.output


async def test_a_version_can_change_the_mode():
    harness, _ = harness_with("hi", loop=True)
    agent = Agent("support", tools=[fetch], harness=harness, version="v1", versions={
        "v1": {"mode": "chat"},
        "v2": {"mode": "research", "depth": "deep"},
        "v3": {"depth": "fast"},
    })
    assert agent.mode.name == "chat" and agent.max_steps == 12
    deep = agent.use("v2")
    assert (deep.mode.name, deep.depth, deep.max_steps) == ("research", "deep", 60)
    assert "record_source" in deep.tools and "record_source" not in agent.tools

    based = Agent("support", mode="cowork", harness=harness, version="v3",
                  versions={"v3": {"depth": "fast"}})
    assert (based.mode.name, based.depth) == ("cowork", "fast")


def test_a_blueprint_declares_the_mode(harness):
    blueprint = Blueprint.from_text("""
defaults: {memory: false}
subagents:
  reader:
    description: Reads sources.
    mode: research
    depth: fast
agents:
  analyst:
    mode: research
    depth: deep
    tools: [fetch]
    subagents: [reader]
  helper:
    mode: {name: cowork, depth: fast, documents: false}
  plain:
    instructions: Just answer.
""")
    analyst = blueprint.build("analyst", tools=[fetch], harness=harness)
    assert (analyst.mode.name, analyst.depth, analyst.max_steps) == (
        "research", "deep", 60)
    reader = analyst.subagents["reader"]
    assert reader.mode.name == "research" and reader.max_steps == 12

    helper = blueprint.build("helper", harness=harness)
    assert helper.mode.name == "cowork" and helper.workspace is not None
    assert "parse_document" not in helper.tools
    assert blueprint.build("plain", harness=harness).mode is None


def test_a_sub_agent_spec_keeps_what_it_said_over_its_mode(harness):
    parent = Agent("lead", tools=[fetch], harness=harness)
    moded = parent.add_subagent(SubAgentSpec(name="a", mode="cowork", depth="fast"))
    assert moded.max_steps == 25 and moded.workspace is not None

    said = parent.add_subagent(
        SubAgentSpec(name="b", mode="cowork", max_steps=5, workspace="none"))
    assert said.max_steps == 5 and said.workspace is None

    plain = parent.add_subagent(SubAgentSpec(name="c"))
    assert plain.max_steps == 12 and plain.mode is None


def test_governance_inventory_records_the_mode():
    from agent_harness.governance import Governance

    governance = Governance()
    harness = Harness.testing(governance=governance)
    Agent("analyst", mode="research", depth="deep", tools=[fetch], harness=harness)
    entry = governance.inventory.agents["analyst"]
    assert (entry["mode"], entry["depth"]) == ("research", "deep")


def test_a_mode_of_your_own_can_be_registered(harness):
    def triage(depth=None):
        return Mode(name="triage", max_steps=3, todos=True,
                    opening="Sort it, do not solve it.")

    modes.register_mode("triage", triage)
    try:
        agent = Agent("desk", mode="triage", harness=harness)
        assert agent.max_steps == 3 and "todo_write" in agent.tools
        assert agent.assembler.mode == "Sort it, do not solve it."
    finally:
        modes.MODES.pop("triage")
    with pytest.raises(ConfigurationError):
        Mode(name="x", depth="ludicrous")


def test_the_cli_takes_a_mode_and_a_depth(tmp_path, capsys):
    from agent_harness.cli import build_parser

    args = build_parser().parse_args([
        "run", "write it up", "--mode", "cowork", "--depth", "deep",
        "--workspace", str(tmp_path)])
    assert (args.mode, args.depth, args.workspace) == ("cowork", "deep", str(tmp_path))
    assert args.max_steps is None
    with pytest.raises(SystemExit):
        build_parser().parse_args(["run", "x", "--mode", "telepathy"])
