"""Spec compiler (blueprint → provider payload) and the deliverable store."""

from __future__ import annotations

from agent_harness import (
    Agent,
    Artifact,
    DeliverableStore,
    FakeProvider,
    Harness,
    ModelRouter,
    SkillRegistry,
    SpecCompiler,
    SubAgentSpec,
    ToolRegistry,
    tool,
)

MODEL = "claude-sonnet-5"


@tool
def lookup(topic: str) -> str:
    """Look something up.

    Args:
        topic: what to look up
    """
    return topic


@tool(tags=["write"])
def publish(text: str) -> str:
    """Publish something.

    Args:
        text: what to publish
    """
    return "published"


REGISTRY = ToolRegistry([lookup, publish])


# --- spec compiler -----------------------------------------------------------

def test_a_blueprint_becomes_a_resolved_payload():
    spec = SubAgentSpec(name="researcher", description="Finds things out.",
                        instructions="Cite every claim.", tools=["lookup"],
                        tier="fast", max_steps=6, max_tokens=2048)
    compiled = SpecCompiler().compile(spec, tools=REGISTRY)

    assert compiled.name == "researcher"
    assert compiled.model == "claude-haiku-4-5"       # the tier resolved to a model
    assert compiled.effort == "low"
    assert [t.name for t in compiled.tools] == ["lookup"]   # the allowlist held
    assert compiled.max_steps == 6 and compiled.max_tokens == 2048
    assert "You are researcher" in compiled.system
    assert "Cite every claim." in compiled.system
    assert "lookup" in compiled.system


def test_the_payload_is_a_real_provider_request():
    spec = SubAgentSpec(name="worker", description="Works.", tools=["lookup"])
    compiled = SpecCompiler().compile(spec, tools=REGISTRY)
    request = compiled.to_request([__import__("agent_harness").Message.user("go")])

    assert request.model == compiled.model
    assert request.system == compiled.system
    assert [t.name for t in request.tools] == ["lookup"]
    assert request.messages[0].text == "go"


def test_an_explicit_model_beats_the_tier():
    spec = SubAgentSpec(name="w", description="d", model="gpt-4.1", tier="fast")
    assert SpecCompiler().compile(spec, tools=REGISTRY).model == "gpt-4.1"


def test_no_allowlist_means_every_tool_offered():
    spec = SubAgentSpec(name="w", description="d")     # tools=None
    compiled = SpecCompiler().compile(spec, tools=REGISTRY)
    assert {t.name for t in compiled.tools} == {"lookup", "publish"}


def test_an_empty_allowlist_means_no_tools():
    spec = SubAgentSpec(name="w", description="d", tools=[])
    compiled = SpecCompiler().compile(spec, tools=REGISTRY)
    assert compiled.tools == []
    assert "Tools you can call" not in compiled.system


def test_an_output_contract_is_compiled_into_the_prompt():
    spec = SubAgentSpec(
        name="extractor", description="Extracts.",
        output_schema={"type": "object", "properties": {"total": {"type": "number"}},
                       "required": ["total"]},
    )
    compiled = SpecCompiler().compile(spec, tools=REGISTRY)
    assert compiled.response_schema is not None
    assert "single JSON object" in compiled.system
    assert '"total"' in compiled.system
    assert compiled.to_request().response_schema == compiled.response_schema


def test_skills_appear_as_a_catalogue_not_as_bodies(tmp_path):
    folder = tmp_path / "refunds"
    folder.mkdir()
    (folder / "SKILL.md").write_text(
        "---\nname: refunds\ndescription: How refunds work.\n---\n\nThe long body.\n")
    spec = SubAgentSpec(name="w", description="d", tools=[])
    compiled = SpecCompiler().compile(spec, tools=REGISTRY,
                                      skills=SkillRegistry.from_dir(tmp_path))
    assert "refunds: How refunds work." in compiled.system
    assert "The long body." not in compiled.system


def test_the_estimate_is_computed_before_anything_is_spent():
    spec = SubAgentSpec(name="w", description="d", instructions="x" * 4000,
                        model="claude-opus-5")
    compiled = SpecCompiler().compile(spec, tools=REGISTRY)
    assert compiled.estimated_input_tokens > 900
    assert compiled.estimated_cost_usd > 0


def test_explain_says_what_the_sub_agent_may_do():
    spec = SubAgentSpec(name="reader", description="Reads.", tools=["lookup"],
                        deny=["publish"], workspace="isolated", allow_shell=True,
                        tier="deep")
    text = SpecCompiler().compile(spec, tools=REGISTRY).explain()
    assert "reader" in text
    assert "claude-opus-5" in text
    assert "lookup" in text
    assert "isolated + shell" in text
    assert "publish" in text            # the denial is visible


def test_a_custom_router_is_honoured():
    from agent_harness import RouteRule

    router = ModelRouter(rules=[RouteRule("*research*", model="gemini-2.5-flash")])
    spec = SubAgentSpec(name="research_agent", description="Research things.")
    assert SpecCompiler(router=router).compile(spec).model == "gemini-2.5-flash"


def test_the_harness_exposes_a_compiler():
    harness = Harness.testing(FakeProvider())
    compiled = harness.compiler.compile(SubAgentSpec(name="w", description="d"))
    assert compiled.model


# --- deliverable store --------------------------------------------------------

def test_artefacts_are_versioned_not_overwritten():
    store = DeliverableStore()
    first = store.put(Artifact(name="report.md", content="draft one"))
    second = store.put(Artifact(name="report.md", content="draft two"))

    assert first.version == 1 and second.version == 2
    assert store.get("report.md").content == "draft two"      # latest by default
    assert store.get("report.md", version=1).content == "draft one"
    assert len(store.versions("report.md")) == 2
    assert len(store) == 2


def test_a_digest_identifies_the_contents():
    store = DeliverableStore()
    a = store.put(Artifact(name="a.txt", content="same"))
    b = store.put(Artifact(name="b.txt", content="same"))
    c = store.put(Artifact(name="c.txt", content="different"))
    assert a.digest == b.digest and a.digest != c.digest


def test_the_manifest_lists_the_latest_of_each():
    store = DeliverableStore()
    store.put(Artifact(name="report.md", content="v1", produced_by="writer"))
    store.put(Artifact(name="report.md", content="v2 longer", produced_by="writer"))
    store.put(Artifact(name="data.csv", content="a,b", produced_by="analyst"))

    manifest = store.manifest()
    assert {row["name"] for row in manifest} == {"report.md", "data.csv"}
    report = next(r for r in manifest if r["name"] == "report.md")
    assert report["version"] == 2 and report["produced_by"] == "writer"


def test_artefacts_can_be_traced_back_to_their_run():
    store = DeliverableStore()
    store.put(Artifact(name="a.md", content="x"), run_id="run-1")
    store.put(Artifact(name="b.md", content="y"), run_id="run-2")
    assert [a.name for a in store.for_run("run-1")] == ["a.md"]


def test_the_store_survives_a_restart(tmp_path):
    store = DeliverableStore(tmp_path)
    stored = store.put(Artifact(name="report.md", content="the report"))
    assert stored.path and "report" in stored.path

    reopened = DeliverableStore(tmp_path)
    assert reopened.names == ["report.md"]
    assert reopened.get("report.md").content == "the report"
    assert "report.md" in reopened


def test_a_name_with_awkward_characters_is_still_safe_on_disk(tmp_path):
    store = DeliverableStore(tmp_path)
    stored = store.put(Artifact(name="../../escape.md", content="x"))
    assert tmp_path in __import__("pathlib").Path(stored.path).parents


async def test_what_an_agent_produces_lands_in_the_store():
    provider = FakeProvider(["done"])
    harness = Harness.testing(provider)
    agent = Agent("writer", provider=provider, model=MODEL, harness=harness,
                  memory=True)
    agent.produce("summary.md", "# Summary\n\nIt went well.")
    result = await agent.run("write it up")

    assert harness.deliverables.names == ["summary.md"]
    assert result.artifacts and result.artifacts[0].name == "summary.md"
    assert harness.report()["deliverables"][0]["produced_by"] == "writer"


async def test_sub_agent_artefacts_reach_the_store_through_the_parent():
    provider = FakeProvider([
        __import__("agent_harness").tool_call("delegate", agent_name="writer",
                                              task="write it"),
        "the sub-agent's answer",
        "all done",
    ])
    harness = Harness.testing(provider)
    parent = Agent("manager", provider=provider, model=MODEL, harness=harness,
                   memory=True,
                   subagents=[SubAgentSpec(name="writer", description="Writes.")])
    child = parent.subagents["writer"]
    child.produce("draft.md", "the draft")

    await parent.run("get it written")
    assert "draft.md" in harness.deliverables.names
