"""Every box in the architecture diagram has an implementation, and it works.

This is the completeness guard. `preview-01.png` in the repository root is the
architecture this library implements; each entry below is one box from it. If a
component is dropped or renamed, this fails rather than the gap going unnoticed.
"""

from __future__ import annotations

import inspect

import agent_harness as ah

MODEL = "claude-sonnet-5"


# --- 1 · PLAN & SCOPE ---------------------------------------------------------

def test_plan_and_scope_boxes_exist():
    # Intake & scoping: a request becomes a typed goal with constraints.
    plan = ah.Plan(goal="ship the report", constraints=["British English"])
    assert plan.goal and plan.constraints

    # Definition of done: acceptance tests, written before the work starts.
    plan.definition_of_done = ["every number is sourced"]
    assert plan.definition_of_done

    # Work breakdown: a task graph that marks what can run in parallel.
    plan.tasks = [ah.Task(id="t1", statement="a"), ah.Task(id="t2", statement="b"),
                  ah.Task(id="t3", statement="c", depends_on=["t1", "t2"])]
    assert [[t.id for t in wave] for wave in plan.waves()] == [["t1", "t2"], ["t3"]]

    # Cost & parallelism estimate.
    assert hasattr(ah.Orchestrator, "estimate")


# --- 2 · THE ACCOUNTABLE MANAGER ---------------------------------------------

def test_the_manager_and_the_staffing_gate_exist():
    assert callable(ah.Orchestrator)
    for stage in ("plan", "staff", "execute", "consolidate", "review", "run"):
        assert callable(getattr(ah.Orchestrator, stage)), stage
    # The staffing decision itself: reuse from the bench, or build.
    bench = ah.Bench.standard()
    assert bench.find("validate this against the rules") is not None
    assert bench.find("xyzzy plugh frobnicate") is None


# --- 3 · SUB-AGENT SUPPLY -----------------------------------------------------

def test_the_bench_has_every_pre_defined_sub_agent_from_the_diagram():
    assert set(ah.Bench.standard().names) == {
        "research", "planner", "document_extractor", "data_analyst",
        "validator", "compliance_checker", "drafting", "report_writer",
    }


def test_the_factory_covers_every_aspect_it_writes():
    spec = ah.SubAgentSpec(name="x", description="d")
    for field in ("name", "instructions",       # blueprint builder, prompt author
                  "skills",                     # playbook binder
                  "tools",                      # tool allowlist + system-access binder
                  "model", "tier", "effort",    # model & effort
                  "workspace", "allow_shell",   # workspace isolation
                  "output_schema"):             # input/output contract
        assert field in type(spec).model_fields, field
    assert callable(ah.SubAgentFactory)


# --- 4 · SUB-AGENTS AT WORK ---------------------------------------------------

def test_work_assignment_and_hand_back_are_implemented():
    signature = inspect.signature(ah.Orchestrator.__init__)
    assert "max_concurrency" in signature.parameters   # concurrency cap
    assert "task_timeout" in signature.parameters      # deadline per sub-agent
    assert "task_retries" in signature.parameters      # retry per sub-agent

    task = ah.Task(id="t", statement="s")
    assert hasattr(task, "partial")                    # partial delivery kept
    assert hasattr(task, "cost_usd")                   # cost per sub-agent
    assert "children" in ah.RunResult.model_fields     # trace per sub-agent


# --- 5 · CONSOLIDATE & REVIEW -------------------------------------------------

def test_consolidate_review_and_rework_exist():
    assert callable(ah.Orchestrator.consolidate)
    assert callable(ah.Orchestrator.review)
    review = ah.Review(accepted=False, gaps=["source the cost figures"])
    assert review.gaps                                  # rework re-plans the gaps
    assert "max_rework" in inspect.signature(ah.Orchestrator.__init__).parameters


# --- CAPABILITY & CONNECTED SYSTEMS ------------------------------------------

def test_system_access_is_reachable_through_mcp():
    assert callable(ah.MCPServer) and callable(ah.MCPClient)
    stdio = ah.MCPServer(name="files", command="npx", args=["-y", "server"])
    http = ah.MCPServer(name="api", url="https://mcp.example/rpc")
    assert stdio.transport == "stdio" and http.transport == "http"
    assert "allowed_tools" in type(stdio).model_fields   # granted capability only


def test_every_native_tool_from_the_tooling_panel_exists():
    from agent_harness import toolkits

    # Read · Write · Transform, and sandboxed compute, come from a workspace.
    assert callable(toolkits.make_python_tool)
    assert callable(toolkits.make_corpus_search)         # search across the corpus
    assert callable(toolkits.make_fetch_tool)            # web fetch & search
    assert callable(toolkits.parse_document)             # document parsing & OCR
    assert callable(toolkits.bar_chart)                  # chart & report rendering
    assert callable(toolkits.render_report)


def test_a_workspace_provides_read_write_transform():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        workspace = ah.Workspace(tmp)
        names = {t.name for t in workspace.tools()}
        assert {"fs_read", "fs_write", "fs_list", "fs_delete"} <= names
        workspace.allow_shell = True
        assert "shell" in {t.name for t in workspace.tools()}


# --- MEMORY & CONTEXT MANAGEMENT ---------------------------------------------

async def test_all_four_memory_scopes_behave_as_the_diagram_describes():
    manager = ah.MemoryManager(ah.InMemoryStore(), semantic=False)

    # user.md — loaded in full, permanent.
    await manager.user.remember("bills in EUR")
    assert "bills in EUR" in await manager.user.block()

    # session — the whole conversation plus its artefacts, this session only.
    manager.session.add_message(ah.Message.user("hello"))
    assert manager.session.messages

    # orchestrator — a digest only, never the whole history.
    for i in range(40):
        await manager.orchestrator.finding(f"finding {i}")
    assert manager.orchestrator.digest(limit=3).count("finding") <= 4

    # sub-agent — nothing carried in, only resources carried out, dies with the task.
    sub = manager.subagent("extract totals", "extractor")
    sub.produce(ah.Artifact(name="totals.csv", content="1,2"))
    assert [a.name for a in sub.close()] == ["totals.csv"]
    assert sub.artifacts == []

    # Session close → user memory update.
    assert await manager.close_session("learned something") is not None


# --- RUNTIME PLATFORM · THE HARNESS CORE -------------------------------------

def test_every_harness_core_component_exists():
    harness = ah.Harness()
    core = {
        "Context Assembler": ah.ContextAssembler,
        "Agent Loop Engine": ah.Agent,
        "Hook Engine": harness.hooks,
        "Spec Compiler": harness.compiler,
        "Workspace Broker": harness.workspaces,
        "Session Store": harness.sessions,
        "Model Router": harness.router,
        "Permission & Policy Gate": harness.policy,
        "Context Compactor": ah.ContextCompactor,
        "Budget & Rate Guard": (harness.guard, harness.rate),
        "Concurrency Scheduler": harness.scheduler,
        "Tool Registry": ah.ToolRegistry,
    }
    for box, component in core.items():
        assert component is not None, box


def test_the_spec_compiler_turns_a_blueprint_into_a_provider_payload():
    spec = ah.SubAgentSpec(name="w", description="Works.", tools=[])
    compiled = ah.SpecCompiler().compile(spec)
    request = compiled.to_request([ah.Message.user("go")])
    assert request.model and request.system and request.messages


# --- CROSS-CUTTING RAILS · INSTITUTIONAL MEMORY ------------------------------

def test_every_institutional_memory_rail_exists():
    harness = ah.Harness()
    assert harness.sessions is not None                  # session & thread
    assert harness.checkpoints is not None               # checkpoints
    assert harness.journal is not None                   # run journal
    assert harness.workspaces.shared() is not None       # shared workspace
    assert harness.memory_store is not None              # long-term memory
    assert callable(ah.SemanticMemory)                   # semantic recall
    assert harness.deliverables is not None              # deliverable store
    assert harness.cache is not None                     # result cache


# --- CROSS-CUTTING RAILS · ASSURANCE & CONTROL -------------------------------

def test_every_assurance_rail_exists():
    harness = ah.Harness()
    assert harness.tracer is not None                    # end-to-end tracing
    assert harness.guard.report()["by_agent"] == {}      # cost attribution
    assert callable(ah.Evaluator)                        # quality evaluation
    assert harness.audit is not None                     # audit trail
    assert harness.guardrails is not None                # guardrails
    assert harness.control is not None                   # stop control
    assert harness.replayer is not None                  # replay & time travel
    assert harness.health is not None                    # service health


def test_the_run_report_surfaces_every_rail():
    report = ah.Harness().report()
    for key in ("budget", "rate", "cache", "scheduler", "guardrails", "health",
                "control", "deliverables", "audit_entries", "journal_entries",
                "spans"):
        assert key in report, key


# --- the whole thing, running --------------------------------------------------

async def test_the_complete_harness_runs_one_job_end_to_end(tmp_path):
    """Plan, staff, run in parallel, consolidate, review — with every rail on."""
    import json

    plan = {
        "goal": "Summarise Q3",
        "definition_of_done": ["revenue and costs are both cited"],
        "tasks": [
            {"id": "t1", "statement": "Research the revenue figures",
             "depends_on": []},
            {"id": "t2", "statement": "Research the cost figures", "depends_on": []},
            {"id": "t3", "statement": "Write the report", "depends_on": ["t1", "t2"]},
        ],
    }

    def respond(request):
        text = request.messages[-1].text
        if "costed plan" in text:
            return json.dumps(plan)
        if "Review this deliverable" in text:
            return json.dumps({"accepted": True, "score": 0.95,
                               "summary": "meets the bar"})
        if "Consolidate these sub-agent results" in text:
            return "Q3: revenue 4.2M, costs 3.1M."
        return "figure found and cited"

    provider = ah.FakeProvider([respond], loop=True)
    harness = ah.Harness.local(tmp_path / "state", trace=False)
    harness.provider = provider
    boss = ah.Orchestrator("boss", harness=harness, provider=provider, model=MODEL,
                           max_concurrency=4, review=True, budget=ah.Budget(max_usd=5))

    result = await boss.run("Summarise how Q3 went, with the numbers cited.")

    # It produced the answer.
    assert result.output == "Q3: revenue 4.2M, costs 3.1M."
    assert result.data["review"]["accepted"] is True

    done = ah.Plan(**result.data["plan"])
    assert all(t.status == "done" for t in done.tasks)
    assert all(t.agent for t in done.tasks)               # every task was staffed

    # And every rail recorded what it should have.
    report = harness.report()
    assert report["budget"]["calls"] > 0
    assert report["budget"]["by_agent"]                   # cost attributed per agent
    assert report["health"]["status"] in {"healthy", "unknown"}
    assert report["audit_entries"] > 0
    assert harness.audit.verify()[0]                      # the chain is intact
    assert report["journal_entries"] > 0
    assert any(a["name"] == "plan.json" for a in report["deliverables"])
    assert report["scheduler"]["completed"] >= 3          # tasks went through the pool

    # The run is reproducible: checkpoints exist and can be walked.
    assert (tmp_path / "state" / "audit.jsonl").exists()
    assert (tmp_path / "state" / "journal.jsonl").exists()


async def test_a_single_agent_uses_every_rail_without_being_asked(tmp_path):
    provider = ah.FakeProvider([ah.tool_call("calculate", expression="2+2"),
                                "The answer is 4."])
    harness = ah.Harness.local(tmp_path / "state", trace=False)
    from agent_harness.toolkits import calculate

    agent = ah.Agent("assistant", "Answer questions.", provider=provider,
                     model=MODEL, harness=harness, tools=[calculate])
    result = await agent.run("what is 2+2?")

    assert result.output == "The answer is 4."
    assert harness.audit.verify()[0]
    assert harness.health.component(MODEL, "model").calls == 2
    assert harness.health.component("calculate", "tool").calls == 1
    assert len(await harness.checkpoints.history(result.run_id)) == 2
    assert harness.journal.entries
    assert (await harness.sessions.load(result.session_id)).messages


# --- the AI-facing docs stay in step with the code ----------------------------

def test_llms_txt_is_regenerated_from_the_repository():
    """`llms.txt` / `llms-full.txt` are generated. If they drift, this fails."""
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [sys.executable, str(root / "scripts" / "build_llms_txt.py"), "--check"],
        capture_output=True, text=True, cwd=root,
    )
    assert result.returncode == 0, (
        f"{result.stdout}{result.stderr}\n"
        "run `python scripts/build_llms_txt.py` and commit the result"
    )


def test_the_index_points_at_files_that_exist():
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    index = (root / "llms.txt").read_text(encoding="utf-8")
    referenced = {
        line.split("/main/", 1)[1].split(")", 1)[0]
        for line in index.splitlines()
        if "/main/" in line and line.startswith("- [")
    }
    assert referenced, "the index links to nothing"
    missing = [path for path in referenced if not (root / path).exists()]
    assert not missing, f"llms.txt links to files that do not exist: {missing}"


def test_every_public_export_appears_in_the_full_docs():
    from pathlib import Path

    import agent_harness as ah

    root = Path(__file__).resolve().parent.parent
    full = (root / "llms-full.txt").read_text(encoding="utf-8")
    missing = [name for name in ah.__all__
               if name != "__version__" and f"`{name}" not in full]
    assert not missing, f"undocumented in llms-full.txt: {missing}"


# --- the package layout the library presents ----------------------------------

def test_the_feature_areas_are_packages_with_a_clear_shape():
    """mcp/, providers/, memory/, runtime/, toolkits/, subagents/, guardrails/."""
    import importlib
    from pathlib import Path

    root = Path(ah.__file__).parent
    for package, members in {
        "providers": ["Provider", "AnthropicProvider", "FakeProvider"],
        "mcp": ["MCPServer", "MCPClient", "MCPManager"],
        "memory": ["MemoryManager", "SemanticMemory"],
        "runtime": ["Budget", "PolicyGate", "Tracer"],
        "toolkits": ["parse_document", "bar_chart", "make_python_tool"],
        "subagents": ["SubAgentSpec", "Bench", "SubAgentFactory", "build_agent"],
        "guardrails": ["Guardrails", "AgentGuardrails", "RequireTools"],
    }.items():
        assert (root / package / "__init__.py").exists(), f"{package} is not a package"
        module = importlib.import_module(f"agent_harness.{package}")
        for member in members:
            assert hasattr(module, member), f"{package}.{member} is missing"


def test_memory_can_be_stored_in_every_backend_the_docs_promise():
    from agent_harness.memory import providers

    assert set(providers.BACKENDS) >= {
        "memory", "file", "sqlite", "postgres", "mysql", "mongo", "redis",
        "dynamodb", "elasticsearch", "s3", "azure", "gcs", "http",
    }
    # Each one is scoped by a trace and implements the same contract.
    for name in providers.BACKENDS:
        cls = providers.get_backend(name)
        for method in ("append", "all", "clear", "read_doc", "write_doc", "search"):
            assert hasattr(cls, method), f"{name} has no {method}"


def test_the_old_module_paths_still_import():
    """0.1.0 shipped these as modules; moving them must not break an import."""
    from agent_harness.runtime.guardrails import Guardrails as FromRuntime
    from agent_harness.subagent import Bench, SubAgentSpec, build_agent

    assert SubAgentSpec is ah.SubAgentSpec
    assert Bench is ah.Bench and callable(build_agent)
    assert FromRuntime is ah.Guardrails
