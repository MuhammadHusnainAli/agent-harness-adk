"""Modes: what kind of work an agent does, and how hard it works at it.

    Agent("helper", mode="chat")
    Agent("analyst", mode="research", depth="deep", tools=[search, http_fetch])
    Agent("colleague", mode="cowork", workspace=Workspace("./project"))

An agent with no mode is the plain loop, exactly as you configured it. A mode is
a way of working laid over that loop:

- **chat** — a conversation. The agent keeps the thread between calls, so the
  second `run()` is the second turn rather than a stranger's first.
- **research** — a report someone can check. The agent plans the questions,
  records every source it relies on in a ledger, and cites by number. A citation
  to something it never recorded, or a source it never actually read, is refused.
- **cowork** — a task handed over and carried through. The agent keeps a todo
  list, works in a workspace, may ask you a question, spins up helpers for the
  parts that can run side by side, and hands back the files it produced.

`depth` is the second dial — `fast`, `balanced` or `deep`. It sets how many
steps the agent may take, how many sources a report needs, how many helpers it
may start, and — when you name one — which model tier does the work.

A mode only fills in what you left unset: `Agent(mode="cowork", max_steps=10)`
takes ten steps. To tune a mode itself, build it:

    from agent_harness import modes

    Agent("analyst", mode=modes.research("deep", min_sources=12))
    Agent("colleague", mode=modes.cowork(ask=my_question_handler, python=True))
"""

from __future__ import annotations

import inspect
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from .errors import ConfigurationError, ToolError
from .tools import Tool, ToolContext
from .types import Artifact, Source, Todo

__all__ = [
    "Mode",
    "Notebook",
    "MODES",
    "DEPTHS",
    "chat",
    "research",
    "cowork",
    "resolve_mode",
    "register_mode",
    "citations",
]

DEPTHS: tuple[str, ...] = ("fast", "balanced", "deep")

_DEPTH_ALIASES = {
    "quick": "fast", "light": "fast",
    "standard": "balanced", "normal": "balanced", "medium": "balanced",
    "default": "balanced",
    "slow": "deep", "thorough": "deep", "extended": "deep",
}

_MODE_ALIASES = {
    "deep_research": "research", "deep-research": "research",
    "deepresearch": "research", "co-work": "cowork", "co_work": "cowork",
}

#: Tools carrying this tag belong to the agent that owns the run — its todo
#: list, its line to the person. A sub-agent never inherits them.
LEAD = "lead"

# Evidence is kept by reference, so this bounds what a very long run can pin.
_EVIDENCE_CHARS = 4_000_000
_MAX_TODOS = 60
_MAX_FILE_CHARS = 400_000
_MAX_FILES = 200

Asker = Callable[[str, list[str]], Any]


# ---------------------------------------------------------------------------
# citations
# ---------------------------------------------------------------------------
_FENCED = re.compile(r"```.*?```", re.DOTALL)
_CITE = re.compile(r"(?<!\w)\[(\d{1,4}(?:\s*[,;–-]\s*\d{1,4})*)\]")
_HAS_SOURCE_LIST = re.compile(
    r"^\s{0,3}(?:#{1,6}\s*|\*\*)?(?:sources|references|bibliography)\b",
    re.IGNORECASE | re.MULTILINE,
)


def citations(text: str) -> set[int]:
    """The source numbers a text cites: `[3]`, `[1, 2]` and `[4-6]` all count.

    Code fences are skipped, and so is an index like `rows[0]` — a bracket
    hard against a word is not a citation.
    """
    found: set[int] = set()
    for group in _CITE.findall(_FENCED.sub(" ", text or "")):
        for part in re.split(r"[,;]", group):
            ends = [int(n) for n in re.split(r"[–-]", part) if n.strip().isdigit()]
            if len(ends) == 2 and 0 < ends[1] - ends[0] <= 50:
                found.update(range(ends[0], ends[1] + 1))
            else:
                found.update(ends)
    return found


def _normalise(ref: str) -> str:
    """A reference as it is compared: case, a trailing slash and a fragment
    are not what makes two sources different."""
    text = " ".join(str(ref).split()).casefold().rstrip(".,;")
    if "://" in text:
        text = text.split("#", 1)[0].rstrip("/")
    return text


# ---------------------------------------------------------------------------
# the notebook
# ---------------------------------------------------------------------------
class Notebook:
    """What a mode keeps while it works: the todo list and the source ledger.

    One per run. Sub-agents get a `child()` — their own todo list, but the same
    ledger, so a source a helper records has one number across the whole run
    and the lead can cite it as it was handed back.
    """

    def __init__(self, *, sources: list[Source] | None = None,
                 evidence: list[str] | None = None, ledger: bool = False,
                 tracking: bool = False) -> None:
        self.todos: list[Todo] = []
        self.sources: list[Source] = sources if sources is not None else []
        self._evidence: list[str] = evidence if evidence is not None else []
        #: This run keeps a source ledger, so its helpers write in it too.
        self.ledger = ledger
        #: Keep what tools returned, so a recorded source can be checked against it.
        self.tracking = tracking
        self.questions = 0
        #: Bumped on every change, so the loop knows when to report progress.
        self.revision = 0

    def child(self) -> Notebook:
        return Notebook(sources=self.sources, evidence=self._evidence,
                        ledger=self.ledger, tracking=self.tracking)

    # ---- todos ---------------------------------------------------------
    def write_todos(self, items: Iterable[Todo]) -> list[Todo]:
        self.todos[:] = list(items)
        self.revision += 1
        return self.todos

    def open_todos(self) -> list[Todo]:
        return [t for t in self.todos if t.open]

    def todo_summary(self) -> str:
        if not self.todos:
            return "no items"
        counts = {status: sum(1 for t in self.todos if t.status == status)
                  for status in ("done", "in_progress", "pending", "skipped")}
        parts = [f"{n} {status.replace('_', ' ')}" for status, n in counts.items() if n]
        return f"{len(self.todos)} items · " + " · ".join(parts)

    # ---- evidence and sources -------------------------------------------
    def saw(self, *texts: str) -> None:
        """Remember what the run has read — tool arguments and tool results."""
        if not self.tracking:
            return
        for text in texts:
            if text:
                self._evidence.append(text)
        total = sum(len(t) for t in self._evidence)
        while total > _EVIDENCE_CHARS and len(self._evidence) > 1:
            total -= len(self._evidence.pop(0))

    def mentions(self, ref: str) -> bool:
        """Did anything this run read actually mention `ref`?"""
        needle = _normalise(ref)
        if len(needle) < 3:
            return False
        needles = {needle, needle.split("://", 1)[-1]}
        if needle.split("://", 1)[-1].startswith("www."):
            needles.add(needle.split("://", 1)[-1][4:])
        for text in self._evidence:
            lowered = text.casefold()
            if any(n in lowered for n in needles):
                return True
        return False

    def find(self, ref: str) -> Source | None:
        key = _normalise(ref)
        return next((s for s in self.sources if _normalise(s.ref) == key), None)

    def record(self, ref: str, finding: str = "", *, title: str = "",
               agent: str = "") -> Source:
        """Add a source to the ledger, or return the number it already has."""
        existing = self.find(ref)
        if existing is not None:
            if finding and finding not in existing.finding:
                existing.finding = f"{existing.finding}\n{finding}".strip()
            if title and not existing.title:
                existing.title = title
            self.revision += 1
            return existing
        source = Source(id=len(self.sources) + 1, ref=" ".join(ref.split()),
                        title=title.strip(), finding=finding.strip(), agent=agent)
        self.sources.append(source)
        self.revision += 1
        return source

    # ---- what must outlive compaction -------------------------------------
    def pins(self) -> list[str]:
        """The state a long run cannot afford to lose when its context is
        compacted: what is left to do, and which number means which source."""
        out: list[str] = []
        if self.todos:
            out.append("Your todo list: " + "; ".join(t.line() for t in self.todos))
        if self.sources:
            out.append("Sources recorded so far: " + "; ".join(
                f"{s.line()}{f' ({s.finding[:160]})' if s.finding else ''}"
                for s in self.sources))
        return out

    def progress(self) -> dict[str, Any]:
        return {"todos": [t.model_dump() for t in self.todos],
                "sources": len(self.sources), "summary": self.todo_summary()}


# ---------------------------------------------------------------------------
# the mode
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Mode:
    """A way of working, resolved at one depth. Build one with `chat()`,
    `research()` or `cowork()`; `dataclasses.replace` changes a single field."""

    name: str
    depth: str = "balanced"
    description: str = ""
    #: The model tier this depth asks for. Only set when a depth was named —
    #: an agent that did not ask for one keeps the router's own choice.
    tier: str | None = None
    max_steps: int = 20
    #: How many specialists the agent may spin up for itself. 0 turns it off.
    helpers: int = 0
    workspace: bool = False
    #: Consecutive runs continue one conversation.
    conversational: bool = False
    #: Keeps a todo list, and may not finish with items still open.
    todos: bool = False
    #: Keeps a source ledger, and may only cite what is in it.
    sources: bool = False
    min_sources: int = 0
    #: A source may only be recorded if something the run read mentions it.
    verify_sources: bool = True
    #: Answers `ask_user`. Without one the agent has no way to ask, and is told so.
    ask: Asker | None = field(default=None, compare=False)
    max_questions: int = 5
    documents: bool = False
    python: bool = False
    #: Files written in the workspace during a run come back as artefacts.
    collect_files: bool = False
    #: The finished answer is kept as this artefact.
    report: str = ""
    #: How many times an unfinished answer is sent back before it is delivered
    #: with `result.violations` saying what is missing.
    retries: int = 2
    #: (tool this line needs, the line). "" always applies; "!name" applies
    #: only when the agent does not have that tool.
    guidance: tuple[tuple[str, str], ...] = ()
    opening: str = ""
    closing: str = ""
    wrap_up: str = "This is your last step. Answer now with what you have."

    def __post_init__(self) -> None:
        if self.depth not in DEPTHS:
            raise ConfigurationError(
                f"depth must be one of {', '.join(DEPTHS)} — got {self.depth!r}")
        if self.max_steps < 1:
            raise ConfigurationError(f"{self.name}: max_steps must be at least 1")
        if not 0 <= self.helpers <= 100:
            raise ConfigurationError(
                f"{self.name}: helpers must be between 0 and 100 — got {self.helpers}")
        if self.min_sources < 0 or self.retries < 0:
            raise ConfigurationError(
                f"{self.name}: min_sources and retries cannot be negative")

    def __str__(self) -> str:
        return self.name

    @property
    def keeps_notebook(self) -> bool:
        return self.todos or self.sources

    # ---- what the model is told -------------------------------------------
    def prompt(self, tool_names: Iterable[str] = ()) -> str:
        """The operating instructions, for the tools this agent actually has."""
        have = set(tool_names)
        lines: list[str] = []
        for needs, text in self.guidance:
            if needs.startswith("!"):
                if needs[1:] in have:
                    continue
            elif needs and needs not in have:
                continue
            lines.append(text)
        body = "\n".join(f"{n}. {line}" for n, line in enumerate(lines, 1))
        return "\n\n".join(p for p in (self.opening, body, self.closing) if p)

    # ---- the tools a mode adds ---------------------------------------------
    def tools(self, workspace: Any = None) -> list[Tool]:
        built: list[Tool] = []
        if self.todos:
            built.append(_todo_tool())
        if self.sources:
            built.append(_source_tool(self))
        if self.ask is not None:
            built.append(_ask_tool(self))
        if workspace is not None and self.documents:
            from .toolkits.documents import make_document_tool

            built.append(make_document_tool(workspace))
        if workspace is not None and self.python:
            from .toolkits.compute import make_python_tool

            built.append(make_python_tool(workspace))
        return built

    # ---- has it finished? ----------------------------------------------------
    def unmet(self, output: str, notebook: Notebook | None) -> list[str]:
        """What this mode still requires of an answer. Empty means done."""
        if notebook is None:
            return []
        problems: list[str] = []
        if self.todos:
            left = notebook.open_todos()
            if left:
                names = "; ".join(t.content for t in left[:8])
                more = f" (and {len(left) - 8} more)" if len(left) > 8 else ""
                problems.append(
                    f"items on your todo list are still open: {names}{more} — "
                    "finish them, or mark each one `skipped` with a note saying why")
        if self.sources:
            ids = {s.id for s in notebook.sources}
            cited = citations(output)
            unknown = sorted(cited - ids)
            if unknown:
                problems.append(
                    f"you cite {', '.join(f'[{n}]' for n in unknown)} but no such "
                    "source is recorded — cite only the numbers `record_source` "
                    "gave you")
            if len(ids) < self.min_sources:
                problems.append(
                    f"{len(ids)} of the {self.min_sources} sources this report "
                    "needs are recorded — read more and record each with "
                    "`record_source`, or say in the report what could not be found")
            elif ids and not cited & ids:
                problems.append(
                    "the report cites none of the sources you recorded — put [n] "
                    "straight after each claim that source supports")
        return problems

    @staticmethod
    def feedback(problems: Iterable[str]) -> str:
        """The message the agent is given so it can finish properly."""
        body = "\n".join(f"- {p}" for p in problems)
        return f"That is not finished yet:\n{body}\n\nPut it right, then answer again."

    # ---- handing it over -----------------------------------------------------
    def close(self, output: str, notebook: Notebook | None) -> str:
        """The answer as it is delivered: a report gains its source list."""
        if not (self.sources and notebook and notebook.sources and output):
            return output
        if _HAS_SOURCE_LIST.search(_FENCED.sub(" ", output)):
            return output
        cited = citations(output)
        rows = [s for s in notebook.sources if s.id in cited]
        heading = "Sources" if rows else "Sources consulted"
        listing = "\n".join(f"- {s.line()}" for s in rows or notebook.sources)
        return f"{output.rstrip()}\n\n## {heading}\n{listing}"

    async def artifacts(self, output: str, *, agent: str, workspace: Any = None,
                        before: dict[str, tuple[int, int]] | None = None
                        ) -> list[Artifact]:
        """What the run leaves behind: the report, and the files it wrote."""
        out: list[Artifact] = []
        if self.report and output:
            out.append(Artifact(name=self.report, content=output,
                                media_type="text/markdown", produced_by=agent))
        if self.collect_files and workspace is not None and before is not None:
            out.extend(await _collect(workspace, before, agent))
        return out


# ---------------------------------------------------------------------------
# tools
# ---------------------------------------------------------------------------
def _notebook(ctx: ToolContext | None, tool: str) -> Notebook:
    notebook = ctx.state.get("notebook") if ctx is not None else None
    if notebook is None:
        raise ToolError(f"{tool} only works inside a run of an agent with a mode",
                        tool=tool)
    return notebook


def _todo_tool() -> Tool:
    def todo_write(todos: list[dict[str, Any]], ctx: ToolContext | None = None) -> str:
        notebook = _notebook(ctx, "todo_write")
        if len(todos) > _MAX_TODOS:
            raise ToolError(f"a todo list holds at most {_MAX_TODOS} items — group "
                            "the small ones", tool="todo_write")
        items: list[Todo] = []
        for n, raw in enumerate(todos, 1):
            if not isinstance(raw, dict):
                raise ToolError(f"item {n} must be an object with `content` and "
                                "`status`", tool="todo_write")
            content = str(raw.get("content") or "").strip()
            status = str(raw.get("status") or "pending").strip().lower()
            status = {"completed": "done", "complete": "done", "doing": "in_progress",
                      "in-progress": "in_progress", "todo": "pending",
                      "cancelled": "skipped", "canceled": "skipped"}.get(status, status)
            note = str(raw.get("note") or "").strip()
            if not content:
                raise ToolError(f"item {n} has no `content`", tool="todo_write")
            if status not in ("pending", "in_progress", "done", "skipped"):
                raise ToolError(
                    f"item {n}: status must be pending, in_progress, done or "
                    f"skipped — got {status!r}", tool="todo_write")
            if status == "skipped" and not note:
                raise ToolError(
                    f"item {n} is skipped without a `note` — say why it was not "
                    "done", tool="todo_write")
            items.append(Todo(content=content, status=status, note=note))
        notebook.write_todos(items)
        listing = "\n".join(t.line() for t in notebook.todos)
        return f"{notebook.todo_summary()}\n{listing}".strip()

    return Tool(
        todo_write,
        name="todo_write",
        description=(
            "Write your todo list: the whole list, every time. Call it before you "
            "start to lay out the steps, then again whenever something changes — "
            "an item started, finished, added or dropped. You may not finish while "
            "an item is still pending or in progress."
        ),
        parameters={
            "type": "object",
            "properties": {
                "todos": {
                    "type": "array",
                    "description": "Every item on the list, in order.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string",
                                        "description": "What is to be done, in one "
                                                       "line."},
                            "status": {"type": "string",
                                       "enum": ["pending", "in_progress", "done",
                                                "skipped"]},
                            "note": {"type": "string",
                                     "description": "The outcome, or why it was "
                                                    "skipped. Required when "
                                                    "skipped."},
                        },
                        "required": ["content", "status"],
                    },
                },
            },
            "required": ["todos"],
        },
        tags=["builtin", "mode", LEAD],
    )


def _source_tool(mode: Mode) -> Tool:
    def record_source(ref: str, finding: str, title: str = "",
                      ctx: ToolContext | None = None) -> str:
        notebook = _notebook(ctx, "record_source")
        ref = " ".join(str(ref).split())
        if not ref:
            raise ToolError("a source needs a `ref` — its URL, title, file path or "
                            "id", tool="record_source")
        if not str(finding).strip():
            raise ToolError("say what this source established in `finding`",
                            tool="record_source")
        if (mode.verify_sources and notebook.tracking
                and notebook.find(ref) is None and not notebook.mentions(ref)):
            raise ToolError(
                f"nothing you have read in this run mentions {ref!r}. Record a "
                "source only after a tool has returned it, and give the URL, "
                "title or id exactly as it appeared.", tool="record_source")
        source = notebook.record(ref, str(finding), title=title,
                                 agent=ctx.agent if ctx else "")
        return (f"Recorded as [{source.id}] — cite it as [{source.id}]. "
                f"{len(notebook.sources)} sources recorded so far.")

    return Tool(
        record_source,
        name="record_source",
        description=(
            "Record a source you have actually read and what it established. It "
            "returns the number to cite it by: put [n] after each claim it "
            "supports. Recording the same source again returns the same number."
        ),
        parameters={
            "type": "object",
            "properties": {
                "ref": {"type": "string",
                        "description": "The URL, document title, file path or id, "
                                       "exactly as it appeared in what you read."},
                "finding": {"type": "string",
                            "description": "What this source established — the "
                                           "fact, figure or quotation you will "
                                           "rely on."},
                "title": {"type": "string",
                          "description": "A readable title, if the ref is a URL "
                                         "or an id."},
            },
            "required": ["ref", "finding"],
        },
        tags=["builtin", "mode", "research"],
    )


def _ask_tool(mode: Mode) -> Tool:
    async def ask_user(question: str, options: list[str] | None = None,
                       ctx: ToolContext | None = None) -> str:
        notebook = _notebook(ctx, "ask_user")
        if not str(question).strip():
            raise ToolError("there is no question to ask", tool="ask_user")
        if notebook.questions >= mode.max_questions:
            return (f"You have asked {mode.max_questions} questions, which is the "
                    "limit for one run. Decide this one yourself and say what you "
                    "assumed.")
        notebook.questions += 1
        assert mode.ask is not None
        answer = mode.ask(question, list(options or []))
        if inspect.isawaitable(answer):
            answer = await answer
        answer = str(answer or "").strip()
        return answer or ("(no answer) — make the reasonable assumption and say "
                          "what you assumed.")

    return Tool(
        ask_user,
        name="ask_user",
        description=(
            "Ask the person you are working for one question and wait for the "
            "answer. Use it only for a decision that is theirs to make and that "
            "changes what you do next — never for something you can look up or "
            "reasonably assume."
        ),
        parameters={
            "type": "object",
            "properties": {
                "question": {"type": "string",
                             "description": "One clear question, with the context "
                                            "needed to answer it."},
                "options": {"type": "array", "items": {"type": "string"},
                            "description": "The choices, if there are only a few."},
            },
            "required": ["question"],
        },
        tags=["builtin", "mode", "human", LEAD],
    )


async def _collect(workspace: Any, before: dict[str, tuple[int, int]],
                   agent: str) -> list[Artifact]:
    """Every file the run created or changed in its workspace, as an artefact.

    From a sandbox the file is brought down to this machine, so the artefact
    outlives the sandbox it was written in.
    """
    import mimetypes

    out: list[Artifact] = []
    for relative in (await workspace.achanged(before))[:_MAX_FILES]:
        media_type = mimetypes.guess_type(relative)[0] or "text/plain"
        content = ""
        try:
            path = str(await workspace.materialize(relative))
        except (ToolError, OSError):
            # Too large to bring down, or gone again: say where it is instead.
            uri = getattr(workspace, "uri", None)
            path = uri(relative) if uri else str(workspace.root / relative)
        else:
            try:
                from pathlib import Path

                if Path(path).stat().st_size <= _MAX_FILE_CHARS:
                    content = Path(path).read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                # Not text: the path is what is worth handing back.
                media_type = (mimetypes.guess_type(relative)[0]
                              or "application/octet-stream")
        out.append(Artifact(name=relative, content=content, path=path,
                            media_type=media_type, produced_by=agent))
    return out


# ---------------------------------------------------------------------------
# the three modes
# ---------------------------------------------------------------------------
def _depth(value: str | None) -> tuple[str, str | None]:
    """(the depth to work at, the tier it asks for).

    No depth named means balanced work on whatever model the router picks —
    naming one is what asks for a tier.
    """
    if value is None:
        return "balanced", None
    name = str(value).strip().lower().replace("-", "_")
    name = _DEPTH_ALIASES.get(name, name)
    if name not in DEPTHS:
        raise ConfigurationError(
            f"depth must be one of {', '.join(DEPTHS)} — got {value!r}")
    return name, name


def chat(depth: str | None = None, *, max_steps: int | None = None) -> Mode:
    """A conversation: quick turns, and the thread kept between them."""
    level, tier = _depth(depth)
    return Mode(
        name="chat", depth=level, tier=tier,
        description="A conversation that keeps its thread between turns.",
        max_steps=max_steps or {"fast": 6, "balanced": 12, "deep": 20}[level],
        conversational=True,
        opening=("You are in a conversation. Answer the message in front of you, "
                 "directly, at the length it deserves — a short question gets a "
                 "short answer."),
        guidance=(
            ("", "Earlier turns are context. Do not ask for what you were already "
                 "told, and do not repeat what you already said."),
            ("", "If a request is ambiguous in a way that changes the answer, ask "
                 "one question instead of guessing."),
            ("", "Reach for a tool when it gets you a fact you do not have. Do not "
                 "turn a question into a project."),
        ),
    )


def research(depth: str | None = None, *, min_sources: int | None = None,
             verify_sources: bool = True, helpers: int | None = None,
             max_steps: int | None = None, report: str = "report.md") -> Mode:
    """A cited report: plan the questions, read, record sources, cite by number.

    Args:
        depth: `fast` (a quick answer from a couple of sources), `balanced`, or
            `deep` (many sources, helpers reading in parallel).
        min_sources: how many recorded sources the report needs.
        verify_sources: refuse a source nothing in the run actually mentioned.
        helpers: how many specialists it may spin up to read in parallel.
        max_steps: the step ceiling, if the depth's own is not right.
        report: the artefact the finished report is kept as; "" keeps none.
    """
    level, tier = _depth(depth)
    table = {"fast": (12, 2, 0), "balanced": (30, 4, 3), "deep": (60, 8, 6)}
    steps, sources, parallel = table[level]
    need = sources if min_sources is None else min_sources
    return Mode(
        name="research", depth=level, tier=tier,
        description="A cited report built from sources the agent read and recorded.",
        max_steps=max_steps or steps,
        helpers=parallel if helpers is None else helpers,
        todos=True, sources=True, min_sources=need, verify_sources=verify_sources,
        report=report,
        opening=("You are doing research. The deliverable is a report someone else "
                 "can check, so every claim in it has to lead back to something you "
                 "read."),
        guidance=(
            ("", "Plan first: break the request into the questions the report must "
                 "answer and write them down with `todo_write`. Mark each one done "
                 "when you have the evidence for it."),
            ("", "Read before you write. Use your tools to find sources and read "
                 "them; do not answer from memory, and do not stop at the first "
                 "source that agrees with you."),
            ("spawn_agent", "Questions that do not depend on each other can be "
                            "researched side by side: `spawn_agent` one specialist "
                            "per question, in the same turn. Tell each to record "
                            "its sources and cite them by number."),
            ("", "Each time a source gives you something the report will rely on, "
                 "call `record_source` with the reference exactly as it appeared "
                 "and what it established. It returns the number to cite."),
            ("", "Cite by number: [n] straight after the claim it supports. Never "
                 "cite a number you were not given, and never state a figure, a "
                 "date or a quotation without one."),
            ("", "Where sources disagree, say so and cite both. Where you found "
                 "nothing, say that — do not fill the gap."),
        ),
        closing=("Write the report itself: the answer first, then the evidence for "
                 "it. Do not add a list of sources at the end — it is appended from "
                 "what you recorded." + (
                     f" This report needs at least {need} "
                     f"source{'' if need == 1 else 's'}." if need else "")),
        wrap_up=("This is your last step, and you cannot call tools in it. Write "
                 "the report now from the sources you have recorded, cite them by "
                 "number, and say plainly what you could not establish."),
    )


def cowork(depth: str | None = None, *, ask: Asker | None = None,
           documents: bool = True, python: bool = False,
           helpers: int | None = None, max_steps: int | None = None,
           max_questions: int = 5) -> Mode:
    """A task handed over and carried through: a plan, a workspace, files back.

    Args:
        depth: `fast` (a short task, alone), `balanced`, or `deep` (a long task,
            with helpers working in parallel).
        ask: `ask(question, options)` — sync or async — answers the agent's
            questions. Without it the agent cannot ask, and is told to assume.
        documents: give it `parse_document` for PDFs, DOCX, CSV and the rest.
        python: give it `run_python` in the workspace. Every run asks for
            approval, so the harness needs an approver.
        helpers: how many specialists it may spin up to work in parallel.
        max_steps: the step ceiling, if the depth's own is not right.
        max_questions: how many times it may ask in one run.
    """
    level, tier = _depth(depth)
    table = {"fast": (25, 0), "balanced": (60, 3), "deep": (120, 8)}
    steps, parallel = table[level]
    return Mode(
        name="cowork", depth=level, tier=tier,
        description="A task carried through in a workspace, with files handed back.",
        max_steps=max_steps or steps,
        helpers=parallel if helpers is None else helpers,
        workspace=True, conversational=True, todos=True,
        ask=ask, max_questions=max_questions, documents=documents, python=python,
        collect_files=True,
        opening=("You are working alongside a person on a task they have handed "
                 "over. Carry it through to a finished result — do not hand back a "
                 "plan and stop."),
        guidance=(
            ("", "Lay out the steps with `todo_write` before you take the first "
                 "one, and keep the list true as you go: mark an item in progress "
                 "when you start it and done when it is done, and add what you "
                 "discover along the way."),
            ("fs_write", "Work in the workspace. What you were given is there "
                         "(`fs_list`, `fs_read`), and what you produce goes there "
                         "(`fs_write`). A deliverable is a file, not a paragraph "
                         "describing one."),
            ("parse_document", "`parse_document` reads PDFs, Word files, "
                               "spreadsheets and the like."),
            ("run_python", "`run_python` is for anything that should be computed "
                           "rather than estimated."),
            ("shell", "`shell` runs commands in the workspace: install what you "
                      "need, run what you wrote, and read the output rather than "
                      "assuming it worked."),
            ("spawn_agent", "Parts that do not depend on each other can run side "
                            "by side: `spawn_agent` one specialist per part, in the "
                            "same turn, and check what each hands back."),
            ("ask_user", "`ask_user` is for a decision that is the person's to "
                         "make and that changes what you do next. Do not ask for "
                         "what you can look up or reasonably assume."),
            ("!ask_user", "You cannot ask questions in this run. Where something "
                          "is unclear, make the reasonable assumption and say what "
                          "you assumed."),
            ("", "If an action is refused or needs an approval you do not get, do "
                 "not work around it. Say what was blocked."),
            ("", "Check your own work before you call it done: re-read what you "
                 "wrote, and run what can be run."),
        ),
        closing=("Finish with a short hand-over: what you did, the files that hold "
                 "it, what you assumed, and anything still open."),
        wrap_up=("This is your last step, and you cannot call tools in it. Hand "
                 "over now: what is finished, which files hold it, and exactly "
                 "what is still open."),
    )


#: The modes an agent can be built in, by name.
MODES: dict[str, Callable[..., Mode]] = {
    "chat": chat,
    "research": research,
    "cowork": cowork,
}


def register_mode(name: str, factory: Callable[..., Mode]) -> None:
    """Add a mode of your own. `factory(depth)` returns the `Mode`."""
    key = str(name).strip().lower()
    if not key:
        raise ConfigurationError("a mode needs a name")
    MODES[key] = factory


def resolve_mode(mode: str | Mode | None, depth: str | None = None) -> Mode | None:
    """Whatever `Agent(mode=..., depth=...)` was given, as a `Mode` or None."""
    if mode is None or mode == "":
        if depth is not None:
            raise ConfigurationError(
                f"depth={depth!r} needs a mode — pass mode=\"chat\", \"research\" "
                "or \"cowork\"")
        return None
    if isinstance(mode, Mode):
        if depth is None or _depth(depth)[0] == mode.depth:
            return mode
        raise ConfigurationError(
            f"this {mode.name} mode was built at depth {mode.depth!r}; pass the "
            f"depth to the mode itself — modes.{mode.name}({depth!r})")
    if isinstance(mode, dict):
        options = dict(mode)
        name = options.pop("name", None)
        if not name:
            raise ConfigurationError("a mode given as a mapping needs a `name`")
        factory = _factory(name)
        level = options.pop("depth", depth)
        try:
            return factory(level, **options)
        except TypeError as exc:
            raise ConfigurationError(f"{name} mode: {exc}") from None
    if not isinstance(mode, str):
        raise ConfigurationError(
            f"mode must be a name, a Mode or a mapping — got {type(mode).__name__}")
    return _factory(mode)(depth)


def _factory(name: str) -> Callable[..., Mode]:
    key = str(name).strip().lower().replace(" ", "_")
    key = _MODE_ALIASES.get(key, key)
    if key not in MODES:
        raise ConfigurationError(
            f"unknown mode {name!r}; known: {', '.join(sorted(MODES))}")
    return MODES[key]
