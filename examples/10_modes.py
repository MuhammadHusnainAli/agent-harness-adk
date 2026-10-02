"""Three ways of working — chat, research, cowork — and how deep each one goes."""

from __future__ import annotations

import asyncio
import tempfile

from _common import pick_provider

from agent_harness import Agent, FakeProvider, Harness, Workspace, modes, tool_call
from agent_harness.toolkits import make_corpus_search

NOTES = {
    "q3-results.md": "Q3 revenue was 4.2M, up 12% on the year. Costs were 3.1M.",
    "q4-outlook.md": "Guidance for Q4 is flat: revenue between 4.1M and 4.3M.",
    "board-minutes.md": "The board approved the Q4 hiring freeze on 2 October.",
}


def todos(*items: tuple[str, str]):
    return tool_call("todo_write",
                     todos=[{"content": c, "status": s} for c, s in items])


# Only used when there is no API key: what a model would do in each mode.
CHAT = ["Noted — you are planning the Q4 review.", "The Q4 review."]
RESEARCH = [
    todos(("How did Q3 go?", "in_progress"), ("What is expected for Q4?", "pending")),
    [tool_call("search_corpus", query="Q3 revenue costs"),
     tool_call("search_corpus", query="Q4 guidance")],
    [tool_call("record_source", ref="q3-results.md",
               finding="Q3 revenue 4.2M (+12%), costs 3.1M"),
     tool_call("record_source", ref="q4-outlook.md",
               finding="Q4 guidance flat, 4.1M to 4.3M")],
    todos(("How did Q3 go?", "done"), ("What is expected for Q4?", "done")),
    "Q3 revenue was 4.2M, up 12%, against costs of 3.1M [1]. Q4 is guided flat, "
    "between 4.1M and 4.3M [2].",
]
COWORK = [
    todos(("Read the notes", "in_progress"), ("Write the briefing", "pending")),
    tool_call("fs_read", path="notes.txt"),
    tool_call("ask_user", question="Which format should the briefing be in?",
              options=["Markdown", "Plain text"]),
    tool_call("fs_write", path="briefing.md",
              content="# Q4 briefing\n\n- Revenue guided flat.\n- Hiring is frozen.\n"),
    todos(("Read the notes", "done"), ("Write the briefing", "done")),
    "The briefing is in briefing.md. I assumed it is for the board.",
]


async def main() -> None:
    provider, model = pick_provider()
    scripted = provider is not None

    def harness(script: list) -> Harness:
        return Harness(provider=FakeProvider(script)) if scripted else Harness()

    # --- chat: the second run is the second turn ----------------------------
    pal = Agent("pal", mode="chat", depth="fast", model=model, harness=harness(CHAT))
    await pal.run("I am planning the Q4 review.")
    reply = await pal.run("What am I planning?")
    print(f"chat      {reply.output}")

    # --- research: every claim leads back to something it read ---------------
    analyst = Agent(
        "analyst", mode="research", depth="fast", model=model,
        tools=[make_corpus_search(NOTES)], harness=harness(RESEARCH),
    )
    report = await analyst.run("How did Q3 go, and what is expected for Q4?")
    print(f"\nresearch  {report.output}")
    print(f"          {len(report.sources)} sources · "
          f"unmet: {report.violations or 'nothing'}")

    # --- cowork: a task carried through, and the files handed back -----------
    async def answer(question: str, options: list[str]) -> str:
        print(f"\n          (asked: {question} → {options[0]})")
        return options[0] if options else "Your call."

    with tempfile.TemporaryDirectory() as folder:
        workspace = Workspace(folder)
        workspace.write("notes.txt", "\n".join(NOTES.values()))
        colleague = Agent(
            "colleague", mode=modes.cowork("fast", ask=answer), model=model,
            workspace=workspace, harness=harness(COWORK),
        )
        done = await colleague.run("Turn notes.txt into a short Q4 briefing.")
        print(f"cowork    {done.output}")
        for item in done.todos:
            print(f"          {item.line()}")
        print(f"          files: {[a.name for a in done.artifacts]}")


if __name__ == "__main__":
    asyncio.run(main())
