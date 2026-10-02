"""Chats in a database: owned, listed, continued by another process, never lost.

Uses SQLite so it runs anywhere. Change the URL to `postgresql://…`,
`mongodb://…`, `redis://…` or `azure://container` and nothing else changes.
"""

from __future__ import annotations

import asyncio
import tempfile

from _common import pick_provider

from agent_harness import Agent, ConfigurationError, FakeProvider, Harness

ADA = {"user_id": "ada", "tenant_id": "acme"}
BOB = {"user_id": "bob", "tenant_id": "acme"}


async def main() -> None:
    provider, model = pick_provider()
    scripted = provider is not None
    url = f"sqlite:///{tempfile.mkdtemp()}/chats.db"

    def replica(script: list, who: dict) -> Agent:
        """One process of several serving the same chats."""
        harness = (Harness.on(url, provider=FakeProvider(script)) if scripted
                   else Harness.on(url))
        return Agent("support", "Answer briefly.", mode="chat", model=model,
                     harness=harness, trace=who)

    report = await Harness.on(url).sessions.check()
    print(f"{report['store']}: "
          + ", ".join(f"{s['step']} {'ok' if s['ok'] else 'FAILED'}"
                      for s in report["steps"]))

    # --- one process starts a chat for Ada ---------------------------------
    first = replica(["Noted: order 4182."], ADA)
    started = await first.run("My order number is 4182.")
    await first.harness.aclose()

    # --- another continues it, by its id ---------------------------------------
    second = replica(["Your order is 4182."], ADA)
    reply = await second.run("What was my order number?", session=started.session_id)
    print(f"\ncontinued  {reply.output}")

    # --- it is Ada's: listed for her, and not there at all for Bob ---------------
    for chat in await second.harness.sessions.list(**ADA):
        print(f"ada's      {chat.id}  v{chat.version}  {len(chat.messages)} messages"
              f"  “{chat.title}”")
    bob = replica(["never asked"], BOB)
    try:
        await bob.run("What was her order number?", session=started.session_id)
    except ConfigurationError as refused:
        print(f"bob        {refused}")

    # --- two requests on one chat: both turns are kept ----------------------------
    one = replica(["Tomorrow."], ADA)
    two = replica(["By courier."], ADA)
    await asyncio.gather(one.run("When does it arrive?", session=started.session_id),
                         two.run("How is it shipped?", session=started.session_id))
    final = await one.harness.sessions.load(started.session_id)
    print(f"\nboth kept  {len(final.messages)} messages, version {final.version}:")
    for message in final.messages[-4:]:
        print(f"           {message.role}: {message.text}")


if __name__ == "__main__":
    asyncio.run(main())
