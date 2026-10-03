"""Hand an agent files: it sends each to the model as the model can take it.

A spreadsheet and a note become text. An image goes as an image. A recording
goes as audio to a model that listens, and is transcribed first for one that
does not.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from _common import pick_provider

from agent_harness import Agent, Harness, attach
from agent_harness.voice import FakeSpeech, tone, wav


async def main() -> None:
    provider, model = pick_provider([
        "Ada leads with 9; the chart agrees; the caller wants a refund."])
    folder = Path(tempfile.mkdtemp())
    (folder / "scores.csv").write_text("name,score\nada,9\nbob,7\n")
    (folder / "chart.png").write_bytes(b"\x89PNG\r\n\x1a\n" + bytes(64))
    (folder / "call.wav").write_bytes(wav(tone(300), 16_000))

    # Something to transcribe with, for models that cannot listen.
    harness = Harness(speech=FakeSpeech(["I would like a refund, please."]))
    if provider is not None:
        harness.provider = provider
        provider.modalities = frozenset({"image"})      # as a text-and-vision model
    agent = Agent("reader", "Answer from what you are given.", model=model,
                  harness=harness)

    for name in ("scores.csv", "chart.png", "call.wav"):
        block = attach(folder / name)
        native = agent.provider.accepts(block, agent.model)
        print(f"{name:<12} {block.type:<9} "
              f"{'sent as it is' if native else 'turned into text first'}")

    result = await agent.run(
        "Who leads, does the chart agree, and what did the caller want?",
        attachments=[folder / "scores.csv", folder / "chart.png", folder / "call.wav"])
    print(f"\n{result.output or result.error}")

    if provider is not None:
        sent = provider.requests[0].messages[0].content
        print("\nwhat the model was sent:")
        for block in sent:
            text = getattr(block, "text", "")
            print(f"  {block.type:<6} {text.splitlines()[0][:60] if text else block}")


if __name__ == "__main__":
    asyncio.run(main())
