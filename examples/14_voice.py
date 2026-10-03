"""A voice agent: it hears when you speak, answers aloud, and stops when you
speak over it.

Runs with no key and no microphone: the "speech" is a tone, the transcriber is
scripted, and the voice is a stand-in whose audio is as long as the words — so
the timing, the turn-taking and the interruption are the real thing.

With a microphone:  agent-harness voice              (pip install sounddevice)
From a recording:   agent-harness voice --input question.wav --output answer.wav
"""

from __future__ import annotations

import asyncio

from _common import pick_provider

from agent_harness import Agent, FakeProvider, Harness, tool, tool_call
from agent_harness.voice import FakeSpeech, VoiceAgent, duration_ms, silence, tone

RATE = 16_000


@tool
def find_booking(name: str) -> str:
    """Find a restaurant booking.

    Args:
        name: who the booking is under.
    """
    return f"{name}: a table for two at eight, by the window"


def said(ms: int) -> bytes:
    """Someone saying something, with the quiet around it."""
    return silence(250, RATE, noise=40) + tone(ms, RATE) + silence(700, RATE, noise=40)


async def microphone(*parts):
    """Audio in 20 ms chunks. An Event in the list waits for the test to set it."""
    for part in parts:
        if isinstance(part, asyncio.Event):
            await part.wait()
            continue
        for offset in range(0, len(part), 640):
            yield part[offset:offset + 640]
            await asyncio.sleep(0.001)


async def main() -> None:
    pick_provider()                # prints the notice; the script below is the model
    harness = Harness(provider=FakeProvider([
        tool_call("find_booking", name="Ada"),
        "You have a table for two at eight, by the window.",
        "Of course. The kitchen opens at six, and the full menu runs until ten. "
        "There is a tasting menu as well, which takes about two hours. "
        "And on Fridays there is live music from nine.",
        "Sorry — go ahead.",
    ], stream_words=True, stream_delay=0.002))
    agent = Agent("concierge", "Help callers with their bookings.", mode="voice",
                  tools=[find_booking], harness=harness)
    speech = FakeSpeech(["Do I have a booking? It is Ada.",
                         "Tell me everything about the evening.",
                         "Actually, never mind."], delay=0.004)
    voice = VoiceAgent(agent, speech=speech, tool_filler="One moment.",
                       greeting="Hello, how can I help?")

    # The caller waits for the first answer, then talks over the second.
    answered, answering = asyncio.Event(), asyncio.Event()
    turn, audio, chunks = 0, 0, 0
    async for event in voice.run(microphone(
            said(900), answered, said(800), answering, said(600))):
        if event.type == "transcript":
            turn, chunks = turn + 1, 0
            print(f"\nyou   › {event.text}")
        elif event.type == "tool":
            print(f"        (looking up: {event.text})")
        elif event.type == "audio":
            audio += len(event.audio)
            if event.text:
                print(f"agent › {event.text}")
            chunks += 1
            if turn == 2 and chunks == 40:
                answering.set()                 # four seconds in: speak over it
        elif event.type == "interrupted":
            print("        — interrupted —")
        elif event.type == "turn_end" and event.data.get("first_audio_ms"):
            answered.set()
            print(f"        [first sound after {event.data['first_audio_ms']:.0f} ms, "
                  f"{event.data['total_ms']:.0f} ms in all]")

    print(f"\n{duration_ms(b'0' * audio, RATE) / 1000:.1f}s of speech · "
          f"{len(voice.history)} messages kept · session {voice.session_id}")
    print("what the agent remembers of the answer it was cut off in:")
    cut = next(m.text for m in voice.history if "interrupted here" in m.text)
    print(f"  {cut}")
    await harness.aclose()


if __name__ == "__main__":
    asyncio.run(main())
