git add -A && git commit -m "feat: handoffs — another agent takes over the conversation""""Voice: agents you talk to.

Two ways to build one, and they are used the same way.

    from agent_harness import Agent
    from agent_harness.voice import OpenAISpeech, RealtimeAgent, VoiceAgent

    agent = Agent("concierge", "Help callers with their bookings.", mode="voice",
                  tools=[find_booking])

    # Listen, think, speak — any model, every rail
    voice = VoiceAgent(agent, speech=OpenAISpeech())

    # One speech-to-speech model — the lowest latency
    voice = RealtimeAgent(agent, provider="openai")        # or "gemini"

    async for event in voice.run(microphone()):
        if event.type == "audio":         speaker.write(event.audio)
        elif event.type == "interrupted": speaker.flush()

| | `VoiceAgent` | `RealtimeAgent` |
|---|---|---|
| how | speech → text → your agent → speech | audio straight into a realtime model |
| models | any of the twenty providers | OpenAI Realtime, Gemini Live |
| time to first sound | a transcription and a first sentence | lowest there is |
| budgets, guardrails, memory, audit | all of them, unchanged | tools run under every rail; what is *said* is checked after it is said |
| turn-taking | here: `EnergyVAD`, or your own | the model's own |

Nothing here needs a package beyond what the harness already depends on.
"""

from .audio import (
    Resampler,
    duration_ms,
    resample,
    rms,
    silence,
    tone,
    ulaw_decode,
    ulaw_encode,
    unwav,
    wav,
)
from .pipeline import VoiceAgent, VoiceEvent
from .realtime import GeminiLive, OpenAIRealtime, RealtimeAgent
from .speech import FakeSpeech, OpenAISpeech, SpeechToText, TextToSpeech
from .text import SpeechChunker, speakable
from .vad import VAD, EnergyVAD, VADEvent
from .ws import ConnectionClosed, WebSocket

__all__ = [
    "VoiceAgent",
    "RealtimeAgent",
    "VoiceEvent",
    "OpenAIRealtime",
    "GeminiLive",
    "OpenAISpeech",
    "FakeSpeech",
    "SpeechToText",
    "TextToSpeech",
    "VAD",
    "EnergyVAD",
    "VADEvent",
    "SpeechChunker",
    "speakable",
    "WebSocket",
    "ConnectionClosed",
    "Resampler",
    "resample",
    "rms",
    "duration_ms",
    "wav",
    "unwav",
    "ulaw_decode",
    "ulaw_encode",
    "tone",
    "silence",
]
