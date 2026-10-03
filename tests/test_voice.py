"""Voice: hearing when someone speaks, answering aloud, and stopping when
spoken over — with the agent behind it unchanged.

Everything here runs offline. Speech is a scripted stand-in, audio is tones and
silence, and the realtime models are played by a scripted socket that speaks
their protocol. One test drives the WebSocket client against a real server
written here; another, against the `websockets` library if it is installed.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import struct
from typing import Any

import httpx
import pytest

from agent_harness import (
    Agent,
    Budget,
    ConfigurationError,
    FakeProvider,
    Harness,
    HookEngine,
    PermissionDenied,
    ProviderError,
    tool,
    tool_call,
)
from agent_harness.voice import (
    ConnectionClosed,
    EnergyVAD,
    FakeSpeech,
    GeminiLive,
    OpenAIRealtime,
    OpenAISpeech,
    RealtimeAgent,
    Resampler,
    SpeechChunker,
    VoiceAgent,
    WebSocket,
    duration_ms,
    resample,
    rms,
    silence,
    speakable,
    tone,
    ulaw_decode,
    ulaw_encode,
    unwav,
    wav,
)
from agent_harness.voice.audio import samples

RATE = 16_000


# --- audio ---------------------------------------------------------------------------

def test_loudness_and_length_are_measured():
    assert rms(silence(100)) == 0
    assert 6000 < rms(tone(100, amplitude=9000)) < 6700        # a sine: peak / √2
    assert duration_ms(tone(250), RATE) == 250
    assert rms(b"") == 0 and rms(b"\x01") == 0                  # an odd byte is dropped


def test_resampling_keeps_the_length_and_the_loudness():
    original = tone(1000, RATE)
    up = resample(original, RATE, 24_000)
    back = resample(up, 24_000, RATE)
    assert abs(len(up) // 2 - 24_000) <= 2 and abs(len(back) // 2 - 16_000) <= 2
    assert abs(rms(up) - rms(original)) < 20 and abs(rms(back) - rms(original)) < 20
    assert resample(original, RATE, RATE) is original


def test_a_stream_resampled_in_chunks_has_no_seams():
    """Chunk by chunk must give what the whole gives — a join that clicks is
    heard fifty times a second."""
    source = tone(600, 24_000, frequency=310)
    whole = samples(resample(source, 24_000, RATE))
    stream = Resampler(24_000, RATE)
    pieces = b"".join(stream.feed(source[i:i + 963]) for i in range(0, len(source), 963))
    chunked = samples(pieces)
    assert abs(len(chunked) - len(whole)) <= 1
    # No sample jumps further than the tone itself ever moves between samples.
    steps = [abs(b - a) for a, b in zip(chunked, chunked[1:], strict=False)]
    assert max(steps) < 1500
    assert max(abs(a - b) for a, b in zip(whole, chunked, strict=False)) < 60


def test_wav_and_mulaw_round_trip():
    pcm = tone(120, 8000)
    back, rate = unwav(wav(pcm, 8000))
    assert (back, rate) == (pcm, 8000)
    decoded = ulaw_decode(ulaw_encode(pcm))
    assert len(decoded) == len(pcm)
    # Lossy by design, but a voice survives it.
    assert abs(rms(decoded) - rms(pcm)) < 60
    assert len(ulaw_encode(pcm)) == len(pcm) // 2


# --- hearing when someone speaks ---------------------------------------------------------

def hear(vad: EnergyVAD, audio: bytes, chunk: int = 333) -> list:
    events = []
    for offset in range(0, len(audio), chunk):
        events += vad.feed(audio[offset:offset + chunk])
    return events + vad.flush()


def test_an_utterance_is_found_between_the_silences():
    vad = EnergyVAD(RATE)
    events = hear(vad, silence(400, noise=60) + tone(900) + silence(800, noise=60))
    assert [e.kind for e in events] == ["start", "end"]
    start, end = events
    assert start.at_ms == 400
    # Declared over half a second after the last sound, not before.
    assert end.at_ms == 400 + 900 + 500
    # The audio handed on has the lead-in kept and the long tail dropped.
    assert 1100 <= duration_ms(end.audio, RATE) <= 1260


def test_a_click_is_not_a_turn_and_says_so():
    vad = EnergyVAD(RATE)
    # Too short even to count as having started: nothing at all.
    assert hear(vad, silence(300) + tone(60) + silence(700)) == []
    # Long enough to start, too short to be speech: started, then taken back.
    vad.reset()
    events = hear(vad, silence(300) + tone(120) + silence(700))
    assert [e.kind for e in events] == ["start", "cancel"]


def test_a_pause_inside_a_sentence_does_not_end_it():
    vad = EnergyVAD(RATE, silence_ms=500)
    events = hear(vad, silence(200) + tone(400) + silence(300) + tone(400) + silence(700))
    assert [e.kind for e in events] == ["start", "end"]
    assert duration_ms(events[1].audio, RATE) > 1100

    quick = EnergyVAD(RATE, silence_ms=200)
    assert [e.kind for e in hear(quick, silence(200) + tone(400) + silence(300)
                                 + tone(400) + silence(700))] == [
        "start", "end", "start", "end"]


def test_the_room_is_learned_so_a_hum_is_not_speech():
    vad = EnergyVAD(RATE)
    hum = tone(1500, amplitude=600, frequency=100)            # a steady fan
    assert hear(vad, hum) == []
    assert vad.threshold > 1000                               # the bar has risen
    # A voice over the hum is still a voice.
    vad.reset()
    voice = samples(tone(700, amplitude=600, frequency=100))
    over = samples(tone(700, amplitude=9000))
    mixed = bytes(b for pair in ((a + b).to_bytes(2, "little", signed=True)
                                 for a, b in zip(voice, over, strict=True)) for b in pair)
    events = hear(vad, hum + mixed + hum[:len(hum) // 2 * 2])
    assert [e.kind for e in events] == ["start", "end"]


def test_nobody_talks_for_ever_and_a_closing_stream_ends_the_turn():
    capped = EnergyVAD(RATE, max_utterance_ms=1000)
    events = hear(capped, silence(200) + tone(2600))
    assert [e.kind for e in events][:2] == ["start", "end"]
    assert duration_ms(events[1].audio, RATE) <= 1000

    open_ended = EnergyVAD(RATE)
    events = hear(open_ended, silence(200) + tone(600))        # no silence after
    assert [e.kind for e in events] == ["start", "end"]

    with pytest.raises(ValueError):
        EnergyVAD(0)


# --- what gets said aloud ------------------------------------------------------------------

def spoken(text: str, size: int = 7) -> list[str]:
    chunker = SpeechChunker()
    out: list[str] = []
    for offset in range(0, len(text), size):
        out += chunker.feed(text[offset:offset + size])
    return out + chunker.flush()


def test_sentences_are_handed_over_as_they_complete_and_the_first_one_sooner():
    assert spoken("Sure, I can help with that order, it shipped on Thursday. "
                  "The total was 4.2 million dollars. Dr. Smith signed for it.") == [
        "Sure, I can help with that order,",       # the first piece ends at a clause
        "it shipped on Thursday.",
        "The total was 4.2 million dollars.",      # a decimal point is not a full stop
        "Dr. Smith signed for it.",                # nor is an abbreviation's
    ]


def test_what_cannot_be_said_is_not_said():
    assert speakable("**Done.** See [the report](https://x.example/r) or "
                     "https://x.example/raw — `code` here.") == (
        "Done. See the report or the link — code here.")
    assert spoken("Here you go:\n```python\nprint(1)\n```\nThat prints one.") == [
        "Here you go:", "That prints one."]
    assert spoken("- first thing\n- second thing\n") == ["first thing", "second thing"]
    assert spoken("...") == [] and spoken("") == []


def test_a_sentence_with_no_end_is_still_spoken():
    long = "word " * 120
    pieces = spoken(long)
    assert len(pieces) > 2 and all(len(p) <= 240 for p in pieces)
    assert " ".join(pieces).split() == long.split()
    # A model that streams one short answer and stops.
    assert spoken("Yes.") == ["Yes."]


# --- speech over HTTP --------------------------------------------------------------------------

async def test_openai_speech_sends_what_the_api_expects():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path.endswith("/audio/transcriptions"):
            return httpx.Response(200, json={"text": "  where is my order  "})
        # An odd-sized first chunk: a sample split across two packets.
        return httpx.Response(200, content=b"\x01\x02\x03" + b"\x04\x05\x06\x07\x08")

    speech = OpenAISpeech("key-1", voice="marin", instructions="Warm.", speed=1.1,
                          language="en",
                          client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))

    assert await speech.transcribe(wav(tone(200), RATE)) == "where is my order"
    stt = seen[0]
    assert stt.headers["authorization"] == "Bearer key-1"
    body = stt.read()
    assert b'name="model"' in body and b"gpt-4o-mini-transcribe" in body
    assert b'name="language"' in body and b'filename="audio.wav"' in body
    assert b"RIFF" in body

    audio = b"".join([chunk async for chunk in speech.synthesize("Hello there.")])
    assert audio == b"\x01\x02\x03\x04\x05\x06\x07\x08"
    assert len(audio) % 2 == 0
    tts = json.loads(seen[1].read())
    assert tts == {"model": "gpt-4o-mini-tts", "voice": "marin", "input": "Hello there.",
                   "response_format": "pcm", "instructions": "Warm.", "speed": 1.1}
    assert speech.rate == 24_000
    await speech.aclose()


async def test_openai_speech_retries_what_might_pass_and_reports_what_will_not(
        monkeypatch):
    calls = {"n": 0}

    def flaky(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(503, text="busy")
        return httpx.Response(200, json={"text": "ok"})

    speech = OpenAISpeech("k", client=httpx.AsyncClient(
        transport=httpx.MockTransport(flaky)))
    assert await speech.transcribe(b"RIFF") == "ok" and calls["n"] == 2

    bad = OpenAISpeech("k", client=httpx.AsyncClient(transport=httpx.MockTransport(
        lambda r: httpx.Response(400, text="unsupported format"))))
    with pytest.raises(ProviderError, match="400: unsupported format"):
        await bad.transcribe(b"x")
    with pytest.raises(ProviderError, match="speech synthesis failed with 400"):
        [c async for c in bad.synthesize("hi")]

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ConfigurationError, match="needs an API key"):
        await OpenAISpeech().transcribe(b"x")


# --- the voice agent ------------------------------------------------------------------------------

def said(ms: int = 700) -> bytes:
    """Someone saying something, with the quiet around it."""
    return silence(200, noise=40) + tone(ms) + silence(700, noise=40)


async def microphone(*audio: Any, chunk_ms: int = 20, pace: float = 0.0):
    """Audio a chunk at a time. An `asyncio.Event` in the list waits for it."""
    size = RATE * chunk_ms // 1000 * 2
    for item in audio:
        if isinstance(item, asyncio.Event):
            await item.wait()
            continue
        for offset in range(0, len(item), size):
            yield item[offset:offset + size]
            await asyncio.sleep(pace)


def concierge(script: list, **kw: Any) -> tuple[Agent, FakeProvider]:
    provider = FakeProvider(script, stream_words=True, **kw)
    agent = Agent("concierge", "Help callers.", mode="voice",
                  harness=Harness.testing(provider), memory=False)
    return agent, provider


async def test_a_spoken_question_gets_a_spoken_answer():
    agent, provider = concierge(["Your order shipped on Thursday. It arrives Monday."])
    speech = FakeSpeech(["Where is my order?"])
    voice = VoiceAgent(agent, speech=speech)

    events = [e async for e in voice.run(microphone(said()))]

    kinds = [e.type for e in events]
    assert kinds[:3] == ["speech_started", "speech_stopped", "transcript"]
    assert kinds[-1] == "turn_end" and "text" in kinds and "audio" in kinds
    assert events[2].text == "Where is my order?"
    assert "".join(e.text for e in events if e.type == "text") == (
        "Your order shipped on Thursday. It arrives Monday.")
    # Spoken a sentence at a time, in order, each as it was ready.
    assert speech.spoken == ["Your order shipped on Thursday.", "It arrives Monday."]
    labelled = [e.text for e in events if e.type == "audio" and e.text]
    assert labelled == speech.spoken
    audio = b"".join(e.audio for e in events if e.type == "audio")
    assert duration_ms(audio, RATE) == pytest.approx(
        sum(len(s) for s in speech.spoken) * 60, abs=5)
    # What the transcriber was given is a WAV of the utterance.
    heard, rate = unwav(speech.heard[0])
    assert rate == RATE and 800 < duration_ms(heard, RATE) < 1200

    end = events[-1]
    assert end.text == "Your order shipped on Thursday. It arrives Monday."
    assert end.data["interrupted"] is False
    for mark in ("stt_ms", "first_token_ms", "first_audio_ms", "total_ms"):
        assert end.data[mark] is not None and end.data[mark] >= 0
    assert end.data["first_audio_ms"] <= end.data["total_ms"]
    assert voice.latency["turns"] == 1
    assert agent.harness.health.component("voice.first_audio", "voice").calls == 1
    assert provider.requests[0].system.count("Write for the ear") == 1


async def test_the_first_sound_does_not_wait_for_the_last_word():
    """The answer starts being spoken while the model is still writing it."""
    agent, _ = concierge(
        ["Right, let me check that for you. " + "And then there is more to say. " * 6],
        stream_delay=0.004)
    voice = VoiceAgent(agent, speech=FakeSpeech(["Check my order."]))

    order: list[str] = []
    async for event in voice.respond("Check my order."):
        if event.type in ("text", "audio") and (not order or order[-1] != event.type):
            order.append(event.type)

    first_audio = order.index("audio")
    assert "text" in order[first_audio + 1:], order     # text kept coming after it
    turn = voice.turns[-1]
    assert turn["first_audio_ms"] < turn["total_ms"] / 2


async def test_the_conversation_is_kept_and_the_agent_keeps_its_tools():
    @tool
    def find_booking(name: str) -> str:
        """Find a booking.

        Args:
            name: who it is under.
        """
        return f"{name}: table for two at eight"

    provider = FakeProvider([
        tool_call("find_booking", name="Ada"),
        "You have a table for two at eight.",
        "It is under Ada.",
    ], stream_words=True)
    harness = Harness.testing(provider)
    agent = Agent("concierge", mode="voice", tools=[find_booking], harness=harness,
                  memory=False, trace={"user_id": "ada"})
    speech = FakeSpeech(["Do I have a booking? It is Ada.", "What name is it under?"])
    voice = VoiceAgent(agent, speech=speech, tool_filler="One moment.")

    events = [e async for e in voice.run(microphone(said(), said()))]

    assert [e.text for e in events if e.type == "tool"] == ["find_booking"]
    # Something is said while the lookup runs, so it is not a silence.
    assert speech.spoken == ["One moment.", "You have a table for two at eight.",
                             "It is under Ada."]
    ends = [e for e in events if e.type == "turn_end"]
    assert [e.data["steps"] for e in ends] == [2, 1]
    # The second turn was asked with the first — tool call and all — behind it.
    second = provider.requests[2].messages
    assert second[0].text == "Do I have a booking? It is Ada."
    assert second[-1].text == "What name is it under?"
    assert any(b.type == "tool_result" for m in second for b in m.content)
    # And it is a session like any other: owned, saved, resumable.
    saved = await harness.sessions.load(voice.session_id)
    assert saved.user_id == "ada" and saved.title == "Do I have a booking? It is Ada."
    assert [m.text for m in saved.messages if m.text][-1] == "It is under Ada."
    assert any(e.action == "tool_call" for e in harness.audit.entries)

    again = VoiceAgent(agent, speech=FakeSpeech([]), session=voice.session_id)
    await again._open()
    assert len(again.history) == len(saved.messages)


async def test_speaking_over_an_answer_stops_it_and_only_what_was_heard_is_kept():
    agent, provider = concierge([
        "The first thing to know is this. " + "Then there is a great deal more. " * 8,
        "Sorry — go ahead.",
    ])
    speech = FakeSpeech(["Tell me everything.", "Actually, stop."], delay=0.01)
    voice = VoiceAgent(agent, speech=speech)
    speaking = asyncio.Event()

    events = []
    async for event in voice.run(microphone(
            said(), speaking, said(500), pace=0.001)):
        events.append(event)
        if event.type == "audio" and sum(e.type == "audio" for e in events) == 6:
            speaking.set()                 # now talk over it

    kinds = [e.type for e in events]
    cut = kinds.index("interrupted")
    assert kinds[cut - 1] != "turn_end"
    # The rest of that answer was never sent: far less audio than it would be.
    first_answer_audio = sum(len(e.audio) for e in events[:cut] if e.type == "audio")
    assert duration_ms(b"\0" * first_answer_audio, RATE) < 6000
    # The second turn is answered normally.
    assert [e.text for e in events if e.type == "transcript"] == [
        "Tell me everything.", "Actually, stop."]
    assert events[-1].type == "turn_end" and events[-1].text == "Sorry — go ahead."

    # The agent is told what the person heard, not what it wrote.
    second = provider.requests[1].messages
    assert second[0].text == "Tell me everything."
    assert second[1].role == "assistant"
    assert second[1].text.startswith("The first thing to know is this.")
    assert second[1].text.endswith("[the user interrupted here; the rest was not heard]")
    assert len(second[1].text) < 300
    assert second[2].text == "Actually, stop."
    assert voice.turns[0]["interrupted"] and not voice.turns[1]["interrupted"]
    assert any(e.action == "voice_interrupted" for e in agent.harness.audit.entries)
    # Nothing is left running behind it.
    assert agent.harness.control.running_agents == []


async def test_a_pause_that_was_mid_sentence_is_answered_once():
    """They carried on before anything was said back: one turn, not two."""
    agent, provider = concierge(["Paris is sunny."])
    speech = FakeSpeech(lambda audio: "weather in Paris" if len(audio) > 40_000
                        else "weather", stt_delay=0.15)
    voice = VoiceAgent(agent, speech=speech)

    events = [e async for e in voice.run(microphone(said(500), said(500), pace=0.0005))]

    transcripts = [e.text for e in events if e.type == "transcript"]
    assert transcripts == ["weather in Paris"]
    assert "interrupted" not in [e.type for e in events]
    assert [e.text for e in events if e.type == "turn_end"] == ["Paris is sunny."]
    # Both halves were transcribed together, as one recording, and the model
    # was asked once.
    assert len(speech.heard[-1]) > 1.9 * len(speech.heard[0])
    assert len(provider.requests) == 1


async def test_without_barge_in_what_is_said_meanwhile_waits_its_turn():
    agent, _ = concierge(["First answer, in full.", "Second answer."])
    speech = FakeSpeech(["one", "two"], delay=0.005)
    voice = VoiceAgent(agent, speech=speech, barge_in=False)
    speaking = asyncio.Event()

    events = []
    async for event in voice.run(microphone(said(), speaking, said(), pace=0.001)):
        events.append(event)
        if event.type == "audio":
            speaking.set()

    assert "interrupted" not in [e.type for e in events]
    assert [e.text for e in events if e.type == "turn_end"] == [
        "First answer, in full.", "Second answer."]


async def test_silence_noise_and_failures_do_not_break_the_conversation():
    # Nothing intelligible was said: no turn, no model call.
    agent, provider = concierge(["unused"])
    voice = VoiceAgent(agent, speech=FakeSpeech(["   "]))
    events = [e async for e in voice.run(microphone(said()))]
    assert events[-1].data == {"skipped": "nothing was said"} and provider.requests == []

    # The model fails: the person hears an apology, not silence.
    agent, _ = concierge([ProviderError("the model is down", provider="fake")])
    speech = FakeSpeech(["hello"])
    voice = VoiceAgent(agent, speech=speech)
    events = [e async for e in voice.run(microphone(said()))]
    assert speech.spoken == ["Sorry, something went wrong on my side."]
    assert "the model is down" in next(e.text for e in events if e.type == "error")
    assert events[-1].type == "turn_end"

    # One sentence cannot be spoken: the rest still is.
    class Hoarse(FakeSpeech):
        async def synthesize(self, text: str):
            if "second" in text:
                raise ProviderError("voice unavailable", provider="fake")
            async for chunk in super().synthesize(text):
                yield chunk

    agent, _ = concierge(["The first sentence. The second sentence. The third one."])
    speech = Hoarse(["go"])
    events = [e async for e in VoiceAgent(agent, speech=speech).run(microphone(said()))]
    assert [e.text for e in events if e.type == "audio" and e.text] == [
        "The first sentence.", "The third one."]
    assert any("could not speak" in e.text for e in events if e.type == "error")

    # The transcriber fails: reported, and the next turn still works.
    class Deaf(FakeSpeech):
        async def transcribe(self, audio, **kw):
            if not self.heard:
                self.heard.append(audio)
                raise ProviderError("stt down", provider="fake")
            return await super().transcribe(audio, **kw)

    agent, _ = concierge(["Hello."])
    events = [e async for e in VoiceAgent(agent, speech=Deaf(["hi"])).run(
        microphone(said(), said()))]
    assert "stt down" in next(e.text for e in events if e.type == "error")
    assert events[-1].type == "turn_end" and events[-1].text == "Hello."


async def test_greeting_push_to_talk_and_the_rate_you_asked_for():
    agent, _ = concierge(["It is nine o'clock."])
    speech = FakeSpeech(["What time is it?"])
    voice = VoiceAgent(agent, speech=speech, greeting="Hello, how can I help?",
                       output_rate=8000)

    events = [e async for e in voice.run(microphone(said()))]
    assert events[0].type == "audio" and events[0].text == "Hello, how can I help?"
    assert all(e.data["rate"] == 8000 for e in events if e.type == "audio")
    spoken_ms = sum(len(s) for s in speech.spoken) * 60
    assert duration_ms(b"".join(e.audio for e in events if e.type == "audio"),
                       8000) == pytest.approx(spoken_ms, abs=10)
    assert voice.history[0].text == "Hello, how can I help?"

    # Push-to-talk: one recording in, one answer out, no detector involved.
    agent, _ = concierge(["Ten past nine."])
    voice = VoiceAgent(agent, speech=FakeSpeech(["And now?"]))
    kinds = [e.type async for e in voice.listen(tone(600))]
    assert kinds[0] == "transcript" and kinds[-1] == "turn_end"


def test_a_voice_agent_needs_ears_and_a_mouth():
    agent, _ = concierge(["x"])
    with pytest.raises(ConfigurationError, match="needs something to listen and speak"):
        VoiceAgent(agent)
    with pytest.raises(ConfigurationError, match="same rate"):
        VoiceAgent(agent, speech=FakeSpeech(), rate=8000, vad=EnergyVAD(16_000))
    # The harness's own is used when none is given.
    agent.harness.speech = FakeSpeech()
    assert VoiceAgent(agent).tts is agent.harness.speech


async def test_a_budget_still_stops_a_voice_agent():
    provider = FakeProvider(["A long answer. " * 20] * 3, stream_words=True)
    harness = Harness.testing(provider, budget=Budget(max_output_tokens=30))
    agent = Agent("concierge", mode="voice", harness=harness, memory=False)
    voice = VoiceAgent(agent, speech=FakeSpeech(["one", "two"]))

    events = [e async for e in voice.run(microphone(said(), said()))]

    ends = [e for e in events if e.type == "turn_end"]
    assert len(ends) == 2 and len(provider.requests) == 1     # the second was not made
    assert ends[1].data["budget_exceeded"] == "output_tokens"
    assert ends[1].text == "I have reached my limit for this conversation."


# --- the websocket client --------------------------------------------------------------------------

class Server:
    """The server side of RFC 6455, enough to hold the client to it."""

    def __init__(self, script: Any, *, status: str = "101 Switching Protocols",
                 accept: str | None = None) -> None:
        self.script, self.status, self.accept = script, status, accept
        self.request = ""
        self.unmasked = False
        self.frames: list[tuple[int, bytes]] = []

    async def __aenter__(self) -> Server:
        self.server = await asyncio.start_server(self._serve, "127.0.0.1", 0)
        self.port = self.server.sockets[0].getsockname()[1]
        self.url = f"ws://127.0.0.1:{self.port}/live?model=x"
        return self

    async def __aexit__(self, *exc: Any) -> None:
        self.server.close()
        await self.server.wait_closed()

    async def _serve(self, reader, writer) -> None:
        self.reader, self.writer = reader, writer
        self.request = (await reader.readuntil(b"\r\n\r\n")).decode()
        key = next(line.split(":", 1)[1].strip() for line in self.request.split("\r\n")
                   if line.lower().startswith("sec-websocket-key"))
        accept = self.accept or base64.b64encode(hashlib.sha1(
            (key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode()).digest()).decode()
        writer.write((f"HTTP/1.1 {self.status}\r\nUpgrade: websocket\r\n"
                      f"Connection: Upgrade\r\nSec-WebSocket-Accept: {accept}\r\n\r\n"
                      ).encode() + (b"not allowed" if "101" not in self.status else b""))
        await writer.drain()
        if "101" in self.status:
            try:
                await self.script(self)
            except (asyncio.IncompleteReadError, ConnectionError):
                pass
        writer.close()

    async def send(self, payload: bytes | str, *, opcode: int | None = None,
                   final: bool = True) -> None:
        data = payload.encode() if isinstance(payload, str) else payload
        code = opcode if opcode is not None else (1 if isinstance(payload, str) else 2)
        head = bytes([(0x80 if final else 0) | code])
        if len(data) < 126:
            head += bytes([len(data)])
        elif len(data) < 65536:
            head += bytes([126]) + struct.pack("!H", len(data))
        else:
            head += bytes([127]) + struct.pack("!Q", len(data))
        self.writer.write(head + data)
        await self.writer.drain()

    async def recv(self) -> tuple[int, bytes]:
        first, second = await self.reader.readexactly(2)
        size = second & 0x7F
        if size == 126:
            (size,) = struct.unpack("!H", await self.reader.readexactly(2))
        elif size == 127:
            (size,) = struct.unpack("!Q", await self.reader.readexactly(8))
        if not second & 0x80:
            self.unmasked = True
        key = await self.reader.readexactly(4)
        data = await self.reader.readexactly(size)
        frame = (first & 0x0F, bytes(b ^ key[i % 4] for i, b in enumerate(data)))
        self.frames.append(frame)
        return frame


async def test_the_websocket_client_speaks_the_protocol():
    big = bytes(range(256)) * 600                       # needs the 64-bit length

    async def script(server: Server) -> None:
        assert await server.recv() == (1, b'{"hello": "world"}')
        await server.send('{"hello": "client"}')
        code, payload = await server.recv()             # a 16-bit length frame
        assert code == 2 and payload == b"\x07" * 1000
        await server.send(big)
        await server.send("frag", final=False)          # one message, three frames
        await server.send(b"", opcode=9)                # ...and a ping between them
        await server.send("men", opcode=0, final=False)
        await server.send("ted ✓", opcode=0)
        assert (await server.recv())[0] == 10           # the pong
        await server.send(struct.pack("!H", 1000) + b"bye", opcode=8)
        assert (await server.recv())[0] == 8            # the close is answered

    async with Server(script) as server:
        ws = await WebSocket.connect(server.url, headers={"Authorization": "Bearer k"})
        await ws.send('{"hello": "world"}')
        assert await ws.recv() == '{"hello": "client"}'
        await ws.send(b"\x07" * 1000)
        assert await ws.recv() == big
        assert await ws.recv() == "fragmented ✓"
        with pytest.raises(ConnectionClosed) as closed:
            await ws.recv()
        assert (closed.value.code, closed.value.reason) == (1000, "bye")
        with pytest.raises(ConnectionClosed):
            await ws.send("too late")
        await asyncio.sleep(0.05)

    assert "GET /live?model=x HTTP/1.1" in server.request
    assert "Authorization: Bearer k" in server.request
    assert "Sec-WebSocket-Version: 13" in server.request
    assert not server.unmasked                          # every client frame is masked


async def test_a_refused_or_wrong_handshake_is_an_error_that_says_why():
    async def nothing(server: Server) -> None:
        return None

    async with Server(nothing, status="401 Unauthorized") as server:
        with pytest.raises(Exception, match="refused the websocket: HTTP/1.1 401"):
            await WebSocket.connect(server.url)
    async with Server(nothing, accept="bm90IHRoZSByaWdodCBrZXk=") as server:
        with pytest.raises(Exception, match="did not complete the websocket handshake"):
            await WebSocket.connect(server.url)
    with pytest.raises(Exception, match="could not connect"):
        await WebSocket.connect("ws://127.0.0.1:9/x", timeout=2)
    with pytest.raises(Exception, match="not a websocket URL"):
        await WebSocket.connect("https://example.com")


async def test_the_websocket_client_against_the_reference_library():
    websockets = pytest.importorskip("websockets")
    big = "x" * 200_000

    async def echo(connection):
        async for message in connection:
            await connection.send(message)

    async with websockets.serve(echo, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        async with await WebSocket.connect(f"ws://127.0.0.1:{port}/") as ws:
            for message in ("héllo", b"\x00\x01\xff" * 5000, big):
                await ws.send(message)
                assert await ws.recv() == message


# --- realtime: OpenAI --------------------------------------------------------------------------------

class Line:
    """A realtime model, played by a script: it is told what the client sent,
    and pushes back what the server would."""

    def __init__(self, react: Any) -> None:
        self.react = react
        self.sent: list[dict[str, Any]] = []
        self.inbox: asyncio.Queue[str | None] = asyncio.Queue()
        self.url = ""
        self.headers: dict[str, str] = {}
        self.closed = False

    async def connect(self, url: str, headers: dict[str, str]) -> Line:
        self.url, self.headers = url, headers
        return self

    def push(self, *messages: dict[str, Any]) -> None:
        for message in messages:
            self.inbox.put_nowait(json.dumps(message))

    async def send(self, raw: str) -> None:
        if self.closed:
            raise ConnectionClosed(1000, "closed")
        message = json.loads(raw)
        self.sent.append(message)
        self.react(self, message)

    def __aiter__(self) -> Line:
        return self

    async def __anext__(self) -> str:
        item = await self.inbox.get()
        if item is None:
            raise StopAsyncIteration
        return item

    async def close(self) -> None:
        if not self.closed:
            self.closed = True
            self.inbox.put_nowait(None)

    def of(self, kind: str) -> list[dict[str, Any]]:
        return [m for m in self.sent if m.get("type") == kind]


PCM = base64.b64encode(tone(100, 24_000)).decode()


def openai_model(line: Line, message: dict[str, Any]) -> None:
    kind = message["type"]
    if kind == "session.update":
        line.push({"type": "session.created"}, {"type": "session.updated"})
    elif kind == "input_audio_buffer.append" and len(
            line.of("input_audio_buffer.append")) == 5:
        line.push(
            {"type": "input_audio_buffer.speech_started"},
            {"type": "input_audio_buffer.speech_stopped"},
            {"type": "conversation.item.input_audio_transcription.completed",
             "transcript": "Do I have a booking?"},
            {"type": "response.function_call_arguments.done", "call_id": "call_1",
             "name": "find_booking", "arguments": '{"name": "Ada"}'},
            {"type": "response.done", "response": {
                "status": "completed", "output": [{"type": "function_call"}],
                "usage": {"input_tokens": 120, "output_tokens": 10}}})
    elif kind == "response.create" and line.of("conversation.item.create"):
        line.push(
            {"type": "response.output_audio.delta", "delta": PCM},
            {"type": "response.output_audio_transcript.delta", "delta": "A table "},
            {"type": "response.output_audio.delta", "delta": PCM},
            {"type": "response.output_audio_transcript.delta", "delta": "for two."},
            {"type": "response.done", "response": {
                "status": "completed", "output": [{"type": "message"}],
                "usage": {"input_tokens": 200, "output_tokens": 40}}})


def booking_agent(**kw: Any) -> Agent:
    @tool
    def find_booking(name: str) -> str:
        """Find a booking.

        Args:
            name: who it is under.
        """
        return f"{name}: table for two at eight"

    return Agent("concierge", "Help callers with bookings.", mode="voice",
                 tools=[find_booking], memory=False,
                 harness=Harness.testing(FakeProvider()), **kw)


async def call_in(ms: int = 400, rate: int = RATE):
    audio = tone(ms, rate)
    size = rate // 50 * 2
    for offset in range(0, len(audio), size):
        yield audio[offset:offset + size]
        await asyncio.sleep(0)


async def test_openai_realtime_carries_the_agent_and_its_rails(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    agent = booking_agent(trace={"user_id": "ada"})
    line = Line(openai_model)
    voice = RealtimeAgent(agent, "openai", rate=RATE, output_rate=8000, linger=5,
                          connect=line.connect, voice="cedar",
                          turn_detection="server_vad", silence_ms=400)

    events = [e async for e in voice.run(call_in())]

    # --- what was opened, and with what
    assert line.url == "wss://api.openai.com/v1/realtime?model=gpt-realtime"
    assert line.headers == {"Authorization": "Bearer sk-test"}
    session = line.sent[0]["session"]
    assert line.sent[0]["type"] == "session.update" and session["type"] == "realtime"
    assert session["output_modalities"] == ["audio"]
    assert session["audio"]["input"] == {
        "format": {"type": "audio/pcm", "rate": 24_000},
        "turn_detection": {"type": "server_vad", "create_response": True,
                           "interrupt_response": True, "silence_duration_ms": 400},
        "transcription": {"model": "gpt-4o-mini-transcribe"}}
    assert session["audio"]["output"] == {
        "format": {"type": "audio/pcm", "rate": 24_000}, "voice": "cedar"}
    assert "Help callers with bookings." in session["instructions"]
    assert "Write for the ear" in session["instructions"]
    assert session["tools"][0]["name"] == "find_booking"
    assert session["tools"][0]["type"] == "function"
    assert session["tools"][0]["parameters"]["required"] == ["name"]

    # --- the audio went up at the model's rate, not the caller's
    sent_audio = b"".join(base64.b64decode(m["audio"])
                          for m in line.of("input_audio_buffer.append"))
    assert duration_ms(sent_audio, 24_000) == pytest.approx(400, abs=5)

    # --- the model's tool call ran here, under the agent's rails
    output = next(m for m in line.of("conversation.item.create"))
    assert output["item"] == {"type": "function_call_output", "call_id": "call_1",
                              "output": "Ada: table for two at eight"}
    assert any(e.action == "tool_call" and e.target == "find_booking"
               for e in agent.harness.audit.entries)

    # --- and what came back
    kinds = [e.type for e in events]
    assert kinds == ["speech_started", "speech_stopped", "transcript", "tool",
                     "audio", "text", "audio", "text", "turn_end"]
    assert events[2].text == "Do I have a booking?"
    assert all(e.data["rate"] == 8000 for e in events if e.type == "audio")
    assert duration_ms(b"".join(e.audio for e in events if e.type == "audio"),
                       8000) == pytest.approx(200, abs=5)
    assert events[-1].text == "A table for two."
    assert events[-1].data["first_audio_ms"] is not None

    # --- it is accounted for and kept like any run
    assert (voice.usage.input_tokens, voice.usage.output_tokens) == (320, 50)
    assert agent.harness.guard.usage.total_tokens == 370
    saved = await agent.harness.sessions.load(voice.session_id)
    assert [m.text for m in saved.messages] == ["Do I have a booking?",
                                                "A table for two."]
    assert saved.user_id == "ada" and saved.usage.total_tokens == 370
    assert line.closed and agent.harness.control.running_agents == []


async def test_a_tool_the_agent_may_not_use_is_refused_to_the_model_too(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    agent = booking_agent(guardrails={"forbid_tools": ["find_booking"]})
    line = Line(openai_model)
    events = [e async for e in RealtimeAgent(agent, connect=line.connect, linger=5,
                                             rate=RATE).run(call_in())]
    output = line.of("conversation.item.create")[0]["item"]["output"]
    assert output.startswith("Not permitted:")
    assert events[-1].type == "turn_end"


async def test_speaking_over_a_realtime_answer_is_reported_and_remembered(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    def model(line: Line, message: dict[str, Any]) -> None:
        if message["type"] == "session.update":
            line.push({"type": "session.updated"})
        elif message["type"] == "input_audio_buffer.append" and len(line.sent) == 3:
            line.push(
                {"type": "input_audio_buffer.speech_stopped"},
                {"type": "conversation.item.input_audio_transcription.completed",
                 "transcript": "Tell me everything."},
                {"type": "response.output_audio.delta", "delta": PCM},
                {"type": "response.output_audio_transcript.delta",
                 "delta": "It began long ago"},
                {"type": "input_audio_buffer.speech_started"},       # they cut in
                {"type": "response.done", "response": {
                    "status": "cancelled", "output": [{"type": "message"}],
                    "usage": {"input_tokens": 50, "output_tokens": 9}}})

    agent = booking_agent()
    line = Line(model)
    voice = RealtimeAgent(agent, connect=line.connect, linger=5, rate=RATE)
    events = [e async for e in voice.run(call_in())]

    kinds = [e.type for e in events]
    assert kinds.count("interrupted") == 1                 # said once, not twice
    assert kinds.index("interrupted") > kinds.index("audio")
    assert events[-1].data["interrupted"] is True
    assert voice.history[-1].text == (
        "It began long ago — [the user interrupted here; the rest was not heard]")


async def test_realtime_respects_a_budget_a_block_and_a_missing_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    # A budget is a budget, whatever the conversation is carried on.
    agent = booking_agent()
    agent.harness.reset_budget(Budget(max_tokens=100))
    line = Line(openai_model)
    events = [e async for e in RealtimeAgent(agent, connect=line.connect, linger=5,
                                             rate=RATE).run(call_in())]
    assert events[-1].type == "error" and events[-1].data["budget_exceeded"] == "tokens"

    # Where the call goes is checked like any model call, before it is opened.
    hooks = HookEngine()

    @hooks.on("model_egress")
    def no_voice(ctx):
        if ctx.data.get("purpose") == "realtime":
            ctx.block("voice may not leave the region")

    agent = Agent("concierge", harness=Harness.testing(FakeProvider(), hooks=hooks),
                  memory=False)
    line = Line(openai_model)
    with pytest.raises(PermissionDenied, match="may not leave the region"):
        [e async for e in RealtimeAgent(agent, connect=line.connect).run(call_in())]
    assert line.sent == [] and line.url == ""

    monkeypatch.delenv("OPENAI_API_KEY")
    with pytest.raises(ConfigurationError, match="set OPENAI_API_KEY"):
        [e async for e in RealtimeAgent(booking_agent(), connect=line.connect).run(
            call_in())]
    with pytest.raises(ConfigurationError, match="unknown realtime provider"):
        RealtimeAgent(booking_agent(), "telepathy")
    with pytest.raises(ConfigurationError, match="realtime openai"):
        RealtimeAgent(booking_agent(), "openai", nonsense=1)


async def test_a_model_error_and_a_dropped_line_end_the_conversation_cleanly(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    def model(line: Line, message: dict[str, Any]) -> None:
        if message["type"] == "session.update":
            line.push({"type": "error", "error": {"message": "model not found"}})
            line.inbox.put_nowait("{not json")
            line.inbox.put_nowait(None)                    # and the line drops

    line = Line(model)
    agent = booking_agent()
    events = [e async for e in RealtimeAgent(agent, connect=line.connect, linger=1,
                                             rate=RATE).run(call_in())]
    assert [e.type for e in events] == ["error"] and events[0].text == "model not found"
    assert agent.harness.control.running_agents == []


# --- realtime: Gemini ---------------------------------------------------------------------------------

def gemini_model(line: Line, message: dict[str, Any]) -> None:
    audio = [m for m in line.sent if "realtimeInput" in m]
    if "setup" in message:
        line.push({"setupComplete": {}})
    elif "realtimeInput" in message and len(audio) == 4:
        line.push(
            {"serverContent": {"inputTranscription": {"text": "Do I have "}}},
            {"serverContent": {"inputTranscription": {"text": "a booking?"}}},
            {"toolCall": {"functionCalls": [
                {"id": "fc_1", "name": "find_booking", "args": {"name": "Ada"}}]}})
    elif "toolResponse" in message:
        line.push(
            {"serverContent": {"modelTurn": {"parts": [
                {"inlineData": {"data": PCM, "mimeType": "audio/pcm;rate=24000"}}]},
                "outputTranscription": {"text": "A table for two."}}},
            {"serverContent": {"turnComplete": True},
             "usageMetadata": {"promptTokenCount": 90, "responseTokenCount": 30}})


async def test_gemini_live_speaks_its_own_protocol_for_the_same_agent(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "g-key")
    agent = booking_agent()
    line = Line(gemini_model)
    voice = RealtimeAgent(agent, GeminiLive(voice="Kore", silence_ms=600),
                          rate=8000, connect=line.connect, linger=5)

    events = [e async for e in voice.run(call_in(300, 8000))]

    assert line.url == (
        "wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage."
        "v1beta.GenerativeService.BidiGenerateContent?key=g-key")
    setup = line.sent[0]["setup"]
    assert setup["model"] == "models/gemini-3.8-live"
    assert setup["generationConfig"] == {
        "responseModalities": ["AUDIO"],
        "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": "Kore"}}}}
    assert "Help callers with bookings." in setup["systemInstruction"]["parts"][0]["text"]
    assert setup["realtimeInputConfig"] == {
        "automaticActivityDetection": {"silenceDurationMs": 600}}
    declared = setup["tools"][0]["functionDeclarations"][0]
    assert declared["name"] == "find_booking"
    assert "additionalProperties" not in json.dumps(declared)

    # The caller's 8 kHz went up as the 16 kHz Live listens at.
    first = next(m for m in line.sent if "realtimeInput" in m)["realtimeInput"]["audio"]
    assert first["mimeType"] == "audio/pcm;rate=16000"
    up = b"".join(base64.b64decode(m["realtimeInput"]["audio"]["data"])
                  for m in line.sent if "realtimeInput" in m)
    assert duration_ms(up, 16_000) == pytest.approx(300, abs=5)

    assert next(m for m in line.sent if "toolResponse" in m) == {"toolResponse": {
        "functionResponses": [{"id": "fc_1", "name": "find_booking",
                               "response": {"result": "Ada: table for two at eight"}}]}}

    assert [e.type for e in events] == ["transcript", "transcript", "tool", "audio",
                                        "text", "turn_end"]
    assert [m.text for m in voice.history] == ["Do I have a booking?", "A table for two."]
    assert (voice.usage.input_tokens, voice.usage.output_tokens) == (90, 30)

    assert OpenAIRealtime().parse({"type": "something.new"}) == []
    assert GeminiLive().parse({"somethingNew": {}}) == []
    monkeypatch.delenv("GEMINI_API_KEY")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(ConfigurationError, match="set GEMINI_API_KEY"):
        GeminiLive().url()


async def test_realtime_over_a_real_socket_end_to_end(monkeypatch):
    """The agent, the client and a server speaking the protocol, with nothing
    stood in for between them."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    async def model(server: Server) -> None:
        heard = 0
        while True:
            code, payload = await server.recv()
            if code == 8:
                return
            message = json.loads(payload)
            if message["type"] == "session.update":
                await server.send(json.dumps({"type": "session.updated"}))
            elif message["type"] == "input_audio_buffer.append":
                heard += 1
                if heard == 3:
                    for reply in (
                            {"type": "input_audio_buffer.speech_stopped"},
                            {"type": "response.output_audio.delta", "delta": PCM},
                            {"type": "response.output_audio_transcript.delta",
                             "delta": "Hello."},
                            {"type": "response.done", "response": {
                                "status": "completed", "output": [],
                                "usage": {"input_tokens": 5, "output_tokens": 2}}}):
                        await server.send(json.dumps(reply))

    async with Server(model) as server:
        voice = RealtimeAgent(booking_agent(), OpenAIRealtime(
            base_url=f"ws://127.0.0.1:{server.port}/v1/realtime"), rate=RATE, linger=5)
        events = [e async for e in voice.run(call_in())]

    assert [e.type for e in events] == ["speech_stopped", "audio", "text", "turn_end"]
    assert "GET /v1/realtime?model=gpt-realtime" in server.request
    assert "Authorization: Bearer sk-test" in server.request
    assert voice.history[-1].text == "Hello."


# --- where the audio goes, and the command line ------------------------------------------------------

async def test_audio_leaving_for_transcription_is_checked_like_a_model_call():
    hooks = HookEngine()
    asked: list[str] = []

    @hooks.on("model_egress")
    def residency(ctx):
        asked.append(ctx.data.get("purpose") or "model")
        if ctx.data.get("purpose") == "transcription":
            ctx.block("recordings stay in the region")

    provider = FakeProvider(["never asked"])
    agent = Agent("concierge", mode="voice", memory=False,
                  harness=Harness.testing(provider, hooks=hooks))
    speech = FakeSpeech(["hello"])

    with pytest.raises(PermissionDenied, match="recordings stay in the region"):
        [e async for e in VoiceAgent(agent, speech=speech).run(microphone(said()))]

    assert asked == ["transcription"] and speech.heard == [] and provider.requests == []
    assert any(e.action == "model_egress" and e.decision == "deny"
               for e in agent.harness.audit.entries)


def test_the_cli_answers_a_recording_aloud(tmp_path, capsys, monkeypatch):
    import agent_harness.voice as voice_module
    from agent_harness import cli
    from agent_harness.llm_providers import register_provider

    class Spoken(FakeProvider):
        name = "spoken"

        def __init__(self, **kw):
            super().__init__(["It is nine o'clock."], **kw)

    register_provider("spoken", Spoken)
    monkeypatch.setattr(voice_module, "OpenAISpeech",
                        lambda **kw: FakeSpeech(["What time is it?"]))
    question, answer = tmp_path / "question.wav", tmp_path / "answer.wav"
    # Recorded at another rate than the agent listens at: converted on the way in.
    question.write_bytes(wav(silence(300, 8000, noise=40) + tone(700, 8000), 8000))

    code = cli.main(["voice", "--input", str(question), "--output", str(answer),
                     "--provider", "spoken", "--model", "fake-1", "--no-memory"])

    out = capsys.readouterr()
    assert code == 0, out.err
    assert "you   › What time is it?" in out.out
    assert "agent › It is nine o'clock." in out.out and "first sound after" in out.out
    pcm, rate = unwav(answer.read_bytes())
    assert rate == 16_000 and duration_ms(pcm, rate) == pytest.approx(19 * 60, abs=5)

    args = cli.build_parser().parse_args(["voice", "--realtime", "gemini"])
    assert args.realtime == "gemini" and args.input is None


# --- holding up under odd input ----------------------------------------------------------------------

def test_the_detector_hears_the_same_thing_however_the_audio_is_cut_up():
    """A network delivers audio in whatever sizes it likes — including odd
    numbers of bytes. The answer must not depend on it."""
    import random

    rng = random.Random(7)
    audio = (silence(300, noise=50) + tone(700) + silence(650, noise=50) + tone(90)
             + silence(600, noise=50) + tone(1200) + silence(900, noise=50))

    def events_for(sizes) -> list[tuple[str, float, int]]:
        vad, out, offset = EnergyVAD(RATE), [], 0
        for size in sizes:
            out += vad.feed(audio[offset:offset + size])
            offset += size
        out += vad.feed(audio[offset:]) + vad.flush()
        return [(e.kind, e.at_ms, len(e.audio)) for e in out]

    whole = events_for([len(audio)])
    assert [kind for kind, _, _ in whole] == ["start", "end", "start", "cancel",
                                              "start", "end"]
    for _ in range(25):
        sizes = []
        while sum(sizes) < len(audio):
            sizes.append(rng.choice([1, 3, 7, 160, 319, 320, 321, 640, 4097]))
        assert events_for(sizes) == whole


def test_resampling_is_the_same_however_the_stream_is_cut_up():
    import random

    rng = random.Random(11)
    for src, dst in ((24_000, 16_000), (16_000, 24_000), (16_000, 8_000),
                     (8_000, 24_000), (44_100, 16_000)):
        source = tone(400, src, frequency=330)
        whole = resample(source, src, dst)
        # The last input sample has no neighbour to interpolate towards.
        assert abs(len(whole) // 2 - dst * 0.4) <= dst // src + 2
        for _ in range(6):
            stream, out, offset = Resampler(src, dst), b"", 0
            while offset < len(source):
                size = rng.choice([1, 2, 5, 481, 960, 2048])
                out += stream.feed(source[offset:offset + size])
                offset += size
            assert out == whole, (src, dst)


def test_listening_costs_a_small_fraction_of_real_time():
    """A detector that cannot keep up with the microphone is no detector."""
    import time

    minute = (silence(400, noise=40) + tone(900) + silence(700, noise=40)) * 30
    vad = EnergyVAD(RATE)
    began = time.perf_counter()
    events = hear(vad, minute, chunk=640)
    took = time.perf_counter() - began
    assert len([e for e in events if e.kind == "end"]) == 30
    assert took < duration_ms(minute, RATE) / 1000 / 10        # at least 10× real time

    chunker = SpeechChunker()
    began = time.perf_counter()
    pieces = []
    for _ in range(2000):
        pieces += chunker.feed("and so it goes on, ")
    pieces += chunker.flush()
    assert time.perf_counter() - began < 2.0 and len(pieces) > 100


async def test_the_websocket_client_at_every_length_the_framing_changes_at():
    sizes = [0, 1, 125, 126, 127, 65_535, 65_536, 65_537, 300_000]

    async def echo(server: Server) -> None:
        while True:
            code, payload = await server.recv()
            if code == 8:
                return
            await server.send(payload if code == 2 else payload.decode())

    async with Server(echo) as server:
        async with await WebSocket.connect(server.url) as ws:
            for size in sizes:
                binary = bytes((i * 31) % 256 for i in range(size))
                await ws.send(binary)
                assert await ws.recv() == binary
                text = "é" * (size // 2)
                await ws.send(text)
                assert await ws.recv() == text
            # Two tasks sending at once must not interleave their frames.
            await asyncio.gather(*(ws.send(bytes([n]) * 70_000) for n in range(6)))
            got = sorted([await ws.recv() for _ in range(6)])
            assert got == [bytes([n]) * 70_000 for n in range(6)]
    assert not server.unmasked

    async def too_much(server: Server) -> None:
        await server.send(b"x" * 5000)
        await asyncio.sleep(0.2)

    async with Server(too_much) as server:
        ws = await WebSocket.connect(server.url, max_size=1000)
        with pytest.raises(ConnectionClosed) as refused:
            await ws.recv()
        assert refused.value.code == 1009


async def test_many_conversations_at_once_do_not_cross():
    """Twenty callers on one harness: each hears its own answer, and each
    conversation is its own session."""
    harness = Harness.testing(FakeProvider(
        [lambda request: f"Answer for {request.messages[-1].text}."], loop=True,
        stream_words=True))

    async def caller(n: int) -> tuple[str, str, str]:
        agent = Agent(f"concierge-{n}", mode="voice", harness=harness, memory=False,
                      trace={"user_id": f"user-{n}"})
        voice = VoiceAgent(agent, speech=FakeSpeech([f"question {n}"]))
        events = [e async for e in voice.run(microphone(said(), pace=0.0002))]
        return (next(e.text for e in events if e.type == "transcript"),
                events[-1].text, voice.session_id)

    results = await asyncio.gather(*(caller(n) for n in range(20)))

    assert [r[:2] for r in results] == [
        (f"question {n}", f"Answer for question {n}.") for n in range(20)]
    assert len({r[2] for r in results}) == 20
    for n in (0, 7, 19):
        mine = await harness.sessions.list(user_id=f"user-{n}")
        assert [s.id for s in mine] == [results[n][2]]
    assert harness.control.running_agents == []


async def test_a_caller_who_hangs_up_mid_answer_leaves_nothing_running():
    agent, _ = concierge(["A very long answer indeed. " * 40], stream_delay=0.002)
    voice = VoiceAgent(agent, speech=FakeSpeech(["hello"], delay=0.005))
    before = len(asyncio.all_tasks())

    stream = voice.run(microphone(said(), silence(60_000), pace=0.001))
    async for event in stream:
        if event.type == "audio":
            break                                  # they hung up
    await stream.aclose()
    await asyncio.sleep(0.05)

    assert len(asyncio.all_tasks()) == before      # no task outlives the call
    assert agent.harness.control.running_agents == []
    # What they heard of it is still on the record.
    assert voice.history and "interrupted here" in voice.history[-1].text
