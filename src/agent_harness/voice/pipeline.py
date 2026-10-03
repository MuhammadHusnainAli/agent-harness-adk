"""A voice agent built from parts: listen, think, speak.

    voice = VoiceAgent(agent, speech=OpenAISpeech())

    async for event in voice.run(microphone()):          # 16-bit PCM in
        if event.type == "audio":
            speaker.write(event.audio)                   # 16-bit PCM out
        elif event.type == "interrupted":
            speaker.flush()                              # they spoke over it

Speech is turned into text, the agent answers as it always does, and the answer
is spoken — so the agent behind the voice is an ordinary `Agent`: any of the
twenty providers, its tools, its memory, its budget, its guardrails, its audit
trail. Nothing about the rails changes because the conversation is aloud.

What makes it feel like talking to someone rather than to a queue:

- **It starts speaking before it has finished thinking.** The answer is cut into
  sentences as the model writes them — the first at a clause, sooner still — and
  each is synthesised while the next is being written.
- **It stops when you speak.** Speech over an answer cancels the rest of it, and
  the conversation remembers only what was actually said aloud.
- **It waits for you to finish.** A pause that turns out to be mid-sentence —
  you carry on before it has replied — is joined to what follows and answered
  once, not twice.

Every turn reports how long each part took (`turn_end`), because latency you
cannot see is latency you cannot fix.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from ..context import close_open_tool_calls
from ..errors import ConfigurationError, PermissionDenied
from ..llm_providers.base import CompletionRequest
from ..runtime.session import Session
from ..types import Message, RunResult, Usage
from .audio import Resampler, duration_ms, wav
from .text import SpeechChunker
from .vad import VAD, EnergyVAD, VADEvent

__all__ = ["VoiceAgent", "VoiceEvent"]

_INTERRUPTED = " — [the user interrupted here; the rest was not heard]"
# A turn abandoned before any reply is kept to join the next — but not for ever.
_CARRY_MS = 30_000


@dataclass
class VoiceEvent:
    """One thing that happened in a spoken conversation.

    `speech_started` / `speech_stopped` — the person began or finished talking.
    `transcript` — what they said.  `text` — what the agent is writing.
    `audio` — PCM to play.  `tool` — the agent is using a tool.
    `interrupted` — stop playing: they spoke over the answer.
    `turn_end` — the turn is over; `data` carries the timings.  `error`.
    """

    type: str
    text: str = ""
    audio: bytes = b""
    data: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:  # never the audio
        extra = f" {len(self.audio)} bytes" if self.audio else ""
        return f"<VoiceEvent {self.type}{extra} {self.text[:60]!r}>"


class VoiceAgent:
    """An agent you talk to: speech in, the agent's answer spoken back.

    Args:
        agent: the agent that does the thinking. `Agent(mode="voice")` writes
            for the ear; any agent will do.
        speech: something that both transcribes and speaks (`OpenAISpeech`).
            Or give `stt=` and `tts=` separately; left out, the harness's own
            `Harness(speech=...)` is used.
        vad: what decides when the person has started and stopped. The default
            listens to loudness against the room; see `EnergyVAD`.
        rate: the sample rate of the audio you feed in.
        output_rate: the rate you want back, if not the voice's own — 8000 for
            a phone line, say.
        barge_in: let the person interrupt. Off, what they say while the agent
            is speaking waits its turn.
        barge_in_ms: how much speech it takes to interrupt, on top of what the
            detector already needs — so a cough does not cut an answer short.
        language: the language being spoken, if you know it.
        greeting: said once, when the conversation opens.
        tool_filler: said if the agent reaches for a tool before saying anything
            — "One moment." — so a lookup is not a silence.
        error_line, budget_line: what is said when the agent fails, or is
            stopped by its budget, before it has said anything.
        session: a session id to carry on from.
    """

    def __init__(self, agent: Any, *, speech: Any = None, stt: Any = None,
                 tts: Any = None, vad: VAD | None = None, rate: int = 16_000,
                 output_rate: int | None = None, barge_in: bool = True,
                 barge_in_ms: int = 120, language: str | None = None,
                 greeting: str | None = None, tool_filler: str | None = None,
                 error_line: str = "Sorry, something went wrong on my side.",
                 budget_line: str = "I have reached my limit for this conversation.",
                 session: str | None = None, prefetch: int = 2) -> None:
        speech = speech if speech is not None else agent.harness.speech
        self.agent = agent
        self.stt = stt if stt is not None else speech
        self.tts = tts if tts is not None else speech
        if self.stt is None or self.tts is None:
            raise ConfigurationError(
                f"{agent.name}: a voice agent needs something to listen and speak "
                "with — VoiceAgent(agent, speech=OpenAISpeech()), or "
                "Harness(speech=...)")
        self.rate = rate
        self.vad: VAD = vad if vad is not None else EnergyVAD(rate)
        if self.vad.rate != rate:
            raise ConfigurationError(
                f"the detector listens at {self.vad.rate} Hz but the audio is "
                f"{rate} Hz — build it with the same rate")
        self.output_rate = output_rate or self.tts.rate
        self.barge_in = barge_in
        self.barge_in_ms = barge_in_ms
        self.language = language
        self.greeting = greeting
        self.tool_filler = tool_filler
        self.error_line = error_line
        self.budget_line = budget_line
        self.prefetch = max(1, prefetch)

        #: The conversation as the agent will next see it: what was said, and
        #: of each answer only what was actually spoken.
        self.history: list[Message] = []
        self.usage = Usage()
        #: The timings of every finished turn, newest last.
        self.turns: list[dict[str, Any]] = []
        self._session_id = session
        self._session: Session | None = None
        #: Who has the conversation, once it has been handed on.
        self._holder = ""
        self._saved = 0
        self._carry = b""
        self._turn: asyncio.Task[Any] | None = None
        self._interrupting = False
        self._last: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # the conversation's record
    # ------------------------------------------------------------------
    @property
    def session_id(self) -> str:
        return self._session.id if self._session is not None else (
            self._session_id or "")

    async def _open(self) -> Session:
        if self._session is None:
            # A voice leaves this machine twice — to be transcribed, and as the
            # words to speak. Both are put to the same check as a model call,
            # so a residency policy that covers the model covers the audio.
            for role, service in (("transcription", self.stt), ("speech", self.tts)):
                model = (getattr(service, "stt_model", None) if role == "transcription"
                         else getattr(service, "tts_model", None)) or role
                egress = await self.agent.hooks.emit(
                    "model_egress", agent=self.agent.name, provider=service,
                    model=model, purpose=role,
                    request=CompletionRequest(model=model))
                if egress.blocked:
                    self.agent.harness.audit.record(
                        self.agent.name, "model_egress", target=model,
                        decision="deny", reason=egress.reason, purpose=role)
                    raise PermissionDenied(
                        f"{self.agent.name} may not send audio for {role}: "
                        f"{egress.reason}", tool=model, reason=egress.reason)
            if self._session_id:
                loaded = self.agent._may_have(
                    await self.agent.harness.sessions.load(self._session_id))
                self._session = loaded
                self.history = close_open_tool_calls(list(loaded.messages))
                self._holder = (loaded.metadata.get("handoff") or {}).get("active", "")
                self._saved = len(self.history)
            else:
                self._session = self.agent._may_have(Session(agent=self.agent.name))
        return self._session

    async def _save(self, usage: Usage | None = None) -> None:
        session = await self._open()
        added = self.history[self._saved:]
        if not added and usage is None:
            return
        session.messages = list(self.history)
        if self._holder:
            session.metadata["handoff"] = {
                **(session.metadata.get("handoff") or {"entry": self.agent.name}),
                "active": self._holder}
        if not session.title and added:
            session.title = " ".join(added[0].text.split())[:80]
        carrier = RunResult(agent=self.agent.name, session_id=session.id,
                            usage=usage or Usage())
        if usage is not None:
            session.usage += usage
        self._session = await self.agent._save_session(
            session, carrier, added=added, rewritten=False, run_id=carrier.run_id)
        # If someone else had written to the session meanwhile, their turns are
        # in it now; the conversation carries on from the whole of it.
        self.history = list(self._session.messages)
        self._saved = len(self.history)

    # ------------------------------------------------------------------
    # text in, speech out
    # ------------------------------------------------------------------
    async def respond(self, text: str, *,
                      began: float | None = None) -> AsyncIterator[VoiceEvent]:
        """Answer one thing that was said, aloud: `text` and `audio` events as
        they are ready, then `turn_end`.

        Close the iterator early and the answer stops there; the conversation
        keeps only what had been spoken.
        """
        await self._open()
        began = began if began is not None else time.monotonic()
        events: asyncio.Queue[VoiceEvent | None] = asyncio.Queue()
        order: asyncio.Queue[tuple[str, asyncio.Queue[Any]] | None] = asyncio.Queue()
        gate = asyncio.Semaphore(self.prefetch)
        synths: list[asyncio.Task[Any]] = []
        state: dict[str, Any] = {"spoken": [], "current": "", "audio": False,
                                 "said": 0, "text": text}
        self._last = state
        timing: dict[str, float] = {}

        async def synth(piece: str, sink: asyncio.Queue[Any]) -> None:
            try:
                async with gate:
                    async for chunk in self.tts.synthesize(piece):
                        await sink.put(chunk)
            except asyncio.CancelledError:
                raise
            except Exception as exc:        # one sentence lost, not the turn
                await sink.put(exc)
            finally:
                sink.put_nowait(None)

        def say(piece: str) -> None:
            sink: asyncio.Queue[Any] = asyncio.Queue()
            state["said"] += 1
            synths.append(asyncio.create_task(synth(piece, sink)))
            order.put_nowait((piece, sink))

        async def think() -> RunResult | None:
            result: RunResult | None = None
            chunker = SpeechChunker()
            try:
                speaker = (self.agent.handoff_agent(self._holder)
                           if self._holder else None) or self.agent
                async for event in speaker.stream(text, messages=list(self.history)):
                    if event.type == "text" and event.text:
                        timing.setdefault("first_token", time.monotonic())
                        await events.put(VoiceEvent("text", text=event.text))
                        for piece in chunker.feed(event.text):
                            say(piece)
                    elif event.type == "tool_call":
                        await events.put(VoiceEvent("tool", text=event.text,
                                                    data=dict(event.data)))
                        if self.tool_filler and not state["said"]:
                            say(self.tool_filler)
                    elif event.type == "run_end":
                        result = event.data.get("result")
                for piece in chunker.flush():
                    say(piece)
                if result is not None and result.error and not state["said"]:
                    await events.put(VoiceEvent("error", text=result.error))
                    say(self.error_line)
                elif (result is not None and result.budget_exceeded
                      and not state["said"]):
                    # Stopped by its budget before it could answer: say so,
                    # rather than leave the person listening to nothing.
                    say(self.budget_line)
            finally:
                order.put_nowait(None)
            return result

        async def speak() -> None:
            resampler = Resampler(self.tts.rate, self.output_rate)
            while True:
                item = await order.get()
                if item is None:
                    return
                piece, sink = item
                state["current"] = piece
                first = True
                while True:
                    chunk = await sink.get()
                    if chunk is None:
                        break
                    if isinstance(chunk, Exception):
                        await events.put(VoiceEvent(
                            "error", text=f"could not speak: {chunk}"))
                        break
                    timing.setdefault("first_audio", time.monotonic())
                    state["audio"] = True
                    await events.put(VoiceEvent(
                        "audio", audio=resampler.feed(chunk),
                        text=piece if first else "", data={"rate": self.output_rate}))
                    first = False
                state["spoken"].append(piece)
                state["current"] = ""

        thinker = asyncio.create_task(think())
        speaker = asyncio.create_task(speak())

        async def finish() -> None:
            await asyncio.gather(thinker, speaker, return_exceptions=True)
            events.put_nowait(None)

        finisher = asyncio.create_task(finish())
        completed = False
        try:
            while True:
                event = await events.get()
                if event is None:
                    break
                yield event
            failure = thinker.exception() or speaker.exception()
            if failure is not None:
                raise failure
            result = thinker.result()
            completed = True

            answer = " ".join(state["spoken"])
            if result is not None and not result.error:
                self.history = list(result.messages)
                self._holder = result.active_agent or self._holder
            else:
                self.history = [*self.history, Message.user(text),
                                Message.assistant(answer or self.error_line)]
            usage = result.usage if result is not None else None
            if usage is not None:
                self.usage += usage
            metrics = self._metrics(began, timing, interrupted=False,
                                    result=result, answer=answer)
            await self._save(usage)
            yield VoiceEvent("turn_end", text=answer, data=metrics)
        finally:
            for task in (thinker, speaker, finisher, *synths):
                task.cancel()
            await asyncio.gather(thinker, speaker, finisher, *synths,
                                 return_exceptions=True)
            if not completed:
                self._cut_short(state, began, timing)

    def _cut_short(self, state: dict[str, Any], began: float,
                   timing: dict[str, float]) -> None:
        """The answer was stopped part-way. Remember what was heard of it —
        not what was written — so the agent does not think it said things the
        person never got."""
        heard = " ".join([*state["spoken"], state["current"]]).strip()
        state["heard"] = heard
        if not state["audio"]:
            return          # nothing reached them: the turn never happened
        self.history = [*self.history, Message.user(state["text"]),
                        Message.assistant(f"{heard}{_INTERRUPTED}")]
        self._metrics(began, timing, interrupted=True, result=None, answer=heard)
        self.agent.harness.audit.record(
            self.agent.name, "voice_interrupted", target=self.session_id,
            decision="ok", heard=len(heard))

    def _metrics(self, began: float, timing: dict[str, float], *, interrupted: bool,
                 result: RunResult | None, answer: str) -> dict[str, Any]:
        now = time.monotonic()

        def since(mark: str) -> float | None:
            return (round((timing[mark] - began) * 1000, 1)
                    if mark in timing else None)

        metrics: dict[str, Any] = {
            "stt_ms": None,
            "first_token_ms": since("first_token"),
            "first_audio_ms": since("first_audio"),
            "total_ms": round((now - began) * 1000, 1),
            "interrupted": interrupted,
            "steps": result.steps if result is not None else 0,
            "cost_usd": result.cost_usd if result is not None else 0.0,
            "error": result.error if result is not None else None,
            "budget_exceeded": result.budget_exceeded if result is not None else None,
            "chars": len(answer),
        }
        self.turns.append(metrics)
        if metrics["first_audio_ms"] is not None:
            # The number that decides whether it feels like a conversation.
            self.agent.harness.health.record(
                "voice.first_audio", metrics["first_audio_ms"], kind="voice")
        return metrics

    async def say(self, text: str) -> AsyncIterator[VoiceEvent]:
        """Speak a line of your own — a greeting, a notice — and put it in the
        conversation as something the agent said."""
        await self._open()
        resampler = Resampler(self.tts.rate, self.output_rate)
        first = True
        async for chunk in self.tts.synthesize(text):
            yield VoiceEvent("audio", audio=resampler.feed(chunk),
                             text=text if first else "",
                             data={"rate": self.output_rate})
            first = False
        self.history = [*self.history, Message.assistant(text)]
        await self._save()

    # ------------------------------------------------------------------
    # speech in, speech out
    # ------------------------------------------------------------------
    async def listen(self, pcm: bytes) -> AsyncIterator[VoiceEvent]:
        """Answer one recorded utterance — push-to-talk. `pcm` is everything
        that was said, at `rate`."""
        began = time.monotonic()
        text = (await self.stt.transcribe(wav(pcm, self.rate), media_type="audio/wav",
                                          language=self.language)).strip()
        heard = time.monotonic()
        if not text:
            yield VoiceEvent("turn_end", data={"skipped": "nothing was said"})
            return
        yield VoiceEvent("transcript", text=text,
                         data={"stt_ms": round((heard - began) * 1000, 1)})
        async for event in self.respond(text, began=began):
            if event.type == "turn_end":
                event.data["stt_ms"] = round((heard - began) * 1000, 1)
            yield event

    async def _answer(self, pcm: bytes, out: asyncio.Queue[VoiceEvent | None]) -> None:
        """One turn of a live conversation, into the outgoing queue."""
        audio = self._carry + pcm
        self._last = {}
        try:
            async for event in self.listen(audio):
                await out.put(event)
            self._carry = b""
        except asyncio.CancelledError:
            if self._last.get("audio"):
                # They heard part of the answer and spoke over it.
                self._carry = b""
                out.put_nowait(VoiceEvent(
                    "interrupted", text=self._last.get("heard", ""),
                    data={"heard_chars": len(self._last.get("heard", ""))}))
            elif duration_ms(audio, self.rate) <= _CARRY_MS:
                # Nothing had been said back yet: this was the first half of a
                # longer turn, and it is answered together with the rest.
                self._carry = audio
            raise
        except Exception as exc:
            self._carry = b""
            await out.put(VoiceEvent("error", text=f"{type(exc).__name__}: {exc}"))
            self.agent.harness.audit.record(
                self.agent.name, "voice_turn", target=self.session_id,
                decision="error", reason=str(exc)[:300])

    async def run(self, audio: AsyncIterator[bytes]) -> AsyncIterator[VoiceEvent]:
        """A whole conversation. Feed it the microphone — 16-bit mono PCM at
        `rate`, in chunks of any size — and play what comes back.

        It ends when the audio does, after the last answer has been spoken.
        """
        await self._open()
        out: asyncio.Queue[VoiceEvent | None] = asyncio.Queue()
        utterances: asyncio.Queue[bytes | None] = asyncio.Queue()
        heard_ms = 0.0
        speech_at: float | None = None

        def interrupt() -> None:
            if self._turn is not None and not self._turn.done():
                self._interrupting = True
                self._turn.cancel()

        async def handle(event: VADEvent) -> None:
            nonlocal speech_at
            if event.kind == "start":
                speech_at = heard_ms
                await out.put(VoiceEvent("speech_started", data={"at_ms": event.at_ms}))
            elif event.kind == "cancel":
                speech_at = None
                await out.put(VoiceEvent("speech_stopped", data={"ignored": True}))
            elif event.kind == "end":
                speech_at = None
                await out.put(VoiceEvent(
                    "speech_stopped",
                    data={"at_ms": event.at_ms,
                          "ms": round(duration_ms(event.audio, self.rate))}))
                await utterances.put(event.audio)

        async def hear() -> None:
            nonlocal heard_ms, speech_at
            try:
                async for chunk in audio:
                    heard_ms += duration_ms(chunk, self.rate)
                    for event in self.vad.feed(chunk):
                        await handle(event)
                    if (self.barge_in and speech_at is not None
                            and heard_ms - speech_at >= self.barge_in_ms):
                        speech_at = None
                        interrupt()
                for event in self.vad.flush():
                    await handle(event)
            finally:
                utterances.put_nowait(None)

        async def work() -> None:
            try:
                if self.greeting:
                    async for event in self.say(self.greeting):
                        await out.put(event)
                while True:
                    pcm = await utterances.get()
                    if pcm is None:
                        return
                    self._turn = asyncio.create_task(self._answer(pcm, out))
                    try:
                        await self._turn
                    except asyncio.CancelledError:
                        if not self._interrupting:
                            raise
                        self._interrupting = False
                    finally:
                        self._turn = None
            finally:
                out.put_nowait(None)

        listener = asyncio.create_task(hear())
        worker = asyncio.create_task(work())
        try:
            while True:
                event = await out.get()
                if event is None:
                    break
                yield event
            for task in (listener, worker):
                if task.done() and not task.cancelled() and task.exception():
                    raise task.exception()  # type: ignore[misc]
        finally:
            tasks = [t for t in (listener, worker, self._turn) if t is not None]
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            self._turn = None
            with contextlib.suppress(Exception):
                await self._save()

    @property
    def latency(self) -> dict[str, float | None]:
        """Typical time to the first sound of an answer, over the turns so far."""
        marks = sorted(t["first_audio_ms"] for t in self.turns
                       if t.get("first_audio_ms") is not None)
        if not marks:
            return {"turns": 0, "median_first_audio_ms": None,
                    "worst_first_audio_ms": None}
        return {"turns": len(marks), "median_first_audio_ms": marks[len(marks) // 2],
                "worst_first_audio_ms": marks[-1]}

    def __repr__(self) -> str:  # pragma: no cover - debugging affordance
        return f"<VoiceAgent {self.agent.name} {len(self.history)} messages>"
