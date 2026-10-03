"""Realtime voice: audio straight into a speech-to-speech model.

    voice = RealtimeAgent(agent, provider="openai")            # or "gemini"

    async for event in voice.run(microphone()):
        if event.type == "audio":         speaker.write(event.audio)
        elif event.type == "interrupted": speaker.flush()

One model listens, thinks and speaks, over one open connection. There is no
transcription step and no synthesis step, so this is as fast as a voice agent
gets, and the model hears tone as well as words.

It is still the same `Agent`. Its instructions become the model's; its tools
are offered to it, and each call the model makes goes through `Agent.call_tool`
— the guardrails, the hooks, the permission gate, the audit trail and the
budget, as in any run. Where the connection goes is put to the same
`model_egress` check as any model call, so a residency policy covers voice too.

What cannot be the same: the model speaks before anyone can read what it said.
Output guardrails here are applied to the transcript afterwards and recorded,
not enforced in the moment. Where an answer must be checked *before* it is
heard, use `VoiceAgent`, which has the text first.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import os
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from ..context import close_open_tool_calls
from ..errors import (
    BudgetExceeded,
    ConfigurationError,
    GuardrailTripped,
    PermissionDenied,
)
from ..llm_providers.base import CompletionRequest
from ..runtime.session import Session
from ..types import Message, RunResult, Usage, new_id
from .audio import Resampler
from .pipeline import _INTERRUPTED, VoiceEvent
from .ws import ConnectionClosed, WebSocket

__all__ = ["RealtimeAgent", "OpenAIRealtime", "GeminiLive", "BACKENDS"]

Signal = tuple[Any, ...]
Connect = Callable[[str, dict[str, str]], Awaitable[Any]]


# ---------------------------------------------------------------------------
# the two wire protocols
# ---------------------------------------------------------------------------
class OpenAIRealtime:
    """OpenAI's Realtime API, over its WebSocket.

    Args:
        model: a realtime model.
        voice: the voice it speaks in.
        turn_detection: "semantic_vad" waits for the *thought* to finish, not
            just the sound; "server_vad" ends a turn on silence, and takes
            `silence_ms`, `threshold` and `prefix_ms`; None leaves turns to you.
        transcription: the model that writes down what the person said. None
            turns their transcript off.
        language: the language being spoken, to help that transcription.
    """

    name = "openai"
    #: The API takes and returns 24 kHz PCM.
    input_rate = 24_000
    output_rate = 24_000

    def __init__(self, model: str = "gpt-realtime", *, voice: str = "marin",
                 api_key: str | None = None,
                 base_url: str = "wss://api.openai.com/v1/realtime",
                 turn_detection: str | None = "semantic_vad",
                 silence_ms: int | None = None, threshold: float | None = None,
                 prefix_ms: int | None = None,
                 transcription: str | None = "gpt-4o-mini-transcribe",
                 language: str | None = None) -> None:
        self.model, self.voice = model, voice
        self.api_key = api_key
        self.base_url = base_url
        self.turn_detection = turn_detection
        self.silence_ms, self.threshold, self.prefix_ms = silence_ms, threshold, prefix_ms
        self.transcription = transcription
        self.language = language

    def url(self) -> str:
        return f"{self.base_url}?model={self.model}"

    def headers(self) -> dict[str, str]:
        key = self.api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ConfigurationError(
                "the OpenAI realtime API needs a key — set OPENAI_API_KEY, or pass "
                "api_key=")
        return {"Authorization": f"Bearer {key}"}

    def setup(self, instructions: str, tools: list[Any]) -> list[dict[str, Any]]:
        listening: dict[str, Any] = {
            "format": {"type": "audio/pcm", "rate": self.input_rate}}
        if self.turn_detection:
            detection: dict[str, Any] = {"type": self.turn_detection,
                                         "create_response": True,
                                         "interrupt_response": True}
            if self.turn_detection == "server_vad":
                for field, value in (("silence_duration_ms", self.silence_ms),
                                     ("threshold", self.threshold),
                                     ("prefix_padding_ms", self.prefix_ms)):
                    if value is not None:
                        detection[field] = value
            listening["turn_detection"] = detection
        else:
            listening["turn_detection"] = None
        if self.transcription:
            listening["transcription"] = {
                "model": self.transcription,
                **({"language": self.language} if self.language else {})}
        session: dict[str, Any] = {
            "type": "realtime",
            "model": self.model,
            "output_modalities": ["audio"],
            "audio": {"input": listening,
                      "output": {"format": {"type": "audio/pcm",
                                            "rate": self.output_rate},
                                 "voice": self.voice}},
            "instructions": instructions,
        }
        if tools:
            session["tools"] = [{"type": "function", "name": t.name,
                                 "description": t.description,
                                 "parameters": t.parameters} for t in tools]
            session["tool_choice"] = "auto"
        return [{"type": "session.update", "session": session}]

    def audio(self, pcm: bytes) -> dict[str, Any]:
        return {"type": "input_audio_buffer.append",
                "audio": base64.b64encode(pcm).decode()}

    def text(self, text: str) -> list[dict[str, Any]]:
        return [{"type": "conversation.item.create",
                 "item": {"type": "message", "role": "user",
                          "content": [{"type": "input_text", "text": text}]}},
                {"type": "response.create"}]

    def tool_result(self, call_id: str, name: str, output: str) -> list[dict[str, Any]]:
        return [{"type": "conversation.item.create",
                 "item": {"type": "function_call_output", "call_id": call_id,
                          "output": output}},
                {"type": "response.create"}]

    def cancel(self) -> list[dict[str, Any]]:
        return [{"type": "response.cancel"}]

    def parse(self, message: dict[str, Any]) -> list[Signal]:
        kind = message.get("type", "")
        if kind in ("session.created", "session.updated"):
            return [("ready",)]
        if kind == "input_audio_buffer.speech_started":
            return [("speech_started",)]
        if kind == "input_audio_buffer.speech_stopped":
            return [("speech_stopped",)]
        if kind == "conversation.item.input_audio_transcription.completed":
            return [("transcript", message.get("transcript", ""))]
        if kind in ("response.output_audio.delta", "response.audio.delta"):
            return [("audio", base64.b64decode(message.get("delta", "")))]
        if kind in ("response.output_audio_transcript.delta",
                    "response.audio_transcript.delta", "response.output_text.delta",
                    "response.text.delta"):
            return [("text", message.get("delta", ""))]
        if kind == "response.function_call_arguments.done":
            try:
                args = json.loads(message.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            return [("tool", message.get("call_id", ""), message.get("name", ""), args)]
        if kind == "response.done":
            response = message.get("response") or {}
            used = response.get("usage") or {}
            usage = Usage(input_tokens=int(used.get("input_tokens") or 0),
                          output_tokens=int(used.get("output_tokens") or 0), calls=1)
            out: list[Signal] = []
            if response.get("status") == "cancelled":
                out.append(("interrupted",))
            # A response that only called a function is not the end of the turn.
            calls = any(item.get("type") == "function_call"
                        for item in response.get("output") or [])
            out.append(("usage", usage))
            if not calls:
                out.append(("turn_end",))
            return out
        if kind == "error":
            error = message.get("error") or {}
            return [("error", error.get("message") or json.dumps(error))]
        return []


class GeminiLive:
    """Google's Gemini Live API, over its WebSocket.

    Args:
        model: a Live model.
        voice: a prebuilt voice name; None is the model's default.
        api_key: defaults to `GEMINI_API_KEY`, then `GOOGLE_API_KEY`.
        silence_ms: how long a pause ends a turn; None is the model's own.
        detect_speech: False turns the model's own turn detection off.
    """

    name = "gemini"
    #: Live listens at 16 kHz and speaks at 24 kHz.
    input_rate = 16_000
    output_rate = 24_000

    _URL = ("wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage."
            "v1beta.GenerativeService.BidiGenerateContent")

    def __init__(self, model: str = "gemini-3.8-live", *, voice: str | None = None,
                 api_key: str | None = None, base_url: str | None = None,
                 silence_ms: int | None = None, detect_speech: bool = True,
                 language: str | None = None) -> None:
        self.model, self.voice = model, voice
        self.api_key = api_key
        self.base_url = base_url or self._URL
        self.silence_ms = silence_ms
        self.detect_speech = detect_speech
        self.language = language

    def url(self) -> str:
        key = (self.api_key or os.environ.get("GEMINI_API_KEY")
               or os.environ.get("GOOGLE_API_KEY"))
        if not key:
            raise ConfigurationError(
                "the Gemini Live API needs a key — set GEMINI_API_KEY, or pass "
                "api_key=")
        return f"{self.base_url}?key={key}"

    def headers(self) -> dict[str, str]:
        return {}

    def setup(self, instructions: str, tools: list[Any]) -> list[dict[str, Any]]:
        from ..llm_providers.gemini import _clean_schema

        generation: dict[str, Any] = {"responseModalities": ["AUDIO"]}
        speech: dict[str, Any] = {}
        if self.voice:
            speech["voiceConfig"] = {"prebuiltVoiceConfig": {"voiceName": self.voice}}
        if self.language:
            speech["languageCode"] = self.language
        if speech:
            generation["speechConfig"] = speech
        model = self.model if self.model.startswith("models/") else f"models/{self.model}"
        setup: dict[str, Any] = {
            "model": model,
            "generationConfig": generation,
            "systemInstruction": {"parts": [{"text": instructions}]},
            "inputAudioTranscription": {},
            "outputAudioTranscription": {},
        }
        detection: dict[str, Any] = {}
        if not self.detect_speech:
            detection["disabled"] = True
        if self.silence_ms is not None:
            detection["silenceDurationMs"] = self.silence_ms
        if detection:
            setup["realtimeInputConfig"] = {"automaticActivityDetection": detection}
        if tools:
            setup["tools"] = [{"functionDeclarations": [
                {"name": t.name, "description": t.description,
                 "parameters": _clean_schema(t.parameters)} for t in tools]}]
        return [{"setup": setup}]

    def audio(self, pcm: bytes) -> dict[str, Any]:
        return {"realtimeInput": {"audio": {
            "data": base64.b64encode(pcm).decode(),
            "mimeType": f"audio/pcm;rate={self.input_rate}"}}}

    def text(self, text: str) -> list[dict[str, Any]]:
        return [{"clientContent": {
            "turns": [{"role": "user", "parts": [{"text": text}]}],
            "turnComplete": True}}]

    def tool_result(self, call_id: str, name: str, output: str) -> list[dict[str, Any]]:
        return [{"toolResponse": {"functionResponses": [
            {"id": call_id, "name": name, "response": {"result": output}}]}}]

    def cancel(self) -> list[dict[str, Any]]:
        return []       # the server stops by itself when it hears the person

    def parse(self, message: dict[str, Any]) -> list[Signal]:
        out: list[Signal] = []
        if "setupComplete" in message:
            out.append(("ready",))
        content = message.get("serverContent") or {}
        heard = (content.get("inputTranscription") or {}).get("text")
        if heard:
            out.append(("transcript_delta", heard))
        for part in (content.get("modelTurn") or {}).get("parts") or []:
            inline = part.get("inlineData") or {}
            if inline.get("data"):
                out.append(("audio", base64.b64decode(inline["data"])))
        said = (content.get("outputTranscription") or {}).get("text")
        if said:
            out.append(("text", said))
        if content.get("interrupted"):
            out.append(("interrupted",))
        for call in (message.get("toolCall") or {}).get("functionCalls") or []:
            out.append(("tool", call.get("id", ""), call.get("name", ""),
                        call.get("args") or {}))
        used = message.get("usageMetadata")
        if used:
            out.append(("usage_total", Usage(
                input_tokens=int(used.get("promptTokenCount") or 0),
                output_tokens=int(used.get("responseTokenCount") or 0), calls=1)))
        if content.get("turnComplete"):
            out.append(("turn_end",))
        if "goAway" in message:
            out.append(("error", "the server is about to close this connection"))
        return out


#: The realtime backends, by name.
BACKENDS: dict[str, Callable[..., Any]] = {"openai": OpenAIRealtime,
                                           "gemini": GeminiLive}


# ---------------------------------------------------------------------------
# the agent
# ---------------------------------------------------------------------------
class RealtimeAgent:
    """An agent you talk to, through a speech-to-speech model.

    Args:
        agent: the agent whose instructions, tools and rails apply.
        provider: "openai", "gemini", or a backend you built — `OpenAIRealtime(
            voice="cedar", turn_detection="server_vad")`.
        rate: the sample rate of the audio you feed in. It is converted to what
            the model takes; left out, feed the model's own rate.
        output_rate: the rate you want back, if not the model's.
        session: a session id whose transcript this carries on from.
        linger: seconds to wait, once your audio has ended, for an answer that
            is still being spoken.
        connect: how to open the socket — yours, for a proxy or a test.
        options: passed to the backend, when `provider` is a name.
    """

    def __init__(self, agent: Any, provider: Any = "openai", *,
                 rate: int | None = None, output_rate: int | None = None,
                 session: str | None = None, linger: float = 20.0,
                 connect: Connect | None = None, **options: Any) -> None:
        if isinstance(provider, str):
            if provider not in BACKENDS:
                raise ConfigurationError(
                    f"unknown realtime provider {provider!r}; known: "
                    f"{', '.join(sorted(BACKENDS))}")
            try:
                provider = BACKENDS[provider](**options)
            except TypeError as exc:
                raise ConfigurationError(f"realtime {provider}: {exc}") from None
        elif options:
            raise ConfigurationError(
                f"{', '.join(sorted(options))}: pass these to the backend itself")
        self.agent = agent
        self.backend = provider
        self.rate = rate or provider.input_rate
        self.output_rate = output_rate or provider.output_rate
        self.linger = linger
        self._connect: Connect = connect or (
            lambda url, headers: WebSocket.connect(url, headers=headers))
        self.history: list[Message] = []
        self.usage = Usage()
        self.turns: list[dict[str, Any]] = []
        self._session_id = session
        self._session: Session | None = None
        self._saved = 0
        self._socket: Any = None

    @property
    def session_id(self) -> str:
        return self._session.id if self._session is not None else (
            self._session_id or "")

    async def _open(self) -> Session:
        if self._session is None:
            if self._session_id:
                self._session = self.agent._may_have(
                    await self.agent.harness.sessions.load(self._session_id))
                self.history = close_open_tool_calls(list(self._session.messages))
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
        if not session.title and added:
            session.title = " ".join(added[0].text.split())[:80]
        if usage is not None:
            session.usage += usage
        carrier = RunResult(agent=self.agent.name, session_id=session.id,
                            usage=usage or Usage())
        self._session = await self.agent._save_session(
            session, carrier, added=added, rewritten=False, run_id=carrier.run_id)
        self.history = list(self._session.messages)
        self._saved = len(self.history)

    async def _instructions(self) -> str:
        system = await self.agent.assembler.build(tool_names=self.agent.tools.names)
        if not self.history:
            return system
        # A realtime model starts with no memory of an earlier call; the
        # transcript so far is what it is given to carry on from.
        lines = [f"{m.role}: {m.text}" for m in self.history[-40:] if m.text]
        return (f"{system}\n\n## The conversation so far\n" + "\n".join(lines))

    async def send_text(self, text: str) -> None:
        """Say something to the model in writing, mid-conversation."""
        if self._socket is None:
            raise ConfigurationError("there is no open conversation to send text to")
        for message in self.backend.text(text):
            await self._socket.send(json.dumps(message))

    async def interrupt(self) -> None:
        """Stop the answer that is being spoken."""
        if self._socket is not None:
            for message in self.backend.cancel():
                await self._socket.send(json.dumps(message))

    async def run(self, audio: AsyncIterator[bytes]) -> AsyncIterator[VoiceEvent]:
        """A whole conversation: feed it the microphone, play what comes back."""
        agent, backend, harness = self.agent, self.backend, self.agent.harness
        await self._open()
        run_id = new_id("run")
        instructions = await self._instructions()
        tools = agent.tools.schemas()

        # Where this connection goes is somebody's business: the same check a
        # model call gets, so a residency policy covers a voice call too.
        egress = await agent.hooks.emit(
            "model_egress", agent=agent.name, run_id=run_id, provider=backend,
            model=backend.model, purpose="realtime",
            request=CompletionRequest(model=backend.model, system=instructions))
        if egress.blocked:
            harness.audit.record(agent.name, "model_egress", target=backend.model,
                                 decision="deny", run_id=run_id, reason=egress.reason)
            raise PermissionDenied(
                f"{agent.name} may not open a realtime session with "
                f"{backend.model}: {egress.reason}", tool=backend.model,
                reason=egress.reason)

        harness.control.check(f"{agent.name} realtime")
        socket = await self._connect(backend.url(), backend.headers())
        self._socket = socket
        harness.control.enter(agent.name, run_id)
        harness.audit.record(agent.name, "run_start", target="realtime voice",
                             run_id=run_id, model=backend.model)

        out: asyncio.Queue[VoiceEvent | None] = asyncio.Queue()
        guard = harness.guard
        state: dict[str, Any] = {"heard": "", "said": "", "speaking": False,
                                 "busy": 0, "began": None, "first_audio": None,
                                 "interrupted": False, "total": Usage()}
        tool_tasks: set[asyncio.Task[Any]] = set()
        idle = asyncio.Event()
        idle.set()

        async def send(messages: list[dict[str, Any]]) -> None:
            for message in messages:
                await socket.send(json.dumps(message))

        async def call(call_id: str, name: str, args: dict[str, Any]) -> None:
            state["busy"] += 1
            idle.clear()
            try:
                await out.put(VoiceEvent("tool", text=name, data={"args": args}))
                outcome = await agent.call_tool(name, args, run_id=run_id, guard=guard)
                await send(backend.tool_result(call_id, name, outcome.content))
            except ConnectionClosed:
                pass
            finally:
                state["busy"] -= 1

        async def end_turn() -> None:
            heard, said = state["heard"].strip(), state["said"].strip()
            if heard or said:
                # What was actually said, checked now that there is text to check.
                try:
                    if heard:
                        heard = harness.guardrails.check(heard, where="input",
                                                         label="voice")
                    if said:
                        said = agent.content_guardrails.check(
                            said, where="output", label=agent.name)
                except GuardrailTripped as exc:
                    harness.audit.record(agent.name, "guardrail", target="realtime",
                                         decision="warn", run_id=run_id,
                                         reason=str(exc)[:300])
                    await out.put(VoiceEvent("error", text=f"guardrail: {exc}"))
                if heard:
                    self.history.append(Message.user(heard))
                if said:
                    self.history.append(Message.assistant(
                        said + (_INTERRUPTED if state["interrupted"] else "")))
            metrics = {
                "first_audio_ms": (round((state["first_audio"] - state["began"]) * 1000, 1)
                                   if state["first_audio"] and state["began"] else None),
                "interrupted": state["interrupted"], "chars": len(said)}
            self.turns.append(metrics)
            if metrics["first_audio_ms"] is not None:
                harness.health.record("voice.first_audio", metrics["first_audio_ms"],
                                      kind="voice")
            await out.put(VoiceEvent("turn_end", text=said, data=metrics))
            state.update(heard="", said="", speaking=False, began=None,
                         first_audio=None, interrupted=False)
            await self._save()
            if not state["busy"]:
                idle.set()

        async def receive() -> None:
            resampler = Resampler(backend.output_rate, self.output_rate)
            try:
                async for raw in socket:
                    try:
                        message = json.loads(raw)
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        continue
                    for signal in backend.parse(message):
                        kind = signal[0]
                        if kind == "audio":
                            if state["first_audio"] is None:
                                state["first_audio"] = time.monotonic()
                            state["speaking"] = True
                            idle.clear()
                            await out.put(VoiceEvent(
                                "audio", audio=resampler.feed(signal[1]),
                                data={"rate": self.output_rate}))
                        elif kind == "text":
                            state["said"] += signal[1]
                            idle.clear()
                            await out.put(VoiceEvent("text", text=signal[1]))
                        elif kind == "transcript":
                            state["heard"] = signal[1]
                            await out.put(VoiceEvent("transcript", text=signal[1]))
                        elif kind == "transcript_delta":
                            state["heard"] += signal[1]
                            await out.put(VoiceEvent("transcript", text=signal[1],
                                                     data={"partial": True}))
                        elif kind == "speech_started":
                            await out.put(VoiceEvent("speech_started"))
                            if state["speaking"]:
                                state["interrupted"] = True
                                await out.put(VoiceEvent("interrupted",
                                                         text=state["said"]))
                        elif kind == "speech_stopped":
                            state["began"] = time.monotonic()
                            state["first_audio"] = None
                            idle.clear()
                            await out.put(VoiceEvent("speech_stopped"))
                        elif kind == "interrupted":
                            if not state["interrupted"]:
                                state["interrupted"] = True
                                await out.put(VoiceEvent("interrupted",
                                                         text=state["said"]))
                        elif kind == "tool":
                            task = asyncio.create_task(call(*signal[1:]))
                            tool_tasks.add(task)
                            task.add_done_callback(tool_tasks.discard)
                        elif kind in ("usage", "usage_total"):
                            usage = signal[1]
                            if kind == "usage_total":
                                # Gemini reports the running total, not the step.
                                seen = state["total"]
                                usage = Usage(
                                    input_tokens=max(0, usage.input_tokens
                                                     - seen.input_tokens),
                                    output_tokens=max(0, usage.output_tokens
                                                      - seen.output_tokens), calls=1)
                                state["total"] = signal[1]
                            self.usage += usage
                            (await self._open()).usage += usage
                            guard.record(usage, agent=agent.name, task="realtime")
                        elif kind == "turn_end":
                            await end_turn()
                        elif kind == "error":
                            await out.put(VoiceEvent("error", text=signal[1]))
            except BudgetExceeded as exc:
                harness.audit.record(agent.name, "budget", target=exc.kind,
                                     decision="stop", run_id=run_id)
                await out.put(VoiceEvent("error", text=f"budget: {exc}",
                                         data={"budget_exceeded": exc.kind}))
            finally:
                idle.set()
                out.put_nowait(None)

        async def pump() -> None:
            resampler = Resampler(self.rate, backend.input_rate)
            async for chunk in audio:
                harness.control.check(f"{agent.name} realtime")
                await socket.send(json.dumps(backend.audio(resampler.feed(chunk))))
            # The microphone has closed. Let an answer in flight finish.
            await asyncio.sleep(0)
            with contextlib.suppress(asyncio.TimeoutError, TimeoutError):
                await asyncio.wait_for(self._settled(idle, state), self.linger)
            await socket.close()

        receiver = pumper = None
        try:
            await send(backend.setup(instructions, tools))
            receiver = asyncio.create_task(receive())
            pumper = asyncio.create_task(pump())
            while True:
                event = await out.get()
                if event is None:
                    break
                yield event
            if pumper.done() and not pumper.cancelled() and pumper.exception():
                failure = pumper.exception()
                if not isinstance(failure, ConnectionClosed):
                    raise failure  # type: ignore[misc]
        finally:
            tasks = [t for t in (receiver, pumper, *tool_tasks) if t is not None]
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            with contextlib.suppress(Exception):
                await socket.close()
            self._socket = None
            harness.control.leave(run_id)
            harness.audit.record(agent.name, "run_end", target="realtime voice",
                                 run_id=run_id, decision="ok",
                                 turns=len(self.turns),
                                 tokens=self.usage.total_tokens)
            with contextlib.suppress(Exception):
                await self._save()

    @staticmethod
    async def _settled(idle: asyncio.Event, state: dict[str, Any]) -> None:
        """Wait until nothing is being said and no tool is running — and stays
        that way for a moment, since a tool's answer starts a new response."""
        while True:
            await idle.wait()
            await asyncio.sleep(0.3)
            if idle.is_set() and not state["busy"] and not state["speaking"]:
                return

    def __repr__(self) -> str:  # pragma: no cover - debugging affordance
        return f"<RealtimeAgent {self.agent.name} via {self.backend.name}>"
