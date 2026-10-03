"""Speech in, speech out: what a voice agent listens and talks through.

Two small contracts. Something that transcribes:

    text = await stt.transcribe(wav_bytes, media_type="audio/wav")

and something that speaks, as a stream so the first sound leaves before the
last word is synthesised:

    async for pcm in tts.synthesize("Your order shipped on Thursday."):
        play(pcm)                      # 16-bit mono PCM at tts.rate

`OpenAISpeech` does both against OpenAI's audio endpoints — or anything that
speaks the same protocol, which covers Groq's Whisper, Azure OpenAI, and the
local servers people run for Whisper and Kokoro. Any object with those methods
is as good: Deepgram, ElevenLabs, a model of your own.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator, Callable
from typing import Any, Protocol, runtime_checkable

from ..errors import ConfigurationError, ProviderError
from .audio import tone

__all__ = ["SpeechToText", "TextToSpeech", "OpenAISpeech", "FakeSpeech"]

_EXTENSIONS = {"audio/wav": "wav", "audio/x-wav": "wav", "audio/wave": "wav",
               "audio/mpeg": "mp3", "audio/mp3": "mp3", "audio/mp4": "m4a",
               "audio/x-m4a": "m4a", "audio/ogg": "ogg", "audio/webm": "webm",
               "audio/flac": "flac"}


@runtime_checkable
class SpeechToText(Protocol):
    async def transcribe(self, audio: bytes, *, media_type: str = "audio/wav",
                         language: str | None = None) -> str: ...


@runtime_checkable
class TextToSpeech(Protocol):
    #: The sample rate of the PCM it produces.
    rate: int

    def synthesize(self, text: str) -> AsyncIterator[bytes]: ...


class OpenAISpeech:
    """Transcription and speech over the OpenAI audio API.

        speech = OpenAISpeech()                         # OPENAI_API_KEY
        speech = OpenAISpeech(voice="marin", instructions="Warm, unhurried.")
        speech = OpenAISpeech(base_url="http://localhost:8000/v1", api_key="-",
                              stt_model="whisper-1", tts_model="kokoro")

    One HTTP client is kept open for both, so after the first call a request
    costs no handshake — on a voice turn that is a tenth of a second back.

    Args:
        stt_model, tts_model: the models to transcribe and to speak with.
        voice: the voice to speak in.
        instructions: how to say it — tone, pace — for models that take direction.
        speed: 0.25 to 4.0; 1.0 is natural.
        language: an ISO-639-1 code. Saying it makes transcription faster and
            better than leaving the model to guess.
        client: your own `httpx.AsyncClient` (a proxy, a transport for tests).
    """

    #: OpenAI's `pcm` format: 24 kHz, 16-bit, mono, little-endian.
    rate = 24_000

    def __init__(self, api_key: str | None = None, *,
                 base_url: str = "https://api.openai.com/v1",
                 stt_model: str = "gpt-4o-mini-transcribe",
                 tts_model: str = "gpt-4o-mini-tts", voice: str = "alloy",
                 instructions: str | None = None, speed: float | None = None,
                 language: str | None = None, timeout: float = 60.0,
                 retries: int = 1, client: Any = None) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.stt_model, self.tts_model = stt_model, tts_model
        self.voice = voice
        self.instructions = instructions
        self.speed = speed
        self.language = language
        self.timeout = timeout
        self.retries = retries
        self._client = client

    @property
    def http(self) -> Any:
        if self._client is None:
            import httpx

            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout, connect=10.0),
                limits=httpx.Limits(max_keepalive_connections=8, keepalive_expiry=120))
        return self._client

    def _headers(self) -> dict[str, str]:
        key = self.api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ConfigurationError(
                "OpenAISpeech needs an API key — set OPENAI_API_KEY, or pass api_key=")
        return {"authorization": f"Bearer {key}"}

    @staticmethod
    def _failed(what: str, response: Any) -> ProviderError:
        return ProviderError(
            f"{what} failed with {response.status_code}: {response.text[:300]}",
            provider="openai-speech", status=response.status_code,
            retryable=response.status_code in (408, 429) or response.status_code >= 500)

    async def transcribe(self, audio: bytes, *, media_type: str = "audio/wav",
                         language: str | None = None) -> str:
        """What was said in a recording."""
        fields = {"model": self.stt_model, "response_format": "json"}
        if language or self.language:
            fields["language"] = str(language or self.language)
        name = f"audio.{_EXTENSIONS.get(media_type, 'wav')}"
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                response = await self.http.post(
                    f"{self.base_url}/audio/transcriptions", headers=self._headers(),
                    data=fields, files={"file": (name, audio, media_type)})
            except OSError as exc:
                last = ProviderError(f"transcription could not be reached: {exc}",
                                     provider="openai-speech", retryable=True)
            else:
                if response.status_code < 400:
                    return str(response.json().get("text", "")).strip()
                last = self._failed("transcription", response)
                if not last.retryable:
                    break
            if attempt < self.retries:
                await asyncio.sleep(0.2 * (attempt + 1))
        assert last is not None
        raise last

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        """`text` as speech: 16-bit PCM at 24 kHz, in chunks as it is made."""
        payload: dict[str, Any] = {"model": self.tts_model, "voice": self.voice,
                                   "input": text, "response_format": "pcm"}
        if self.instructions:
            payload["instructions"] = self.instructions
        if self.speed:
            payload["speed"] = self.speed
        odd = b""
        async with self.http.stream("POST", f"{self.base_url}/audio/speech",
                                    headers=self._headers(), json=payload) as response:
            if response.status_code >= 400:
                await response.aread()
                raise self._failed("speech synthesis", response)
            async for chunk in response.aiter_bytes():
                # A sample is two bytes; a chunk boundary may fall between them.
                data = odd + chunk
                odd = data[len(data) - (len(data) % 2):]
                if len(data) > 1:
                    yield data[:len(data) - len(odd)]

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


class FakeSpeech:
    """Scripted speech, for testing a voice agent with no network.

        speech = FakeSpeech(["What is my balance?", "Thanks, goodbye."])

    Each call to `transcribe` returns the next line (or what a function you pass
    makes of the audio). `synthesize` returns a tone whose length follows the
    text — 60 ms a character — in 100 ms chunks, so timing, ordering and
    interruption can all be tested.
    """

    rate = 16_000

    def __init__(self, transcripts: list[str] | Callable[[bytes], str] | None = None,
                 *, ms_per_char: float = 60.0, chunk_ms: int = 100,
                 delay: float = 0.0, stt_delay: float = 0.0) -> None:
        self.transcripts = transcripts if transcripts is not None else []
        self.ms_per_char = ms_per_char
        self.chunk_ms = chunk_ms
        self.delay = delay
        self.stt_delay = stt_delay
        self.heard: list[bytes] = []
        self.spoken: list[str] = []
        self._cursor = 0

    async def transcribe(self, audio: bytes, *, media_type: str = "audio/wav",
                         language: str | None = None) -> str:
        self.heard.append(audio)
        if self.stt_delay:
            await asyncio.sleep(self.stt_delay)
        if callable(self.transcripts):
            return self.transcripts(audio)
        if self._cursor >= len(self.transcripts):
            return ""
        self._cursor += 1
        return self.transcripts[self._cursor - 1]

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        self.spoken.append(text)
        pcm = tone(len(text) * self.ms_per_char, self.rate)
        size = int(self.rate * self.chunk_ms / 1000) * 2
        for offset in range(0, len(pcm), size):
            if self.delay:
                await asyncio.sleep(self.delay)
            yield pcm[offset:offset + size]
