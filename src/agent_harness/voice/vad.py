"""Voice activity detection: when someone starts speaking, and when they stop.

    vad = EnergyVAD(rate=16_000)
    for event in vad.feed(chunk):
        if event.kind == "start": ...          # they began — stop talking over them
        if event.kind == "end":   event.audio  # one whole utterance, ready to transcribe

This is what makes a voice agent feel like a conversation. Stopping too early
cuts people off mid-sentence; too late, and every reply arrives after an awkward
pause. `silence_ms` is that dial.

`EnergyVAD` listens to loudness against the room, which it keeps measuring as
the quietest tenth of the last few seconds — speech has gaps in it, a fan does
not — so it needs nothing installed and copes with a steady hum. It can be
fooled by a door slamming. For a noisy room, anything with the same two methods — a
neural detector such as Silero, WebRTC's — plugs in where this one goes.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .audio import rms

__all__ = ["VAD", "VADEvent", "EnergyVAD"]


@dataclass
class VADEvent:
    """`start` when speech begins; `end` when it stops, carrying the utterance;
    `cancel` when what started turned out to be a noise, not a turn."""

    kind: str                 # "start" | "end" | "cancel"
    at_ms: float = 0.0        # position in the stream
    audio: bytes = b""        # the whole utterance, on "end"


@runtime_checkable
class VAD(Protocol):
    """Anything that can be fed audio and say when speech starts and ends."""

    rate: int

    def feed(self, pcm: bytes) -> list[VADEvent]: ...

    def flush(self) -> list[VADEvent]: ...


class EnergyVAD:
    """Speech by loudness, against a noise floor that adapts.

    Args:
        rate: the sample rate of what it is fed.
        frame_ms: how finely it listens.
        start_ms: how much continuous speech it takes to believe someone has
            started. Lower reacts faster, and to more coughs.
        silence_ms: how long a pause ends the turn. 500 suits conversation;
            raise it for people who think aloud, lower it for quick commands.
        min_speech_ms: an "utterance" shorter than this is a noise, and dropped.
        prefix_ms: audio kept from just before speech was detected, so the first
            consonant is not clipped off.
        max_utterance_ms: nobody is left talking for ever; past this the turn ends.
        ratio: how many times louder than the room speech has to be.
        floor: the quietest level that can count as speech at all.
        room_ms: how far back it listens to judge how loud the room is.
    """

    def __init__(self, rate: int = 16_000, *, frame_ms: int = 20, start_ms: int = 80,
                 silence_ms: int = 500, min_speech_ms: int = 160, prefix_ms: int = 240,
                 max_utterance_ms: int = 30_000, ratio: float = 3.0,
                 floor: float = 250.0, room_ms: int = 3000) -> None:
        if rate <= 0 or frame_ms <= 0:
            raise ValueError("rate and frame_ms must be positive")
        self.rate = rate
        self.frame_ms = frame_ms
        self.frame_bytes = int(rate * frame_ms / 1000) * 2
        self.start_frames = max(1, round(start_ms / frame_ms))
        self.silence_frames = max(1, round(silence_ms / frame_ms))
        self.min_speech_frames = max(1, round(min_speech_ms / frame_ms))
        self.max_frames = max(1, round(max_utterance_ms / frame_ms))
        self.ratio = ratio
        self.floor = floor
        self._prefix: deque[bytes] = deque(maxlen=max(1, round(prefix_ms / frame_ms)))
        self._levels: deque[float] = deque(maxlen=max(10, round(room_ms / frame_ms)))
        self.reset()

    def reset(self) -> None:
        self._rest = b""
        self._noise = 0.0           # how loud the room is, as last measured
        self._levels.clear()
        self._frames = 0
        self._speaking = False
        self._loud = 0              # consecutive loud frames while not speaking
        self._quiet = 0             # consecutive quiet frames while speaking
        self._voiced = 0            # loud frames in the current utterance
        self._utterance: list[bytes] = []
        self._prefix.clear()

    @property
    def speaking(self) -> bool:
        return self._speaking

    @property
    def threshold(self) -> float:
        """How loud a frame has to be, right now, to count as speech."""
        return max(self.floor, self._noise * self.ratio)

    def feed(self, pcm: bytes) -> list[VADEvent]:
        events: list[VADEvent] = []
        data = self._rest + pcm
        size = self.frame_bytes
        whole = len(data) - len(data) % size
        self._rest = data[whole:]
        for offset in range(0, whole, size):
            events.extend(self._frame(data[offset:offset + size]))
        return events

    def flush(self) -> list[VADEvent]:
        """The stream has ended: whoever was mid-sentence has finished it."""
        events: list[VADEvent] = []
        if self._speaking:
            events.extend(self._finish())
        self._rest = b""
        return events

    def _frame(self, frame: bytes) -> list[VADEvent]:
        self._frames += 1
        level = rms(frame)
        # The room is the quietest tenth of what was recently heard. Speech is
        # full of gaps that fall back to it; a hum has none, so the hum *is* the
        # room. A tenth rather than the minimum, so one dropped packet of pure
        # silence does not make the whole room seem quiet.
        self._levels.append(level)
        ranked = sorted(self._levels)
        self._noise = ranked[len(ranked) // 10]
        loud = level > self.threshold
        events: list[VADEvent] = []

        if not self._speaking:
            self._prefix.append(frame)
            self._loud = self._loud + 1 if loud else 0
            if self._loud >= self.start_frames:
                self._speaking = True
                self._quiet = 0
                self._voiced = self._loud
                self._utterance = list(self._prefix)
                self._prefix.clear()
                events.append(VADEvent(
                    "start", at_ms=(self._frames - self._loud) * self.frame_ms))
            return events

        self._utterance.append(frame)
        if loud:
            self._voiced += 1
            self._quiet = 0
        else:
            self._quiet += 1
        if self._quiet >= self.silence_frames or len(self._utterance) >= self.max_frames:
            events.extend(self._finish())
        return events

    def _finish(self) -> list[VADEvent]:
        voiced, frames = self._voiced, self._utterance
        self._speaking = False
        self._loud = self._quiet = self._voiced = 0
        self._utterance = []
        if voiced < self.min_speech_frames:
            # A click, a cough: not a turn — and whoever stopped to listen
            # should be told they can carry on.
            return [VADEvent("cancel", at_ms=self._frames * self.frame_ms)]
        # The trailing silence that ended the turn is not part of what was said,
        # bar a short tail so the last word is not clipped.
        keep = len(frames) - max(0, min(self._tail(frames), len(frames)))
        return [VADEvent("end", at_ms=self._frames * self.frame_ms,
                         audio=b"".join(frames[:keep]))]

    def _tail(self, frames: list[bytes]) -> int:
        """How many frames of the closing silence to drop."""
        quiet = 0
        threshold = self.threshold
        for frame in reversed(frames):
            if rms(frame) > threshold:
                break
            quiet += 1
        return max(0, quiet - 5)            # keep 100 ms of it
