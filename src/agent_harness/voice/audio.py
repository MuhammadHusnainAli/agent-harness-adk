"""PCM in plain Python: just enough signal handling for a voice agent.

Everything here is 16-bit mono little-endian PCM — what microphones, telephony
bridges and every speech API settle on. No numpy, and no `audioop`, which left
the standard library in 3.13: a 20 ms frame is a few hundred samples, and
looping over that is microseconds.
"""

from __future__ import annotations

import io
import math
import sys
import wave
from array import array

__all__ = ["samples", "rms", "duration_ms", "Resampler", "resample", "wav", "unwav",
           "ulaw_decode", "ulaw_encode", "tone", "silence"]

_BIG = sys.byteorder == "big"


def samples(pcm: bytes) -> array:
    """PCM bytes as signed 16-bit samples. A trailing odd byte is dropped."""
    data = array("h")
    data.frombytes(pcm[:len(pcm) - (len(pcm) % 2)])
    if _BIG:
        data.byteswap()
    return data


def _bytes(data: array) -> bytes:
    if _BIG:
        data = array("h", data)
        data.byteswap()
    return data.tobytes()


def rms(pcm: bytes) -> float:
    """How loud a stretch of audio is: 0 is silence, 32768 is full scale."""
    data = samples(pcm)
    if not data:
        return 0.0
    return math.sqrt(sum(x * x for x in data) / len(data))


def duration_ms(pcm: bytes, rate: int) -> float:
    return len(pcm) / 2 / rate * 1000


class Resampler:
    """Changes the sample rate of a stream, a chunk at a time.

    It remembers where one chunk ended, so the next begins exactly there —
    resampling each chunk on its own leaves a click at every join, and a voice
    that clicks fifty times a second does not sound like a voice.
    """

    def __init__(self, src: int, dst: int) -> None:
        if src <= 0 or dst <= 0:
            raise ValueError("sample rates must be positive")
        self.src, self.dst = src, dst
        self._made = 0              # output samples produced so far
        self._taken = 0             # input samples consumed before this chunk
        self._last: int | None = None
        self._odd = b""

    def feed(self, pcm: bytes) -> bytes:
        if self.src == self.dst:
            return pcm
        pcm = self._odd + pcm
        self._odd = pcm[len(pcm) - (len(pcm) % 2):]
        data = samples(pcm)
        if not data:
            return b""
        # The last sample of the previous chunk sits at index -1 of this one.
        previous = self._last if self._last is not None else data[0]
        out = array("h")
        src, dst, count, made, taken = self.src, self.dst, len(data), self._made, self._taken
        while True:
            # Where output sample `made` falls among the input samples — kept in
            # whole numbers, so the answer does not depend on how the stream
            # happened to be cut into chunks.
            position = made * src
            index = position // dst - taken
            if index >= count - 1:
                break
            left = previous if index < 0 else data[index]
            right = data[index + 1]
            out.append(int(left + (right - left) * ((position % dst) / dst)))
            made += 1
        self._made, self._taken = made, taken + count
        self._last = data[-1]
        return _bytes(out)


def resample(pcm: bytes, src: int, dst: int) -> bytes:
    """A whole recording at another sample rate."""
    return Resampler(src, dst).feed(pcm)


def wav(pcm: bytes, rate: int) -> bytes:
    """PCM wrapped as a WAV file, for an API that wants a file."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(pcm)
    return buffer.getvalue()


def unwav(data: bytes) -> tuple[bytes, int]:
    """(mono 16-bit PCM, sample rate) from a WAV file. Stereo is mixed down."""
    with wave.open(io.BytesIO(data), "rb") as source:
        if source.getsampwidth() != 2:
            raise ValueError(f"only 16-bit WAV is read here, not "
                             f"{source.getsampwidth() * 8}-bit")
        channels, rate = source.getnchannels(), source.getframerate()
        pcm = source.readframes(source.getnframes())
    if channels > 1:
        data_in = samples(pcm)
        pcm = _bytes(array("h", (
            sum(data_in[i:i + channels]) // channels
            for i in range(0, len(data_in) - channels + 1, channels))))
    return pcm, rate


# ---- G.711 µ-law: what a phone line carries ---------------------------------
_BIAS, _CLIP = 0x84, 32635


def _ulaw_table() -> list[int]:
    table = []
    for byte in range(256):
        value = ~byte & 0xFF
        sample = (((value & 0x0F) << 3) + _BIAS) << ((value & 0x70) >> 4)
        sample -= _BIAS
        table.append(-sample if value & 0x80 else sample)
    return table


_ULAW = _ulaw_table()


def ulaw_decode(data: bytes) -> bytes:
    """8-bit µ-law to 16-bit PCM."""
    return _bytes(array("h", (_ULAW[b] for b in data)))


def ulaw_encode(pcm: bytes) -> bytes:
    """16-bit PCM to 8-bit µ-law."""
    out = bytearray()
    for sample in samples(pcm):
        sign = 0x80 if sample < 0 else 0
        magnitude = min(abs(sample), _CLIP) + _BIAS
        exponent = max(magnitude.bit_length() - 8, 0)
        mantissa = (magnitude >> (exponent + 3)) & 0x0F
        out.append(~(sign | (exponent << 4) | mantissa) & 0xFF)
    return bytes(out)


# ---- signals to test with -------------------------------------------------------
def tone(ms: float, rate: int = 16_000, *, frequency: float = 220.0,
         amplitude: int = 9_000) -> bytes:
    """A steady tone — a stand-in for a voice, for tests and examples."""
    count = int(rate * ms / 1000)
    return _bytes(array("h", (
        int(amplitude * math.sin(2 * math.pi * frequency * n / rate))
        for n in range(count))))


def silence(ms: float, rate: int = 16_000, *, noise: int = 0) -> bytes:
    """Quiet, optionally with a little room noise in it."""
    count = int(rate * ms / 1000)
    if not noise:
        return bytes(count * 2)
    # A fixed pattern rather than randomness, so a test hears the same room twice.
    return _bytes(array("h", (((n * 7919) % (2 * noise + 1)) - noise
                              for n in range(count))))
