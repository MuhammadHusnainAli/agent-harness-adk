"""Turning a model's stream of tokens into things worth saying aloud.

A voice agent cannot wait for the whole answer before it starts speaking, and
cannot speak half a word. `SpeechChunker` hands over text a sentence at a time
as it arrives — and the first piece sooner still, at a clause, because the gap
before the first sound is the one people notice.
"""

from __future__ import annotations

import re

__all__ = ["SpeechChunker", "speakable"]

_SENTENCE = re.compile(r"""[.!?…]+["'”’)\]]*(?=\s)|\n+""")
_CLAUSE = re.compile(r"[,;:—–](?=\s)")
# A full stop after one of these is not the end of a sentence.
_ABBREVIATIONS = ("mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc",
                  "e.g", "i.e", "no", "inc", "ltd", "approx")

_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_URL = re.compile(r"https?://\S+")
_CODE_FENCE = re.compile(r"```.*?```", re.DOTALL)
_MARKS = re.compile(r"[*_`#>|~]+")
_BULLET = re.compile(r"^\s*(?:[-•]|\d+[.)])\s+", re.MULTILINE)
_SPACE = re.compile(r"\s+")


def speakable(text: str) -> str:
    """Text with what cannot be said removed: markdown, code fences, raw URLs."""
    text = _CODE_FENCE.sub(" ", text)
    text = _LINK.sub(r"\1", text)
    text = _URL.sub("the link", text)
    text = _BULLET.sub("", text)
    text = _MARKS.sub("", text)
    return _SPACE.sub(" ", text).strip()


class SpeechChunker:
    """Feed it deltas; it gives back pieces ready for a voice.

    Args:
        first_min: the first piece may end at a comma once it is this long.
        min_chars: a later piece shorter than this waits for the next sentence —
            a voice that says "Yes." and stops sounds broken.
        max_chars: a sentence longer than this is cut at a space, so a model
            that never uses a full stop still gets spoken.
    """

    def __init__(self, *, first_min: int = 28, min_chars: int = 12,
                 max_chars: int = 240) -> None:
        self.first_min = first_min
        self.min_chars = min_chars
        self.max_chars = max_chars
        self._buffer = ""
        self._emitted = 0

    def feed(self, delta: str) -> list[str]:
        self._buffer += delta
        out: list[str] = []
        while True:
            cut = self._cut()
            if cut is None:
                break
            piece, self._buffer = self._buffer[:cut], self._buffer[cut:]
            said = speakable(piece)
            if said and any(c.isalnum() for c in said):
                out.append(said)
                self._emitted += 1
        return out

    def flush(self) -> list[str]:
        """The stream has ended: say whatever is left."""
        said = speakable(self._buffer)
        self._buffer = ""
        if said and any(c.isalnum() for c in said):
            self._emitted += 1
            return [said]
        return []

    def _cut(self) -> int | None:
        text = self._buffer
        # Code is not spoken, and a fence is never split: it leaves as one
        # piece, which `speakable` then drops whole.
        fence = text.find("```")
        if fence == 0:
            close = text.find("```", 3)
            return close + 3 if close != -1 else None
        if fence != -1:
            text = text[:fence]
            if not text.strip():
                return fence
        elif text.endswith("`"):
            return None                 # perhaps the start of a fence: wait
        for match in _SENTENCE.finditer(text):
            end = match.end()
            if end < self.min_chars and "\n" not in match.group():
                continue
            before = text[:match.start()].rsplit(None, 1)[-1].lower() if text[
                :match.start()].strip() else ""
            if match.group().startswith(".") and (
                    before.rstrip(".") in _ABBREVIATIONS or len(before) == 1):
                continue
            return end
        if self._emitted == 0 and len(text) >= self.first_min:
            for match in _CLAUSE.finditer(text):
                if match.end() >= self.first_min:
                    return match.end()
        if len(text) > self.max_chars:
            space = text.rfind(" ", 0, self.max_chars)
            return space + 1 if space > 0 else self.max_chars
        return fence if fence > 0 else None
