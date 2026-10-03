"""Getting an attachment to a model in a form the model can take.

    await agent.run("What does the contract say about notice?",
                    attachments=["contract.pdf", "call.mp3", "whiteboard.jpg"])

Each attachment goes one of two ways. Where the model reads that kind of thing —
Claude and GPT a PDF, Gemini a recording or a video — it is sent as it is, and
the model sees the pages, hears the voice. Where it cannot, the harness turns
it into text first: a file is read with `parse_document`, a recording is
transcribed if the harness has been given something to transcribe with. So the
same line works on every provider, and says plainly when it cannot.
"""

from __future__ import annotations

import asyncio
import mimetypes
import tempfile
from pathlib import Path
from typing import Any

from .errors import ConfigurationError
from .types import DocumentBlock, MediaBlock, TextBlock, attach

__all__ = ["prepare_attachments", "document_text", "MAX_INLINE_BYTES"]

#: The most that is sent inline. Past this a provider wants an upload, not a
#: request body — and most reject the request outright.
MAX_INLINE_BYTES = 20 * 1024 * 1024

_TEXT_LIMIT = 400_000


def document_text(block: MediaBlock) -> str:
    """The text of a file attachment, read on this machine."""
    from .toolkits.documents import parse_document

    if block.path:
        target = Path(block.path)
        if not target.is_file():
            raise ConfigurationError(f"attachment {block.label!r} was not found at "
                                     f"{block.path}")
        return str(parse_document(target, max_chars=_TEXT_LIMIT)["text"])
    if not block.data:
        raise ConfigurationError(
            f"attachment {block.label!r} is only a URL, and this model cannot fetch "
            "it — pass the file itself")
    suffix = (Path(block.name).suffix or mimetypes.guess_extension(block.media_type)
              or ".txt")
    with tempfile.TemporaryDirectory(prefix="agent-harness-attach-") as folder:
        target = Path(folder) / f"attachment{suffix}"
        target.write_bytes(block.read())
        return str(parse_document(target, max_chars=_TEXT_LIMIT)["text"])


async def prepare_attachments(items: Any, provider: Any, model: str, *,
                              speech: Any = None) -> list[Any]:
    """Blocks ready to send to `model`: native where it takes them, text where
    it does not. Raises `ConfigurationError` for what cannot be done either way."""
    out: list[Any] = []
    for item in items:
        block = attach(item)
        if block.path and not Path(block.path).is_file():
            raise ConfigurationError(
                f"attachment {block.label!r} was not found at {block.path}")
        size = block.size()
        if size > MAX_INLINE_BYTES:
            raise ConfigurationError(
                f"attachment {block.label!r} is {size / 1_048_576:.0f} MB; the most "
                f"that can be sent inline is {MAX_INLINE_BYTES // 1_048_576} MB — "
                "send a smaller file, or a URL the provider can fetch")

        if provider.accepts(block, model):
            out.append(block)
        elif isinstance(block, DocumentBlock):
            try:
                text = await asyncio.to_thread(document_text, block)
            except ConfigurationError:
                raise
            except Exception as exc:
                raise ConfigurationError(
                    f"attachment {block.label!r} could not be read: {exc}") from None
            out.append(TextBlock(text=f"[Attached file: {block.label}]\n{text}"))
        elif block.type == "audio":
            if speech is None or not hasattr(speech, "transcribe"):
                raise ConfigurationError(
                    f"{model} cannot listen to {block.label!r}. Use a model that "
                    "takes audio (Gemini, or an OpenAI audio model), or give the "
                    "harness something to transcribe with: Harness(speech=...)")
            if not block.inline:
                raise ConfigurationError(
                    f"attachment {block.label!r} is only a URL — pass the recording "
                    "itself so it can be transcribed")
            text = await speech.transcribe(block.read(), media_type=block.media_type)
            out.append(TextBlock(
                text=f"[Transcript of the recording {block.label}]\n{text}"))
        else:
            raise ConfigurationError(
                f"{model} cannot take {block.type} input, so {block.label!r} cannot "
                "be sent" + (" — Gemini models take video" if block.type == "video"
                             else ""))
    return out
