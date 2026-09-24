"""A scripted provider. No network, no key — the way you test an agent.

    agent = Agent("qa", provider=FakeProvider(["hello", tool_call("add", a=1, b=2)]))
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable
from typing import Any, ClassVar

from ..errors import ProviderError
from ..types import Message, ModelResponse, TextBlock, ToolUseBlock, Usage
from .base import CompletionRequest, Provider

Script = str | Message | ModelResponse | ToolUseBlock | Callable[[CompletionRequest], Any]


def tool_call(name: str, /, **args: Any) -> ToolUseBlock:
    """Sugar for scripting a tool call into a FakeProvider timeline."""
    return ToolUseBlock(name=name, input=args)


class FakeProvider(Provider):
    name: ClassVar[str] = "fake"
    default_model: ClassVar[str] = "fake-1"
    BASE_URL: ClassVar[str] = "http://fake.invalid"

    display_name: ClassVar[str] = "Fake (scripted)"
    description: ClassVar[str] = "A scripted provider for tests: no network, no key, no cost."
    auth_type: ClassVar[str] = "none"
    capabilities: ClassVar[frozenset[str]] = frozenset({"streaming", "tools", "embeddings"})

    def __init__(self, responses: list[Script] | None = None, *,
                 default: str = "done", loop: bool = False, **kw: Any) -> None:
        super().__init__(api_key="fake", **kw)
        self.responses: list[Script] = list(responses or [])
        self.default = default
        self.loop = loop
        self.requests: list[CompletionRequest] = []
        self.cursor = 0

    def queue(self, *responses: Script) -> FakeProvider:
        self.responses.extend(responses)
        return self

    def _next(self, req: CompletionRequest) -> Any:
        if self.cursor >= len(self.responses):
            if self.loop and self.responses:
                self.cursor = 0
            else:
                return self.default
        item = self.responses[self.cursor]
        self.cursor += 1
        return item(req) if callable(item) else item

    async def complete(self, req: CompletionRequest) -> ModelResponse:
        self.requests.append(req)
        item = self._next(req)
        if isinstance(item, ModelResponse):
            return item
        if isinstance(item, BaseException):
            raise item
        if isinstance(item, Message):
            message = item
        elif isinstance(item, ToolUseBlock):
            message = Message(role="assistant", content=[item])
        elif isinstance(item, list):
            message = Message(role="assistant", content=item)
        elif isinstance(item, str):
            message = Message.assistant(item)
        else:
            raise ProviderError(f"FakeProvider cannot script {type(item)!r}",
                                provider=self.name)
        stop = "tool_use" if message.tool_uses else "end_turn"
        words = sum(len(b.text.split()) for b in message.content if isinstance(b, TextBlock))
        usage = Usage(input_tokens=sum(len(m.text) for m in req.messages) // 4,
                      output_tokens=max(words, 1), calls=1)
        return self._finish(message=message, stop_reason=stop, usage=usage,
                            model=req.model or self.default_model, raw={})

    async def embed(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        return [hash_embedding(t) for t in texts]


def hash_embedding(text: str, dims: int = 256) -> list[float]:
    """A deterministic bag-of-words embedding with no model behind it.

    It is not semantic in the learned sense, but it is free, offline and stable,
    which makes it a sane default for semantic recall until a real embedder is
    plugged in. Token hashing + L2 normalisation gives usable cosine scores for
    overlapping vocabulary.
    """
    vec = [0.0] * dims
    tokens = [t for t in "".join(c.lower() if c.isalnum() else " " for c in text).split() if t]
    for token in tokens:
        digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
        idx = int.from_bytes(digest[:4], "big") % dims
        sign = 1.0 if digest[4] % 2 else -1.0
        vec[idx] += sign
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec] if norm else vec
