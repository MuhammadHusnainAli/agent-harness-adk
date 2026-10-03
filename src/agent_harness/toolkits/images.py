"""Image generation, as a tool: a prompt in, pictures out — with reference images.

    from agent_harness import Agent, ImageGenerator

    images = ImageGenerator()                       # from the environment
    agent = Agent("designer", tools=images.tools(), workspace="./studio")
    await agent.run("A poster for the autumn sale, in the style of this photo.",
                    attachments=["storefront.jpg"])

The engine is whichever has a key set — Gemini (`GEMINI_API_KEY`), OpenAI
(`OPENAI_API_KEY`), Replicate (`REPLICATE_API_TOKEN`) — or the ones you name,
tried in order. Every one is spoken to over plain HTTP.

    images = ImageGenerator(["gemini", "openai"], aspect_ratio="16:9")
    made = await images.generate("a red fox in snow", references=["fox.jpg"])
    made[0].save("fox.png")

**Reference images** are what the picture is made from or made to look like: a
photo to edit, a style to follow, a character to keep. The model names them by
what it can see — a file in its workspace, an image it generated earlier, an
attachment that came with the task, a web address — and never by a path on
this machine.

**Progress.** Every generation is a job. `generate` waits for it; `submit`
returns it at once, to be watched:

    job = await images.submit("a city at dusk", count=4)
    while not job.done:
        print(job.describe())            # "running on replicate — 40%, 12s"
        await asyncio.sleep(2)
    made = job.result()

An agent gets the same: `generate_image` waits a while, and a job that outlives
the wait is collected with `image_status`.

A generation is something that fails. A call that may succeed later is retried
with back-off; an engine that cannot answer is passed over for the next; one
that keeps failing is left alone for a minute; a job that is cancelled or runs
out of time is cancelled at the provider too. A prompt a provider *refuses* is
not retried and is not taken to another provider: the model is told it was
refused, and why. What comes back is checked to be an image before it is kept.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import inspect
import os
import re
import struct
import time
import zlib
from collections import OrderedDict
from collections.abc import Callable, Iterable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlsplit

from ..errors import ConfigurationError, ToolError
from ..llm_providers.resilience import CircuitBreaker, RetryPolicy, retry_after
from ..tools import Permission, Tool, tool
from ..types import Artifact, ImageBlock, MediaBlock, Usage, attach, new_id
from ._net import is_private_host

__all__ = [
    "ImageGenerator",
    "ImageEngine",
    "ImageRequest",
    "ImageJob",
    "ImageProgress",
    "ImageFailure",
    "GeneratedImage",
    "Reference",
    "GeminiImages",
    "OpenAIImages",
    "ReplicateImages",
    "FakeImages",
    "image_engines",
    "image_tools",
]

#: Set to an engine name, or several separated by commas, to choose the engines
#: of a generator that was not given any.
ENGINE_ENV = "AGENT_HARNESS_IMAGES"

RATIOS = ("1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9")
_ALIASES = {"square": "1:1", "landscape": "16:9", "wide": "16:9", "portrait": "9:16",
            "tall": "9:16", "auto": "", "any": "", "none": ""}
_EXTENSIONS = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp",
               "image/gif": "gif"}
_MAX_PROMPT = 8_000
_Timeout = (TimeoutError, asyncio.TimeoutError)
_SLUG = re.compile(r"[^a-z0-9]+")
_PERCENT = re.compile(r"(\d{1,3})\s?%")
#: What a provider says when the *content* was the problem, not the request.
_REFUSED = re.compile(
    r"safety|moderat|content.?policy|policy.?violation|prohibited|nsfw|flagged|"
    r"sensitive|blocked|not allowed|violat", re.IGNORECASE)


# ----------------------------------------------------------------------
# images, read without a library
# ----------------------------------------------------------------------
def sniff(data: bytes) -> tuple[str, int, int] | None:
    """`(media type, width, height)` of an image, from its first bytes.

    None when the bytes are not a PNG, JPEG, WebP or GIF — which is how an
    error page saved as `photo.png` is told from a photo.
    """
    if len(data) < 16:
        return None
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        width, height = struct.unpack(">II", data[16:24]) if len(data) >= 24 else (0, 0)
        return "image/png", width, height
    if data[:3] == b"\xff\xd8\xff":
        index = 2
        while index + 9 < len(data):
            if data[index] != 0xFF:
                index += 1
                continue
            marker = data[index + 1]
            if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB,
                          0xCD, 0xCE, 0xCF):
                height, width = struct.unpack(">HH", data[index + 5:index + 9])
                return "image/jpeg", width, height
            if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
                index += 2
                continue
            index += 2 + struct.unpack(">H", data[index + 2:index + 4])[0]
        return "image/jpeg", 0, 0
    if data[:6] in (b"GIF87a", b"GIF89a"):
        width, height = struct.unpack("<HH", data[6:10])
        return "image/gif", width, height
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        kind = data[12:16]
        if kind == b"VP8X" and len(data) >= 30:
            width = int.from_bytes(data[24:27], "little") + 1
            height = int.from_bytes(data[27:30], "little") + 1
            return "image/webp", width, height
        if kind == b"VP8 " and len(data) >= 30:
            width, height = struct.unpack("<HH", data[26:30])
            return "image/webp", width & 0x3FFF, height & 0x3FFF
        if kind == b"VP8L" and len(data) >= 25:
            bits = int.from_bytes(data[21:25], "little")
            return "image/webp", (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1
        return "image/webp", 0, 0
    return None


def solid_png(width: int, height: int, rgb: tuple[int, int, int]) -> bytes:
    """A PNG of one colour. What `FakeImages` draws."""
    def chunk(kind: bytes, body: bytes) -> bytes:
        return (struct.pack(">I", len(body)) + kind + body
                + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF))

    row = b"\x00" + bytes(rgb) * width
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(row * height, 9)) + chunk(b"IEND", b""))


def _size(count: int) -> str:
    return f"{count / 1_048_576:.1f} MB" if count >= 1_048_576 else f"{max(1, count // 1024)} KB"


def _slug(text: str, fallback: str = "image") -> str:
    slug = _SLUG.sub("-", text.lower()).strip("-")
    return "-".join(slug.split("-")[:6])[:48].strip("-") or fallback


# ----------------------------------------------------------------------
# what goes in, what comes out
# ----------------------------------------------------------------------
@dataclass
class Reference:
    """An image a generation is given: to edit, to follow, to keep a likeness of."""

    data: bytes
    media_type: str = "image/png"
    name: str = "reference"

    @property
    def b64(self) -> str:
        return base64.b64encode(self.data).decode()

    @property
    def data_uri(self) -> str:
        return f"data:{self.media_type};base64,{self.b64}"


@dataclass
class ImageRequest:
    """One generation, as every engine is asked for it."""

    prompt: str
    references: list[Reference] = field(default_factory=list)
    #: e.g. "16:9"; empty to leave it to the model.
    aspect_ratio: str = ""
    count: int = 1
    #: A resolution hint for models that take one: "1K", "2K", "4K".
    size: str = ""
    seed: int | None = None
    #: Passed to the engine as they are: `quality`, `background`, a model's own.
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class GeneratedImage:
    """One picture that was made."""

    data: bytes
    media_type: str = "image/png"
    width: int = 0
    height: int = 0
    prompt: str = ""
    engine: str = ""
    model: str = ""
    #: What it is called: the file name it was, or would be, saved under.
    name: str = ""
    #: Where it was saved; empty when it was not.
    path: str = ""
    #: Anything the model said alongside it.
    text: str = ""

    @property
    def extension(self) -> str:
        return _EXTENSIONS.get(self.media_type, "png")

    def block(self) -> ImageBlock:
        """The image as something a model can be shown."""
        return ImageBlock.from_bytes(self.data, self.media_type, name=self.name or "image")

    def save(self, path: str | Path) -> Path:
        target = Path(path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.data)
        return target

    def describe(self) -> str:
        shape = f"{self.width}x{self.height} " if self.width else ""
        return f"{self.path or self.name} — {shape}{self.extension}, {_size(len(self.data))}"

    def __repr__(self) -> str:  # never the bytes
        return f"<GeneratedImage {self.describe()}>"


class ImageFailure(Exception):
    """One engine could not make the image. Raised by engines, read by `ImageGenerator`.

    ``retryable``    the same request may succeed in a moment (429, 5xx, timeout)
    ``retry_after``  seconds the engine asked us to wait, when it said
    ``refused``      the provider declined the *content*. Not retried, and not
                     taken to another provider.
    ``unhealthy``    counts towards leaving the engine alone for a while
    """

    def __init__(self, message: str, *, retryable: bool = False,
                 retry_after: float | None = None, refused: bool = False,
                 unhealthy: bool = True, status: int | None = None) -> None:
        super().__init__(message)
        self.retryable = retryable and not refused
        self.retry_after = retry_after
        self.refused = refused
        self.unhealthy = unhealthy and not refused
        self.status = status


@dataclass
class ImageProgress:
    """Where a job has got to. Handed to `on_progress`."""

    job: str
    status: str
    message: str = ""
    #: 0.0 to 1.0 when the engine reports how far along it is; None when it does not.
    fraction: float | None = None
    engine: str = ""
    elapsed: float = 0.0


Reporter = Callable[..., None]


class ImageJob:
    """One generation under way, or finished.

    ``status`` is `queued`, `running`, `done`, `failed` or `cancelled`.
    """

    def __init__(self, prompt: str, count: int = 1) -> None:
        self.id = new_id("img")
        self.prompt = prompt
        self.count = count
        self.status = "queued"
        self.message = ""
        self.fraction: float | None = None
        self.engine = ""
        self.images: list[GeneratedImage] = []
        self.error: BaseException | None = None
        self.started = time.monotonic()
        self.finished: float | None = None
        #: Has what it made been handed to whoever asked? (For `image_status`.)
        self.collected = False
        self._task: asyncio.Task[Any] | None = None

    @property
    def done(self) -> bool:
        return self.status in ("done", "failed", "cancelled")

    @property
    def elapsed(self) -> float:
        return (self.finished or time.monotonic()) - self.started

    def result(self) -> list[GeneratedImage]:
        """What was made. Raises what went wrong, or `ToolError` if it is not over."""
        if self.error is not None:
            raise self.error
        if not self.done:
            raise ToolError(f"job {self.id} is still {self.describe()}", tool="generate_image")
        return self.images

    async def wait(self, timeout: float | None = None) -> list[GeneratedImage]:
        """Wait for it to finish. On a timeout the job keeps running."""
        task = self._task
        if task is not None and not task.done():
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout)
            except _Timeout:
                raise ToolError(f"job {self.id} is still {self.describe()}",
                                tool="generate_image") from None
            except asyncio.CancelledError:
                if not task.cancelled():
                    raise              # the waiter was cancelled, not the job
            except Exception:  # noqa: S110 - kept on the job, raised by result()
                pass
        return self.result()

    def cancel(self) -> bool:
        """Stop it. The engine is told to stop too, where it can be."""
        if self.done or self._task is None:
            return False
        self._task.cancel()
        return True

    def describe(self) -> str:
        """One line on where it stands."""
        took = f"{self.elapsed:.0f}s"
        if self.status == "done":
            return f"done — {len(self.images)} image{'s' * (len(self.images) != 1)} in {took}"
        if self.status == "failed":
            return f"failed after {took}: {self.error}"
        if self.status == "cancelled":
            return f"cancelled after {took}"
        where = f" on {self.engine}" if self.engine else ""
        how_far = f" — {self.fraction:.0%}" if self.fraction is not None else ""
        detail = f" ({self.message})" if self.message else ""
        return f"{self.status}{where}{how_far}, {took}{detail}"

    def __repr__(self) -> str:  # pragma: no cover - debugging affordance
        return f"<ImageJob {self.id} {self.describe()}>"


# ----------------------------------------------------------------------
# engines
# ----------------------------------------------------------------------
def _redact(text: str, secrets: Iterable[str]) -> str:
    for secret in secrets:
        if secret and len(secret) >= 4:
            text = text.replace(secret, "***")
    return text


class ImageEngine:
    """One image service, spoken to over HTTP.

    Subclass it for a service that is not here: `generate` takes the request and
    an `httpx.AsyncClient` and returns the images, as bytes or as
    `GeneratedImage`s. `self.send` makes a request and turns a bad answer into
    an `ImageFailure` of the right kind; `report(...)` says how far along it is.
    The generator brings the retries, the fallback, the references and the jobs.
    """

    name = ""
    url = ""
    env: tuple[str, ...] = ()
    needs_key = True
    default_model = ""
    #: How many reference images one request may carry; 0 for none.
    max_references = 0

    def __init__(self, api_key: str | None = None, *, model: str | None = None,
                 base_url: str | None = None, headers: dict[str, str] | None = None,
                 **options: Any) -> None:
        self._api_key = api_key
        self.model = model or self.default_model
        self._base_url = base_url
        self.headers = dict(headers or {})
        #: Sent with every request of this engine.
        self.options = options

    # The environment is read when the engine is used, not when it is built.
    @property
    def api_key(self) -> str:
        if self._api_key:
            return self._api_key
        return next((os.environ[v] for v in self.env if os.environ.get(v)), "")

    @property
    def base_url(self) -> str:
        return (self._base_url or self.url).rstrip("/")

    def ready(self) -> str:
        """Empty when the engine can be called; otherwise what is missing."""
        if self.needs_key and not self.api_key:
            return f"no API key — set {' or '.join(self.env) or 'api_key='}"
        return ""

    def secrets(self) -> list[str]:
        return [self.api_key]

    def failure(self, response: Any) -> ImageFailure | None:
        """The failure an HTTP answer amounts to; None for a good one."""
        status = response.status_code
        if 200 <= status < 300:
            return None
        detail = " ".join(response.text[:600].split())[:300]
        if status in (401, 403) and not _REFUSED.search(detail):
            return ImageFailure("the API key was refused", status=status)
        if status == 402 or (status == 429 and re.search(
                r"quota|credit|billing|insufficient", detail, re.IGNORECASE)):
            return ImageFailure("out of quota or credit", status=status)
        if status == 429:
            return ImageFailure("rate limited", retryable=True, status=status,
                                retry_after=retry_after(response.headers, detail))
        if status == 408 or status >= 500:
            return ImageFailure(f"the engine answered {status}", retryable=True,
                                status=status, retry_after=retry_after(
                                    response.headers, rate_limited=False))
        if _REFUSED.search(detail):
            return ImageFailure(f"the provider refused this: {detail}", refused=True,
                                status=status)
        return ImageFailure(f"the request was rejected ({status}): {detail}",
                            unhealthy=False, status=status)

    async def send(self, client: Any, method: str, url: str, **kw: Any) -> Any:
        """One HTTP request; its JSON back. Raises `ImageFailure` for a bad answer."""
        headers = {**self.headers, **(kw.pop("headers", None) or {})}
        response = await client.request(method, url, headers=headers, **kw)
        problem = self.failure(response)
        if problem is not None:
            raise problem
        try:
            return response.json()
        except ValueError:
            raise ImageFailure("answered with something that is not JSON",
                               retryable=True) from None

    async def download(self, client: Any, url: str) -> bytes:
        """Fetch a finished image from where the engine left it."""
        last: ImageFailure | None = None
        for attempt in range(3):
            response = await client.get(url, follow_redirects=True)
            if response.status_code == 200 and response.content:
                return response.content
            last = ImageFailure(f"the finished image could not be fetched "
                                f"({response.status_code})", retryable=True)
            await asyncio.sleep(0.5 * (attempt + 1))
        raise last or ImageFailure("the finished image could not be fetched")

    async def generate(self, request: ImageRequest, client: Any,
                       report: Reporter) -> list[Any]:
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover - debugging affordance
        return f"<ImageEngine {self.name} {self.model}>"


class GeminiImages(ImageEngine):
    """Gemini's image models ("Nano Banana"), and Imagen, through the Gemini API.

    Reference images are sent with the prompt: the model edits them, combines
    them, or keeps to their style. An `imagen-…` model is text-to-image only.
    """

    name = "gemini"
    url = "https://generativelanguage.googleapis.com/v1beta"
    env = ("GEMINI_API_KEY", "GOOGLE_API_KEY")
    default_model = "gemini-2.5-flash-image"
    max_references = 14
    #: The finish reasons that mean the content was declined.
    _DECLINED = ("SAFETY", "PROHIBITED", "BLOCKLIST", "RECITATION", "SPII")

    def _headers(self) -> dict[str, str]:
        return {"x-goog-api-key": self.api_key}

    async def generate(self, request: ImageRequest, client: Any,
                       report: Reporter) -> list[Any]:
        if self.model.startswith("imagen"):
            return await self._imagen(request, client)
        # One picture a call: ask as many times as pictures are wanted.
        made = await asyncio.gather(*(self._one(request, client, n)
                                      for n in range(request.count)))
        return [image for batch in made for image in batch][:request.count]

    async def _one(self, request: ImageRequest, client: Any, n: int) -> list[Any]:
        parts: list[dict[str, Any]] = [{"text": request.prompt}]
        parts += [{"inlineData": {"mimeType": r.media_type, "data": r.b64}}
                  for r in request.references]
        config: dict[str, Any] = {"responseModalities": ["TEXT", "IMAGE"]}
        shape = {k: v for k, v in (("aspectRatio", request.aspect_ratio),
                                   ("imageSize", request.size)) if v}
        if shape:
            config["imageConfig"] = shape
        if request.seed is not None:
            config["seed"] = request.seed + n
        config.update(request.options.get("generation_config") or {})
        data = await self.send(
            client, "POST", f"{self.base_url}/models/{self.model}:generateContent",
            headers=self._headers(),
            json={"contents": [{"role": "user", "parts": parts}],
                  "generationConfig": config})
        blocked = (data.get("promptFeedback") or {}).get("blockReason")
        if blocked:
            raise ImageFailure(f"the provider refused this prompt ({blocked})",
                               refused=True)
        candidates = data.get("candidates") or []
        candidate = candidates[0] if candidates and isinstance(candidates[0], dict) else {}
        found = (candidate.get("content") or {}).get("parts") or []
        said = " ".join(p["text"] for p in found
                        if isinstance(p, dict) and p.get("text")).strip()
        images: list[Any] = []
        for part in found:
            blob = (part.get("inlineData") or part.get("inline_data")
                    if isinstance(part, dict) else None)
            if blob and blob.get("data"):
                images.append(GeneratedImage(
                    data=_decode(blob["data"]), text=said,
                    media_type=blob.get("mimeType") or blob.get("mime_type") or "image/png"))
        if images:
            return images
        reason = str(candidate.get("finishReason") or "")
        if any(mark in reason for mark in self._DECLINED):
            raise ImageFailure(f"the provider refused this ({reason})"
                               + (f": {said[:200]}" if said else ""), refused=True)
        raise ImageFailure("the model answered without an image"
                           + (f" ({reason})" if reason else "")
                           + (f": {said[:200]}" if said else ""),
                           retryable=True, unhealthy=False)

    async def _imagen(self, request: ImageRequest, client: Any) -> list[Any]:
        if request.references:
            raise ImageFailure(f"{self.model} takes no reference images",
                               unhealthy=False)
        parameters: dict[str, Any] = {"sampleCount": min(request.count, 4)}
        if request.aspect_ratio:
            parameters["aspectRatio"] = request.aspect_ratio
        data = await self.send(
            client, "POST", f"{self.base_url}/models/{self.model}:predict",
            headers=self._headers(),
            json={"instances": [{"prompt": request.prompt}], "parameters": parameters})
        made = [GeneratedImage(data=_decode(p["bytesBase64Encoded"]),
                               media_type=p.get("mimeType") or "image/png")
                for p in data.get("predictions") or []
                if isinstance(p, dict) and p.get("bytesBase64Encoded")]
        if not made:
            raise ImageFailure("the provider refused this prompt, or filtered every "
                               "image it made", refused=True)
        return made


class OpenAIImages(ImageEngine):
    """OpenAI's image API — or anything that speaks it (`base_url=`).

    With reference images the request goes to the edits endpoint, which takes
    several for `gpt-image-1`.
    """

    name = "openai"
    url = "https://api.openai.com/v1"
    env = ("OPENAI_API_KEY",)
    default_model = "gpt-image-1"
    max_references = 16
    _SIZES = {"1:1": "1024x1024", "landscape": "1536x1024", "portrait": "1024x1536"}

    def _dimensions(self, ratio: str) -> str:
        if not ratio:
            return "1024x1024" if self.model.startswith("dall-e") else "auto"
        width, _, height = ratio.partition(":")
        shape = ("1:1" if width == height else
                 "landscape" if int(width) > int(height) else "portrait")
        if self.model == "dall-e-3":
            return {"1:1": "1024x1024", "landscape": "1792x1024",
                    "portrait": "1024x1792"}[shape]
        return self._SIZES[shape]

    async def generate(self, request: ImageRequest, client: Any,
                       report: Reporter) -> list[Any]:
        fields: dict[str, Any] = {
            "model": self.model, "prompt": request.prompt, "n": request.count,
            "size": self._dimensions(request.aspect_ratio),
            **self.options, **{k: v for k, v in request.options.items()
                               if k != "generation_config"}}
        if self.model.startswith("dall-e"):
            fields.setdefault("response_format", "b64_json")
        auth = {"Authorization": f"Bearer {self.api_key}"}
        if request.references:
            if self.model == "dall-e-3":
                raise ImageFailure("dall-e-3 takes no reference images", unhealthy=False)
            files = [("image[]", (f"{_slug(r.name, 'reference')}."
                                  f"{_EXTENSIONS.get(r.media_type, 'png')}",
                                  r.data, r.media_type)) for r in request.references]
            data = await self.send(client, "POST", f"{self.base_url}/images/edits",
                                   headers=auth, files=files,
                                   data={k: str(v) for k, v in fields.items()})
        else:
            data = await self.send(client, "POST", f"{self.base_url}/images/generations",
                                   headers=auth, json=fields)
        kind = f"image/{fields.get('output_format', 'png')}".replace("jpg", "jpeg")
        made: list[Any] = []
        for item in data.get("data") or []:
            if not isinstance(item, dict):
                continue
            if item.get("b64_json"):
                made.append(GeneratedImage(data=_decode(item["b64_json"]), media_type=kind,
                                           text=item.get("revised_prompt") or ""))
            elif item.get("url"):
                made.append(GeneratedImage(data=await self.download(client, item["url"]),
                                           text=item.get("revised_prompt") or ""))
        if not made:
            raise ImageFailure("the engine answered without an image", retryable=True)
        return made


class ReplicateImages(ImageEngine):
    """Any image model hosted on Replicate — FLUX, Ideogram, Seedream, the rest.

    A prediction is started and then watched: this is the engine whose progress
    is the provider's own. `reference_key` is the name the model gives its image
    input (`input_image`, `image`, `image_input`); `reference_list=True` for a
    model that takes several. `inputs` are sent with every request.
    """

    name = "replicate"
    url = "https://api.replicate.com/v1"
    env = ("REPLICATE_API_TOKEN", "REPLICATE_API_KEY")
    default_model = "black-forest-labs/flux-kontext-pro"
    max_references = 1

    def __init__(self, api_key: str | None = None, *, model: str | None = None,
                 reference_key: str = "input_image", reference_list: bool = False,
                 inputs: dict[str, Any] | None = None, poll_interval: float = 1.5,
                 **kw: Any) -> None:
        super().__init__(api_key, model=model, **kw)
        self.reference_key = reference_key
        self.reference_list = reference_list
        self.inputs = dict(inputs or {})
        self.poll_interval = poll_interval
        if reference_list:
            self.max_references = 10

    async def generate(self, request: ImageRequest, client: Any,
                       report: Reporter) -> list[Any]:
        made = await asyncio.gather(*(self._one(request, client, report, n)
                                      for n in range(request.count)))
        return [image for batch in made for image in batch][:request.count]

    async def _one(self, request: ImageRequest, client: Any, report: Reporter,
                   n: int) -> list[Any]:
        auth = {"Authorization": f"Bearer {self.api_key}"}
        inputs: dict[str, Any] = {"prompt": request.prompt, **self.inputs}
        if request.aspect_ratio:
            inputs["aspect_ratio"] = request.aspect_ratio
        if request.seed is not None:
            inputs["seed"] = request.seed + n
        if request.references:
            uris = [r.data_uri for r in request.references]
            inputs[self.reference_key] = uris if self.reference_list else uris[0]
        inputs.update({k: v for k, v in request.options.items()
                       if k != "generation_config"})
        name, _, version = self.model.partition(":")
        if version:
            started = await self.send(client, "POST", f"{self.base_url}/predictions",
                                      headers=auth, json={"version": version,
                                                          "input": inputs})
        else:
            started = await self.send(
                client, "POST", f"{self.base_url}/models/{name}/predictions",
                headers=auth, json={"input": inputs})
        watch = (started.get("urls") or {}).get("get") or (
            f"{self.base_url}/predictions/{started.get('id')}")
        stop = (started.get("urls") or {}).get("cancel")
        state, misses, wait = started, 0, self.poll_interval
        try:
            while True:
                status = str(state.get("status") or "")
                if status == "succeeded":
                    break
                if status in ("failed", "canceled"):
                    why = " ".join(str(state.get("error") or status).split())[:300]
                    raise ImageFailure(
                        f"the provider refused this: {why}" if _REFUSED.search(why)
                        else f"the prediction {status}: {why}",
                        refused=bool(_REFUSED.search(why)), unhealthy=False)
                marks = _PERCENT.findall(str(state.get("logs") or "")[-400:])
                report(status or "starting",
                       fraction=min(int(marks[-1]), 100) / 100 if marks else None)
                await asyncio.sleep(wait)
                wait = min(wait * 1.3, 5.0)
                try:
                    state = await self.send(client, "GET", watch, headers=auth)
                    misses = 0
                except ImageFailure as exc:
                    # A status check that fails says nothing about the picture.
                    misses += 1
                    if not exc.retryable or misses >= 5:
                        raise
        except BaseException:
            if stop:
                # Whatever ended the wait — a timeout, a cancel — stop paying for it.
                try:
                    await asyncio.shield(client.post(stop, headers=auth))
                except BaseException:  # noqa: S110 - best effort, on the way out
                    pass
            raise
        output = state.get("output")
        links = [output] if isinstance(output, str) else [
            o for o in output or [] if isinstance(o, str)]
        if not links:
            raise ImageFailure("the prediction finished without an image")
        report("downloading", fraction=1.0)
        return [GeneratedImage(data=await self.download(client, link)) for link in links]


class FakeImages(ImageEngine):
    """An engine that draws without a network or a key — for tests and examples.

    Each image is a small PNG of one colour, chosen by the prompt, so the same
    prompt gives the same picture. `steps` and `delay` make a job take time and
    report progress; `fail` is raised instead, once per entry.

        images = ImageGenerator(FakeImages(steps=4, delay=0.01))
    """

    name = "fake"
    needs_key = False
    default_model = "fake-image-1"
    max_references = 8

    def __init__(self, *, steps: int = 0, delay: float = 0.0, side: int = 64,
                 fail: Iterable[BaseException] = (), name: str = "fake") -> None:
        super().__init__()
        self.name = name
        self.steps, self.delay, self.side = steps, delay, side
        self.fail = list(fail)
        #: Every request this engine was given, in order.
        self.requests: list[ImageRequest] = []

    async def generate(self, request: ImageRequest, client: Any,
                       report: Reporter) -> list[Any]:
        self.requests.append(request)
        if self.fail:
            raise self.fail.pop(0)
        for step in range(self.steps):
            report("processing", fraction=step / self.steps)
            await asyncio.sleep(self.delay)
        width, _, height = (request.aspect_ratio or "1:1").partition(":")
        scale = self.side / max(int(width), int(height))
        made = []
        for n in range(request.count):
            seed = f"{request.prompt}|{len(request.references)}|{request.seed}|{n}"
            colour = hashlib.sha256(seed.encode()).digest()[:3]
            made.append(solid_png(max(1, round(int(width) * scale)),
                                  max(1, round(int(height) * scale)),
                                  (colour[0], colour[1], colour[2])))
        return made


class _Function(ImageEngine):
    """A function of your own as an engine: `(request) -> images`, sync or async."""

    needs_key = False
    max_references = 16

    def __init__(self, fn: Callable[..., Any]) -> None:
        super().__init__()
        self.fn = fn
        self.name = getattr(fn, "__name__", "") or "custom"
        self.model = self.name

    async def generate(self, request: ImageRequest, client: Any,
                       report: Reporter) -> list[Any]:
        made = self.fn(request)
        if inspect.isawaitable(made):
            made = await made
        return list(made) if isinstance(made, (list, tuple)) else [made]


def _decode(text: str) -> bytes:
    try:
        return base64.b64decode(text, validate=False)
    except (binascii.Error, ValueError):
        raise ImageFailure("the engine sent an image that could not be decoded") from None


#: In the order they are preferred when the choice is left to the environment.
ENGINES: dict[str, type[ImageEngine]] = {
    "gemini": GeminiImages, "openai": OpenAIImages, "replicate": ReplicateImages}
_NAMES = {"google": "gemini", "nano-banana": "gemini", "nanobanana": "gemini",
          "imagen": "gemini", "gpt-image": "openai", "dall-e": "openai",
          "dalle": "openai", "flux": "replicate"}


def image_engines() -> list[dict[str, Any]]:
    """Every engine that ships, and whether this environment can call it."""
    out = []
    for name, kind in ENGINES.items():
        engine = kind()
        out.append({"name": name, "ready": not engine.ready(), "missing": engine.ready(),
                    "needs": " or ".join(kind.env), "model": kind.default_model})
    return out


def _engine(spec: Any, api_key: str | None = None, model: str | None = None) -> ImageEngine:
    if isinstance(spec, ImageEngine):
        return spec
    kw: dict[str, Any] = {**({"api_key": api_key} if api_key else {}),
                          **({"model": model} if model else {})}
    if isinstance(spec, type) and issubclass(spec, ImageEngine):
        return spec(**kw)
    if isinstance(spec, str):
        name = spec.strip().lower()
        name = _NAMES.get(name, name)
        if name not in ENGINES:
            raise ConfigurationError(
                f"no image engine named {spec!r}; there are: {', '.join(ENGINES)}")
        return ENGINES[name](**kw)
    if callable(spec):
        return _Function(spec)
    raise ConfigurationError(
        "an image engine is a name, an ImageEngine, or a function — got "
        f"{type(spec).__name__}")


# ----------------------------------------------------------------------
# the generator
# ----------------------------------------------------------------------
class ImageGenerator:
    """Image generation with reference images, jobs, retries and fallback.

    ``engine``            a name, an `ImageEngine`, a function, or a list to fall
                          back through. None: `$AGENT_HARNESS_IMAGES`, else every
                          engine with a key set.
    ``model``             the model of a single named engine
    ``aspect_ratio``      the shape when none is asked for: "1:1", "16:9", …
    ``max_count``         the most images one request may ask for
    ``max_references`` / ``max_reference_bytes``  how many reference images, and
                          how large each may be
    ``timeout``           seconds for one attempt at one engine
    ``deadline``          seconds for the whole job, retries and fallbacks included
    ``retries``           tries after the first, per engine (or pass `retry=`)
    ``max_images``        the most images this generator will ever make; None for
                          no limit. A guard against a loop that will not stop.
    ``cost_per_image``    USD, charged to the run's budget for each image made
    ``output_dir``        where images are saved when there is no workspace
    ``allow_private``     let a reference be fetched from a private address
    ``on_progress``       called with an `ImageProgress` as a job moves
    """

    def __init__(
        self,
        engine: Any = None,
        *,
        api_key: str | None = None,
        model: str | None = None,
        aspect_ratio: str = "",
        max_count: int = 4,
        max_references: int = 8,
        max_reference_bytes: int = 20 * 1024 * 1024,
        timeout: float = 180.0,
        deadline: float = 600.0,
        retries: int = 2,
        retry: RetryPolicy | None = None,
        max_images: int | None = None,
        max_concurrency: int = 3,
        cost_per_image: float | None = None,
        output_dir: str | Path | None = None,
        directory: str = "images",
        allow_private: bool = False,
        on_progress: Callable[[ImageProgress], Any] | None = None,
        failure_threshold: int = 3,
        cooldown: float = 60.0,
        transport: Any = None,
        client: Any = None,
    ) -> None:
        specs = list(engine) if isinstance(engine, (list, tuple)) else (
            [] if engine is None else [engine])
        if (api_key or model) and len(specs) != 1:
            raise ConfigurationError(
                "api_key= and model= go with one named engine; for several, pass "
                "each as an engine of its own — GeminiImages(model=...)")
        self._engines: list[ImageEngine] | None = (
            [_engine(spec, api_key, model) for spec in specs] if specs else None)
        self.aspect_ratio = self._ratio(aspect_ratio)
        self.max_count = max(1, int(max_count))
        self.max_references = max(0, int(max_references))
        self.max_reference_bytes = int(max_reference_bytes)
        self.timeout = float(timeout)
        self.deadline = float(deadline)
        self.retry = retry or RetryPolicy(max_retries=max(0, int(retries)),
                                          initial_delay=1.0, max_delay=15.0,
                                          max_retry_after=30.0)
        self.max_images = max_images
        self.max_concurrency = max(1, int(max_concurrency))
        self.cost_per_image = cost_per_image
        self.output_dir = Path(output_dir).expanduser() if output_dir else None
        self.directory = directory.strip("/") or "images"
        self.allow_private = allow_private
        self.on_progress = on_progress
        self._failure_threshold = failure_threshold
        self._cooldown = cooldown
        self._transport = transport
        self._client = client
        self._breakers: dict[str, CircuitBreaker] = {}
        self._gates: dict[int, asyncio.Semaphore] = {}
        self._resolved: dict[str, bool] = {}
        #: The jobs of this generator, newest last.
        self.jobs: OrderedDict[str, ImageJob] = OrderedDict()
        #: The newest images, by name — so one can be the next one's reference.
        self.recent: OrderedDict[str, GeneratedImage] = OrderedDict()
        #: How many images have been made.
        self.made = 0
        self.stats: dict[str, dict[str, Any]] = {}

    # ---- engines ----------------------------------------------------------
    def _resolve(self) -> list[ImageEngine]:
        if self._engines is None:
            named = [n.strip() for n in os.environ.get(ENGINE_ENV, "").split(",")
                     if n.strip()]
            self._engines = ([_engine(n) for n in named] if named else
                             [e for e in (kind() for kind in ENGINES.values())
                              if not e.ready()])
        return self._engines

    @property
    def engines(self) -> list[str]:
        """The engines a generation goes through, in order."""
        return [engine.name for engine in self._resolve()]

    def _breaker(self, name: str) -> CircuitBreaker:
        if name not in self._breakers:
            self._breakers[name] = CircuitBreaker(self._failure_threshold, self._cooldown)
        return self._breakers[name]

    def _count(self, name: str, outcome: str, error: str = "") -> None:
        entry = self.stats.setdefault(name, {"made": 0, "failed": 0, "last_error": ""})
        entry[outcome] += 1
        if error:
            entry["last_error"] = error

    # ---- the request ----------------------------------------------------------
    @staticmethod
    def _ratio(value: Any) -> str:
        text = str(value or "").strip().lower().replace("x", ":").replace("/", ":")
        text = _ALIASES.get(text, text)
        if text and text not in RATIOS:
            raise ToolError(f"aspect_ratio is one of {', '.join(RATIOS)} — got {value!r}",
                            tool="generate_image")
        return text

    def _check(self, prompt: Any, count: Any) -> tuple[str, int]:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ToolError("the prompt is empty — describe the image to make",
                            tool="generate_image")
        prompt = prompt.strip()
        if len(prompt) > _MAX_PROMPT:
            raise ToolError(f"the prompt is {len(prompt)} characters; the most is "
                            f"{_MAX_PROMPT}", tool="generate_image")
        try:
            wanted = max(1, min(int(count or 1), self.max_count))
        except (TypeError, ValueError):
            wanted = 1
        if self.max_images is not None:
            left = self.max_images - self.made - sum(
                j.count for j in self.jobs.values() if not j.done)
            if left <= 0:
                raise ToolError(f"this generator's limit of {self.max_images} images "
                                "is used up", tool="generate_image")
            wanted = min(wanted, left)
        return prompt, wanted

    # ---- references ------------------------------------------------------------
    def _accept(self, data: bytes, name: str) -> Reference:
        if len(data) > self.max_reference_bytes:
            raise ToolError(f"reference {name!r} is {_size(len(data))}; the most is "
                            f"{_size(self.max_reference_bytes)}", tool="generate_image")
        kind = sniff(data)
        if kind is None:
            raise ToolError(f"reference {name!r} is not a PNG, JPEG, WebP or GIF image",
                            tool="generate_image")
        return Reference(data=data, media_type=kind[0], name=name)

    async def _fetch(self, url: str, client: Any) -> bytes:
        """A reference from the web: public addresses only, images only, capped."""
        for _ in range(4):
            parts = urlsplit(url)
            host = (parts.hostname or "").lower()
            if parts.scheme not in ("http", "https") or not host:
                raise ToolError(f"reference {url[:120]!r} is not an http(s) address",
                                tool="generate_image")
            if not self.allow_private and await is_private_host(host, self._resolved):
                raise ToolError(f"reference {host} is a private or loopback address",
                                tool="generate_image")
            try:
                async with client.stream("GET", url, follow_redirects=False) as response:
                    if response.status_code in (301, 302, 303, 307, 308):
                        url = urljoin(url, response.headers.get("location", ""))
                        continue
                    if response.status_code != 200:
                        raise ToolError(f"reference {url[:120]} answered "
                                        f"{response.status_code}", tool="generate_image")
                    chunks, total = [], 0
                    async for chunk in response.aiter_bytes():
                        total += len(chunk)
                        if total > self.max_reference_bytes:
                            raise ToolError(
                                f"reference {url[:120]} is larger than "
                                f"{_size(self.max_reference_bytes)}", tool="generate_image")
                        chunks.append(chunk)
                    return b"".join(chunks)
            except ToolError:
                raise
            except Exception as exc:
                raise ToolError(f"reference {url[:120]} could not be fetched "
                                f"({type(exc).__name__})", tool="generate_image") from None
        raise ToolError("the reference redirects too many times", tool="generate_image")

    async def _reference(self, item: Any, client: Any, *, trusted: bool,
                         workspace: Any = None, attachments: Iterable[Any] = ()
                         ) -> Reference:
        """One reference, from whatever names it.

        `trusted` is the difference between your code and a model: your code may
        name any file on this machine; a model may name only what it was shown.
        """
        if isinstance(item, Reference):
            return self._accept(item.data, item.name)
        if isinstance(item, GeneratedImage):
            return self._accept(item.data, item.name or "image")
        if isinstance(item, (bytes, bytearray)):
            return self._accept(bytes(item), "reference")
        if isinstance(item, MediaBlock):
            if item.inline:
                try:
                    return self._accept(item.read(), item.label)
                except OSError:
                    raise ToolError(f"reference {item.label!r} is no longer there",
                                    tool="generate_image") from None
            return self._accept(await self._fetch(item.url or "", client), item.label)
        if isinstance(item, Path):
            item = str(item) if trusted else item.as_posix()
        if not isinstance(item, str) or not item.strip():
            raise ToolError("a reference is the name of an image", tool="generate_image")
        text = item.strip()
        if text.startswith("data:image/"):
            try:
                return self._accept(base64.b64decode(text.split(",", 1)[1]), "reference")
            except (IndexError, binascii.Error, ValueError):
                raise ToolError("that data: reference could not be decoded",
                                tool="generate_image") from None
        if text.startswith(("http://", "https://")):
            name = text.rsplit("/", 1)[-1].split("?")[0] or "reference"
            return self._accept(await self._fetch(text, client), name)

        # An image made earlier, by its name or its path.
        for known in reversed(self.recent.values()):
            if text in (known.name, known.path) or text == Path(known.name).stem:
                return self._accept(known.data, known.name)
        # Something that came with the task.
        given = []
        for raw in attachments:
            try:
                block = raw if isinstance(raw, MediaBlock) else attach(raw)
            except Exception:  # noqa: S112 - not an attachment we can read
                continue
            if block.type == "image":
                given.append(block)
        position = re.fullmatch(r"attachment[:\s#]*(\d+)", text.lower())
        for number, block in enumerate(given, 1):
            if text in (block.name, block.label) or (position and int(position[1]) == number):
                return await self._reference(block, client, trusted=True)
        if workspace is not None:
            try:
                return self._accept(await workspace.aread_bytes(text), text)
            except Exception as exc:
                # Not a file there. But a file that is there and is no image,
                # or is too large, is worth saying so.
                if "not a PNG" in str(exc) or "the most is" in str(exc):
                    raise
        if trusted:
            file = Path(text).expanduser()
            if file.is_file():
                return self._accept(file.read_bytes(), file.name)
            raise ToolError(f"there is no image at {text!r}", tool="generate_image")
        have = [k.path or k.name for k in self.recent.values()][-6:]
        have += [f"attachment:{n} ({b.label})" for n, b in enumerate(given, 1)]
        raise ToolError(
            f"there is no image called {text!r}. A reference is "
            + ("a file in the workspace, " if workspace is not None else "")
            + "an image generated earlier, an attachment, or an http(s) address"
            + (f". Known: {', '.join(have)}" if have else ""), tool="generate_image")

    # ---- making -------------------------------------------------------------------
    @asynccontextmanager
    async def _http(self):
        if self._client is not None:
            yield self._client
            return
        import httpx

        from .. import __version__

        async with httpx.AsyncClient(
                timeout=self.timeout, follow_redirects=False, transport=self._transport,
                headers={"user-agent": f"agent-harness/{__version__}"}) as client:
            yield client

    def _gate(self) -> asyncio.Semaphore:
        loop = id(asyncio.get_running_loop())
        if loop not in self._gates:
            self._gates.clear()
            self._gates[loop] = asyncio.Semaphore(self.max_concurrency)
        return self._gates[loop]

    def _report(self, job: ImageJob, status: str, message: str = "", *,
                fraction: float | None = None,
                listener: Callable[[ImageProgress], Any] | None = None) -> None:
        job.message = message
        if fraction is not None or status != job.status:
            job.fraction = fraction
        if status in ("queued", "running", "done", "failed", "cancelled"):
            job.status = status
        update = ImageProgress(job=job.id, status=status, message=message,
                               fraction=fraction, engine=job.engine, elapsed=job.elapsed)
        for callback in (listener, self.on_progress):
            if callback is None:
                continue
            try:
                answer = callback(update)
                if inspect.isawaitable(answer):
                    task = asyncio.ensure_future(answer)
                    task.add_done_callback(lambda t: t.cancelled() or t.exception())
            except Exception:  # noqa: S110 - a listener must not fail the job
                pass

    async def _ask(self, engine: ImageEngine, request: ImageRequest, client: Any,
                   job: ImageJob, listener: Any) -> list[Any]:
        """One engine, tried until it answers, cannot, or the time is up."""
        missing = engine.ready()
        if missing:
            raise ImageFailure(missing)
        if len(request.references) > engine.max_references:
            raise ImageFailure(
                f"{engine.model or engine.name} takes "
                + (f"at most {engine.max_references} reference image"
                   f"{'s' * (engine.max_references != 1)}" if engine.max_references
                   else "no reference images")
                + f"; {len(request.references)} were given", unhealthy=False)

        def report(status: str, message: str = "", *, fraction: float | None = None) -> None:
            self._report(job, "running", message or status, fraction=fraction,
                         listener=listener)

        attempt = 0
        while True:
            left = self.deadline - job.elapsed
            if left <= 0:
                raise ImageFailure("the job ran out of time", unhealthy=False)
            try:
                return await asyncio.wait_for(
                    engine.generate(request, client, report), min(self.timeout, left))
            except ImageFailure as exc:
                failure = exc
            except asyncio.CancelledError:
                raise
            except _Timeout:
                failure = ImageFailure(
                    f"no image within {min(self.timeout, left):g}s", retryable=True)
            except Exception as exc:
                kind, module = type(exc).__name__, type(exc).__module__ or ""
                if module.startswith(("httpx", "httpcore")):
                    failure = ImageFailure(
                        "timed out" if "Timeout" in kind else
                        f"could not be reached ({kind})", retryable=True)
                else:
                    failure = ImageFailure(
                        _redact(f"{kind}: {exc}", engine.secrets())[:300])
            policy = self.retry
            if not failure.retryable or attempt >= policy.max_retries:
                raise failure
            wait = failure.retry_after
            if wait is None:
                wait = policy.backoff(attempt)
            elif wait > policy.max_retry_after:
                raise failure
            if job.elapsed + wait >= self.deadline:
                raise failure
            attempt += 1
            report("retrying", f"{engine.name}: {failure} — trying again "
                               f"({attempt} of {policy.max_retries})")
            await asyncio.sleep(wait)

    def _finish(self, raw: Iterable[Any], request: ImageRequest,
                engine: ImageEngine) -> list[GeneratedImage]:
        """What an engine returned, checked to be images and measured."""
        made: list[GeneratedImage] = []
        for item in raw:
            image = item if isinstance(item, GeneratedImage) else (
                GeneratedImage(data=bytes(item)) if isinstance(item, (bytes, bytearray))
                else None)
            if image is None:
                continue
            kind = sniff(image.data)
            if kind is None:
                continue
            image.media_type, image.width, image.height = kind
            image.prompt, image.engine = request.prompt, engine.name
            image.model = image.model or engine.model
            made.append(image)
        return made[:request.count]

    async def _run(self, job: ImageJob, request: ImageRequest, references: Iterable[Any],
                   context: dict[str, Any], listener: Any) -> list[GeneratedImage]:
        async with self._gate(), self._http() as client:
            self._report(job, "running", "reading the reference images"
                         if references else "starting", listener=listener)
            request.references = [
                await self._reference(item, client, trusted=context["trusted"],
                                      workspace=context["workspace"],
                                      attachments=context["attachments"])
                for item in references]
            engines = self._resolve()
            if not engines:
                raise ToolError(
                    "no image engine is set up — set " + ", ".join(
                        e["needs"].split(" or ")[0] for e in image_engines())
                    + ", or pass engine=", tool="generate_image")
            problems: list[str] = []
            for engine in engines:
                breaker = self._breaker(engine.name)
                if not breaker.allow():
                    problems.append(f"{engine.name}: left alone after repeated failures, "
                                    f"for another {breaker.retry_in():.0f}s")
                    continue
                job.engine = engine.name
                self._report(job, "running", f"asking {engine.name}", listener=listener)
                try:
                    raw = await self._ask(engine, request, client, job, listener)
                    made = self._finish(raw, request, engine)
                    if not made:
                        raise ImageFailure("what came back was not an image")
                except asyncio.CancelledError:
                    if breaker.state == "half_open":
                        breaker.record_failure()
                    raise
                except ImageFailure as exc:
                    message = _redact(str(exc), engine.secrets())
                    self._count(engine.name, "failed", message)
                    if exc.unhealthy:
                        breaker.record_failure()
                    else:
                        breaker.record_success()
                    if exc.refused:
                        # Declined for what was asked. Another provider is not
                        # a way round that.
                        raise ToolError(f"{engine.name} would not make this image — "
                                        f"{message}. Change the prompt or the "
                                        "references.", tool="generate_image") from None
                    problems.append(f"{engine.name}: {message}")
                    continue
                breaker.record_success()
                self._count(engine.name, "made")
                return made
            raise ToolError("image generation failed — " + "; ".join(problems),
                            tool="generate_image")

    async def _store(self, images: list[GeneratedImage], name: str,
                     context: dict[str, Any]) -> None:
        """Name each image and put it where it is kept."""
        workspace = context["workspace"]
        base = _slug(name) if name else _slug(images[0].prompt)
        for number, image in enumerate(images, 1):
            stem = base if len(images) == 1 else f"{base}-{number}"
            taken, attempt = True, 1
            while taken:
                image.name = f"{stem if attempt == 1 else f'{stem}-{attempt}'}.{image.extension}"
                relative = f"{self.directory}/{image.name}"
                if workspace is not None:
                    taken = await workspace.aexists(relative)
                elif self.output_dir is not None:
                    taken = (self.output_dir / image.name).exists()
                else:
                    taken = image.name in self.recent
                attempt += 1
            if workspace is not None:
                relative = f"{self.directory}/{image.name}"
                if hasattr(workspace, "sandbox"):
                    await workspace.awrite(relative, image.data)
                else:
                    workspace._writable()
                    target = workspace.resolve(relative)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(image.data)
                image.path = relative
            elif self.output_dir is not None:
                image.path = str(image.save(self.output_dir / image.name))
            self.recent[image.name] = image
            while len(self.recent) > 24:
                self.recent.popitem(last=False)

    async def submit(self, prompt: str, *, references: Iterable[Any] = (),
                     aspect_ratio: str | None = None, count: int = 1, size: str = "",
                     seed: int | None = None, name: str = "",
                     on_progress: Callable[[ImageProgress], Any] | None = None,
                     _context: dict[str, Any] | None = None, **options: Any) -> ImageJob:
        """Start a generation and return its job at once.

        Watch it with `job.describe()` and `job.done`, wait with `await
        job.wait()`, stop it with `job.cancel()`. What is wrong with the request
        itself — an empty prompt, too many references — is raised here.
        """
        prompt, wanted = self._check(prompt, count)
        refs = list(references or ())
        if len(refs) > self.max_references:
            raise ToolError(f"{len(refs)} reference images were given; the most is "
                            f"{self.max_references}", tool="generate_image")
        ratio = self.aspect_ratio if aspect_ratio is None else (
            self._ratio(aspect_ratio) or self.aspect_ratio)
        request = ImageRequest(prompt=prompt, aspect_ratio=ratio, count=wanted,
                               size=str(size or ""), seed=seed, options=dict(options))
        context = _context or {"trusted": True, "workspace": None, "attachments": (),
                               "state": {}}
        job = ImageJob(prompt, wanted)
        self.jobs[job.id] = job
        while len(self.jobs) > 64:
            self.jobs.popitem(last=False)

        async def work() -> None:
            try:
                images = await self._run(job, request, refs, context, on_progress)
                await self._store(images, name, context)
                job.images = images
                self.made += len(images)
                self._charge(len(images), context)
                job.finished = time.monotonic()
                self._report(job, "done", "", fraction=1.0, listener=on_progress)
            except asyncio.CancelledError:
                job.finished = time.monotonic()
                job.error = ToolError(f"job {job.id} was cancelled", tool="generate_image")
                self._report(job, "cancelled", listener=on_progress)
                raise
            except Exception as exc:
                job.finished = time.monotonic()
                job.error = exc if isinstance(exc, ToolError) else ToolError(
                    f"image generation failed: {exc}", tool="generate_image")
                self._report(job, "failed", str(job.error), listener=on_progress)

        job._task = asyncio.get_running_loop().create_task(work())
        return job

    def _charge(self, count: int, context: dict[str, Any]) -> None:
        if not self.cost_per_image or not count:
            return
        spent = Usage(cost_usd=self.cost_per_image * count)
        state = context.get("state") or {}
        result, guard = state.get("result"), state.get("guard")
        if result is not None:
            result.usage += spent
        if guard is not None:
            try:
                guard.record(spent, agent=getattr(result, "agent", ""))
            except Exception:  # noqa: S110 - the loop stops the run at its next check
                pass

    async def generate(self, prompt: str, *, references: Iterable[Any] = (),
                       aspect_ratio: str | None = None, count: int = 1, size: str = "",
                       seed: int | None = None, name: str = "",
                       on_progress: Callable[[ImageProgress], Any] | None = None,
                       **options: Any) -> list[GeneratedImage]:
        """Make images and return them. Raises `ToolError` when none could be made.

        `references` may be paths, URLs, bytes, `ImageBlock`s, or images made
        earlier. Cancelling this call cancels the job.
        """
        job = await self.submit(prompt, references=references, aspect_ratio=aspect_ratio,
                                count=count, size=size, seed=seed, name=name,
                                on_progress=on_progress, **options)
        try:
            return await job.wait()
        except asyncio.CancelledError:
            job.cancel()
            raise

    def job(self, job_id: str) -> ImageJob:
        try:
            return self.jobs[job_id.strip()]
        except KeyError:
            known = ", ".join(list(self.jobs)[-5:]) or "none"
            raise ToolError(f"there is no job {job_id!r}; the latest are: {known}",
                            tool="image_status") from None

    # ---- as tools ----------------------------------------------------------------
    def tools(self, **options: Any) -> list[Tool]:
        """`generate_image` and `image_status`, for an agent. See `image_tools`."""
        return image_tools(self, **options)

    def __repr__(self) -> str:  # pragma: no cover - debugging affordance
        return f"<ImageGenerator {', '.join(e.name for e in self._engines or []) or 'auto'}>"


def image_tools(generator: ImageGenerator | None = None, *, name: str = "generate_image",
                status_name: str = "image_status", wait: float = 120.0, show: bool = True,
                references: Iterable[Any] = (), permission: Permission = "allow",
                **options: Any) -> list[Tool]:
    """The tools an agent makes images with.

    ``generator``   the generator; one is made from `options` when not given
    ``wait``        seconds `generate_image` waits before handing back a job to
                    check on with `image_status`
    ``show``        show the model what was made, so it can judge it and try again
    ``references``  images every generation is given — a brand's style, a
                    character sheet — besides the ones the model names
    ``permission``  "ask" to approve every generation: each one costs money
    """
    images = generator or ImageGenerator(**options)
    always = list(references)
    tags = ["builtin", "images"]

    def context(ctx: Any) -> dict[str, Any]:
        state = getattr(ctx, "state", None) or {}
        return {"trusted": False, "workspace": getattr(ctx, "workspace", None),
                "attachments": state.get("attachments") or (), "state": state}

    def handed(job: ImageJob) -> Any:
        """What the model reads when a job is over."""
        made = job.result()
        job.collected = True
        lines = [f"Generated {len(made)} image{'s' * (len(made) != 1)} with "
                 f"{made[0].engine} ({made[0].model}) in {job.elapsed:.0f}s:"]
        lines += [f"- {image.describe()}" for image in made]
        said = next((image.text for image in made if image.text), "")
        if said:
            lines.append(f"The model said: {said[:400]}")
        lines.append("To change one, generate again with it as a reference: "
                     f"references=[\"{made[0].path or made[0].name}\"].")
        if not show:
            return "\n".join(lines)
        return ["\n".join(lines), *(image.block() for image in made)]

    def pending(job: ImageJob) -> str:
        return (f"Still working: job {job.id} is {job.describe()}. Call "
                f"{status_name}(job_id=\"{job.id}\") to collect it — do not start "
                "the same image again.")

    def keep(job: ImageJob, ctx: Any) -> None:
        result = (getattr(ctx, "state", None) or {}).get("result")
        if result is None or job.status != "done":
            return
        have = {a.name for a in result.artifacts}
        for image in job.images:
            if image.name in have:
                continue
            result.artifacts.append(Artifact(
                name=image.name, path=image.path or None, media_type=image.media_type,
                produced_by=getattr(ctx, "agent", ""),
                # Kept inline only when it was saved nowhere.
                **({} if image.path else {"data": base64.b64encode(image.data).decode()})))

    @tool(name=name, tags=tags, permission=permission)
    async def generate_image(prompt: str, references: list[str] | None = None,
                             aspect_ratio: str = "", count: int = 1, name: str = "",
                             ctx: Any = None) -> Any:
        """Generate an image from a description — or from other images, to edit or follow them.

        Say everything that matters in the prompt: subject, setting, style,
        lighting, any text to appear. To change an existing image, pass it as a
        reference and describe the change.

        Args:
            prompt: what the image should show.
            references: images to work from — a file in the workspace, the name of an image generated earlier, an attachment ("attachment:1"), or an http(s) address.
            aspect_ratio: the shape, e.g. "1:1", "16:9", "9:16", "4:3". Leave empty for the default.
            count: how many variations to make.
            name: a short file name for the result, without an extension.
        """
        job = await images.submit(
            prompt, references=[*always, *(references or [])], aspect_ratio=aspect_ratio,
            count=count, name=name, _context=context(ctx))
        try:
            await job.wait(wait)
        except ToolError:
            if not job.done:
                return pending(job)
            raise
        except asyncio.CancelledError:
            job.cancel()
            raise
        keep(job, ctx)
        return handed(job)

    @tool(name=status_name, tags=tags)
    async def image_status(job_id: str = "", wait: float = 0.0, cancel: bool = False,
                           ctx: Any = None) -> Any:
        """Check on an image that is still being generated, and collect it when it is done.

        Args:
            job_id: the job to check. Leave empty to list the latest jobs.
            wait: seconds to wait for it to finish before answering, at most 120.
            cancel: stop the job instead.
        """
        if not job_id.strip():
            latest = list(images.jobs.values())[-8:]
            return "\n".join(f"{j.id}: {j.describe()} — {j.prompt[:60]!r}"
                             for j in latest) or "No images have been asked for."
        job = images.job(job_id)
        if cancel:
            return (f"Job {job.id} is being cancelled." if job.cancel()
                    else f"Job {job.id} is already over: {job.describe()}")
        if not job.done and wait > 0:
            try:
                await job.wait(max(0.0, min(float(wait), 120.0)))
            except ToolError:
                pass
        if not job.done:
            return pending(job)
        keep(job, ctx)
        return handed(job)

    return [generate_image, image_status]
