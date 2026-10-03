"""Image generation: every engine's wire format, references, jobs and what goes wrong."""

from __future__ import annotations

import asyncio
import base64
import json

import httpx
import pytest

from agent_harness import (
    Agent,
    Budget,
    FakeProvider,
    Harness,
    ImageBlock,
    ImageGenerator,
    Workspace,
    cli,
    tool_call,
)
from agent_harness.errors import ConfigurationError, ToolError
from agent_harness.llm_providers.resilience import RetryPolicy
from agent_harness.toolkits import FakeImages, image_engines
from agent_harness.toolkits.images import (
    GeminiImages,
    ImageFailure,
    OpenAIImages,
    ReplicateImages,
    sniff,
    solid_png,
)
from agent_harness.types import ToolResultBlock

FAST = RetryPolicy(max_retries=2, initial_delay=0.0, max_delay=0.0, jitter=0.0,
                   max_retry_after=1.0)
KEYS = ("GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY", "REPLICATE_API_TOKEN",
        "REPLICATE_API_KEY", "AGENT_HARNESS_IMAGES")
RED = solid_png(8, 4, (200, 30, 30))
BLUE = solid_png(4, 4, (30, 30, 200))
B64 = base64.b64encode(RED).decode()
JPEG = (b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
        b"\xff\xc0\x00\x11\x08\x00\x20\x00\x30\x03\x01\x22\x00\x02\x11\x01\x03\x11\x01")


@pytest.fixture(autouse=True)
def no_keys(monkeypatch):
    for name in KEYS:
        monkeypatch.delenv(name, raising=False)


def served(handler, engine, **kw):
    """A generator whose engine is answered by `handler`, and the requests it saw."""
    seen: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    kw.setdefault("retry", FAST)
    return ImageGenerator(engine, transport=httpx.MockTransport(record), **kw), seen


def gemini_ok(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={"candidates": [{"finishReason": "STOP", "content": {
        "parts": [{"text": "Here is your fox."},
                  {"inlineData": {"mimeType": "image/png", "data": B64}}]}}]})


# --- reading images -----------------------------------------------------------------

def test_an_image_is_recognised_and_measured_from_its_bytes():
    assert sniff(RED) == ("image/png", 8, 4)
    assert sniff(JPEG) == ("image/jpeg", 48, 32)
    assert sniff(b"GIF89a" + (5).to_bytes(2, "little") + (7).to_bytes(2, "little")
                 + b"\x00" * 8) == ("image/gif", 5, 7)
    webp = b"RIFF\x00\x00\x00\x00WEBPVP8X\x0a\x00\x00\x00\x00\x00\x00\x00" + (
        (99).to_bytes(3, "little") + (49).to_bytes(3, "little"))
    assert sniff(webp) == ("image/webp", 100, 50)
    assert sniff(b"<html>an error page saved as photo.png</html>") is None
    assert sniff(b"") is None


# --- engines, on the wire -------------------------------------------------------------

async def test_gemini_sends_the_prompt_and_the_references_and_reads_the_image():
    images, seen = served(gemini_ok, GeminiImages("g-secret-key"))
    made = await images.generate("a red fox in snow", references=[BLUE, JPEG],
                                 aspect_ratio="16:9", seed=7, size="2K")
    request = seen[0]
    body = json.loads(request.content)
    assert request.url.path == "/v1beta/models/gemini-2.5-flash-image:generateContent"
    assert request.headers["x-goog-api-key"] == "g-secret-key"
    assert "g-secret-key" not in str(request.url)
    parts = body["contents"][0]["parts"]
    assert parts[0] == {"text": "a red fox in snow"}
    assert [p["inlineData"]["mimeType"] for p in parts[1:]] == ["image/png", "image/jpeg"]
    assert base64.b64decode(parts[1]["inlineData"]["data"]) == BLUE
    assert body["generationConfig"] == {
        "responseModalities": ["TEXT", "IMAGE"], "seed": 7,
        "imageConfig": {"aspectRatio": "16:9", "imageSize": "2K"}}

    image = made[0]
    assert image.data == RED and (image.width, image.height) == (8, 4)
    assert (image.engine, image.model) == ("gemini", "gemini-2.5-flash-image")
    assert image.text == "Here is your fox." and image.name == "a-red-fox-in-snow.png"
    assert isinstance(image.block(), ImageBlock) and "bytes" not in repr(image).lower()

    # One picture a call: three wanted is three calls.
    images, seen = served(gemini_ok, GeminiImages("k"))
    assert len(await images.generate("a fox", count=3)) == 3 and len(seen) == 3


async def test_imagen_is_asked_the_other_way_and_takes_no_references():
    def imagen(request):
        return httpx.Response(200, json={"predictions": [
            {"bytesBase64Encoded": B64, "mimeType": "image/png"}] * 2})

    images, seen = served(imagen, GeminiImages("k", model="imagen-4.0-generate-001"))
    made = await images.generate("a fox", count=2, aspect_ratio="portrait")
    assert seen[0].url.path.endswith("/models/imagen-4.0-generate-001:predict")
    assert json.loads(seen[0].content) == {
        "instances": [{"prompt": "a fox"}],
        "parameters": {"sampleCount": 2, "aspectRatio": "9:16"}}
    assert len(made) == 2
    with pytest.raises(ToolError, match="imagen-4.0-generate-001 takes no reference"):
        await images.generate("a fox", references=[BLUE])


async def test_openai_generates_and_edits_with_references():
    def openai(request):
        return httpx.Response(200, json={"data": [{"b64_json": B64}]})

    images, seen = served(openai, OpenAIImages("sk-secret-key", quality="high"))
    await images.generate("a fox", aspect_ratio="3:2", count=2)
    assert seen[0].url.path == "/v1/images/generations"
    assert seen[0].headers["authorization"] == "Bearer sk-secret-key"
    assert json.loads(seen[0].content) == {"model": "gpt-image-1", "prompt": "a fox",
                                           "n": 2, "size": "1536x1024", "quality": "high"}

    await images.generate("make the fox blue", references=[RED, BLUE])
    edit = seen[1]
    assert edit.url.path == "/v1/images/edits"
    assert edit.headers["content-type"].startswith("multipart/form-data")
    assert edit.content.count(b'name="image[]"') == 2 and RED in edit.content
    assert b'name="prompt"\r\n\r\nmake the fox blue' in edit.content

    # An older model, and a server that answers with a link.
    def dalle(request):
        if request.url.host == "cdn.example":
            return httpx.Response(200, content=RED)
        return httpx.Response(200, json={"data": [{"url": "https://cdn.example/a.png",
                                                   "revised_prompt": "A red fox."}]})

    images, seen = served(dalle, OpenAIImages("k", model="dall-e-3"))
    made = await images.generate("a fox", aspect_ratio="9:16")
    assert json.loads(seen[0].content)["size"] == "1024x1792"
    assert json.loads(seen[0].content)["response_format"] == "b64_json"
    assert made[0].data == RED and made[0].text == "A red fox."


async def test_replicate_starts_a_prediction_watches_it_and_reports_progress():
    states = [{"status": "starting"}, {"status": "processing", "logs": "12%|█\n 40%|███"},
              {"status": "processing", "logs": "40%|███\n 85%|██████"},
              {"status": "succeeded", "output": ["https://replicate.delivery/out.png"]}]

    def replicate(request):
        if request.url.host == "replicate.delivery":
            return httpx.Response(200, content=RED)
        if request.method == "POST":
            return httpx.Response(201, json={
                "id": "p1", "status": "starting",
                "urls": {"get": "https://api.replicate.com/v1/predictions/p1",
                         "cancel": "https://api.replicate.com/v1/predictions/p1/cancel"}})
        return httpx.Response(200, json=states.pop(0))

    updates = []
    images, seen = served(replicate, ReplicateImages("r8-secret", poll_interval=0.001),
                          on_progress=updates.append)
    made = await images.generate("a fox", references=[BLUE], aspect_ratio="1:1", seed=3)
    start = json.loads(seen[0].content)
    assert seen[0].url.path == "/v1/models/black-forest-labs/flux-kontext-pro/predictions"
    assert seen[0].headers["authorization"] == "Bearer r8-secret"
    assert start["input"]["prompt"] == "a fox" and start["input"]["seed"] == 3
    assert start["input"]["aspect_ratio"] == "1:1"
    assert start["input"]["input_image"].startswith("data:image/png;base64,")
    assert made[0].data == RED

    fractions = [u.fraction for u in updates if u.fraction is not None]
    assert fractions == [0.4, 0.85, 1.0, 1.0]
    assert [u.status for u in updates][-1] == "done"
    assert all(u.job == updates[0].job and u.engine in ("", "replicate") for u in updates)


async def test_a_replicate_job_that_fails_or_is_abandoned_is_cancelled_there():
    cancelled: list[str] = []

    def replicate(request):
        if request.url.path.endswith("/cancel"):
            cancelled.append(request.url.path)
            return httpx.Response(200, json={})
        if request.method == "POST":
            return httpx.Response(201, json={"id": "p1", "status": "starting", "urls": {
                "get": "https://api.replicate.com/v1/predictions/p1",
                "cancel": "https://api.replicate.com/v1/predictions/p1/cancel"}})
        return httpx.Response(200, json={"status": "processing"})

    images, _ = served(replicate, ReplicateImages("k", poll_interval=0.01))
    job = await images.submit("a fox")
    await asyncio.sleep(0.05)
    assert job.status == "running" and "running on replicate" in job.describe()
    assert job.cancel()
    with pytest.raises(ToolError, match="was cancelled"):
        await job.wait()
    assert job.status == "cancelled" and cancelled == ["/v1/predictions/p1/cancel"]

    def failing(request):
        if request.method == "POST":
            return httpx.Response(201, json={"id": "p2", "status": "starting"})
        return httpx.Response(200, json={"status": "failed", "error": "NSFW content detected"})

    images, _ = served(failing, [ReplicateImages("k", poll_interval=0.001),
                                 FakeImages(name="second")])
    with pytest.raises(ToolError, match="replicate would not make this image"):
        await images.generate("a fox")
    assert "second" not in images.stats             # a refusal is not taken elsewhere


# --- failing, retrying, falling back ---------------------------------------------------

async def test_a_throttled_call_is_retried_and_a_spurious_empty_answer_too():
    answers = [httpx.Response(429, headers={"retry-after": "0"}), httpx.Response(503),
               httpx.Response(200, json={"candidates": [{"finishReason": "STOP", "content": {
                   "parts": [{"text": "Let me think about that."}]}}]})]
    images, seen = served(lambda r: answers.pop(0) if answers else gemini_ok(r),
                          GeminiImages("k"), retry=RetryPolicy(
                              max_retries=3, initial_delay=0, max_delay=0, jitter=0))
    updates = []
    made = await images.generate("a fox", on_progress=updates.append)
    assert len(made) == 1 and len(seen) == 4
    retried = [u.message for u in updates if "trying again" in u.message]
    assert len(retried) == 3 and "rate limited" in retried[0]
    assert "answered without an image" in retried[2]


async def test_the_next_engine_makes_it_when_one_cannot():
    def handler(request):
        if "googleapis" in request.url.host:
            return httpx.Response(401, json={"error": {"message": "API key not valid"}})
        return httpx.Response(200, json={"data": [{"b64_json": B64}]})

    images, seen = served(handler, [GeminiImages("g-secret-key"), OpenAIImages("sk-key")])
    made = await images.generate("a fox")
    assert made[0].engine == "openai" and images.engines == ["gemini", "openai"]
    assert images.stats["gemini"]["last_error"] == "the API key was refused"
    assert [r.url.host for r in seen] == ["generativelanguage.googleapis.com",
                                          "api.openai.com"]


async def test_a_refused_prompt_is_not_retried_and_not_taken_to_another_provider():
    for answer in [
            httpx.Response(200, json={"promptFeedback": {"blockReason": "PROHIBITED_CONTENT"}}),
            httpx.Response(200, json={"candidates": [{"finishReason": "IMAGE_SAFETY"}]}),
            httpx.Response(400, json={"error": {"code": "moderation_blocked",
                                                "message": "rejected by the safety system"}})]:
        images, seen = served(lambda r, a=answer: a,
                              [GeminiImages("k"), FakeImages(name="second")])
        with pytest.raises(ToolError, match="gemini would not make this image") as caught:
            await images.generate("something it will not draw")
        assert "Change the prompt" in str(caught.value)
        assert len(seen) == 1 and "second" not in images.stats


async def test_when_every_engine_fails_the_reasons_are_given_without_the_keys():
    def handler(request):
        if "googleapis" in request.url.host:
            raise httpx.ConnectError("no route to host")
        if "openai" in request.url.host:
            return httpx.Response(429, json={"error": {
                "message": "You exceeded your current quota sk-secret-key"}})
        return httpx.Response(200, text="<html>gateway</html>")

    images, seen = served(handler, [GeminiImages("g-secret-key"),
                                    OpenAIImages("sk-secret-key"),
                                    ReplicateImages("r8-secret-key")])
    with pytest.raises(ToolError) as caught:
        await images.generate("a fox")
    message = str(caught.value)
    assert "gemini: could not be reached (ConnectError)" in message
    assert "openai: out of quota or credit" in message
    assert "replicate: answered with something that is not JSON" in message
    assert "secret-key" not in message
    assert [r.url.host for r in seen].count("generativelanguage.googleapis.com") == 3


async def test_what_comes_back_must_be_an_image_and_a_slow_engine_times_out():
    def page(request):
        return [b"<html>Service unavailable</html>"]

    with pytest.raises(ToolError, match="page: what came back was not an image"):
        await ImageGenerator(page).generate("a fox")

    async def slow(request):
        await asyncio.sleep(5)

    images = ImageGenerator([slow, FakeImages()], timeout=0.05, retry=RetryPolicy.none())
    made = await images.generate("a fox")
    assert made[0].engine == "fake"
    assert "no image within 0.05s" in images.stats["slow"]["last_error"]

    stuck = ImageGenerator(slow, timeout=0.05, deadline=0.12, retry=RetryPolicy(
        max_retries=50, initial_delay=0, jitter=0))
    started = asyncio.get_running_loop().time()
    with pytest.raises(ToolError, match="slow: "):
        await stuck.generate("a fox")
    assert asyncio.get_running_loop().time() - started < 1.0


async def test_an_engine_that_keeps_failing_is_left_alone():
    engine = FakeImages(fail=[ImageFailure("down")] * 2)
    images = ImageGenerator(engine, failure_threshold=2, retry=RetryPolicy.none())
    for _ in range(2):
        with pytest.raises(ToolError, match="fake: down"):
            await images.generate("a fox")
    with pytest.raises(ToolError, match="left alone after repeated failures"):
        await images.generate("a fox")
    assert len(engine.requests) == 2


async def test_nothing_set_up_and_things_set_up_wrong_say_so(monkeypatch):
    with pytest.raises(ToolError, match="no image engine is set up — set GEMINI_API_KEY"):
        await ImageGenerator().generate("a fox")
    with pytest.raises(ToolError, match="gemini: no API key — set GEMINI_API_KEY"):
        await ImageGenerator("gemini").generate("a fox")
    with pytest.raises(ConfigurationError, match="no image engine named 'midjourney'"):
        ImageGenerator("midjourney")
    with pytest.raises(ConfigurationError, match="go with one named engine"):
        ImageGenerator(["gemini", "openai"], model="x")
    with pytest.raises(ToolError, match="aspect_ratio is one of"):
        ImageGenerator(aspect_ratio="7:3")

    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("GOOGLE_API_KEY", "k")
    assert ImageGenerator().engines == ["gemini", "openai"]
    monkeypatch.setenv("AGENT_HARNESS_IMAGES", "flux, nano-banana")
    assert ImageGenerator().engines == ["replicate", "gemini"]
    assert {e["name"]: e["ready"] for e in image_engines()} == {
        "gemini": True, "openai": True, "replicate": False}
    assert ImageGenerator("gemini", model="gemini-3-pro-image-preview")._resolve()[
        0].model == "gemini-3-pro-image-preview"


# --- the request --------------------------------------------------------------------

async def test_the_request_is_checked_and_counts_are_capped():
    engine = FakeImages()
    images = ImageGenerator(engine, max_count=3, max_images=4, max_references=2,
                            aspect_ratio="wide")
    with pytest.raises(ToolError, match="prompt is empty"):
        await images.generate("   ")
    with pytest.raises(ToolError, match="the most is 8000"):
        await images.generate("x" * 9000)
    with pytest.raises(ToolError, match="3 reference images were given; the most is 2"):
        await images.generate("a fox", references=[RED, RED, RED])
    assert not engine.requests

    assert len(await images.generate("a fox", count=99)) == 3        # clamped
    assert engine.requests[0].aspect_ratio == "16:9"                # the default shape
    assert len(await images.generate("a fox", count=3, aspect_ratio="4x3")) == 1
    assert engine.requests[1].aspect_ratio == "4:3" and images.made == 4
    with pytest.raises(ToolError, match="limit of 4 images is used up"):
        await images.generate("a fox")


async def test_references_come_from_files_urls_blocks_and_earlier_images(tmp_path):
    photo = tmp_path / "photo.jpg"
    photo.write_bytes(JPEG)
    (tmp_path / "notes.png").write_text("not an image at all")

    def web(request):
        if request.url.path == "/moved":
            return httpx.Response(302, headers={"location": "https://93.184.216.34/ref.png"})
        if request.url.path == "/big.png":
            return httpx.Response(200, content=RED + b"\x00" * 5000)
        return httpx.Response(200 if request.url.path == "/ref.png" else 404, content=BLUE)

    engine = FakeImages()
    images = ImageGenerator(engine, transport=httpx.MockTransport(web),
                            max_reference_bytes=2000)
    first = await images.generate("a fox", name="fox", references=[
        photo, str(photo), RED, ImageBlock.from_bytes(BLUE, name="blue.png"),
        "https://93.184.216.34/ref.png", "https://93.184.216.34/moved",
        "data:image/png;base64," + B64])
    kinds = [(r.media_type, len(r.data)) for r in engine.requests[0].references]
    assert kinds == [("image/jpeg", len(JPEG)), ("image/jpeg", len(JPEG)),
                     ("image/png", len(RED)), ("image/png", len(BLUE)),
                     ("image/png", len(BLUE)), ("image/png", len(BLUE)),
                     ("image/png", len(RED))]

    await images.generate("the same fox, at night", references=["fox.png"])   # by name
    await images.generate("again", references=[first[0]])                     # by object
    assert engine.requests[1].references[0].data == first[0].data
    assert engine.requests[2].references[0].data == first[0].data

    for bad, why in [
            (str(tmp_path / "notes.png"), "is not a PNG, JPEG, WebP or GIF"),
            (str(tmp_path / "missing.png"), "there is no image at"),
            ("https://93.184.216.34/nope.png", "answered 404"),
            ("https://93.184.216.34/big.png", "is larger than"),
            ("http://127.0.0.1/x.png", "private or loopback"),
            ("http://169.254.169.254/latest/meta-data", "private or loopback"),
            ("ftp://example.com/x.png", "there is no image at")]:
        with pytest.raises(ToolError, match=why):
            await images.generate("a fox", references=[bad])


# --- jobs ----------------------------------------------------------------------------

async def test_a_job_is_watched_waited_for_and_collected():
    images = ImageGenerator(FakeImages(steps=4, delay=0.02))
    job = await images.submit("a city at dusk", count=2)
    assert job.status in ("queued", "running") and not job.done and job.id in images.jobs
    with pytest.raises(ToolError, match="is still"):
        job.result()
    with pytest.raises(ToolError, match="is still running on fake"):
        await job.wait(0.03)
    assert not job.done                                # a timed-out wait leaves it running
    seen = set()
    while not job.done:
        seen.add(job.describe().split(",")[0])
        await asyncio.sleep(0.01)
    assert any("%" in line for line in seen)
    assert len(job.result()) == 2 and job.fraction == 1.0
    assert job.describe().startswith("done — 2 images in")
    assert not job.cancel()                            # nothing left to cancel
    with pytest.raises(ToolError, match="there is no job 'img_nope'"):
        images.job("img_nope")


async def test_cancelling_the_caller_cancels_the_job_and_only_so_many_run_at_once():
    images = ImageGenerator(FakeImages(steps=50, delay=0.02), max_concurrency=1)
    call = asyncio.ensure_future(images.generate("a fox"))
    await asyncio.sleep(0.05)
    waiting = await images.submit("a second fox")
    await asyncio.sleep(0.02)
    assert waiting.status == "queued"                  # one at a time was asked for
    call.cancel()
    with pytest.raises(asyncio.CancelledError):
        await call
    await asyncio.sleep(0.02)
    first = next(iter(images.jobs.values()))
    assert first.status == "cancelled" and "cancelled after" in first.describe()
    waiting.cancel()


# --- as tools -----------------------------------------------------------------------

async def test_an_agent_generates_into_its_workspace_and_edits_from_references(tmp_path):
    (tmp_path / "storefront.jpg").write_bytes(JPEG)
    engine = FakeImages()
    images = ImageGenerator(engine, cost_per_image=0.04)
    shown: list[list] = []

    def model(request):
        last = request.messages[-1]
        shown.append(list(last.content))
        turn = sum(1 for m in request.messages if m.role == "assistant")
        if turn == 0:
            return tool_call("generate_image", prompt="a poster for the autumn sale",
                             references=["storefront.jpg", "attachment:1"],
                             aspect_ratio="3:4", name="poster")
        if turn == 1:
            return tool_call("generate_image", prompt="the same, with warmer light",
                             references=["images/poster.png"], count=2)
        return "The poster is in images/poster.png, with two warmer variations."

    agent = Agent("designer", tools=images.tools(), memory=False,
                  workspace=Workspace(tmp_path),
                  harness=Harness.testing(FakeProvider([model], loop=True)))
    result = await agent.run("Make a poster in the style of this photo.",
                             attachments=[ImageBlock.from_bytes(BLUE, name="logo.png")])

    assert result.output.startswith("The poster is in images/poster.png")
    assert sorted(p.name for p in (tmp_path / "images").iterdir()) == [
        "poster.png", "the-same-with-warmer-light-1.png", "the-same-with-warmer-light-2.png"]
    assert sniff((tmp_path / "images/poster.png").read_bytes()) == ("image/png", 48, 64)
    # The references reached the engine: the workspace file, the attachment, the poster.
    assert [r.media_type for r in engine.requests[0].references] == ["image/jpeg", "image/png"]
    assert engine.requests[0].references[1].data == BLUE
    assert engine.requests[1].references[0].data == (tmp_path / "images/poster.png").read_bytes()

    # The model read where it went, and was shown it.
    answer = next(b for b in shown[1] if isinstance(b, ToolResultBlock))
    assert "Generated 1 image with fake (fake-image-1)" in answer.content
    assert "- images/poster.png — 48x64 png" in answer.content
    assert 'references=["images/poster.png"]' in answer.content
    assert sum(isinstance(b, ImageBlock) for b in shown[1]) == 1
    assert sum(isinstance(b, ImageBlock) for b in shown[2]) == 2

    assert [(a.name, a.path, a.media_type) for a in result.artifacts if a.media_type == "image/png"] == [
        ("poster.png", "images/poster.png", "image/png"),
        ("the-same-with-warmer-light-1.png", "images/the-same-with-warmer-light-1.png", "image/png"),
        ("the-same-with-warmer-light-2.png", "images/the-same-with-warmer-light-2.png", "image/png")]
    assert 0.12 <= result.usage.cost_usd < 0.125        # three images at 0.04, plus the model


async def test_a_model_may_only_name_what_it_was_shown(tmp_path):
    secret = tmp_path / "secret.png"
    secret.write_bytes(RED)
    generate, _ = ImageGenerator(FakeImages()).tools()
    for reference in [str(secret), "/etc/passwd", "../secret.png"]:
        outcome = await generate.run("c", {"prompt": "a fox", "references": [reference]})
        assert outcome.is_error and "there is no image called" in outcome.content
    outcome = await generate.run("c", {"prompt": "a fox", "aspect_ratio": "7:3"})
    assert outcome.is_error and "aspect_ratio is one of" in outcome.content

    # With no workspace the image is kept in memory, and can still be built on.
    made = await generate.run("c", {"prompt": "a fox", "name": "fox"})
    assert not made.is_error and "- fox.png — 64x64 png" in made.content
    assert len(made.media) == 1 and made.media[0].read()[:4] == b"\x89PNG"
    again = await generate.run("c", {"prompt": "now at night", "references": ["fox"]})
    assert not again.is_error

    quiet, _ = ImageGenerator(FakeImages()).tools(show=False, references=[BLUE])
    outcome = await quiet.run("c", {"prompt": "a fox"})
    assert not outcome.media and "Generated 1 image" in outcome.content


async def test_a_job_that_outlives_the_wait_is_collected_with_image_status():
    images = ImageGenerator(FakeImages(steps=6, delay=0.03))
    generate, status = images.tools(wait=0.05)
    assert (await status.invoke({})) == "No images have been asked for."

    said = await generate.invoke({"prompt": "a detailed map of a city"})
    assert said.startswith("Still working: job img_") and "do not start the same" in said
    job_id = said.split("job ")[1].split(" ")[0]
    assert 'image_status(job_id="' + job_id in said

    assert "Still working" in await status.invoke({"job_id": job_id})
    assert f"{job_id}: running on fake" in await status.invoke({})
    text, picture = await status.invoke({"job_id": job_id, "wait": 5})
    assert "Generated 1 image with fake" in text and isinstance(picture, ImageBlock)

    second = await generate.invoke({"prompt": "another map"})
    stopped = await status.invoke({"job_id": second.split("job ")[1].split(" ")[0],
                                   "cancel": True})
    assert "is being cancelled" in stopped
    missing = await status.run("c", {"job_id": "img_nope"})
    assert missing.is_error and "there is no job" in missing.content


async def test_what_images_cost_counts_against_the_budget():
    def model(request):
        return tool_call("generate_image", prompt="a fox", count=2)

    images = ImageGenerator(FakeImages(), cost_per_image=0.5)
    agent = Agent("designer", tools=images.tools(), memory=False,
                  budget=Budget(max_usd=1.5),
                  harness=Harness.testing(FakeProvider([model], loop=True)))
    result = await agent.run("keep making foxes")
    assert result.stop_reason == "budget" and images.made <= 4


def test_the_cli_lists_the_engines_and_generates(capsys, monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    assert cli.main(["image"]) == 0
    out = capsys.readouterr().out
    assert "gemini      ready     gemini-2.5-flash-image" in out
    assert "openai      not set   gpt-image-1" in out
