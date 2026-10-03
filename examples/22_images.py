"""Image generation: an agent that makes a picture, is shown it, and works from it.

With an image key in the environment (`GEMINI_API_KEY`, `OPENAI_API_KEY` or
`REPLICATE_API_TOKEN`) this makes real images. Without one it uses `FakeImages`,
which draws a small PNG of one colour — enough to show everything around the
drawing: the job and its progress, the reference images, the fallback, and the
files an agent leaves in its workspace.
"""

from __future__ import annotations

import asyncio
import tempfile

from _common import pick_provider

from agent_harness import Agent, Harness, ImageGenerator, Workspace, tool_call
from agent_harness.toolkits import FakeImages, image_engines
from agent_harness.toolkits.images import ImageFailure, solid_png


async def main() -> None:
    print("engines   ", ", ".join(f"{e['name']}{'' if e['ready'] else ' (not set)'}"
                                 for e in image_engines()), "\n")
    live = any(e["ready"] for e in image_engines())
    # Without a key: an engine that is down, then one that draws — slowly enough
    # to be watched.
    engines = None if live else [
        FakeImages(name="first", fail=[ImageFailure("the engine answered 503")]),
        FakeImages(steps=4, delay=0.05)]
    images = ImageGenerator(engines, aspect_ratio="16:9")

    # --- a job you can watch ------------------------------------------------------
    job = await images.submit("a lighthouse at dusk, oil painting", count=2)
    while not job.done:
        print(f"progress   {job.describe()}")
        await asyncio.sleep(0.06 if not live else 2)
    for image in job.result():
        print(f"made       {image.describe()}")
    print(f"stats      {images.stats}\n")

    # --- a reference image: change what was made, or follow a picture of your own --
    logo = solid_png(32, 32, (240, 180, 40))           # stands in for a file of yours
    edited = await images.generate(
        "the same lighthouse, in winter, with this emblem on the door",
        references=[job.result()[0], logo], name="winter")
    print(f"edited     {edited[0].describe()}  (from 2 references)\n")

    # --- and an agent, with a workspace to keep them in -------------------------------
    provider, model = pick_provider([
        tool_call("generate_image", prompt="a poster for an autumn book fair",
                  aspect_ratio="3:4", name="poster"),
        tool_call("generate_image", prompt="the same poster with a darker background",
                  references=["images/poster.png"], name="poster-dark"),
        "Two versions are in the workspace: images/poster.png and images/poster-dark.png.",
    ])
    harness = Harness(provider=provider)
    with tempfile.TemporaryDirectory() as folder:
        agent = Agent("designer", "Make what is asked for, look at it, and improve it.",
                      tools=images.tools(), workspace=Workspace(folder), model=model,
                      harness=harness, memory=False)
        async for event in agent.stream("A poster for the autumn book fair, and a "
                                        "darker variation."):
            if event.type == "tool_result":
                print(f"tool       {event.data['tool']} → {event.text.splitlines()[1]}")
            elif event.type == "run_end":
                result = event.data["result"]
                print(f"\ndesigner   {result.output}")
                print("files      " + ", ".join(a.path for a in result.artifacts if a.path))
    await harness.aclose()


if __name__ == "__main__":
    asyncio.run(main())
