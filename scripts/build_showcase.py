"""Package selected samples as a GitHub gallery and a 75-second video.

Selections are explicit in examples/showcase/selection.json. The builder
only renders already-generated images; it does not load the image model.
"""

import argparse
from functools import lru_cache
import json
from pathlib import Path
import pickle
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from visual_anagrams.export import AnimationOptions, animation_timeline, export_animations, prompt_overlay, render_frame

STYLE_LABELS = {"quarter-turn": "Oil painting", "upside-down": "Watercolor", "vertical-flip": "Charcoal",
                "color-inversion": "Black & white linocut", "color-inversion-engraving": "Copperplate engraving",
                "inner-circle": "Art nouveau", "patch-shuffle": "Pencil sketch"}
BG, INK, MUTED, ACCENT = "#f5f6f9", "#202b3c", "#637086", "#5146d9"


@lru_cache(None)
def font(size, bold=False):
    windows = Path("C:/Windows/Fonts/seguisb.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf")
    bundled = ROOT / "visual_anagrams/assets/CourierPrime-Regular.ttf"
    return ImageFont.truetype(str(windows if windows.is_file() else bundled), size)


def centered(draw, y, text, size, fill=INK, bold=False, width=1080):
    face = font(size, bold)
    while face.getlength(text) > width - 80 and size > 16:
        size -= 1
        face = font(size, bold)
    draw.text((width / 2, y), text, font=face, fill=fill, anchor="mt")


def still_card(case, sheet, *, intro=False, outro=False):
    frame = Image.new("RGB", (1080, 1080), BG)
    draw = ImageDraw.Draw(frame)
    if intro:
        centered(draw, 65, "ONE IMAGE", 72, bold=True)
        centered(draw, 155, "DIFFERENT PERSPECTIVES", 42, fill=ACCENT, bold=True)
        centered(draw, 225, "Six transforms · six visual styles", 30, fill=MUTED)
    elif outro:
        centered(draw, 70, "VISUAL ANAGRAMS", 60, bold=True)
        centered(draw, 155, "Iris 3B INT8 + Turbo r2", 36, fill=ACCENT)
        centered(draw, 225, "4 denoising steps · 1024 × 1024", 30, fill=MUTED)
    else:
        centered(draw, 65, case["method"].upper(), 52, bold=True)
        centered(draw, 150, STYLE_LABELS.get(case["id"], case["style"]), 36, fill=ACCENT)
        centered(draw, 220, "The same pixels, viewed differently", 28, fill=MUTED)
    for index, prompt in enumerate(case["prompts"]):
        source = sheet.crop((index * 1024, 0, (index + 1) * 1024, 1024)).resize((480, 480), Image.Resampling.LANCZOS)
        source = prompt_overlay(source, prompt, font_size=23)
        frame.paste(source, (40 + index * 520, 320))
    if intro:
        centered(draw, 860, "Watch the image turn, flip and rearrange", 31, bold=True)
        centered(draw, 925, "Generated locally · animated from the saved image", 26, fill=MUTED)
    elif outro:
        centered(draw, 850, "Generated on an RTX 3060 · 12 GB VRAM", 30, bold=True)
        centered(draw, 915, "Prompts label the intended views. Results are experimental.", 25, fill=MUTED)
        centered(draw, 980, "Visual Anagrams: Geng, Park & Owens · Iris: Speridlabs", 23, fill=MUTED)
    else:
        centered(draw, 860, "Watch the transformation", 34, bold=True)
        centered(draw, 925, "Iris 3B INT8 · Turbo r2 · 4 steps", 26, fill=MUTED)
    return frame


def scene_frame(frame, case, number, total):
    frame = frame.resize((1080, 1080), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(frame)
    centered(draw, 38, f"{number:02} / {total:02}   {case['method'].upper()}", 34, bold=True)
    centered(draw, 94, STYLE_LABELS.get(case["id"], case["style"]), 28, fill=ACCENT)
    centered(draw, 1008, "Iris 3B INT8 · Turbo r2 · 4 steps", 24, fill=MUTED)
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "results/showcase_candidates")
    parser.add_argument("--output", type=Path, default=ROOT / "examples/showcase")
    parser.add_argument("--selection", type=Path, default=ROOT / "examples/showcase/selection.json")
    parser.add_argument("--skip-gifs", action="store_true")
    parser.add_argument("--gifs-only", action="store_true", help="Refresh the gallery without re-encoding the combined video")
    args = parser.parse_args()
    cases = json.loads((args.output / "plan.json").read_text(encoding="utf-8"))
    selected = json.loads(args.selection.read_text(encoding="utf-8"))
    cases = [case for case in cases if case["id"] in selected]
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    items = []
    for case in cases:
        seed = selected[case["id"]]
        directory = args.source / case["id"]
        source = directory / f"{seed:04}" / "sample_1024.png"
        with (directory / "metadata.pkl").open("rb") as stream:
            views = pickle.load(stream)["views"]
        with Image.open(source) as original:
            image = original.convert("RGB")
        with Image.open(source.with_name("sample_1024.views.png")) as raw:
            sheet = raw.convert("RGB")
        shutil.copyfile(source, output / f"{case['id']}.png")
        preview = sheet.resize((1024, 512), Image.Resampling.LANCZOS)
        preview.save(output / f"{case['id']}.views.jpg", quality=92)
        if not args.skip_gifs:
            gifs = export_animations(source, views, case["views"], case["prompts"], ["gif"],
                                      AnimationOptions(fps=12, size=384, transition=1.25, hold=0.75, labels=True))
            shutil.copyfile(gifs[0], output / f"{case['id']}.gif")
        item = dict(case, selected_seed=seed, view_seed=case["seed"], style_label=STYLE_LABELS.get(case["id"], case["style"]))
        items.append((item, image, sheet, views))
        print(f"Packaged {case['id']} / seed {seed}", flush=True)

    # Readable contact sheet for GitHub and the video's upload thumbnail.
    contact = Image.new("RGB", (1200, len(items) * 360 + 100), BG)
    draw = ImageDraw.Draw(contact)
    centered(draw, 22, "VISUAL ANAGRAMS · IRIS TURBO", 38, bold=True, width=1200)
    for index, (case, _image, sheet, _views) in enumerate(items):
        y = 100 + index * 360
        draw.text((24, y), f"{case['method']}  /  {case['style_label']}", font=font(26, True), fill=INK)
        resized = sheet.resize((600, 300), Image.Resampling.LANCZOS)
        contact.paste(resized, (24, y + 48))
        for p, prompt in enumerate(case["prompts"]):
            draw.text((660, y + 85 + p * 80), prompt, font=font(25), fill=INK)
    contact.save(output / "gallery.jpg", quality=92)
    if args.gifs_only:
        return

    fps = 30
    options = AnimationOptions(fps=fps, size=1024, transition=2.0, hold=2.0, labels=True)
    timeline = animation_timeline(options)
    video = output / "reddit-showcase.mp4"
    chapters, frames = [], 0
    with imageio.get_writer(video, format="FFMPEG", fps=fps, codec="libx264", quality=None,
                            macro_block_size=1, pixelformat="yuv420p", ffmpeg_log_level="error",
                            output_params=["-crf", "20", "-preset", "medium", "-movflags", "+faststart"]) as writer:
        def write(frame, count=1):
            nonlocal frames
            array = np.asarray(frame)
            for _ in range(count):
                writer.append_data(array)
            frames += count

        intro = still_card(items[0][0], items[0][2], intro=True)
        intro.save(output / "reddit-thumbnail.jpg", quality=94)
        write(intro, fps * 3)
        for index, (case, image, sheet, views) in enumerate(items, 1):
            chapters.append(dict(method=case["method"], style=case["style_label"], start_seconds=frames / fps,
                                  selected_seed=case["selected_seed"]))
            write(still_card(case, sheet), round(fps * 1.5))
            cache = {}
            for t in timeline:
                key = round(t, 12)
                if key not in cache:
                    rendered = render_frame(image, case["views"][1], views[1], t, options, *case["prompts"])
                    cache[key] = scene_frame(rendered, case, index, len(items))
                write(cache[key])
            write(still_card(case, sheet), fps * 2)
            print(f"Video chapter {index}/{len(items)}: {case['method']}", flush=True)
        write(still_card(items[-1][0], items[-1][2], outro=True), fps * 3)

    manifest = dict(model="Aikimi/iris-3b-int8", adapter="pentacoxian-dev/iris-3b-turbo-lora:r2-1000",
                    model_revision="6231647d02bb538b401a675061c11c7a64bba81f",
                    adapter_revision="fe16b3496eb440585993c92d95641c08c6b08263",
                    steps=4, guidance=1, shift=4, resolution=[1024,1024],
                    candidates_per_case=2, examples=[item[0] for item in items],
                    video=dict(file=video.name, fps=fps, size=[1080,1080], duration_seconds=frames/fps,
                               audio=False, chapters=chapters))
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved {video} ({frames/fps:.1f} seconds)", flush=True)


if __name__ == "__main__":
    main()
