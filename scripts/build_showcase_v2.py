"""Select the small candidate batch and render the mixed Standard/Turbo showcase.

This consumes local, trusted metadata.pkl files from generate_showcase_v2.py.
It never loads Iris or generates additional candidates.
"""

import argparse
from functools import lru_cache
import hashlib
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

BG = "#10131d"
INK = "#f6f7fb"
MUTED = "#b5bfd2"
ACCENT = "#8be5df"
SCORER = "openai/clip-vit-large-patch14"
SCORER_REVISION = "32bd64288804d66eefd0ccbe215aa642df71cc41"


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


@lru_cache(None)
def font(size, bold=False):
    path = Path("C:/Windows/Fonts/seguisb.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf")
    if not path.is_file():
        path = ROOT / "visual_anagrams/assets/CourierPrime-Regular.ttf"
    return ImageFont.truetype(str(path), size)


def center(draw, text, y, size=32, fill=INK, bold=False, width=1080):
    while font(size, bold).getlength(text) > width - 64 and size > 14:
        size -= 1
    draw.text((width // 2, y), text, font=font(size, bold), fill=fill, anchor="mt")


def setting(case):
    mode = "Standard Iris" if case["mode"] == "standard" else "Turbo r2"
    return f"{mode} / {case['steps']} steps / shift 8 / adaptive"


def select(plan, source, output):
    import torch
    import torch.nn.functional as F
    from transformers import CLIPModel, CLIPProcessor

    print("Scoring saved views on CPU with held-out CLIP ViT-L/14", flush=True)
    model = CLIPModel.from_pretrained(SCORER, revision=SCORER_REVISION).float().eval()
    processor = CLIPProcessor.from_pretrained(SCORER, revision=SCORER_REVISION, use_fast=False)
    scores, selected = {}, {}
    with torch.inference_mode():
        for case in plan["cases"]:
            prompts = [f"{case['style']} {p}" for p in case["prompts"]]
            rows = []
            for seed in case["seeds"]:
                folder = source / case["id"] / f"{seed:04}"
                run = json.loads((folder / "run.json").read_text(encoding="utf-8"))
                assert hashlib.sha256((folder / "sample_1024.png").read_bytes()).hexdigest() == run["image_sha256"]
                with Image.open(folder / "sample_1024.views.png") as sheet:
                    assert sheet.size == (2048, 1024)
                    views = [sheet.crop((i * 1024, 0, (i + 1) * 1024, 1024)).convert("RGB") for i in range(2)]
                result = model(**processor(text=prompts, images=views, padding=True, return_tensors="pt"))
                matrix = F.normalize(result.image_embeds, dim=-1) @ F.normalize(result.text_embeds, dim=-1).T
                alignment = matrix.diag().tolist()
                rows.append(dict(seed=seed, scores=alignment, weaker_view=min(alignment), similarity_matrix=matrix.tolist(),
                                 minimum_prompt_margin=min(float(matrix[i, i] - matrix[i, 1-i]) for i in range(2))))
            best = max(rows, key=lambda row: row["weaker_view"])
            selected[case["id"]] = best["seed"]
            scores[case["id"]] = rows
            print(f"Selected {case['id']} seed {best['seed']}: {best['scores']}", flush=True)
    write_json(output / "selection.json", selected)
    write_json(output / "candidate-scores.json", dict(scorer=SCORER, revision=SCORER_REVISION,
        rule="Maximize the weaker of the two intended-view cosine similarities within each case. Scores are not human illusion ratings.",
        cases=scores))
    return selected


def comparison(case, sheet):
    frame = Image.new("RGB", (1080, 1080), BG)
    draw = ImageDraw.Draw(frame)
    center(draw, case["method"].upper(), 85, 48, bold=True)
    center(draw, case["style_label"], 155, 34, ACCENT)
    for index, label in enumerate(case["labels"]):
        tile = sheet.crop((index * 1024, 0, (index + 1) * 1024, 1024)).resize((520, 520), Image.Resampling.LANCZOS)
        tile = prompt_overlay(tile, label, font_size=27)
        frame.paste(tile, (10 + 540 * index, 290))
    center(draw, setting(case), 865, 29, MUTED)
    center(draw, "One saved image. Two intended views.", 935, 28)
    return frame


def overlay(frame, case, number, total):
    frame = frame.convert("RGBA")
    banner = Image.new("RGBA", frame.size)
    draw = ImageDraw.Draw(banner)
    # Text overlays sit on the artwork; there is no added canvas or outer border.
    for y in range(155):
        draw.line((0, y, 1080, y), fill=(9, 13, 22, round(220 * (1 - y / 155))))
    draw.text((27, 19), f"{number:02}/{total:02}  {case['method']}", font=font(35, True), fill=INK)
    draw.text((28, 68), f"{case['style_label']}  |  {setting(case)}", font=font(24), fill=INK)
    return Image.alpha_composite(frame, banner).convert("RGB")


def title_card(items, outro=False):
    frame = Image.new("RGB", (1080, 1080), BG)
    draw = ImageDraw.Draw(frame)
    center(draw, "VISUAL ANAGRAMS", 45, 62, bold=True)
    center(draw, "Iris 3B INT8 / Standard + Turbo r2", 125, 31, ACCENT)
    for index, (case, image, _sheet, _views, _record) in enumerate(items):
        x, y = 22 + (index % 3) * 350, 225 + (index // 3) * 365
        tile = image.resize((336, 336), Image.Resampling.LANCZOS)
        frame.paste(prompt_overlay(tile, case["labels"][0], font_size=20), (x, y))
    if outro:
        center(draw, "RTX 3060 12 GB / generated locally", 970, 29)
        center(draw, "Experimental results. Labels show the intended views.", 1021, 25, MUTED)
    else:
        center(draw, "Six methods. Six themes. Watch the image change.", 970, 32, bold=True)
        center(draw, "Rotations: 30 steps / Other transforms: 4 steps", 1021, 27, MUTED)
    return frame


def gallery(items, path):
    canvas = Image.new("RGB", (1200, 1490), BG)
    draw = ImageDraw.Draw(canvas)
    center(draw, "IRIS VISUAL ANAGRAMS / SHOWCASE 2", 23, 34, bold=True, width=1200)
    for index, (case, _image, sheet, _views, _record) in enumerate(items):
        x, y = (index % 2) * 600, 95 + (index // 2) * 460
        draw.text((x + 16, y), case["method"], font=font(28, True), fill=INK)
        draw.text((x + 16, y + 43), f"{case['style_label']} / {case['steps']} steps", font=font(23), fill=ACCENT)
        for view_index, label in enumerate(case["labels"]):
            tile = sheet.crop((view_index * 1024, 0, (view_index + 1) * 1024, 1024)).resize((280, 280), Image.Resampling.LANCZOS)
            canvas.paste(prompt_overlay(tile, label, font_size=18), (x + 16 + view_index * 288, y + 91))
        draw.text((x + 16, y + 387), setting(case), font=font(20), fill=MUTED)
    canvas.save(path, quality=92)


def main():
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "results/showcase_v2_candidates_20261025")
    parser.add_argument("--output", type=Path, default=ROOT / "examples/showcase-v2")
    parser.add_argument("--select-only", action="store_true")
    parser.add_argument("--skip-gifs", action="store_true")
    parser.add_argument("--assets-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(6)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    plan = json.loads((output / "plan.json").read_text(encoding="utf-8"))
    selected_path = output / "selection.json"
    selected = json.loads(selected_path.read_text(encoding="utf-8")) if selected_path.exists() else select(plan, args.source, output)
    if args.select_only:
        return
    items, validation = [], []
    for case in plan["cases"]:
        seed = selected[case["id"]]
        assert seed in case["seeds"]
        directory = args.source / case["id"]
        folder = directory / f"{seed:04}"
        source = folder / "sample_1024.png"
        record = json.loads((folder / "run.json").read_text(encoding="utf-8"))
        assert record["steps"] == case["steps"] and record["shift"] == plan["shift"]
        assert hashlib.sha256(source.read_bytes()).hexdigest() == record["image_sha256"]
        with (directory / "metadata.pkl").open("rb") as stream:
            views = pickle.load(stream)["views"]
        with Image.open(source) as raw:
            image = raw.convert("RGB")
        with Image.open(folder / "sample_1024.views.png") as raw:
            sheet = raw.convert("RGB")
        assert image.size == (1024, 1024) and sheet.size == (2048, 1024)
        # Ensure the saved random permutation and animation end at the actual generated views.
        endpoint_options = AnimationOptions(size=1024, borderless=True, labels=False)
        errors = []
        for view_index, t in enumerate([0., 1.]):
            rendered = render_frame(image, case["views"][1], views[1], t, endpoint_options)
            expected = sheet.crop((1024 * view_index, 0, 1024 * (view_index + 1), 1024))
            errors.append(float(np.abs(np.asarray(rendered).astype(float) - np.asarray(expected)).mean()))
        assert max(errors) < 1, (case["id"], errors)
        shutil.copyfile(source, output / f"{case['id']}.png")
        preview = Image.new("RGB", (1024, 512))
        for i, label in enumerate(case["labels"]):
            tile = sheet.crop((i*1024, 0, (i+1)*1024, 1024)).resize((512, 512), Image.Resampling.LANCZOS)
            preview.paste(prompt_overlay(tile, label, font_size=25), (i*512, 0))
        preview.save(output / f"{case['id']}.views.jpg", quality=92)
        if not args.skip_gifs:
            gif = output / f"{case['id']}.gif"
            if not gif.exists():
                exported = export_animations(source, views, case["views"], case["labels"], ["gif"],
                    AnimationOptions(fps=12, size=384, transition=1.25, hold=1., labels=True, borderless=True))
                shutil.copyfile(exported[0], gif)
        items.append((case, image, sheet, views, record))
        validation.append(dict(case=case["id"], seed=seed, image_sha256=record["image_sha256"],
            endpoint_mean_absolute_pixel_error=errors, steps=record["steps"], shift=record["shift"]))
        print("Packaged", case["id"], flush=True)
    gallery(items, output / "gallery.jpg")
    title_card(items).save(output / "reddit-thumbnail.jpg", quality=93)
    write_json(output / "validation.json", dict(images=validation))
    if args.assets_only:
        return

    fps, frames, chapters = 24, 0, []
    options = AnimationOptions(fps=fps, size=1080, transition=2., hold=2., labels=True, borderless=True)
    # render_frame accepts arbitrary canvas sizes; timeline validation requires a multiple of 16.
    timeline = animation_timeline(AnimationOptions(fps=fps, size=1024, transition=2., hold=2.))
    video = output / "showcase3.mp4"
    temporary = output / ".showcase3.tmp.mp4"
    with imageio.get_writer(temporary, format="FFMPEG", fps=fps, codec="libx264", quality=None,
            macro_block_size=1, pixelformat="yuv420p", ffmpeg_log_level="error",
            output_params=["-crf", "25", "-preset", "medium", "-threads", "6", "-movflags", "+faststart"]) as writer:
        def write(frame, count=1):
            nonlocal frames
            pixels = np.asarray(frame)
            for _ in range(count):
                writer.append_data(pixels)
            frames += count
        write(title_card(items), fps * 2)
        for index, (case, image, sheet, views, _record) in enumerate(items, 1):
            chapters.append(dict(case=case["id"], method=case["method"], start_seconds=frames/fps,
                animation_start_seconds=frames/fps+1, mode=case["mode"], steps=case["steps"]))
            write(overlay(render_frame(image, case["views"][1], views[1], 0, options, *case["labels"]), case, index, len(items)), fps)
            cache = {}
            for t in timeline:
                key = round(t, 12)
                if key not in cache:
                    cache[key] = overlay(render_frame(image, case["views"][1], views[1], t, options, *case["labels"]), case, index, len(items))
                write(cache[key])
            write(comparison(case, sheet), fps)
            print(f"Encoded chapter {index}/{len(items)}: {case['id']}", flush=True)
        write(title_card(items, outro=True), fps * 3)
    temporary.replace(video)
    with imageio.get_reader(video) as reader:
        meta = reader.get_meta_data()
        assert tuple(meta["size"]) == (1080, 1080) and meta["fps"] == fps
        assert abs(meta["duration"] - frames/fps) < .1
    environment = json.loads((args.source / "environment.json").read_text(encoding="utf-8"))
    manifest = dict(version=2, model=environment["model"], model_revision=environment["model_revision"],
        code_revision=environment["code_revision"], text_revision=environment["text_revision"],
        turbo_repo=environment["turbo_repo"], turbo_revision=environment["turbo_revision"],
        turbo_adapter="r2-1000 (Turbo examples only)", resolution=plan["resolution"], shift=plan["shift"],
        quality_method="adaptive", combining="mean", standard_solver_order=2,
        gpu=environment["gpu"], new_images=8, reused_images=1,
        examples=[dict(case, selected_seed=selected[case["id"]], full_prompts=[f"{case['style']} {p}" for p in case["prompts"]],
            generation_seconds=record["seconds"], image_sha256=record["image_sha256"],
            peak_allocated_gib=record["peak_allocated_gib"],
            source=str((args.source/case["id"]/f"{selected[case['id']]:04}").relative_to(ROOT)).replace("\\", "/"))
            for case, _image, _sheet, _views, record in items],
        video=dict(file=video.name, fps=fps, size=[1080,1080], duration_seconds=frames/fps,
            bytes=video.stat().st_size, borderless=True, labels=True, audio=False, chapters=chapters))
    write_json(output / "manifest.json", manifest)
    print(f"Saved {video}: {frames/fps:.1f} s, {video.stat().st_size/1e6:.2f} MB", flush=True)


if __name__ == "__main__":
    main()
