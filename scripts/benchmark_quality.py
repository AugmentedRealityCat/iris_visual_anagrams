"""Run and score a reproducible Iris quality pilot without changing GUI defaults.

The default pilot uses two prompt pairs, one seed and 30 standard steps at
1024px. Use --steps 100 --seeds 0 1 for the larger comparison. Text is
encoded once, the base model stays resident for the guidance sweep, and the
Turbo adapter is applied only after all standard runs finish. Completed runs
are resumable. Scores evaluate the actual saved pixels, without labels.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CASES = [
    dict(id="owl-dog", style="a watercolor painting of", prompts=["an owl", "a dog"],
         views=["identity", "rotate_180"]),
    dict(id="village-horse", style="an oil painting of", prompts=["a snowy mountain village", "a horse"],
         views=["identity", "rotate_cw"]),
]
MODES = [
    dict(id="standard-g3", turbo=False, steps=100, guidance=3.0, reduction="mean"),
    dict(id="standard-g4", turbo=False, steps=100, guidance=4.0, reduction="mean"),
    dict(id="standard-g5", turbo=False, steps=100, guidance=5.0, reduction="mean"),
    dict(id="turbo-r2-mean", turbo=True, steps=4, guidance=1.0, reduction="mean"),
    dict(id="turbo-r2-sum", turbo=True, steps=4, guidance=1.0, reduction="sum", control=True),
]


def write_json(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temporary.replace(path)


def plan_for(seeds, steps=30):
    from visual_anagrams.iris import CODE_REVISION, MODEL_ID, MODEL_REVISION, TEXT_REVISION
    from visual_anagrams.turbo import TURBO_REVISION

    modes = [dict(mode, steps=mode["steps"] if mode["turbo"] else steps) for mode in MODES]
    return dict(cases=CASES, modes=modes, seeds=seeds, size=1024, shift=4.0, solver_order=2,
                model=MODEL_ID, model_revision=MODEL_REVISION, code_revision=CODE_REVISION,
                text_revision=TEXT_REVISION, turbo_revision=TURBO_REVISION, adapter="r2-1000")


def entries(plan):
    for mode in plan["modes"]:
        for case in plan["cases"]:
            for seed in plan["seeds"][:1] if mode.get("control") else plan["seeds"]:
                yield mode, case, seed


def run_directory(output, mode, case, seed):
    return output / mode["id"] / case["id"] / f"{seed:04}"


def generate(output, plan, text_device):
    import torch
    from visual_anagrams.iris import PromptEncoding, encode_prompts, load_model, read_config, resolve_model, sample_iris
    from visual_anagrams.turbo import apply_turbo_lora, resolve_turbo_adapter, sample_turbo
    from visual_anagrams.utils import save_illusion
    from visual_anagrams.views import get_views

    todo = [(m, c, s) for m, c, s in entries(plan)
            if not (run_directory(output, m, c, s) / "run.json").is_file()]
    if not todo:
        print("All generations already complete", flush=True)
        return
    torch.set_num_threads(6)
    directory = resolve_model()
    cfg = read_config(directory)
    cache = output / "prompt-encodings.pt"
    fingerprint = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    if cache.is_file():
        saved = torch.load(cache, map_location="cpu", weights_only=True)
        if saved["plan_hash"] != fingerprint:
            raise ValueError("Prompt cache does not match this benchmark plan")
        print("Reusing cached prompt embeddings", flush=True)
    else:
        prompts = [f"{case['style']} {prompt}" for case in plan["cases"] for prompt in case["prompts"]]
        print(f"Encoding {len(prompts)} prompts once on {text_device}", flush=True)
        start = time.perf_counter()
        try:
            positive, negative = encode_prompts(cfg, prompts, "", device=text_device)
        except torch.OutOfMemoryError:
            if text_device != "cuda":
                raise
            print("Text encoder exceeded currently free VRAM; retrying on CPU", flush=True)
            gc.collect()
            torch.cuda.empty_cache()
            positive, negative = encode_prompts(cfg, prompts, "", device="cpu")
        saved = dict(plan_hash=fingerprint, embeddings=positive.embeddings, mask=positive.mask,
                     negative_embeddings=negative.embeddings, negative_mask=negative.mask,
                     encoding_seconds=time.perf_counter() - start)
        torch.save(saved, cache)
    positive = PromptEncoding(saved["embeddings"], saved["mask"])
    negative = PromptEncoding(saved["negative_embeddings"], saved["negative_mask"])
    print("Loading Iris INT8 once for the benchmark", flush=True)
    model = load_model(directory, cfg, "cuda")
    write_json(output / "environment.json", dict(
        started_utc=datetime.now(timezone.utc).isoformat(), torch=torch.__version__,
        cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
        model_manifest=json.loads((directory / "manifest.json").read_text(encoding="utf-8")),
        prompt_encoding_seconds=saved["encoding_seconds"],
    ))
    applied = False
    warmed = set()
    for number, (mode, case, seed) in enumerate(todo, 1):
        if mode["turbo"] and not applied:
            print("Applying Turbo r2 after standard runs", flush=True)
            apply_turbo_lora(model, resolve_turbo_adapter("r2-1000"))
            applied = True
        case_index = plan["cases"].index(case)
        encoding = PromptEncoding(positive.embeddings[case_index * 2:case_index * 2 + 2],
                                  positive.mask[case_index * 2:case_index * 2 + 2])
        views = get_views(case["views"])
        key = "turbo" if mode["turbo"] else "standard"
        if key not in warmed:
            print(f"Warming {key} model kernels", flush=True)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                warm = model(torch.zeros((1, 3, 1024, 1024), device="cuda"),
                             torch.full((1,), 500.0, device="cuda"),
                             encoding.embeddings[:1].cuda(), y_mask=encoding.mask[:1].cuda()).x
                del warm
            torch.cuda.synchronize()
            warmed.add(key)
        print(f"BENCHMARK {number}/{len(todo)}: {mode['id']} / {case['id']} / seed {seed}", flush=True)
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        kwargs = dict(device="cuda", num_inference_steps=mode["steps"], reduction=mode["reduction"],
                      generator=torch.Generator(device="cuda").manual_seed(seed), shift=4.0)
        if mode["turbo"]:
            image = sample_turbo(model, encoding, views, **kwargs)
        else:
            image = sample_iris(model, encoding, negative, views, guidance_scale=mode["guidance"], order=2, **kwargs)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        peak = torch.cuda.max_memory_allocated() / 2**30
        destination = run_directory(output, mode, case, seed)
        destination.mkdir(parents=True, exist_ok=True)
        save_illusion(image.cpu(), views, destination)
        write_json(destination / "run.json", dict(
            mode=mode, case=case, seed=seed, seconds=elapsed, peak_allocated_gib=peak,
            estimated_network_evaluations=2 * mode["steps"] * (1 if mode["turbo"] else 2),
        ))
        print(f"DONE {mode['id']} / {case['id']} / {seed}: {elapsed:.2f}s, {peak:.2f} GiB allocated", flush=True)
        del image
    del model
    gc.collect()
    torch.cuda.empty_cache()


def report(output, plan, *, evaluator_file="evaluator.json", suffix=""):
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    import torch
    import torch.nn.functional as F
    from transformers import CLIPModel, CLIPProcessor

    torch.set_num_threads(6)
    evaluator_path = output / evaluator_file
    if evaluator_path.is_file():
        evaluator = json.loads(evaluator_path.read_text(encoding="utf-8"))
    else:
        from huggingface_hub import HfApi

        clip_id = "openai/clip-vit-large-patch14" if suffix == "-holdout" else "openai/clip-vit-base-patch32"
        evaluator = dict(model=clip_id, revision=HfApi().model_info(clip_id).sha)
        write_json(evaluator_path, evaluator)
    clip_id = evaluator["model"]
    print(f"Scoring saved outputs with {clip_id} on CPU", flush=True)
    scorer = CLIPModel.from_pretrained(clip_id, revision=evaluator["revision"]).float().eval()
    processor = CLIPProcessor.from_pretrained(clip_id, revision=evaluator["revision"], use_fast=False)
    score_records = []
    font_path = ROOT / "visual_anagrams/assets/CourierPrime-Regular.ttf"
    font = ImageFont.truetype(str(font_path), 17)
    small = ImageFont.truetype(str(font_path), 14)
    panels = []
    with torch.inference_mode():
        for case in plan["cases"]:
            for seed in plan["seeds"]:
                modes = [m for m in plan["modes"] if not m.get("control") or seed == plan["seeds"][0]]
                columns = min(3, len(modes))
                panel = Image.new("RGB", (columns * 512, 70 + math.ceil(len(modes) / columns) * 410), "#eef0f4")
                draw = ImageDraw.Draw(panel)
                draw.text((12, 8), f"{case['id']} | seed {seed} | {case['style']}", fill="#182337", font=font)
                draw.text((12, 32), f"Upright: {case['prompts'][0]} | transformed: {case['prompts'][1]}",
                          fill="#182337", font=small)
                for index, mode in enumerate(modes):
                    left, top = (index % columns) * 512, 70 + (index // columns) * 410
                    directory = run_directory(output, mode, case, seed)
                    path = directory / "sample_1024.png"
                    if not path.is_file() or not (directory / "run.json").is_file():
                        draw.text((left + 12, top), "PENDING " + mode["id"], font=font, fill="black")
                        continue
                    with Image.open(path) as source:
                        image = source.convert("RGB")
                    transformed = image.transpose(Image.Transpose.ROTATE_180 if case["views"][1] == "rotate_180"
                                                  else Image.Transpose.ROTATE_270)
                    inputs = processor(text=[f"{case['style']} {p}" for p in case["prompts"]],
                                       images=[image, transformed], return_tensors="pt", padding=True)
                    values = scorer(**inputs)
                    similarities = F.normalize(values.image_embeds, dim=-1) @ F.normalize(values.text_embeds, dim=-1).T
                    aligned = similarities.diag().tolist()
                    weakest = min(aligned)
                    margin = min(float(similarities[i, i] - similarities[i, 1 - i]) for i in range(2))
                    run = json.loads((directory / "run.json").read_text(encoding="utf-8"))
                    record = dict(case=case["id"], seed=seed, mode=mode["id"], scores=aligned,
                                  weakest_view=weakest, minimum_correct_prompt_margin=margin,
                                  similarity_matrix=similarities.tolist(), seconds=run["seconds"],
                                  peak_allocated_gib=run["peak_allocated_gib"], image=path.relative_to(output).as_posix())
                    score_records.append(record)
                    write_json(directory / f"scores{suffix}.json", record)
                    draw.text((left + 12, top), f"{mode['id']} ({mode['steps']} steps)", font=font, fill="black")
                    for n, viewed in enumerate((image, transformed)):
                        panel.paste(viewed.resize((256, 256), Image.Resampling.LANCZOS), (left + n * 256, top + 30))
                    draw.text((left + 12, top + 298), f"CLIP: {aligned[0]:.3f} / {aligned[1]:.3f}", font=font, fill="black")
                    draw.text((left + 12, top + 326), f"Weaker view: {weakest:.3f}", font=font, fill="black")
                    draw.text((left + 12, top + 354), f"Minimum prompt margin: {margin:+.3f}", font=small, fill="black")
                    draw.text((left + 12, top + 382), f"{run['seconds']:.1f}s | {run['peak_allocated_gib']:.2f} GiB allocated", font=small, fill="black")
                filename = f"comparison{suffix}-{case['id']}-seed-{seed}.jpg"
                panel.save(output / filename, quality=94)
                panels.append((filename, panel))
    aggregate = []
    for mode in plan["modes"]:
        rows = [r for r in score_records if r["mode"] == mode["id"]]
        if rows:
            aggregate.append(dict(mode=mode["id"], samples=len(rows),
                                  mean_weakest_view=float(np.mean([r["weakest_view"] for r in rows])),
                                  mean_minimum_prompt_margin=float(np.mean([r["minimum_correct_prompt_margin"] for r in rows])),
                                  mean_seconds=float(np.mean([r["seconds"] for r in rows]))))
    write_json(output / f"scores{suffix}.json", dict(evaluator=evaluator, note="Pilot scores, not comparable to paper metrics using other evaluators",
                                            results=score_records, summary=aggregate))
    lines = ["# Iris quality pilot", "", "Two prompt pairs, fixed seeds, 1024px, INT8 weights. All outputs are included; no best-seed selection.", "",
             f"Evaluator: `{clip_id}` at revision `{evaluator['revision']}`. Scores measure the weaker of the two views. Prompt margin checks whether each view prefers its own prompt over the other. These are screening metrics, not proof of a convincing human illusion, and are not comparable to scores from different CLIP models.", "",
             "| Mode | Samples | Mean weaker-view score | Mean minimum prompt margin | Mean generation seconds |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for row in aggregate:
        lines.append(f"| {row['mode']} | {row['samples']} | {row['mean_weakest_view']:.4f} | {row['mean_minimum_prompt_margin']:+.4f} | {row['mean_seconds']:.1f} |")
    lines.extend(["", f"Standard modes use {next(m['steps'] for m in plan['modes'] if not m['turbo'])} steps. "
                  f"Seeds: {plan['seeds']}. Baseline Turbo and Sum controls use 4 steps; extra Turbo modes test larger step counts.", "",
                  "Timings exclude prompt encoding, model loading, one warm-up forward per model mode, saving, and scoring. VRAM values are PyTorch allocations and exclude other desktop applications.", ""])
    for filename, _ in panels:
        lines += [f"![{filename}]({filename})", ""]
    (output / f"REPORT{suffix}.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(aggregate, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/quality_pilot_20261024")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--steps", type=int, choices=[30, 100], default=30)
    parser.add_argument("--stage", choices=["all", "generate", "report"], default="all")
    parser.add_argument("--text-device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    plan = plan_for(args.seeds, args.steps)
    plan_path = output / "plan.json"
    if plan_path.is_file() and json.loads(plan_path.read_text(encoding="utf-8")) != plan:
        raise ValueError("Existing benchmark directory has a different plan; use a new --output")
    write_json(plan_path, plan)
    if args.stage in ("all", "generate"):
        generate(output, plan, args.text_device)
    if args.stage in ("all", "report"):
        report(output, plan)


if __name__ == "__main__":
    main()
