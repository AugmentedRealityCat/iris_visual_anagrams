"""Append low-cost Turbo method experiments to a completed quality pilot.

Usage: python scripts/benchmark_quality_experiments.py --output results/quality_pilot_20261024
The base experiment plan is retained. The extended plan, per-step research
diagnostics, and a combined report are saved alongside the original results.
"""

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.benchmark_quality import report, run_directory, write_json
from scripts.quality_methods import CleanPreviewCLIP, ResearchMixer, sample_research


EXTRA_MODES = [
    dict(id="turbo-r2-8step", turbo=True, steps=8, guidance=1.0, reduction="mean", research="mean"),
    dict(id="turbo-r2-16step", turbo=True, steps=16, guidance=1.0, reduction="mean", research="mean"),
    dict(id="turbo-r2-balanced", turbo=True, steps=4, guidance=1.0, reduction="mean", research="preview-balanced"),
    dict(id="turbo-r2-variance", turbo=True, steps=4, guidance=1.0, reduction="mean", research="bounded-variance"),
]


def main():
    import torch
    from visual_anagrams.iris import PromptEncoding, load_model, read_config, resolve_model
    from visual_anagrams.turbo import resolve_turbo_adapter
    from visual_anagrams.utils import save_illusion
    from visual_anagrams.views import get_views

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/quality_pilot_20261024")
    args = parser.parse_args()
    output = args.output.resolve()
    torch.set_num_threads(6)
    original = json.loads((output / "plan.json").read_text(encoding="utf-8"))
    extended = dict(original, modes=original["modes"] + EXTRA_MODES)
    extended["research_notes"] = dict(
        preview_balancing="Clean CLIP previews every two steps below t=.95; softmax(-10*score), "
                          "weights clipped to [.25,.75], then smoothed 50% with previous weights.",
        bounded_variance="Convert v to epsilon; Pearson correlation estimate, raw gain capped at 1.2, "
                         "50% blend (applied gain <=1.1); skip outside .05<t<.95. Not full Anagram-MTL.",
    )
    write_json(output / "experiments-plan.json", extended)
    saved = torch.load(output / "prompt-encodings.pt", map_location="cpu", weights_only=True)
    expected_hash = hashlib.sha256(json.dumps(original, sort_keys=True).encode()).hexdigest()
    if saved["plan_hash"] != expected_hash:
        raise ValueError("The base plan and saved prompt encodings differ")
    todo = [(m, c, s) for m in EXTRA_MODES for c in original["cases"] for s in original["seeds"]
            if not (run_directory(output, m, c, s) / "run.json").is_file()]
    if todo:
        if not (output / "evaluator.json").is_file():
            from huggingface_hub import HfApi

            guide_id = "openai/clip-vit-base-patch32"
            write_json(output / "evaluator.json", dict(model=guide_id, revision=HfApi().model_info(guide_id).sha))
        evaluator = json.loads((output / "evaluator.json").read_text(encoding="utf-8"))
        scorer = CleanPreviewCLIP(evaluator["model"], evaluator["revision"])
        directory = resolve_model()
        model = load_model(directory, read_config(directory), "cuda", turbo_path=resolve_turbo_adapter("r2-1000"))
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            warm = model(torch.zeros((1, 3, 1024, 1024), device="cuda"),
                         torch.full((1,), 500.0, device="cuda"),
                         saved["embeddings"][:1].cuda(), y_mask=saved["mask"][:1].cuda()).x
            del warm
        torch.cuda.synchronize()
        for number, (mode, case, seed) in enumerate(todo, 1):
            case_index = original["cases"].index(case)
            positive = PromptEncoding(saved["embeddings"][case_index * 2:case_index * 2 + 2],
                                      saved["mask"][case_index * 2:case_index * 2 + 2])
            views = get_views(case["views"])
            scorer.set_prompts([f"{case['style']} {prompt}" for prompt in case["prompts"]])
            mixer = ResearchMixer(mode["research"], views, scorer if mode["research"] == "preview-balanced" else None)
            print(f"EXPERIMENT {number}/{len(todo)}: {mode['id']} / {case['id']} / seed {seed}", flush=True)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start = time.perf_counter()
            image = sample_research(model, positive, None, views, mixer, turbo=True, steps=mode["steps"], seed=seed)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            destination = run_directory(output, mode, case, seed)
            destination.mkdir(parents=True, exist_ok=True)
            save_illusion(image.cpu(), views, destination)
            write_json(destination / "run.json", dict(mode=mode, case=case, seed=seed, seconds=elapsed,
                       peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                       estimated_network_evaluations=2 * mode["steps"],
                       research_history=mixer.history))
            print(f"DONE {mode['id']} / {case['id']}: {elapsed:.2f}s", flush=True)
            del image
        del model, scorer
        gc.collect()
        torch.cuda.empty_cache()
    report(output, extended)
    report(output, extended, evaluator_file="holdout-evaluator.json", suffix="-holdout")
    path = output / "REPORT.md"
    with path.open("a", encoding="utf-8") as stream:
        stream.write("\nResearch modes are isolated experiments, not changes to GUI defaults. "
                     "Preview balancing and bounded variance correction are adaptations inspired by Anagram-MTL, "
                     "not a reproduction. The Turbo adapter was trained for four steps; eight and sixteen steps are exploratory. "
                     "See experiments-plan.json and each research run.json for exact settings and per-step weights/gains.\n")
    holdout = output / "REPORT-holdout.md"
    if holdout.is_file():
        with holdout.open("a", encoding="utf-8") as stream:
            stream.write("\nThe preview-balancing method uses CLIP ViT-B/32 internally. This report uses "
                         "CLIP ViT-L/14, which was not used to guide generation. This reduces direct evaluator "
                         "coupling, but both evaluators are from the CLIP family. The pilot has only one seed per "
                         "prompt pair and does not establish statistical significance or human illusion quality.\n")


if __name__ == "__main__":
    main()
