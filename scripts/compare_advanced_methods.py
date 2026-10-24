"""Small real-model check; unlike unit tests, this loads the complete Iris model."""

from pathlib import Path
import argparse
import gc
import json
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    import torch
    from transformers import AutoTokenizer
    from visual_anagrams.advanced_sampling import PreviewScorer, sample_advanced
    from visual_anagrams.iris import PromptEncoding, TEXT_ID, TEXT_REVISION, load_model, read_config, resolve_model
    from visual_anagrams.method_options import MethodOptions
    from visual_anagrams.optimization import subject_token_indices
    from visual_anagrams.utils import save_illusion
    from visual_anagrams.views import get_views

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/advanced_method_check"))
    parser.add_argument("--method", choices=["all", "anagram_mtl", "syncvc"], default="all")
    parser.add_argument("--strength", type=float)
    args = parser.parse_args()
    torch.set_num_threads(6)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    source = Path("results/quality_pilot_20261024")
    saved = torch.load(source / "prompt-encodings.pt", weights_only=True)
    positive = PromptEncoding(saved["embeddings"][:2], saved["mask"][:2])
    negative = PromptEncoding(saved["negative_embeddings"], saved["negative_mask"])
    prompts = ["a watercolor painting of an owl", "a watercolor painting of a dog"]
    names = ["identity", "rotate_180"]
    methods = ["anagram_mtl", "syncvc"] if args.method == "all" else [args.method]
    baseline = output / "plain/owl-dog/0000"
    baseline.mkdir(parents=True, exist_ok=True)
    for name in ["sample_1024.png", "sample_1024.views.png", "run.json"]:
        if not (baseline / name).exists():
            shutil.copy2(source / "standard-g3/owl-dog/0000" / name, baseline / name)
    plan = dict(prompts=prompts, views=names, seed=0, steps=30, guidance=3,
                baseline_reused=str(source / "standard-g3/owl-dog/0000"), methods=methods,
                strength=args.strength)
    plan_path = output / "plan.json"
    if plan_path.is_file() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("Existing method check has different settings; use a new --output")
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    pending = [m for m in methods if not (output / m / "owl-dog/0000/run.json").is_file()]
    if not pending:
        print("All requested samples are complete", flush=True)
        return
    directory = resolve_model()
    model = load_model(directory, read_config(directory), "cuda")
    tokenizer = AutoTokenizer.from_pretrained(TEXT_ID, revision=TEXT_REVISION)
    positions = subject_token_indices(tokenizer, prompts, ["owl", "dog"])
    del tokenizer
    scorer = PreviewScorer(prompts) if "anagram_mtl" in pending else None
    for method in pending:
        options = MethodOptions(method, strength=args.strength)
        destination = output / method / "owl-dog/0000"
        destination.mkdir(parents=True, exist_ok=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        diagnostics = {}
        print("RUNNING", method, "strength", options.strength, flush=True)
        image = sample_advanced(model, positive, negative, get_views(names), options=options,
            prompts=prompts, subject_positions=positions, scorer=scorer, diagnostics=diagnostics,
            generator=torch.Generator(device="cuda").manual_seed(0))
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        save_illusion(image.cpu(), get_views(names), destination)
        record = dict(method=method, seconds=seconds, peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                      seed=0, steps=30, guidance=3, options=vars(options), diagnostics=diagnostics)
        (destination / "run.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        print("DONE", method, "seconds", seconds, "peak", record["peak_allocated_gib"], flush=True)
        del image
        gc.collect()
        torch.cuda.empty_cache()
    print("Requested advanced samples complete", flush=True)


if __name__ == "__main__":
    main()
