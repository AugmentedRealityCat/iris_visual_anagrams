"""Score the small advanced-method check on CPU and publish its comparison."""

import argparse
import gc
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVALUATORS = {
    "guide_b32": ("openai/clip-vit-base-patch32", "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"),
    "holdout_l14": ("openai/clip-vit-large-patch14", "32bd64288804d66eefd0ccbe215aa642df71cc41"),
}


def main():
    import torch
    import torch.nn.functional as F
    from PIL import Image, ImageDraw, ImageFont
    from transformers import CLIPModel, CLIPProcessor

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, default=ROOT / "results/advanced_methods_20261024")
    parser.add_argument("--corrected", type=Path, default=ROOT / "results/advanced_methods_20261024_corrected")
    parser.add_argument("--output", type=Path, default=ROOT / "examples/advanced-methods")
    args = parser.parse_args()
    torch.set_num_threads(6)
    args.output.mkdir(parents=True, exist_ok=True)
    folders = {
        "classic": args.original / "plain/owl-dog/0000",
        "anagram_mtl": args.corrected / "anagram_mtl/owl-dog/0000",
        "syncvc": args.original / "syncvc/owl-dog/0000",
        "mtl_unscaled_failed_trial": args.original / "anagram_mtl/owl-dog/0000",
    }
    images, records = {}, {}
    for name, folder in folders.items():
        record = json.loads((folder / "run.json").read_text(encoding="utf-8"))
        path = folder / "sample_1024.png"
        image = Image.open(path).convert("RGB")
        assert image.size == (1024, 1024)
        images[name] = [image, image.transpose(Image.Transpose.ROTATE_180)]
        record["image_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        record["local_image"] = str(path.relative_to(ROOT)).replace("\\", "/")
        if name == "mtl_unscaled_failed_trial":
            record["note"] = "Initial failed trial used strength*t; final implementation uses strength*t*(1-t)."
        records[name] = record
    prompts = ["a watercolor painting of an owl", "a watercolor painting of a dog"]
    report = dict(prompts=prompts, views=["identity", "rotate_180"], seed=0, steps=30,
                  guidance=3, evaluators={}, runs=records,
                  limitations="One pair and seed. B/32 guides MTL; L/14 is held out. Neither is a human illusion rating.")
    for key, (model_id, revision) in EVALUATORS.items():
        print("Scoring", model_id, "on CPU", flush=True)
        model = CLIPModel.from_pretrained(model_id, revision=revision).float().eval()
        processor = CLIPProcessor.from_pretrained(model_id, revision=revision, use_fast=False)
        results = {}
        with torch.inference_mode():
            for name, viewed in images.items():
                inputs = processor(text=prompts, images=viewed, return_tensors="pt", padding=True)
                values = model(**inputs)
                matrix = F.normalize(values.image_embeds, dim=-1) @ F.normalize(values.text_embeds, dim=-1).T
                aligned = matrix.diag().tolist()
                results[name] = dict(scores=aligned, weaker_view=min(aligned), similarity_matrix=matrix.tolist(),
                    minimum_prompt_margin=min(float(matrix[i, i] - matrix[i, 1-i]) for i in range(2)))
                print(key, name, results[name], flush=True)
        report["evaluators"][key] = dict(model=model_id, revision=revision, results=results)
        del model, processor
        gc.collect()
    sources = ["generate.py", "visual_anagrams/gui.py", "visual_anagrams/iris.py", "visual_anagrams/advanced_sampling.py",
               "visual_anagrams/optimization.py", "visual_anagrams/method_options.py",
               "scripts/compare_advanced_methods.py", "scripts/report_advanced_methods.py"]
    report["final_source_sha256"] = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources}
    (args.output / "scores.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    font_path = str(ROOT / "visual_anagrams/assets/CourierPrime-Regular.ttf")
    font, small = ImageFont.truetype(font_path, 23), ImageFont.truetype(font_path, 19)
    labels = {"classic": "Classic (reused baseline)", "anagram_mtl": "Anagram-MTL / Iris",
              "syncvc": "SyncVC / Iris", "mtl_unscaled_failed_trial": "Failed initial MTL trial (unscaled)"}

    def panel(names, filename):
        sheet = Image.new("RGB", (1024, 86 + len(names) * 590), "#eef0f4")
        draw = ImageDraw.Draw(sheet)
        draw.text((16, 10), "Owl upright | Dog after 180-degree rotation", font=font, fill="#182337")
        draw.text((16, 43), "30 steps / CFG 3 / seed 0 / 1024 px / Iris INT8", font=small, fill="#182337")
        for index, name in enumerate(names):
            top = 86 + index * 590
            draw.text((16, top), labels[name], font=font, fill="#182337")
            for side, viewed in enumerate(images[name]):
                sheet.paste(viewed.resize((512, 512), Image.Resampling.LANCZOS), (512 * side, top + 32))
            score = report["evaluators"]["holdout_l14"]["results"][name]["weaker_view"]
            run = records[name]
            draw.text((16, top + 551), f"L/14 weaker view: {score:.4f} | {run['seconds']:.1f}s | {run['peak_allocated_gib']:.2f} GiB", font=small, fill="#182337")
        sheet.save(args.output / filename, quality=92)

    panel(["classic", "anagram_mtl", "syncvc"], "comparison.jpg")
    panel(["mtl_unscaled_failed_trial"], "failed-initial-mtl.jpg")
    lines = ["# Advanced-method check", "", "One prompt pair and seed, reusing the previous Classic baseline. "
             "Only three new full-size images were generated: the two methods and one corrective MTL trial. "
             "All trials, including the failure, are included.", "", "Prompts: `a watercolor painting of an owl` / "
             "`a watercolor painting of a dog`. Views: identity / 180-degree rotation. Seed 0, 1024 px, "
             "Standard Iris INT8, 30 steps, CFG 3, shift 4. Both methods use one optimization pass every five steps, "
             "strength 0.05 and CPU activation offloading. MTL subjects: `owl | dog`.", "",
             "| Method | B/32 weaker view | Held-out L/14 weaker view | L/14 min prompt margin | Seconds | Peak allocated GiB |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name in folders:
        guide = report["evaluators"]["guide_b32"]["results"][name]
        holdout = report["evaluators"]["holdout_l14"]["results"][name]
        run = records[name]
        lines.append(f"| {labels[name]} | {guide['weaker_view']:.4f} | {holdout['weaker_view']:.4f} | "
                     f"{holdout['minimum_prompt_margin']:+.4f} | {run['seconds']:.1f} | {run['peak_allocated_gib']:.2f} |")
    lines += ["", "![Three-method comparison](comparison.jpg)", "", "## Interpretation", "",
              "Neither adaptation improved this pair on either evaluator. Classic remains the recommended default. "
              "Corrected MTL removed the severe oversaturation, but both subject heads still occupy separate areas. "
              "SyncVC mostly produced a dog and lost the upright owl. The extra optimization cost did not pay off here.", "",
              "CLIP similarities screen prompt alignment; they do not establish whether people see a convincing illusion. "
              "B/32 is used inside MTL, so L/14 is the primary held-out check. One seed does not establish a general ranking.", "",
              "The initial MTL attention edit was too aggressive and produced severe oversaturation. Its `strength*t` update "
              "was replaced with `strength*t*(1-t)`, bounding the edit in signal units near pure noise. "
              "The failed output is preserved below; it is not the final default.", "",
              "![Initial MTL failure](failed-initial-mtl.jpg)", "", "## Reproduce", "",
              "The comparison script reuses `results/quality_pilot_20261024/prompt-encodings.pt` and its Standard CFG-3 "
              "baseline. For fresh prompts, use the public CLI commands in the main README.", "", "```powershell",
              r".\.venv\Scripts\python.exe scripts/compare_advanced_methods.py --method all --output results/my-method-check",
              "```", "", "`scripts/report_advanced_methods.py` scores saved images on CPU. "
              "[scores.json](scores.json) contains pinned evaluator revisions, per-view similarities, "
              "runtimes, per-step optimization diagnostics, image hashes, local source paths, and final source hashes.", "",
              "Hardware: RTX 3060 12 GB, Ryzen 5 5500, 64 GB RAM. Timings exclude model loading, prompt encoding, saving "
              "and scoring. The reused baseline had a warm-up forward; advanced runs did not, so timings are approximate. "
              "VRAM is PyTorch peak allocated memory, excluding desktop applications and the CUDA context.", "",
              "These are Iris adaptations of [Anagram-MTL](https://github.com/Pixtella/Anagram-MTL) and "
              "[SyncVC](https://github.com/mandt-lab/SyncVC), with different attention architecture, flow schedule and "
              "optimization budget. They do not reproduce the papers' DeepFloyd results.", ""]
    (args.output / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print("Saved", args.output, flush=True)


if __name__ == "__main__":
    main()
