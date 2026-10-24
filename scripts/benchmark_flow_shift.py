"""Three four-step Turbo r2 anagrams: Joker upright, Batman rotated 180 degrees."""

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def main():
    import torch
    from visual_anagrams.iris import (CODE_REVISION, MODEL_ID, MODEL_REVISION, TEXT_REVISION,
                                     PromptEncoding, encode_prompts, load_model, read_config, resolve_model)
    from visual_anagrams.turbo import TURBO_REVISION, resolve_turbo_adapter, sample_turbo
    from visual_anagrams.utils import save_illusion
    from visual_anagrams.views import get_views

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/flow_shift_joker_batman_20261025")
    parser.add_argument("--publish", type=Path, default=ROOT / "examples/flow-shift")
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(6)
    prompts = ["a comic book portrait of the Joker from DC Comics, white face, green hair and a red smile",
               "a comic book portrait of Batman from DC Comics wearing his black pointed-eared cowl"]
    plan = dict(prompts=prompts, views=["identity", "rotate_180"], shifts=[2.,4.,6.], seed=0,
                steps=4, guidance=1., reduction="mean", size=1024, adapter="r2-1000",
                model=MODEL_ID, model_revision=MODEL_REVISION, code_revision=CODE_REVISION,
                text_revision=TEXT_REVISION, turbo_revision=TURBO_REVISION)
    plan_path = output / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text(encoding="utf-8")) != plan:
        raise ValueError("Existing output has a different experiment plan; choose a new --output")
    write_json(plan_path, plan)
    pending = [shift for shift in plan["shifts"] if not (output/f"shift-{shift:g}"/"run.json").is_file()]
    if pending:
        directory = resolve_model()
        cfg = read_config(directory)
        fingerprint = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
        cache = output / "prompt-encodings.pt"
        if cache.is_file():
            saved = torch.load(cache, weights_only=True, map_location="cpu")
            assert saved["plan_hash"] == fingerprint
            positive = PromptEncoding(saved["embeddings"], saved["mask"])
        else:
            print("Encoding both prompts once; no negative encoding needed for Turbo", flush=True)
            start = time.perf_counter()
            try:
                positive, _ = encode_prompts(cfg, prompts, device="cuda", encode_negative=False)
            except torch.OutOfMemoryError:
                gc.collect()
                torch.cuda.empty_cache()
                print("Retrying text encoding on CPU", flush=True)
                positive, _ = encode_prompts(cfg, prompts, device="cpu", encode_negative=False)
            torch.save(dict(plan_hash=fingerprint, embeddings=positive.embeddings,
                            mask=positive.mask, seconds=time.perf_counter()-start), cache)
        print("Loading Iris INT8 and Turbo r2 once", flush=True)
        model = load_model(directory, cfg, "cuda", turbo_path=resolve_turbo_adapter("r2-1000"))
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            warm = model(torch.zeros((1,3,1024,1024),device="cuda"), torch.tensor([500.],device="cuda"),
                         positive.embeddings[:1].cuda(),y_mask=positive.mask[:1].cuda()).x
            del warm
        torch.cuda.synchronize()
        write_json(output / "environment.json",dict(torch=torch.__version__,cuda=torch.version.cuda,
                   gpu=torch.cuda.get_device_name(0),model_manifest=json.loads((directory/"manifest.json").read_text(encoding="utf-8"))))
        for shift in pending:
            print("RUN shift",shift,flush=True)
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            pixels = sample_turbo(model, positive, get_views(plan["views"]),
                      num_inference_steps=4, shift=shift, reduction="mean",
                      generator=torch.Generator(device="cuda").manual_seed(0))
            torch.cuda.synchronize()
            seconds = time.perf_counter()-start
            folder = output/f"shift-{shift:g}"
            folder.mkdir(exist_ok=True)
            save_illusion(pixels.cpu(),get_views(plan["views"]),folder)
            write_json(folder/"run.json",dict(shift=shift,seconds=seconds,
                       peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                       image_sha256=hashlib.sha256((folder/"sample_1024.png").read_bytes()).hexdigest()))
            print("DONE",shift,seconds,flush=True)
            del pixels
        del model, positive
        gc.collect()
        torch.cuda.empty_cache()
    report(output,args.publish,plan)


def report(output, published, plan):
    import torch
    import torch.nn.functional as F
    from PIL import Image, ImageDraw, ImageFont
    from transformers import CLIPModel, CLIPProcessor

    published.mkdir(parents=True,exist_ok=True)
    model_id = "openai/clip-vit-large-patch14"
    revision = "32bd64288804d66eefd0ccbe215aa642df71cc41"
    model = CLIPModel.from_pretrained(model_id,revision=revision).float().eval()
    processor = CLIPProcessor.from_pretrained(model_id,revision=revision,use_fast=False)
    panel = Image.new("RGB",(1152,940),"#eef0f4")
    draw = ImageDraw.Draw(panel)
    font_path = str(ROOT/"visual_anagrams/assets/CourierPrime-Regular.ttf")
    font, small = ImageFont.truetype(font_path,21),ImageFont.truetype(font_path,16)
    draw.text((12,8),"Joker upright / Batman after 180-degree rotation",font=font,fill="#182337")
    draw.text((12,38),"Turbo r2 / 4 steps / seed 0 / 1024px / same noise per shift",font=small,fill="#182337")
    results=[]
    with torch.inference_mode():
        for column,shift in enumerate(plan["shifts"]):
            folder=output/f"shift-{shift:g}"
            run=json.loads((folder/"run.json").read_text(encoding="utf-8"))
            image=Image.open(folder/"sample_1024.png").convert("RGB")
            rotated=image.transpose(Image.Transpose.ROTATE_180)
            values=model(**processor(text=plan["prompts"],images=[image,rotated],padding=True,return_tensors="pt"))
            matrix=F.normalize(values.image_embeds,dim=-1)@F.normalize(values.text_embeds,dim=-1).T
            scores=matrix.diag().tolist()
            run.update(scores=scores,weaker_view=min(scores),similarity_matrix=matrix.tolist(),
                       minimum_prompt_margin=min(float(matrix[i,i]-matrix[i,1-i]) for i in range(2)),
                       local_image=f"results/{output.name}/shift-{shift:g}/sample_1024.png")
            results.append(run)
            left=column*384
            draw.text((left+12,76),f"Flow shift {shift:g}",font=font,fill="#182337")
            for row,viewed in enumerate([image,rotated]):
                panel.paste(viewed.resize((384,384),Image.Resampling.LANCZOS),(left,108+row*384))
            draw.text((left+12,886),f"Joker {scores[0]:.3f} / Batman {scores[1]:.3f}",font=small,fill="#182337")
            draw.text((left+12,911),f"{run['seconds']:.1f}s / {run['peak_allocated_gib']:.2f} GiB",font=small,fill="#182337")
            image.save(published/f"shift-{shift:g}.jpg",quality=93)
    panel.save(published/"comparison.jpg",quality=94)
    write_json(published/"results.json",dict(plan=plan,evaluator=dict(model=model_id,revision=revision),results=results,
               note="One seed and prompt pair. CLIP alignment is not a human illusion rating. No CLIP guidance used."))
    lines=["# Turbo flow-shift check", "", "Joker upright; Batman after a 180-degree rotation. "
           "Three new images, one seed, Turbo r2-1000, four steps, CFG 1, Mean combining, 1024x1024. "
           "The only changed setting is flow shift. The same initial noise and subsequent re-noising draws are reused.", "",
           "| Shift | Joker CLIP | Batman CLIP | Weaker view | Minimum prompt margin | Seconds | Peak GiB |",
           "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in results:
        lines.append(f"| {row['shift']:g} | {row['scores'][0]:.4f} | {row['scores'][1]:.4f} | {row['weaker_view']:.4f} | {row['minimum_prompt_margin']:+.4f} | {row['seconds']:.1f} | {row['peak_allocated_gib']:.2f} |")
    lines += ["", "![All three shifts, both orientations](comparison.jpg)", "", "## Findings", "",
              "Shifts 2 and 4 were close. Shift 2 had a slightly stronger Batman/weaker-view score; "
              "shift 4 had a stronger Joker score. Shift 6 scored lowest for Batman. "
              "All three rotated views preferred the Joker prompt over Batman, reflected in the negative "
              "minimum prompt margins. Visually the results keep a dominant Joker face with a smaller, "
              "separate Batman-like face; none is a convincing Joker-to-Batman transformation. "
              "This small check does not justify changing the default shift of 4.", "", "Prompts:", "",
              *[f"- `{prompt}`" for prompt in plan["prompts"]], "",
              "CLIP ViT-L/14 scores the saved pixels against the two full prompts. Higher weaker-view scores "
              "indicate stronger alignment of both orientations, not necessarily a convincing illusion. "
              "One seed cannot establish the best shift generally. These settings do not change the GUI defaults.", "",
              "Hardware: RTX 3060 12 GB. Generation times exclude model loading, prompt encoding, warm-up, "
              "saving and scoring. [Exact settings, scores and image hashes](results.json). "
              "Full-resolution PNGs and prompt cache are saved locally under `results/flow_shift_joker_batman_20261025/`.", "",
              "Reproduce with `python scripts/benchmark_flow_shift.py`; completed generations are reused.", ""]
    (published/"README.md").write_text("\n".join(lines),encoding="utf-8")
    print(json.dumps(results,indent=2),flush=True)


if __name__=="__main__":
    main()
