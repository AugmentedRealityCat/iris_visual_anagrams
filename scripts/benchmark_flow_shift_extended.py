"""Extend the Joker/Batman shift check: six new Turbo images, reusing three baselines."""

import argparse
import gc
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.benchmark_flow_shift import write_json


def folder_for(root, steps, shift):
    return root / f"steps-{steps}-shift-{shift:g}"


def main():
    import torch
    from visual_anagrams.iris import PromptEncoding, load_model, read_config, resolve_model
    from visual_anagrams.turbo import resolve_turbo_adapter, sample_turbo
    from visual_anagrams.utils import save_illusion
    from visual_anagrams.views import get_views

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT/"results/flow_shift_joker_batman_20261025")
    parser.add_argument("--output", type=Path, default=ROOT/"results/flow_shift_extended_joker_batman_20261025")
    parser.add_argument("--publish", type=Path, default=ROOT/"examples/flow-shift-extended")
    args = parser.parse_args()
    torch.set_num_threads(6)
    original = json.loads((args.source/"plan.json").read_text(encoding="utf-8"))
    conditions = [dict(steps=4,shift=float(s)) for s in [1,2,3,4,6,8]]
    conditions += [dict(steps=8,shift=float(s)) for s in [2,4,6]]
    plan = dict(original, conditions=conditions, reused_baselines=[2.,4.,6.], new_images=6)
    plan.pop("steps")
    plan.pop("shifts")
    args.output.mkdir(parents=True,exist_ok=True)
    plan_path = args.output/"plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text(encoding="utf-8")) != plan:
        raise ValueError("Existing extension has a different plan; choose a new --output")
    write_json(plan_path,plan)
    for shift in plan["reused_baselines"]:
        old = args.source/f"shift-{shift:g}"
        new = folder_for(args.output,4,shift)
        new.mkdir(exist_ok=True)
        run = json.loads((old/"run.json").read_text(encoding="utf-8"))
        assert hashlib.sha256((old/"sample_1024.png").read_bytes()).hexdigest() == run["image_sha256"]
        for name in ["sample_1024.png","sample_1024.views.png"]:
            if not (new/name).exists():
                shutil.copy2(old/name,new/name)
        if not (new/"run.json").exists():
            write_json(new/"run.json",dict(run,steps=4,seed=0,reused_from=f"results/{args.source.name}/{old.name}"))
    pending=[c for c in conditions if not (folder_for(args.output,**c)/"run.json").is_file()]
    if pending:
        saved=torch.load(args.source/"prompt-encodings.pt",map_location="cpu",weights_only=True)
        assert saved["plan_hash"] == hashlib.sha256(json.dumps(original,sort_keys=True).encode()).hexdigest()
        positive=PromptEncoding(saved["embeddings"],saved["mask"])
        directory=resolve_model()
        model=load_model(directory,read_config(directory),"cuda",turbo_path=resolve_turbo_adapter("r2-1000"))
        with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
            warm=model(torch.zeros((1,3,1024,1024),device="cuda"),torch.tensor([500.],device="cuda"),
                       positive.embeddings[:1].cuda(),y_mask=positive.mask[:1].cuda()).x
            del warm
        torch.cuda.synchronize()
        write_json(args.output/"environment.json",dict(torch=torch.__version__,cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(0),model_manifest=json.loads((directory/"manifest.json").read_text(encoding="utf-8")),
            reused_prompt_cache=f"results/{args.source.name}/prompt-encodings.pt"))
        for condition in pending:
            print("RUN",condition,flush=True)
            torch.cuda.reset_peak_memory_stats()
            start=time.perf_counter()
            pixels=sample_turbo(model,positive,get_views(plan["views"]),num_inference_steps=condition["steps"],
                      shift=condition["shift"],reduction="mean",generator=torch.Generator(device="cuda").manual_seed(0))
            torch.cuda.synchronize()
            elapsed=time.perf_counter()-start
            folder=folder_for(args.output,**condition)
            folder.mkdir(exist_ok=True)
            save_illusion(pixels.cpu(),get_views(plan["views"]),folder)
            write_json(folder/"run.json",dict(condition,seed=0,seconds=elapsed,
                       peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                       image_sha256=hashlib.sha256((folder/"sample_1024.png").read_bytes()).hexdigest()))
            print("DONE",condition,elapsed,flush=True)
            del pixels
        del model,positive
        gc.collect()
        torch.cuda.empty_cache()
    report(args.output,args.publish,plan)


def report(output,published,plan):
    import torch
    import torch.nn.functional as F
    from PIL import Image,ImageDraw,ImageFont
    from transformers import CLIPModel,CLIPProcessor

    published.mkdir(parents=True,exist_ok=True)
    model_id="openai/clip-vit-large-patch14"
    revision="32bd64288804d66eefd0ccbe215aa642df71cc41"
    model=CLIPModel.from_pretrained(model_id,revision=revision).float().eval()
    processor=CLIPProcessor.from_pretrained(model_id,revision=revision,use_fast=False)
    results=[]
    viewed_images=[]
    with torch.inference_mode():
        for condition in plan["conditions"]:
            folder=folder_for(output,**condition)
            record=json.loads((folder/"run.json").read_text(encoding="utf-8"))
            image=Image.open(folder/"sample_1024.png").convert("RGB")
            viewed=[image,image.transpose(Image.Transpose.ROTATE_180)]
            values=model(**processor(text=plan["prompts"],images=viewed,return_tensors="pt",padding=True))
            matrix=F.normalize(values.image_embeds,dim=-1)@F.normalize(values.text_embeds,dim=-1).T
            scores=matrix.diag().tolist()
            record.update(scores=scores,weaker_view=min(scores),similarity_matrix=matrix.tolist(),
                          minimum_prompt_margin=min(float(matrix[i,i]-matrix[i,1-i]) for i in range(2)),
                          local_image=f"results/{output.name}/{folder.name}/sample_1024.png")
            results.append(record)
            viewed_images.append(viewed)
            image.save(published/f"{folder.name}.jpg",quality=92)
    font_path=str(ROOT/"visual_anagrams/assets/CourierPrime-Regular.ttf")
    font,small=ImageFont.truetype(font_path,20),ImageFont.truetype(font_path,17)

    def panel(indices,filename,title):
        columns=min(3,len(indices))
        rows=(len(indices)+columns-1)//columns
        sheet=Image.new("RGB",(columns*512,76+rows*350),"#eef0f4")
        draw=ImageDraw.Draw(sheet)
        draw.text((12,10),title,font=font,fill="#182337")
        draw.text((12,40),"Each pair: Joker upright | Batman rotated 180 degrees / seed 0",font=small,fill="#182337")
        for position,index in enumerate(indices):
            record=results[index]
            left,top=(position%columns)*512,76+(position//columns)*350
            draw.text((left+10,top),f"{record['steps']} steps / shift {record['shift']:g}",font=font,fill="#182337")
            for side,image in enumerate(viewed_images[index]):
                sheet.paste(image.resize((256,256),Image.Resampling.LANCZOS),(left+side*256,top+30))
            draw.text((left+10,top+296),f"J {record['scores'][0]:.3f} / B {record['scores'][1]:.3f} / min {record['weaker_view']:.3f}",font=small,fill="#182337")
            draw.text((left+10,top+322),f"margin {record['minimum_prompt_margin']:+.3f} / {record['seconds']:.1f}s",font=small,fill="#182337")
        sheet.save(published/filename,quality=94)

    panel(list(range(len(results))),"comparison.jpg","Extended Turbo r2 flow-shift / step-count check")
    panel([i for i,r in enumerate(results) if r["steps"]==4],"four-steps.jpg","Four steps: shifts 1, 2, 3, 4, 6 and 8")
    panel([i for i,r in enumerate(results) if r["steps"]==8],"eight-steps.jpg","Eight steps: shifts 2, 4 and 6")
    write_json(published/"results.json",dict(plan=plan,evaluator=dict(model=model_id,revision=revision),results=results,
        note="Six new images; three reused baselines. Single seed/pair; no CLIP guidance. More than four steps is exploratory for this adapter."))
    lines=["# Extended Turbo flow-shift check", "", "Six new Joker/Batman images and the three reused baselines. "
           "Four-step shifts: 1, 2, 3, 4, 6, 8. Eight-step shifts: 2, 4, 6. Turbo r2-1000, seed 0, "
           "CFG 1, Mean combining, 1024x1024. Prompts, model, LoRA and view mapping are unchanged. "
           "Eight steps are exploratory: this adapter was distilled for four.", "",
           "| Steps | Shift | Joker CLIP | Batman CLIP | Weaker view | Minimum prompt margin | Seconds |",
           "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in results:
        lines.append(f"| {r['steps']} | {r['shift']:g} | {r['scores'][0]:.4f} | {r['scores'][1]:.4f} | {r['weaker_view']:.4f} | {r['minimum_prompt_margin']:+.4f} | {r['seconds']:.1f} |")
    lines += ["", "![All nine configurations](comparison.jpg)", "",
              "[Larger four-step sheet](four-steps.jpg) · [Eight-step sheet](eight-steps.jpg)", "",
              "## Findings", "", "The best weaker-view score was **0.2448 at four steps / shift 8**, "
              "followed by **0.2392 at four steps / shift 3**. Shift 4 / eight steps scored 0.2302. "
              "Across the matched shifts 2, 4 and 6, eight steps had almost the same mean weaker-view score "
              "as four steps (0.2192 versus 0.2196), while taking about twice as long. "
              "All nine minimum prompt margins were negative: the rotated images still preferred Joker over Batman. "
              "Visually every configuration kept a dominant Joker/clown face and a smaller, separate Batman-like face. "
              "The higher CLIP scores did not produce a convincing transformation. Defaults remain unchanged.", "",
              "## Conditions and limitations", "", *[f"- `{p}`" for p in plan["prompts"]], "",
              "Joker is the upright view and Batman is the 180-degree view. All runs start from the same seed and "
              "reuse the original prompt embeddings. Within a step count, changing shift preserves the random "
              "draw sequence. Eight-step runs have additional re-noising draws. All trials are included; no best-seed selection.", "",
              "CLIP ViT-L/14 evaluates prompt alignment of saved pixels, not human illusion quality. A negative minimum "
              "prompt margin means at least one orientation prefers the other prompt. One pair and seed do not establish "
              "a generally optimal shift or step count. GUI defaults remain unchanged.", "",
              "RTX 3060 12 GB. Timings exclude downloads, loading, prompt encoding, warm-up, saving and scoring. "
              "[Exact settings, scores and hashes](results.json). Full-resolution PNGs are under "
              "`results/flow_shift_extended_joker_batman_20261025/`. [Original three-run check](../flow-shift/README.md).", "",
              "Reproduce with `python scripts/benchmark_flow_shift_extended.py`; it reuses the earlier prompt cache "
              "and completed generations. Reports are rebuilt from saved clean PNGs.", ""]
    (published/"README.md").write_text("\n".join(lines),encoding="utf-8")
    print(json.dumps(results,indent=2),flush=True)


if __name__=="__main__":
    main()
