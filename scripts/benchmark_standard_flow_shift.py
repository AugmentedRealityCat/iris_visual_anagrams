"""Two Standard Iris 30-step Joker/Batman anagrams: flow shifts 4 versus 8."""

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.benchmark_flow_shift import write_json


def main():
    import torch
    from iris3b.flow.solver import FlowDPMSolver
    from visual_anagrams.iris import PromptEncoding,load_model,read_config,resolve_model
    from visual_anagrams.utils import save_illusion
    from visual_anagrams.views import get_views

    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"results/standard_flow_shift_joker_batman_20261025")
    parser.add_argument("--publish",type=Path,default=ROOT/"examples/standard-flow-shift")
    args=parser.parse_args()
    torch.set_num_threads(6)
    positive_source=ROOT/"results/flow_shift_joker_batman_20261025"
    negative_source=ROOT/"results/quality_pilot_20261024"
    original=json.loads((positive_source/"plan.json").read_text(encoding="utf-8"))
    negative_plan=json.loads((negative_source/"plan.json").read_text(encoding="utf-8"))
    for key in ["model_revision","code_revision","text_revision"]:
        assert original[key]==negative_plan[key]
    plan=dict(original,shifts=[4.,8.],steps=30,guidance=3.,method="classic",turbo=False,
              adapter=None,turbo_revision=None,solver_order=2,negative_prompt="")
    args.output.mkdir(parents=True,exist_ok=True)
    path=args.output/"plan.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8"))!=plan:
        raise ValueError("Existing directory has different test settings")
    write_json(path,plan)
    pending=[s for s in plan["shifts"] if not (args.output/f"shift-{s:g}"/"run.json").exists()]
    if pending:
        saved=torch.load(positive_source/"prompt-encodings.pt",map_location="cpu",weights_only=True)
        blank=torch.load(negative_source/"prompt-encodings.pt",map_location="cpu",weights_only=True)
        assert saved["plan_hash"]==hashlib.sha256(json.dumps(original,sort_keys=True).encode()).hexdigest()
        assert blank["plan_hash"]==hashlib.sha256(json.dumps(negative_plan,sort_keys=True).encode()).hexdigest()
        positive=PromptEncoding(saved["embeddings"],saved["mask"])
        negative=PromptEncoding(blank["negative_embeddings"],blank["negative_mask"])
        directory=resolve_model()
        cfg=read_config(directory)
        print("Loading Standard Iris INT8 once; no LoRA and no adaptive balancing",flush=True)
        model=load_model(directory,cfg,"cuda",turbo_path=None)
        lora_layers=sum(type(layer).__name__=="Int8LoRALinear" for layer in model.modules())
        assert lora_layers==0
        with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
            warm=model(torch.zeros((1,3,1024,1024),device="cuda"),torch.tensor([500.],device="cuda"),
                       positive.embeddings[:1].cuda(),y_mask=positive.mask[:1].cuda()).x
            del warm
        torch.cuda.synchronize()
        write_json(args.output/"environment.json",dict(torch=torch.__version__,cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(0),lora_layers=lora_layers,
            model_manifest=json.loads((directory/"manifest.json").read_text(encoding="utf-8")),
            positive_encoding_from=f"results/{positive_source.name}/prompt-encodings.pt",
            empty_negative_encoding_from=f"results/{negative_source.name}/prompt-encodings.pt"))
        from visual_anagrams.iris import sample_iris
        for shift in pending:
            print("RUN Standard 30 steps / shift",shift,flush=True)
            torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
            start=time.perf_counter()
            pixels=sample_iris(model,positive,negative,get_views(plan["views"]),num_inference_steps=30,
                guidance_scale=3.,shift=shift,order=2,reduction="mean",
                generator=torch.Generator(device="cuda").manual_seed(0),prediction_mixer=None)
            torch.cuda.synchronize()
            seconds=time.perf_counter()-start
            folder=args.output/f"shift-{shift:g}"
            folder.mkdir(exist_ok=True)
            save_illusion(pixels.cpu(),get_views(plan["views"]),folder)
            write_json(folder/"run.json",dict(shift=shift,steps=30,guidance=3.,method="classic",turbo=False,seed=0,
                solver_order=2,seconds=seconds,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                image_sha256=hashlib.sha256((folder/"sample_1024.png").read_bytes()).hexdigest(),
                noise_levels=FlowDPMSolver.time_grid(30,shift),estimated_network_evaluations=120))
            print("DONE Standard shift",shift,seconds,flush=True)
            del pixels
        del model,positive,negative
        gc.collect();torch.cuda.empty_cache()
    report(args.output,args.publish,plan)


def report(output,published,plan):
    import torch
    import torch.nn.functional as F
    from PIL import Image,ImageDraw,ImageFont
    from transformers import CLIPModel,CLIPProcessor

    published.mkdir(parents=True,exist_ok=True)
    model_id="openai/clip-vit-large-patch14";revision="32bd64288804d66eefd0ccbe215aa642df71cc41"
    model=CLIPModel.from_pretrained(model_id,revision=revision).float().eval()
    processor=CLIPProcessor.from_pretrained(model_id,revision=revision,use_fast=False)
    sheet=Image.new("RGB",(1536,950),"#eef0f4");draw=ImageDraw.Draw(sheet)
    font_path=str(ROOT/"visual_anagrams/assets/CourierPrime-Regular.ttf")
    font,small=ImageFont.truetype(font_path,23),ImageFont.truetype(font_path,20)
    draw.text((14,10),"Standard Iris / 30 steps / CFG 3 / seed 0 / Classic / no LoRA",font=font,fill="#182337")
    draw.text((14,45),"Each pair: Joker upright | Batman rotated 180 degrees",font=small,fill="#182337")
    results=[]
    with torch.inference_mode():
        for row,shift in enumerate(plan["shifts"]):
            folder=output/f"shift-{shift:g}"
            record=json.loads((folder/"run.json").read_text(encoding="utf-8"))
            image=Image.open(folder/"sample_1024.png").convert("RGB")
            views=[image,image.transpose(Image.Transpose.ROTATE_180)]
            values=model(**processor(text=plan["prompts"],images=views,padding=True,return_tensors="pt"))
            matrix=F.normalize(values.image_embeds,dim=-1)@F.normalize(values.text_embeds,dim=-1).T
            scores=matrix.diag().tolist()
            record.update(scores=scores,weaker_view=min(scores),similarity_matrix=matrix.tolist(),
                minimum_prompt_margin=min(float(matrix[i,i]-matrix[i,1-i]) for i in range(2)),
                local_image=f"results/{output.name}/{folder.name}/sample_1024.png")
            results.append(record)
            x,y=0,90+row*430
            draw.text((14,y),f"Flow shift {shift:g}",font=font,fill="#182337")
            for column,view in enumerate(views):sheet.paste(view.resize((384,384),Image.Resampling.LANCZOS),(column*384,y+36))
            draw.text((800,y+64),f"Joker CLIP: {scores[0]:.4f}",font=font,fill="#182337")
            draw.text((800,y+108),f"Batman CLIP: {scores[1]:.4f}",font=font,fill="#182337")
            draw.text((800,y+152),f"Weaker view: {min(scores):.4f}",font=font,fill="#182337")
            draw.text((800,y+196),f"Minimum margin: {record['minimum_prompt_margin']:+.4f}",font=small,fill="#182337")
            draw.text((800,y+240),f"{record['seconds']:.1f}s / {record['peak_allocated_gib']:.2f} GiB",font=small,fill="#182337")
            image.save(published/f"shift-{shift:g}.jpg",quality=94)
    sheet.save(published/"comparison.jpg",quality=94)
    write_json(published/"results.json",dict(plan=plan,evaluator=dict(model=model_id,revision=revision),results=results,
        note="Two new Standard images, same seed/pair. Only flow shift changes. No LoRA/adaptive/CLIP guidance. CLIP scores are alignment measures, not human illusion ratings."))
    lines=["# Standard Iris: 30 steps, flow shift 4 versus 8", "", "Two new images with Standard Iris INT8, "
        "30 DPM steps, solver order 2, CFG 3, Classic/Mean combining, empty negative prompt, seed 0 and "
        "1024x1024 output. No Turbo LoRA, adaptive balancing, MTL or SyncVC. Only flow shift changes. "
        "Joker is upright; Batman is the 180-degree view. The same initial noise and prompt embeddings are reused.", "",
        "| Shift | Joker CLIP | Batman CLIP | Weaker view | Minimum prompt margin | Seconds | Peak GiB |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in results:
        lines.append(f"| {r['shift']:g} | {r['scores'][0]:.4f} | {r['scores'][1]:.4f} | {r['weaker_view']:.4f} | {r['minimum_prompt_margin']:+.4f} | {r['seconds']:.1f} | {r['peak_allocated_gib']:.2f} |")
    lines += ["", "![Both shifts and orientations](comparison.jpg)", "", "## Findings", "",
        "Shift 8 improved the weaker-view score from 0.2388 to 0.2552, mostly by improving Batman "
        "alignment; Joker changed from 0.2908 to 0.2859. The minimum prompt margin moved closer to "
        "zero (-0.0264 to -0.0047), but stayed negative. Both results have sharper comic-book detail "
        "than the earlier Turbo examples while still placing the two faces in separate areas. "
        "This is a small alignment gain for this pair, not a convincing transformation or evidence "
        "for changing the general default.", "", "## Conditions and limitations", "",
        *[f"- `{prompt}`" for prompt in plan["prompts"]], "",
        "CLIP ViT-L/14 evaluates saved clean pixels and is not used during generation. Higher weaker-view "
        "scores indicate stronger alignment of both orientations, not necessarily a convincing illusion. "
        "A negative minimum prompt margin means an orientation prefers the other prompt. One seed and "
        "pair cannot establish a generally optimal shift. Defaults remain unchanged.", "",
        "Prompt embeddings reuse the verified Joker/Batman Qwen cache; the empty negative encoding reuses "
        "the earlier quality pilot's cache from the same model/code/text revisions. Cache plan hashes are "
        "checked before use. The image model runs all 30 steps for both conditions.", "",
        "[Exact settings, pinned revisions, noise schedules, image hashes and scores](results.json). "
        "RTX 3060 12 GB. Processing times exclude downloads, loading, prompt encoding, warm-up, saving and "
        "evaluation. VRAM is PyTorch peak allocated memory. Full-resolution PNGs are stored locally under "
        "`results/standard_flow_shift_joker_batman_20261025/`.", "",
        "Reproduce with `python scripts/benchmark_standard_flow_shift.py`; completed images are reused. "
        "The earlier Joker/Batman and quality-pilot prompt caches are required.", ""]
    (published/"README.md").write_text("\n".join(lines),encoding="utf-8")
    print(json.dumps(results,indent=2),flush=True)


if __name__=="__main__":main()
