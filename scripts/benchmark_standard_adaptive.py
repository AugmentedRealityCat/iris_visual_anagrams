"""Compare 30-step Standard adaptive balancing with cached Classic shifts 4 and 8."""

import argparse
import gc
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.benchmark_flow_shift import write_json


def main():
    import torch
    from visual_anagrams.balancing import AdaptiveBalancer,CleanPreviewScorer
    from visual_anagrams.iris import PromptEncoding,load_model,read_config,resolve_model,sample_iris
    from visual_anagrams.views import get_views
    from visual_anagrams.utils import save_illusion

    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"results/standard_adaptive_joker_batman_20261025")
    parser.add_argument("--publish",type=Path,default=ROOT/"examples/standard-adaptive")
    args=parser.parse_args()
    torch.set_num_threads(6)
    baseline=ROOT/"results/standard_flow_shift_joker_batman_20261025"
    positive_source=ROOT/"results/flow_shift_joker_batman_20261025"
    negative_source=ROOT/"results/quality_pilot_20261024"
    base_plan=json.loads((baseline/"plan.json").read_text(encoding="utf-8"))
    plan=dict(base_plan,method="adaptive",compared_methods=["classic","adaptive"],new_images=2,reused_baselines=2)
    args.output.mkdir(parents=True,exist_ok=True)
    path=args.output/"plan.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8"))!=plan:
        raise ValueError("Existing output has different settings")
    write_json(path,plan)
    for shift in plan["shifts"]:
        old=baseline/f"shift-{shift:g}"
        destination=args.output/"classic"/old.name
        destination.mkdir(parents=True,exist_ok=True)
        record=json.loads((old/"run.json").read_text(encoding="utf-8"))
        assert hashlib.sha256((old/"sample_1024.png").read_bytes()).hexdigest()==record["image_sha256"]
        for name in ["sample_1024.png","sample_1024.views.png"]:
            if not (destination/name).exists():shutil.copy2(old/name,destination/name)
        if not (destination/"run.json").exists():
            write_json(destination/"run.json",dict(record,reused_from=f"results/{baseline.name}/{old.name}"))
    pending=[s for s in plan["shifts"] if not (args.output/"adaptive"/f"shift-{s:g}"/"run.json").exists()]
    if pending:
        positive_plan=json.loads((positive_source/"plan.json").read_text(encoding="utf-8"))
        negative_plan=json.loads((negative_source/"plan.json").read_text(encoding="utf-8"))
        for key in ["model_revision","code_revision","text_revision"]:
            assert plan[key]==positive_plan[key]==negative_plan[key]
        assert plan["prompts"]==positive_plan["prompts"]
        saved=torch.load(positive_source/"prompt-encodings.pt",map_location="cpu",weights_only=True)
        blank=torch.load(negative_source/"prompt-encodings.pt",map_location="cpu",weights_only=True)
        assert saved["plan_hash"]==hashlib.sha256(json.dumps(positive_plan,sort_keys=True).encode()).hexdigest()
        assert blank["plan_hash"]==hashlib.sha256(json.dumps(negative_plan,sort_keys=True).encode()).hexdigest()
        positive=PromptEncoding(saved["embeddings"],saved["mask"])
        negative=PromptEncoding(blank["negative_embeddings"],blank["negative_mask"])
        directory=resolve_model()
        print("Loading Standard Iris INT8 and CPU adaptive scorer; no LoRA",flush=True)
        model=load_model(directory,read_config(directory),"cuda",turbo_path=None)
        lora_layers=sum(type(layer).__name__=="Int8LoRALinear" for layer in model.modules())
        assert lora_layers==0
        scorer=CleanPreviewScorer(plan["prompts"])
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
        for shift in pending:
            print("RUN Standard adaptive / 30 steps / shift",shift,flush=True)
            mixer=AdaptiveBalancer(get_views(plan["views"]),scorer)
            torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
            start=time.perf_counter()
            pixels=sample_iris(model,positive,negative,mixer.views,num_inference_steps=30,guidance_scale=3.,
                order=2,shift=shift,reduction="mean",generator=torch.Generator(device="cuda").manual_seed(0),
                prediction_mixer=mixer)
            torch.cuda.synchronize()
            seconds=time.perf_counter()-start
            destination=args.output/"adaptive"/f"shift-{shift:g}"
            destination.mkdir(parents=True,exist_ok=True)
            save_illusion(pixels.cpu(),mixer.views,destination)
            write_json(destination/"run.json",dict(shift=shift,steps=30,guidance=3.,solver_order=2,method="adaptive",
                turbo=False,seed=0,seconds=seconds,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                image_sha256=hashlib.sha256((destination/"sample_1024.png").read_bytes()).hexdigest(),
                estimated_network_evaluations=120,diagnostics=mixer.diagnostics()))
            print("DONE Standard adaptive shift",shift,seconds,flush=True)
            del pixels,mixer
        del model,positive,negative,scorer
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
    font,small=ImageFont.truetype(font_path,23),ImageFont.truetype(font_path,19)
    draw.text((14,10),"Standard / 30 steps / CFG 3 / seed 0 / Classic (top) vs Adaptive (bottom)",font=font,fill="#182337")
    draw.text((14,45),"Each pair: Joker upright | Batman rotated 180 degrees",font=small,fill="#182337")
    results=[]
    with torch.inference_mode():
        for row,method in enumerate(["classic","adaptive"]):
            for column,shift in enumerate(plan["shifts"]):
                folder=output/method/f"shift-{shift:g}"
                record=json.loads((folder/"run.json").read_text(encoding="utf-8"))
                image=Image.open(folder/"sample_1024.png").convert("RGB")
                views=[image,image.transpose(Image.Transpose.ROTATE_180)]
                values=model(**processor(text=plan["prompts"],images=views,padding=True,return_tensors="pt"))
                matrix=F.normalize(values.image_embeds,dim=-1)@F.normalize(values.text_embeds,dim=-1).T
                scores=matrix.diag().tolist()
                record.update(scores=scores,weaker_view=min(scores),similarity_matrix=matrix.tolist(),
                    minimum_prompt_margin=min(float(matrix[i,i]-matrix[i,1-i]) for i in range(2)),
                    local_image=f"results/{output.name}/{method}/{folder.name}/sample_1024.png")
                if method=="adaptive":
                    record["adaptive_updates"]=sum(step["scores"] is not None for step in record["diagnostics"]["steps"])
                results.append(record)
                x,y=column*768,90+row*430
                draw.text((x+14,y),f"{method} / shift {shift:g}",font=font,fill="#182337")
                for side,view in enumerate(views):sheet.paste(view.resize((384,384),Image.Resampling.LANCZOS),(x+side*384,y+36))
                image.save(published/f"{method}-shift-{shift:g}.jpg",quality=94)
    sheet.save(published/"comparison.jpg",quality=94)
    write_json(published/"results.json",dict(plan=plan,evaluator=dict(model=model_id,revision=revision),results=results,
        note="Two new Standard adaptive images; two cached Classic controls. One pair/seed. B/32 guides balancing, held-out L/14 evaluates saved pixels."))
    lines=["# Standard Iris at 30 steps: adaptive balancing", "", "Two new adaptive images compared with "
        "the two cached Classic images at shifts 4 and 8. Standard Iris INT8, 30 DPM steps, order 2, CFG 3, "
        "1024x1024, seed 0 and the same empty negative prompt. No LoRA, MTL or SyncVC. Within each shift, "
        "only adaptive balancing is changed. Prompt encodings and initial noise are reused.", "",
        "| Method | Shift | Joker CLIP | Batman CLIP | Weaker view | Minimum margin | Seconds | Peak GiB |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in results:
        lines.append(f"| {r['method']} | {r['shift']:g} | {r['scores'][0]:.4f} | {r['scores'][1]:.4f} | {r['weaker_view']:.4f} | {r['minimum_prompt_margin']:+.4f} | {r['seconds']:.1f} | {r['peak_allocated_gib']:.2f} |")
    lines += ["", "![Both shifts and both methods](comparison.jpg)", "", "## Findings", "",
        "Adaptive balancing improved the weaker-view score at both shifts: 0.2388 to 0.2496 at shift 4 "
        "and 0.2552 to 0.2631 at shift 8, approximately 4.5% and 3.1%. Joker alignment decreased slightly "
        "in both cases. The minimum prompt margin worsened at shift 4 and improved at shift 8; all four "
        "margins remained negative. Shift 8 with adaptive had the strongest weaker-view alignment in this "
        "Standard comparison, but the images still place the two faces in separate areas and retain "
        "Joker dominance. Adaptive also introduces streak-like details around the mouth/neck region.", "",
        "The adaptive runs took about 18 seconds longer (roughly 4.5-4.9%); VRAM stayed near 6.92 GiB. "
        "There were 12 balancing updates at shift 4 and 10 at shift 8. The improvement is modest and "
        "comes from one pair and seed. The selected anagram default is now Standard, 30 steps, "
        "guidance 3, shift 8 and adaptive balancing for two geometric views. Classic remains "
        "selectable. This profile is the strongest Standard alignment setting in this check, "
        "not a universally optimal configuration.", "", "## Conditions and limitations", "",
        *[f"- `{p}`" for p in plan["prompts"]], "",
        "CPU CLIP ViT-B/32 scores clean previews every two steps below noise level 0.95. Desired weights "
        "use softmax(-10*score), bounds [0.25,0.75] and 50% smoothing. Each image starts at equal weights. "
        "The ordinary Standard DPM solver is preserved; no attention gradients or variance correction are used.", "",
        "Held-out CLIP ViT-L/14 evaluates saved clean pixels and is not used to guide generation. "
        "Scores measure prompt alignment, not human illusion quality. Negative minimum prompt margin means "
        "an orientation prefers the other prompt. One seed and prompt pair do not establish a universal default.", "",
        "[Scores, schedules, per-step weights, hashes and pinned revisions](results.json). RTX 3060 12 GB. "
        "Times include preview scoring and exclude downloads, loading, prompt encoding, warm-up, saving and "
        "evaluation. [Original Standard shift comparison](../standard-flow-shift/README.md). Full PNGs are "
        "under `results/standard_adaptive_joker_batman_20261025/`.", "",
        "Reproduce with `python scripts/benchmark_standard_adaptive.py`; the earlier Standard baseline "
        "and verified prompt caches are required, and completed runs are reused.", ""]
    (published/"README.md").write_text("\n".join(lines),encoding="utf-8")
    print(json.dumps(results,indent=2),flush=True)


if __name__=="__main__":main()
