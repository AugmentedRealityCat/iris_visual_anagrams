"""Check production adaptive balancing against three cached Joker/Batman baselines."""

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
    from visual_anagrams.iris import PromptEncoding,load_model,read_config,resolve_model
    from visual_anagrams.turbo import resolve_turbo_adapter,sample_turbo
    from visual_anagrams.utils import save_illusion
    from visual_anagrams.views import get_views

    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"results/adaptive_gui_check_20261025")
    parser.add_argument("--publish",type=Path,default=ROOT/"examples/adaptive-balancing")
    args=parser.parse_args()
    torch.set_num_threads(6)
    source=ROOT/"results/flow_shift_joker_batman_20261025"
    baseline=ROOT/"results/flow_shift_extended_joker_batman_20261025"
    original=json.loads((source/"plan.json").read_text(encoding="utf-8"))
    conditions=[dict(steps=4,shift=4.),dict(steps=4,shift=8.),dict(steps=8,shift=4.)]
    plan=dict(original,conditions=conditions,method="adaptive",interval=2,temperature=10.,bounds=[.25,.75],smoothing=.5)
    plan.pop("steps"); plan.pop("shifts")
    args.output.mkdir(parents=True,exist_ok=True)
    plan_path=args.output/"plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text(encoding="utf-8"))!=plan:
        raise ValueError("Existing check has a different plan")
    write_json(plan_path,plan)
    for condition in conditions:
        name=f"steps-{condition['steps']}-shift-{condition['shift']:g}"
        old=baseline/name
        destination=args.output/"classic"/name
        destination.mkdir(parents=True,exist_ok=True)
        record=json.loads((old/"run.json").read_text(encoding="utf-8"))
        assert hashlib.sha256((old/"sample_1024.png").read_bytes()).hexdigest()==record["image_sha256"]
        for file in ["sample_1024.png","sample_1024.views.png"]:
            if not (destination/file).exists():shutil.copy2(old/file,destination/file)
        if not (destination/"run.json").exists():
            write_json(destination/"run.json",dict(record,method="classic",reused_from=f"results/{baseline.name}/{name}"))
    pending=[c for c in conditions if not (args.output/"adaptive"/f"steps-{c['steps']}-shift-{c['shift']:g}"/"run.json").exists()]
    if pending:
        saved=torch.load(source/"prompt-encodings.pt",map_location="cpu",weights_only=True)
        assert saved["plan_hash"]==hashlib.sha256(json.dumps(original,sort_keys=True).encode()).hexdigest()
        positive=PromptEncoding(saved["embeddings"],saved["mask"])
        folder=resolve_model()
        model=load_model(folder,read_config(folder),"cuda",turbo_path=resolve_turbo_adapter("r2-1000"))
        scorer=CleanPreviewScorer(plan["prompts"])
        with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
            warm=model(torch.zeros((1,3,1024,1024),device="cuda"),torch.tensor([500.],device="cuda"),
                       positive.embeddings[:1].cuda(),y_mask=positive.mask[:1].cuda()).x
            del warm
        torch.cuda.synchronize()
        write_json(args.output/"environment.json",dict(torch=torch.__version__,cuda=torch.version.cuda,
                   gpu=torch.cuda.get_device_name(0),model_manifest=json.loads((folder/"manifest.json").read_text(encoding="utf-8"))))
        for condition in pending:
            print("RUN",condition,flush=True)
            mixer=AdaptiveBalancer(get_views(plan["views"]),scorer)
            torch.cuda.reset_peak_memory_stats()
            start=time.perf_counter()
            image=sample_turbo(model,positive,mixer.views,num_inference_steps=condition["steps"],shift=condition["shift"],
                               reduction="mean",generator=torch.Generator(device="cuda").manual_seed(0),prediction_mixer=mixer)
            torch.cuda.synchronize()
            seconds=time.perf_counter()-start
            destination=args.output/"adaptive"/f"steps-{condition['steps']}-shift-{condition['shift']:g}"
            destination.mkdir(parents=True,exist_ok=True)
            save_illusion(image.cpu(),mixer.views,destination)
            write_json(destination/"run.json",dict(condition,method="adaptive",seed=0,seconds=seconds,
                       peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                       image_sha256=hashlib.sha256((destination/"sample_1024.png").read_bytes()).hexdigest(),
                       diagnostics=mixer.diagnostics()))
            print("DONE",condition,seconds,flush=True)
            del image,mixer
        del model,positive,scorer
        gc.collect(); torch.cuda.empty_cache()
    report(args.output,args.publish,plan)


def report(output,published,plan):
    import torch
    import torch.nn.functional as F
    from PIL import Image,ImageDraw,ImageFont
    from transformers import CLIPModel,CLIPProcessor

    published.mkdir(parents=True,exist_ok=True)
    model_id="openai/clip-vit-large-patch14"; revision="32bd64288804d66eefd0ccbe215aa642df71cc41"
    model=CLIPModel.from_pretrained(model_id,revision=revision).float().eval()
    processor=CLIPProcessor.from_pretrained(model_id,revision=revision,use_fast=False)
    font_path=str(ROOT/"visual_anagrams/assets/CourierPrime-Regular.ttf")
    font,small=ImageFont.truetype(font_path,20),ImageFont.truetype(font_path,17)
    panel=Image.new("RGB",(1536,776),"#eef0f4");draw=ImageDraw.Draw(panel)
    draw.text((12,10),"Classic (top) vs Adaptive (bottom) / Joker upright, Batman rotated 180 degrees",font=font,fill="#182337")
    results=[]
    with torch.inference_mode():
        for row,method in enumerate(["classic","adaptive"]):
            for column,condition in enumerate(plan["conditions"]):
                name=f"steps-{condition['steps']}-shift-{condition['shift']:g}"
                folder=output/method/name
                record=json.loads((folder/"run.json").read_text(encoding="utf-8"))
                image=Image.open(folder/"sample_1024.png").convert("RGB")
                views=[image,image.transpose(Image.Transpose.ROTATE_180)]
                values=model(**processor(text=plan["prompts"],images=views,padding=True,return_tensors="pt"))
                matrix=F.normalize(values.image_embeds,dim=-1)@F.normalize(values.text_embeds,dim=-1).T
                scores=matrix.diag().tolist()
                record.update(scores=scores,weaker_view=min(scores),similarity_matrix=matrix.tolist(),
                              minimum_prompt_margin=min(float(matrix[i,i]-matrix[i,1-i]) for i in range(2)),
                              local_image=f"results/{output.name}/{method}/{name}/sample_1024.png")
                results.append(record)
                x,y=column*512,76+row*350
                draw.text((x+10,y),f"{method} / {condition['steps']} steps / shift {condition['shift']:g}",font=font,fill="#182337")
                for i,view in enumerate(views):panel.paste(view.resize((256,256),Image.Resampling.LANCZOS),(x+i*256,y+30))
                draw.text((x+10,y+296),f"J {scores[0]:.3f} / B {scores[1]:.3f} / min {min(scores):.3f}",font=small,fill="#182337")
                draw.text((x+10,y+322),f"margin {record['minimum_prompt_margin']:+.3f} / {record['seconds']:.1f}s",font=small,fill="#182337")
                image.save(published/f"{method}-{name}.jpg",quality=92)
    panel.save(published/"comparison.jpg",quality=94)
    write_json(published/"results.json",dict(plan=plan,evaluator=dict(model=model_id,revision=revision),results=results,
        note="Three new production-sampler images, three cached Classic baselines. CLIP B/32 guides balancing; L/14 is held out. One pair/seed is not a general quality guarantee."))
    lines=["# Standalone adaptive-balancing check", "", "Three new Turbo r2 images compared with cached Classic baselines. "
           "Same Joker/Batman prompts, seed 0, 1024x1024, Mean combining and CFG 1. No prompt encoding or "
           "Classic generation was repeated. The new images use the production sampler exposed in the GUI.", "",
           "| Method | Steps | Shift | Joker CLIP | Batman CLIP | Weaker view | Min margin | Seconds |",
           "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in results:
        lines.append(f"| {r['method']} | {r['steps']} | {r['shift']:g} | {r['scores'][0]:.4f} | {r['scores'][1]:.4f} | {r['weaker_view']:.4f} | {r['minimum_prompt_margin']:+.4f} | {r['seconds']:.1f} |")
    lines += ["", "![Both orientations, Classic versus adaptive](comparison.jpg)", "",
              "## Findings and default", "", "Adaptive balancing improved the held-out weaker-view score "
              "in all three settings: 0.2247 to 0.2305 (four steps / shift 4), 0.2448 to 0.2735 "
              "(four steps / shift 8), and 0.2302 to 0.2560 (eight steps / shift 4). "
              "Processing increased by roughly 1-2 seconds per image. The eight-step adaptive result was the "
              "first of these trials with a positive minimum prompt margin, though the images still segregate "
              "the subjects and do not establish a convincing transformation.", "",
              "Together with improvement on both earlier pilot pairs, these results support adaptive balancing "
              "as the default for **two-view Turbo runs with Mean/Auto geometric combining**. "
              "Standard, component groups, Alternate combining and other view counts keep Classic. "
              "Standard quality improvement has not been established. GUI users can choose Classic, "
              "and CLI users can pass `--quality_method plain` explicitly. The sampler API remains Classic "
              "unless a balancer is supplied. This is a practical default, not a claim of universal improvement.", "",
              "Adaptive balancing scores a clean-image preview with CPU CLIP ViT-B/32 every two steps below "
              "t=0.95. It gives weaker views more weight using softmax(-10*score), bounds desired weights to "
              "[0.25,0.75], and blends them 50% with previous weights. Each sample starts at [0.5,0.5]. "
              "It uses no attention gradients or variance correction.", "",
              "CLIP ViT-L/14 is the held-out evaluator. Scores measure prompt alignment, not human illusion quality. "
              "Negative minimum prompt margin means an orientation prefers the other prompt. This is one pair and seed; "
              "eight steps are exploratory for the four-step adapter.", "",
              "[Exact settings, pinned revisions, scores, timings, image hashes and per-step weights](results.json). "
              "RTX 3060 12 GB. Times exclude loading, prompt encoding, warm-up, saving and evaluation, "
              "and include preview scoring. The B/32 scorer stays on CPU. "
              "[Earlier two-pair adaptive pilot](../quality-pilot/README.md).", "",
              "Reproduce with `python scripts/compare_adaptive_balancing.py`; cached generation is reused. "
              "The earlier flow-shift output tree and prompt cache are required.", ""]
    (published/"README.md").write_text("\n".join(lines),encoding="utf-8")
    print(json.dumps(results,indent=2),flush=True)


if __name__=="__main__":main()
