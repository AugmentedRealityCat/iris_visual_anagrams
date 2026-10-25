"""Check higher flow shifts with the current paired-Turbo adaptive default."""

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
    from visual_anagrams.views import get_views
    from visual_anagrams.utils import save_illusion

    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"results/high_flow_shift_joker_batman_20261025")
    parser.add_argument("--publish",type=Path,default=ROOT/"examples/high-flow-shift")
    args=parser.parse_args()
    torch.set_num_threads(6)
    source=ROOT/"results/flow_shift_joker_batman_20261025"
    previous=ROOT/"results/adaptive_gui_check_20261025"
    original=json.loads((source/"plan.json").read_text(encoding="utf-8"))
    plan=dict(original,shifts=[4.,8.,12.,16.,24.],method="adaptive",new_shifts=[12.,16.,24.],reused_shifts=[4.,8.])
    args.output.mkdir(parents=True,exist_ok=True)
    path=args.output/"plan.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8"))!=plan:
        raise ValueError("Existing directory has a different plan")
    write_json(path,plan)
    for shift in plan["reused_shifts"]:
        old=previous/"adaptive"/f"steps-4-shift-{shift:g}"
        new=args.output/f"shift-{shift:g}"
        record=json.loads((old/"run.json").read_text(encoding="utf-8"))
        assert hashlib.sha256((old/"sample_1024.png").read_bytes()).hexdigest()==record["image_sha256"]
        new.mkdir(exist_ok=True)
        for name in ["sample_1024.png","sample_1024.views.png"]:
            if not (new/name).exists():shutil.copy2(old/name,new/name)
        if not (new/"run.json").exists():
            write_json(new/"run.json",dict(record,reused_from=f"results/{previous.name}/adaptive/{old.name}"))
    pending=[shift for shift in plan["new_shifts"] if not (args.output/f"shift-{shift:g}"/"run.json").exists()]
    if pending:
        saved=torch.load(source/"prompt-encodings.pt",map_location="cpu",weights_only=True)
        assert saved["plan_hash"]==hashlib.sha256(json.dumps(original,sort_keys=True).encode()).hexdigest()
        positive=PromptEncoding(saved["embeddings"],saved["mask"])
        directory=resolve_model()
        model=load_model(directory,read_config(directory),"cuda",turbo_path=resolve_turbo_adapter("r2-1000"))
        scorer=CleanPreviewScorer(plan["prompts"])
        with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
            warm=model(torch.zeros((1,3,1024,1024),device="cuda"),torch.tensor([500.],device="cuda"),
                       positive.embeddings[:1].cuda(),y_mask=positive.mask[:1].cuda()).x
            del warm
        torch.cuda.synchronize()
        write_json(args.output/"environment.json",dict(torch=torch.__version__,cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(0),model_manifest=json.loads((directory/"manifest.json").read_text(encoding="utf-8"))))
        for shift in pending:
            print("RUN shift",shift,flush=True)
            mixer=AdaptiveBalancer(get_views(plan["views"]),scorer)
            torch.cuda.reset_peak_memory_stats()
            start=time.perf_counter()
            pixels=sample_turbo(model,positive,mixer.views,shift=shift,num_inference_steps=4,
                reduction="mean",generator=torch.Generator(device="cuda").manual_seed(0),prediction_mixer=mixer)
            torch.cuda.synchronize()
            seconds=time.perf_counter()-start
            folder=args.output/f"shift-{shift:g}"
            folder.mkdir(exist_ok=True)
            save_illusion(pixels.cpu(),mixer.views,folder)
            write_json(folder/"run.json",dict(shift=shift,steps=4,method="adaptive",seed=0,seconds=seconds,
                peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                image_sha256=hashlib.sha256((folder/"sample_1024.png").read_bytes()).hexdigest(),diagnostics=mixer.diagnostics()))
            print("DONE",shift,seconds,flush=True)
            del pixels,mixer
        del model,positive,scorer
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
    sheet=Image.new("RGB",(1536,776),"#eef0f4");draw=ImageDraw.Draw(sheet)
    font_path=str(ROOT/"visual_anagrams/assets/CourierPrime-Regular.ttf")
    font,small=ImageFont.truetype(font_path,20),ImageFont.truetype(font_path,17)
    draw.text((12,10),"Higher flow shifts / Turbo r2 / four steps / adaptive enabled / seed 0",font=font,fill="#182337")
    draw.text((12,40),"Each pair: Joker upright | Batman rotated 180 degrees",font=small,fill="#182337")
    results=[]
    with torch.inference_mode():
        for index,shift in enumerate(plan["shifts"]):
            folder=output/f"shift-{shift:g}"
            record=json.loads((folder/"run.json").read_text(encoding="utf-8"))
            image=Image.open(folder/"sample_1024.png").convert("RGB")
            views=[image,image.transpose(Image.Transpose.ROTATE_180)]
            values=model(**processor(text=plan["prompts"],images=views,padding=True,return_tensors="pt"))
            matrix=F.normalize(values.image_embeds,dim=-1)@F.normalize(values.text_embeds,dim=-1).T
            scores=matrix.diag().tolist()
            history=record["diagnostics"]["steps"]
            record.update(scores=scores,weaker_view=min(scores),similarity_matrix=matrix.tolist(),
                minimum_prompt_margin=min(float(matrix[i,i]-matrix[i,1-i]) for i in range(2)),
                adaptive_updates=sum(step["scores"] is not None for step in history),
                noise_levels=[step["time"] for step in history],
                local_image=f"results/{output.name}/{folder.name}/sample_1024.png")
            results.append(record)
            x,y=(index%3)*512,76+(index//3)*350
            draw.text((x+10,y),f"Shift {shift:g} / {record['adaptive_updates']} balancing updates",font=font,fill="#182337")
            for side,view in enumerate(views):sheet.paste(view.resize((256,256),Image.Resampling.LANCZOS),(x+side*256,y+30))
            draw.text((x+10,y+296),f"J {scores[0]:.3f} / B {scores[1]:.3f} / min {min(scores):.3f}",font=small,fill="#182337")
            draw.text((x+10,y+322),f"margin {record['minimum_prompt_margin']:+.3f} / {record['seconds']:.1f}s",font=small,fill="#182337")
            image.save(published/f"shift-{shift:g}.jpg",quality=92)
    sheet.save(published/"comparison.jpg",quality=94)
    write_json(published/"results.json",dict(plan=plan,evaluator=dict(model=model_id,revision=revision),results=results,
        note="Three new images and two reused adaptive baselines. One pair/seed. Adaptive is enabled but can make zero updates when its t<.95/every-two-steps gate never opens."))
    lines=["# Higher flow-shift check", "", "Turbo r2-1000, four steps, seed 0, 1024x1024, "
           "CFG 1, Mean combining and adaptive balancing enabled. Joker upright; Batman after a 180-degree "
           "rotation. Shifts 4 and 8 reuse the earlier adaptive results; 12, 16 and 24 are new. Prompts and "
           "random draws are unchanged. No generation defaults are changed by this check.", "",
           "| Shift | Joker CLIP | Batman CLIP | Weaker view | Minimum prompt margin | Balancing updates | Seconds |",
           "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in results:
        lines.append(f"| {r['shift']:g} | {r['scores'][0]:.4f} | {r['scores'][1]:.4f} | {r['weaker_view']:.4f} | {r['minimum_prompt_margin']:+.4f} | {r['adaptive_updates']} | {r['seconds']:.1f} |")
    lines += ["", "![Both orientations at all five shifts](comparison.jpg)", "", "## Findings", "",
        "None of the three higher values beat shift 8 on the weaker-view alignment score. "
        "Shift 8 scored 0.2735; shift 24 was next at 0.2642, followed by 16 at 0.2501 and 12 at 0.2338. "
        "The relationship is not monotonic. Some high shifts beat the default shift 4 on this metric, "
        "but all five rotated views still preferred Joker over Batman. The higher-shift images show "
        "softer, washed-out detail and overlapping facial features; none solves the transformation. "
        "One pair and seed do not justify a new general default.", "", "## Schedule interaction", "",
        "At four steps and shift 24, the noise levels are approximately 1.000, 0.986, 0.960 and 0.889. "
        "Adaptive balancing scores only every second step below t=0.95: no step satisfies both conditions "
        "here, so the weights stay equal. Shifts 4, 8, 12 and 16 each get one update. "
        "This tests the GUI's actual default behavior; it does not force balancing at otherwise-skipped times. "
        "The shift-24 result therefore also reflects the absence of a balancing update.", "", "## Conditions", "",
        *[f"- `{p}`" for p in plan["prompts"]], "",
        "CLIP ViT-B/32 guides adaptive balancing; held-out CLIP ViT-L/14 scores saved pixels. "
        "These are prompt-alignment measures, not human illusion ratings. One pair and seed do not establish "
        "a generally optimal shift. Inspect both orientations for subject separation and visual artifacts.", "",
        "[Scores, exact noise levels, per-step weights and image hashes](results.json). RTX 3060 12 GB. "
        "Times include preview scoring and exclude loading, prompt encoding, warm-up, saving and evaluation. "
        "[Earlier adaptive comparison](../adaptive-balancing/README.md). Full-resolution PNGs are stored "
        "under `results/high_flow_shift_joker_batman_20261025/`.", "",
        "Reproduce with `python scripts/benchmark_high_flow_shift.py`; completed images and the earlier "
        "prompt cache/baselines are reused.", ""]
    (published/"README.md").write_text("\n".join(lines),encoding="utf-8")
    print(json.dumps(results,indent=2),flush=True)


if __name__=="__main__":main()
