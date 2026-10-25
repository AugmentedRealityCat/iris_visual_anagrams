"""Generate the mixed Standard/Turbo showcase efficiently with one text-encoder load."""

import argparse
import gc
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.benchmark_flow_shift import write_json


def full_prompts(case):
    return [f"{case['style']} {p}".strip() for p in case["prompts"]]


def main():
    import torch
    from visual_anagrams.balancing import AdaptiveBalancer,CleanPreviewScorer
    from visual_anagrams.iris import CODE_REVISION,MODEL_ID,MODEL_REVISION,TEXT_REVISION,PromptEncoding,encode_prompts,load_model,read_config,resolve_model,sample_iris
    from visual_anagrams.turbo import TURBO_ID,TURBO_REVISION,apply_turbo_lora,resolve_turbo_adapter,sample_turbo
    from visual_anagrams.utils import save_illusion,save_metadata
    from visual_anagrams.views import get_views

    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan",type=Path,default=ROOT/"examples/showcase-v2/plan.json")
    parser.add_argument("--output",type=Path,default=ROOT/"results/showcase_v2_candidates_20261025")
    args=parser.parse_args()
    plan=json.loads(args.plan.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True,exist_ok=True)
    stored=args.output/"plan.json"
    if stored.exists() and json.loads(stored.read_text(encoding="utf-8"))!=plan:
        raise ValueError("Existing showcase uses a different plan; select a new output")
    write_json(stored,plan)
    torch.set_num_threads(6)
    for case in plan["cases"]:
        rotation=case["views"][1] in ("rotate_cw","rotate_ccw","rotate_180","inner_circle")
        assert case["steps"] == (30 if case["mode"] == "standard" else 4)
        assert not rotation or case["mode"] == "standard"
        directory=args.output/case["id"]
        directory.mkdir(exist_ok=True)
        torch.manual_seed(case["view_seed"])
        views=get_views(case["views"],case.get("view_args"))
        save_metadata(views,SimpleNamespace(views=case["views"],prompts=case["labels"],full_prompts=full_prompts(case),
            seed=case["view_seed"],style=case["style"]),directory)
        write_json(directory/"metadata.json",case)
        if case.get("reuse"):
            source=ROOT/case["reuse"]
            old=json.loads((source/"run.json").read_text(encoding="utf-8"))
            assert old["steps"]==case["steps"] and old["shift"]==plan["shift"]
            assert old.get("mode", "turbo" if old.get("turbo") else "standard") == case["mode"]
            assert hashlib.sha256((source/"sample_1024.png").read_bytes()).hexdigest()==old["image_sha256"]
            for seed in case["seeds"]:
                destination=directory/f"{seed:04}"
                destination.mkdir(exist_ok=True)
                for name in ["sample_1024.png","sample_1024.views.png"]:
                    if not (destination/name).exists():shutil.copy2(source/name,destination/name)
                if not (destination/"run.json").exists():
                    write_json(destination/"run.json",dict(old,reused_from=case["reuse"],case_id=case["id"]))
    pending=[case for case in plan["cases"] if any(not (args.output/case["id"]/f"{seed:04}"/"run.json").exists() for seed in case["seeds"])]
    if not pending:
        print("All showcase candidates complete",flush=True);return
    model_path=resolve_model()
    cfg=read_config(model_path)
    fingerprint=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest()
    cache=args.output/"prompt-encodings.pt"
    if cache.exists():
        saved=torch.load(cache,map_location="cpu",weights_only=True)
        assert saved["plan_hash"]==fingerprint
    else:
        cases=[case for case in plan["cases"] if not case.get("reuse")]
        texts=[p for case in cases for p in full_prompts(case)]
        negatives={case["negative_prompt"] for case in cases if case["mode"]=="standard"}
        assert len(negatives)<=1
        negative=next(iter(negatives), "")
        print("Encoding",len(texts),"prompts once and one shared Standard negative prompt",flush=True)
        start=time.perf_counter()
        try:
            positive,negative_encoding=encode_prompts(cfg,texts,negative,device="cuda")
        except torch.OutOfMemoryError:
            gc.collect();torch.cuda.empty_cache()
            positive,negative_encoding=encode_prompts(cfg,texts,negative,device="cpu")
        saved=dict(plan_hash=fingerprint,case_indices={case["id"]:i*2 for i,case in enumerate(cases)},
            embeddings=positive.embeddings,mask=positive.mask,negative_embeddings=negative_encoding.embeddings,
            negative_mask=negative_encoding.mask,negative_prompt=negative,encoding_seconds=time.perf_counter()-start)
        torch.save(saved,cache)
        del positive,negative_encoding
    print("Loading Standard Iris once; Turbo will be applied after all Standard runs",flush=True)
    model=load_model(model_path,cfg,"cuda",turbo_path=None)
    scorer=CleanPreviewScorer(full_prompts(pending[0]))
    negative=PromptEncoding(saved["negative_embeddings"],saved["negative_mask"])
    write_json(args.output/"environment.json",dict(torch=torch.__version__,cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0),model=MODEL_ID,model_revision=MODEL_REVISION,code_revision=CODE_REVISION,
        text_revision=TEXT_REVISION,turbo_repo=TURBO_ID,turbo_revision=TURBO_REVISION,
        model_manifest=json.loads((model_path/"manifest.json").read_text(encoding="utf-8"))))
    ordered=sorted(pending,key=lambda case:case["mode"]=="turbo")
    applied=False;warmed=set()
    for case in ordered:
        if case["mode"]=="turbo" and not applied:
            print("Applying Turbo r2 for four-step non-rotation runs",flush=True)
            apply_turbo_lora(model,resolve_turbo_adapter("r2-1000"));applied=True
        index=saved["case_indices"][case["id"]]
        positive=PromptEncoding(saved["embeddings"][index:index+2],saved["mask"][index:index+2])
        scorer.set_prompts(full_prompts(case))
        torch.manual_seed(case["view_seed"])
        views=get_views(case["views"],case.get("view_args"))
        if case["mode"] not in warmed:
            with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
                warm=model(torch.zeros((1,3,1024,1024),device="cuda"),torch.tensor([500.],device="cuda"),
                    positive.embeddings[:1].cuda(),y_mask=positive.mask[:1].cuda()).x
                del warm
            torch.cuda.synchronize();warmed.add(case["mode"])
        for seed in case["seeds"]:
            destination=args.output/case["id"]/f"{seed:04}"
            if (destination/"run.json").exists():continue
            print("RUN",case["id"],case["mode"],case["steps"],"steps / seed",seed,flush=True)
            method=case.get("quality_method",plan["quality_method"])
            reduction=case.get("reduction","mean")
            mixer=AdaptiveBalancer(views,scorer) if method=="adaptive" else None
            torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
            start=time.perf_counter()
            common=dict(generator=torch.Generator(device="cuda").manual_seed(seed),shift=plan["shift"],
                reduction=reduction,num_inference_steps=case["steps"],prediction_mixer=mixer)
            image=(sample_turbo(model,positive,views,**common) if applied else
                   sample_iris(model,positive,negative,views,guidance_scale=3.,order=2,**common))
            torch.cuda.synchronize();seconds=time.perf_counter()-start
            destination.mkdir(exist_ok=True)
            save_illusion(image.cpu(),views,destination)
            write_json(destination/"run.json",dict(case_id=case["id"],mode=case["mode"],steps=case["steps"],
                guidance=case["guidance"],shift=plan["shift"],quality_method=method,reduction=reduction,
                seed=seed,view_seed=case["view_seed"],negative_prompt=case["negative_prompt"],
                seconds=seconds,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                image_sha256=hashlib.sha256((destination/"sample_1024.png").read_bytes()).hexdigest(),
                diagnostics=mixer.diagnostics() if mixer else None))
            print("DONE",case["id"],seed,seconds,flush=True)
            del image,mixer
    print("All showcase candidates complete",flush=True)


if __name__=="__main__":main()
