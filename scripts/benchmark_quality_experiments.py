"""Matched 30-step quality experiments, reusing verified showcase prompt caches.

Screen individual changes on a warp and two component examples. Use --confirm
to repeat promising geometric changes on a jigsaw without a broad seed sweep.
"""

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
from visual_anagrams.experiment_options import ExperimentOptions

FEATURES=dict(contrastive=dict(contrastive=True),cross_negative=dict(cross_negative='qwen'),
    schedule=dict(guidance_schedule='middle'),conflict=dict(conflict_projection=True),
    freeze=dict(freeze_below=.6),component=dict(component_guidance=True))
CASE_IDS=['venom-spawn','hulk-loki-jigsaw','vader-castle-hybrid','terminator-rose']
SCREEN=[('venom-spawn',feature) for feature in FEATURES if feature!='component']+[
    ('vader-castle-hybrid','component'),('terminator-rose','component')]


def full_prompts(case):return [f"{case['style']} {p}" for p in case['prompts']]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_negatives(cfg,cases,output,plan_hash):
    import torch
    from huggingface_hub import snapshot_download
    from iris3b.text.qwen3_vl import Qwen3VLTextEncoder
    from visual_anagrams.iris import TEXT_ID,TEXT_REVISION
    from visual_anagrams.negative_planner import plan_negatives
    cache=output/'negative-encodings.pt'
    if cache.exists():
        saved=torch.load(cache,map_location='cpu',weights_only=True)
        assert saved['plan_hash']==plan_hash
        return saved
    folder=snapshot_download(TEXT_ID,revision=TEXT_REVISION,
        allow_patterns=['*.json','*.safetensors','merges.txt','vocab.json'])
    cfg.text_encoder.pretrained=str(folder)
    encoder=Qwen3VLTextEncoder(cfg.text_encoder,device='cpu')
    plans,values,indices={},[],{}
    try:
        encoder.to('cuda')
        with torch.inference_mode():
            for case in cases[:2]:
                print('Planning negatives with Qwen:',case['id'],flush=True)
                plan=plan_negatives(case['prompts'],'qwen',encoder=encoder,style=case['style'])
                plans[case['id']]=plan
                indices[case['id']]=len(values)
                print(json.dumps(plan),flush=True)
                for text in plan['view_negative_prompts']:
                    value=encoder.encode([text])
                    values.append((value.embeddings.cpu(),value.mask.cpu()))
            saved=dict(plan_hash=plan_hash,plans=plans,case_indices=indices,
                embeddings=torch.cat([v[0] for v in values]),mask=torch.cat([v[1] for v in values]))
        torch.save(saved,cache)
        write_json(output/'negative-plans.json',plans)
        return saved
    finally:
        del encoder
        gc.collect();torch.cuda.empty_cache()


def main():
    import torch
    from visual_anagrams.iris import (MODEL_ID,MODEL_REVISION,CODE_REVISION,TEXT_REVISION,
        PromptEncoding,load_model,resolve_model,read_config,sample_iris)
    from visual_anagrams.balancing import AdaptiveBalancer,CleanPreviewScorer
    from visual_anagrams.views import get_views
    from visual_anagrams.utils import save_illusion,save_metadata
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'results/quality_experiments_20261025')
    parser.add_argument('--confirm',nargs='+',choices=[f for f in FEATURES if f!='component'])
    parser.add_argument('--report-only',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(6)
    base=ROOT/'results/showcase_v3_candidates_20261025'
    original=json.loads((base/'plan.json').read_text())
    cases=[dict(next(c for c in original['cases'] if c['id']==name),mode='standard',steps=30,guidance=3) for name in CASE_IDS]
    source_hashes={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in [
        ROOT/'visual_anagrams/iris.py',ROOT/'visual_anagrams/balancing.py',
        ROOT/'visual_anagrams/quality_experiments.py',ROOT/'visual_anagrams/negative_planner.py']}
    plan=dict(cases=cases,features=FEATURES,screen=SCREEN,steps=30,guidance=3,shift=8,solver_order=2,
        resolution=[1024,1024],model=MODEL_ID,model_revision=MODEL_REVISION,code_revision=CODE_REVISION,
        text_revision=TEXT_REVISION,source_hashes=source_hashes,
        baseline_sources={c['id']:f"results/{'showcase_v3_refinements_20261025' if c['id']=='hulk-loki-jigsaw' else 'showcase_v3_candidates_20261025'}/{c['id']}/{c['seeds'][0]:04}" for c in cases},
        criteria='Promote only after visual review and confirmation on a second case, with no material weaker-view regression. Component methods require both perceptual views to remain readable. Scores alone do not establish quality.')
    args.output.mkdir(parents=True,exist_ok=True)
    path=args.output/'plan.json'
    if path.exists() and json.loads(path.read_text())!=json.loads(json.dumps(plan)):
        raise ValueError('Existing benchmark has a different plan or implementation; use another output folder')
    write_json(path,plan)
    plan_hash=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest()
    for case in cases:
        folder=args.output/case['id'];folder.mkdir(exist_ok=True)
        source=ROOT/plan['baseline_sources'][case['id']]
        record=json.loads((source/'run.json').read_text())
        assert (record['steps'],record['guidance'],record['shift'],record['mode'],record['seed'])==(30,3,8,'standard',case['seeds'][0])
        assert sha(source/'sample_1024.png')==record['image_sha256']
        destination=folder/'baseline';destination.mkdir(exist_ok=True)
        for name in ['sample_1024.png','sample_1024.views.png']:
            if not (destination/name).exists():shutil.copy2(source/name,destination/name)
        if not (destination/'run.json').exists():write_json(destination/'run.json',dict(record,
            feature='baseline',experiments=ExperimentOptions().as_dict(),reused_from=plan['baseline_sources'][case['id']]))
        torch.manual_seed(case['view_seed'])
        views=get_views(case['views'],case.get('view_args'))
        save_metadata(views,SimpleNamespace(views=case['views'],prompts=case['labels'],seed=case['view_seed']),folder)
    jobs=[('hulk-loki-jigsaw',feature) for feature in args.confirm] if args.confirm else SCREEN
    pending=[(case_id,feature) for case_id,feature in jobs if not (args.output/case_id/feature/'run.json').exists()]
    if pending and not args.report_only:
        saved=torch.load(base/'prompt-encodings.pt',map_location='cpu',weights_only=True)
        assert saved['plan_hash']==hashlib.sha256(json.dumps(original,sort_keys=True).encode()).hexdigest()
        directory=resolve_model();cfg=read_config(directory)
        negative=PromptEncoding(saved['negative_embeddings'],saved['negative_mask'])
        assert saved['negative_prompt']==''
        planned=prepare_negatives(cfg,cases,args.output,plan_hash)
        print('Loading Standard Iris INT8 once; no Turbo LoRA',flush=True)
        model=load_model(directory,cfg,'cuda',turbo_path=None)
        assert not any(type(m).__name__=='Int8LoRALinear' for m in model.modules())
        scorer=CleanPreviewScorer(full_prompts(cases[0]))
        write_json(args.output/'environment.json',dict(gpu=torch.cuda.get_device_name(0),torch=torch.__version__,
            cuda=torch.version.cuda,model_manifest=json.loads((directory/'manifest.json').read_text())))
        first=cases[0];index=saved['case_indices'][first['id']]
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            warm=model(torch.zeros((1,3,1024,1024),device='cuda'),torch.tensor([500.],device='cuda'),
                saved['embeddings'][index:index+1].cuda(),y_mask=saved['mask'][index:index+1].cuda()).x
            del warm
        torch.cuda.synchronize()
        for case_id,feature in pending:
            case=next(c for c in cases if c['id']==case_id)
            index=saved['case_indices'][case_id]
            positive=PromptEncoding(saved['embeddings'][index:index+2],saved['mask'][index:index+2])
            torch.manual_seed(case['view_seed']);views=get_views(case['views'],case.get('view_args'))
            scorer.set_prompts(full_prompts(case))
            options=ExperimentOptions(**FEATURES[feature])
            options.validate(views,reduction=case['reduction'],quality_method=case['quality_method'])
            mixer=(AdaptiveBalancer(views,scorer,contrastive=options.contrastive,freeze_below=options.freeze_below)
                if case['quality_method']=='adaptive' else None)
            other=None
            if options.cross_negative!='off':
                index=planned['case_indices'][case_id]
                other=PromptEncoding(planned['embeddings'][index:index+2],planned['mask'][index:index+2])
            details={}
            print('RUN',case_id,feature,'30 steps / seed',case['seeds'][0],flush=True)
            torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
            image=sample_iris(
                model,positive,negative,views,num_inference_steps=30,guidance_scale=3,shift=8,order=2,
                reduction=case['reduction'],generator=torch.Generator(device='cuda').manual_seed(case['seeds'][0]),
                prediction_mixer=mixer,experiments=options,view_negative=other,
                component_scorer=scorer if options.component_guidance else None,experiment_diagnostics=details)
            torch.cuda.synchronize();seconds=time.perf_counter()-start
            destination=args.output/case_id/feature;destination.mkdir(exist_ok=True)
            save_illusion(image.cpu(),views,destination)
            write_json(destination/'run.json',dict(case_id=case_id,feature=feature,steps=30,guidance=3,shift=8,
                solver_order=2,mode='standard',quality_method=case['quality_method'],reduction=case['reduction'],
                seed=case['seeds'][0],view_seed=case['view_seed'],seconds=seconds,
                peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,experiments=options.as_dict(),
                image_sha256=sha(destination/'sample_1024.png'),diagnostics=details,
                balancing=mixer.diagnostics() if mixer else None,
                negative_plan=planned['plans'][case_id] if other else None))
            print('DONE',case_id,feature,seconds,flush=True)
            del image,mixer,other
        del model,scorer
        gc.collect();torch.cuda.empty_cache()
    print('Selected benchmark runs complete',flush=True)


if __name__=='__main__':main()
