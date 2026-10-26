"""Small, reproducible Iris depth pilot. Reuses verified Joker/Batman text encodings."""

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def write(path,value):path.write_text(json.dumps(value,indent=2),encoding='utf-8')


def main():
    import torch
    from PIL import Image
    from iris_tools import build_parser,validate_args
    from depth_lab import run_experiment
    from visual_anagrams.iris import resolve_model,read_config,load_model,PromptEncoding
    from visual_anagrams.turbo import resolve_turbo_adapter,sample_turbo
    from visual_anagrams.views import get_views
    from visual_anagrams.iris_tasks import resolve_task_model,load_task,DOWNSTREAM_REVISION
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'results/depth-lab-20261026')
    parser.add_argument('--iterations',type=int,default=16)
    parser.add_argument('--optimization_side',type=int,default=256)
    parser.add_argument('--budget',type=float,default=.12)
    parser.add_argument('--skip_generation',action='store_true')
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(6)
    cached=ROOT/'results/flow_shift_joker_batman_20261025'
    plan=json.loads((cached/'plan.json').read_text())
    saved=torch.load(cached/'prompt-encodings.pt',map_location='cpu',weights_only=True)
    assert saved['plan_hash']==hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest()
    preparation=dict(prompts=plan['prompts'],seed=42,steps=4,shift=8,adapter='r2-1000',
        prompt_cache=str(cached/'prompt-encodings.pt'),prompt_cache_sha256=hashlib.sha256((cached/'prompt-encodings.pt').read_bytes()).hexdigest())
    if (out/'references.json').exists():assert json.loads((out/'references.json').read_text())==preparation
    else:write(out/'references.json',preparation)
    missing=[i for i,name in enumerate(('joker','batman')) if not (out/f'{name}.png').exists()]
    if missing and not args.skip_generation:
        directory=resolve_model();cfg=read_config(directory)
        print('Loading generator for two ordinary four-step portraits; no rotations',flush=True)
        model=load_model(directory,cfg,'cuda',turbo_path=resolve_turbo_adapter('r2-1000'))
        for index in missing:
            name=('joker','batman')[index]
            print('Generating reference',name,flush=True)
            positive=PromptEncoding(saved['embeddings'][index:index+1],saved['mask'][index:index+1])
            image=sample_turbo(model,positive,get_views(['identity']),num_inference_steps=4,shift=8,
                generator=torch.Generator(device='cuda').manual_seed(42),device='cuda')
            pixels=((image[0].detach().cpu().float().permute(1,2,0)+1)*127.5).round().byte().numpy()
            Image.fromarray(pixels).save(out/f'{name}.png')
            del image
        del model;gc.collect();torch.cuda.empty_cache()
    assert (out/'joker.png').exists() and (out/'batman.png').exists()
    print('Loading depth once for the four experiments',flush=True)
    directory=resolve_task_model('depth');runner=load_task('depth',directory,'cuda')
    write(out/'environment.json',dict(gpu=torch.cuda.get_device_name(0),torch=torch.__version__,
        depth_revision=DOWNSTREAM_REVISION,model_manifest=json.loads((directory/'manifest.json').read_text())))
    cases=[
        ('parallax-batman',['--task','parallax','--input',str(out/'batman.png'),'--motion','18']),
        ('stereogram-batman',['--task','stereogram','--input',str(out/'batman.png'),'--texture',str(out/'joker.png')]),
        ('focus-hybrid',['--task','focus','--input',str(ROOT/'results/showcase_v3_candidates_20261025/vader-castle-hybrid/0131/sample_1024.png'),'--aperture','24']),
        (f'depth-anagram-{args.optimization_side}-{args.budget:g}',[
            '--task','depth_anagram','--input',str(out/'joker.png'),'--target',str(out/'batman.png'),
            '--iterations',str(args.iterations),'--optimization_side',str(args.optimization_side),'--edit_budget',str(args.budget)]),
    ]
    for name,options in cases:
        destination=out/name
        if (destination/'run.json').exists():print('Already complete',name,flush=True);continue
        if destination.exists():raise ValueError(f'Incomplete output exists: {destination}; choose another output folder')
        settings=validate_args(build_parser().parse_args(options+['--render_side','512','--duration','3','--fps','24']))
        print('RUN',name,flush=True);torch.cuda.reset_peak_memory_stats()
        with Image.open(settings.input) as image:source=image.convert('RGB')
        details=run_experiment(runner,settings,source,destination)
        torch.cuda.synchronize()
        details['peak_allocated_gib']=torch.cuda.max_memory_allocated()/2**30
        details['settings']={key:str(value) if isinstance(value,Path) else value for key,value in vars(settings).items()}
        write(destination/'run.json',details)
        print('DONE',name,details['seconds'],'seconds',details['peak_allocated_gib'],'GiB',flush=True)
        gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':main()
