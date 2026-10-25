"""Evaluate saved quality experiments on CPU and publish all completed trials."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from scripts.benchmark_flow_shift import write_json
from scripts.build_showcase_v2 import BG,INK,MUTED,ACCENT,font
from visual_anagrams.export import prompt_overlay

SCORER='openai/clip-vit-large-patch14'
REVISION='32bd64288804d66eefd0ccbe215aa642df71cc41'
LABELS=dict(baseline='Baseline (reused)',contrastive='Contrastive balancing',cross_negative='Qwen cross negatives',
    schedule='Scheduled CFG',conflict='Conflict projection',freeze='Freeze balancing below 0.6',component='Component guidance')


def main():
    import numpy as np
    from PIL import Image,ImageDraw
    import torch
    import torch.nn.functional as F
    from transformers import CLIPModel,CLIPProcessor
    from visual_anagrams.views import get_views
    from visual_anagrams.quality_experiments import component_previews
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=ROOT/'results/quality_experiments_20261025')
    parser.add_argument('--output',type=Path,default=ROOT/'examples/quality-experiments')
    args=parser.parse_args()
    torch.set_num_threads(6)
    plan=json.loads((args.input/'plan.json').read_text())
    args.output.mkdir(parents=True,exist_ok=True)
    model=None;processor=None;results=[]
    for case in plan['cases']:
        torch.manual_seed(case['view_seed']);views=get_views(case['views'],case.get('view_args'))
        prompts=[f"{case['style']} {p}" for p in case['prompts']]
        records=[]
        for feature in LABELS:
            folder=args.input/case['id']/feature
            if not (folder/'run.json').exists():continue
            record=json.loads((folder/'run.json').read_text())
            source=folder/'sample_1024.png'
            digest=hashlib.sha256(source.read_bytes()).hexdigest()
            assert digest==record['image_sha256']
            with Image.open(source) as raw:image=raw.convert('RGB')
            assert image.size==(1024,1024)
            if case['reduction']=='sum':
                endpoints=component_previews(torch.from_numpy(np.asarray(image).copy()).permute(2,0,1).float()/127.5-1,views)
            else:
                with Image.open(folder/'sample_1024.views.png') as sheet:
                    endpoints=[sheet.crop((1024*i,0,1024*(i+1),1024)).convert('RGB') for i in range(2)]
            evaluation=folder/'evaluation.json'
            if evaluation.exists():
                scores=json.loads(evaluation.read_text())
                assert scores['image_sha256']==digest and scores['revision']==REVISION and scores['prompts']==prompts
            else:
                if model is None:
                    print('Loading held-out CLIP ViT-L/14 on CPU',flush=True)
                    model=CLIPModel.from_pretrained(SCORER,revision=REVISION).float().eval()
                    processor=CLIPProcessor.from_pretrained(SCORER,revision=REVISION,use_fast=False)
                with torch.inference_mode():
                    values=model(**processor(text=prompts,images=endpoints,padding=True,return_tensors='pt'))
                    matrix=F.normalize(values.image_embeds,dim=-1)@F.normalize(values.text_embeds,dim=-1).T
                own=matrix.diag().tolist()
                scores=dict(model=SCORER,revision=REVISION,prompts=prompts,image_sha256=digest,scores=own,
                    weaker_view=min(own),similarity_matrix=matrix.tolist(),
                    minimum_prompt_margin=min(float(matrix[i,i]-matrix[i,1-i]) for i in range(2)))
                write_json(evaluation,scores)
            balance=record.get('balancing') or (record.get('diagnostics') if feature=='baseline' else {}) or {}
            steps=(record.get('diagnostics') or {}).get('steps',[]) if feature!='baseline' else []
            row=dict(record,evaluation=scores,
                balance_updates=sum(s.get('scores') is not None for s in balance.get('steps',[])),
                projected_steps=sum(bool((s.get('conflict') or {}).get('projected')) for s in steps),
                component_updates=sum((s.get('component_balance') or {}).get('scores') is not None for s in steps),
                image_file=f"{case['id']}-{feature}.png")
            shutil.copyfile(source,args.output/row['image_file'])
            records.append((feature,row,endpoints))
            results.append(dict(row,case_id=case['id']))
            print(case['id'],feature,'scores',scores['scores'],'margin',scores['minimum_prompt_margin'],flush=True)
        if not records:continue
        canvas=Image.new('RGB',(1200,95+430*len(records)),BG)
        draw=ImageDraw.Draw(canvas)
        draw.text((16,15),case['theme']+' / '+case['method'],font=font(31,True),fill=INK)
        for index,(feature,row,endpoints) in enumerate(records):
            y=95+430*index
            draw.text((16,y),LABELS[feature],font=font(28,True),fill=ACCENT)
            for side,endpoint in enumerate(endpoints):
                tile=endpoint.resize((360,360),Image.Resampling.LANCZOS)
                canvas.paste(prompt_overlay(tile,case['display_labels'][side],font_size=20),(side*368+16,y+48))
            score=row['evaluation']
            lines=[f"View scores: {score['scores'][0]:.4f} / {score['scores'][1]:.4f}",
                   f"Weaker view: {score['weaker_view']:.4f}",
                   f"Minimum margin: {score['minimum_prompt_margin']:+.4f}" if case['reduction']=='mean' else 'Component scores use filtered/full views',
                   f"{row['seconds']:.1f} seconds / {row['peak_allocated_gib']:.2f} GiB",
                   f"Balance / component updates: {row['balance_updates']} / {row['component_updates']}",
                   f"Conflict corrections: {row['projected_steps']}"]
            for line_index,line in enumerate(lines):draw.text((765,y+70+line_index*47),line,font=font(20),fill=MUTED)
        canvas.save(args.output/f"{case['id']}.jpg",quality=92)
    write_json(args.output/'results.json',dict(plan=plan,scorer=dict(model=SCORER,revision=REVISION,device='cpu'),results=results))
    if (args.input/'negative-plans.json').exists():shutil.copyfile(args.input/'negative-plans.json',args.output/'negative-plans.json')
    done=sum(r['feature']!='baseline' for r in results)
    lines=['# Iris quality experiments: Standard 30 steps','',
        f'{done} new completed images; four verified cached baselines. RTX 3060 12 GB. No new rotation runs.',
        'All images use 1024 x 1024, guidance 3, shift 8 and DPM order 2. Same prompts, seed and view mapping within each case. '+
        'Geometric controls use Adaptive Mean; components use Classic Sum. Each trial changes one option.','',
        'A is the weaker intended-view CLIP cosine similarity. M is the minimum own-minus-other prompt margin. '+
        'Higher is better for both, but neither establishes human illusion quality. Component subjects can coexist, '+
        'so M is not a selection criterion for components. Inspect the filtered/full views as well as scores.','']
    for case in plan['cases']:
        selected=[r for r in results if r['case_id']==case['id']]
        if not selected:continue
        lines += [f"## {case['theme']}",'',*['- '+p for p in [f"{case['style']} {p}" for p in case['prompts']]],'',
            '| Option | View 1 | View 2 | A | M | Seconds | Peak GiB |','| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
        for row in selected:
            score=row['evaluation'];margin=f"{score['minimum_prompt_margin']:+.4f}" if case['reduction']=='mean' else '—'
            lines.append(f"| {LABELS[row['feature']]} | {score['scores'][0]:.4f} | {score['scores'][1]:.4f} | {score['weaker_view']:.4f} | {margin} | {row['seconds']:.1f} | {row['peak_allocated_gib']:.2f} |")
        lines += ['',f"![All completed trials: {case['theme']}]({case['id']}.jpg)",'']
    decision=args.output/'defaults.json'
    if decision.exists():
        choice=json.loads(decision.read_text())
        lines += ['## Default decision','',choice['summary'],'',*choice.get('findings',[]),'']
    else:lines += ['## Default decision','','Evaluation in progress. Defaults have not yet been changed.','']
    lines += ['## Conditions','',
        'Held-out CLIP ViT-L/14 scores saved clean pixels. Generation uses CPU CLIP ViT-B/32 where applicable; '+
        'these are related evaluator families. Human review is required, especially for subject segregation and artifacts.',
        'The initial screen tests contrastive balancing, Qwen cross negatives (strength 0.5), scheduled CFG '+
        '(noise interval 0.1–0.9, ramps 0.05), conflict projection (strength 0.5), freeze below noise 0.6, '+
        'and component guidance. At 30 steps / shift 8, the lowest model-evaluation noise is about 0.216: '+
        'freeze 0.1 has no effect, whereas freeze 0.6 suppresses the final two balancing updates.',
        'Times exclude encoding, loading, warm-up, saving and evaluation. Baseline times come from the earlier runs; '+
        'treat runtime differences as approximate. Trials are a small pilot, not a universal ranking.',
        '[Full settings, schedules, hashes and scores](results.json). [Qwen plans](negative-plans.json).','',
        'Reproduce the screen with `python scripts/benchmark_quality_experiments.py`; '+
        'repeat selected geometric options on the jigsaw using `--confirm contrastive` (or other option IDs). '+
        'The verified source runs and prompt caches named in the plan are required. '+
        'Publish completed trials with `python scripts/report_quality_experiments.py`.','']
    (args.output/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    print('Report saved:',args.output,flush=True)


if __name__=='__main__':main()
