"""Compare the two four-step puzzle previews with their 30-step refinements."""

import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    import torch
    import torch.nn.functional as F
    from PIL import Image,ImageDraw
    from transformers import CLIPModel,CLIPProcessor
    from scripts.build_showcase_v2 import SCORER,SCORER_REVISION,BG,INK,font,write_json
    torch.set_num_threads(6)
    output=ROOT/'examples/showcase-v3'
    cases=json.loads((output/'plan.json').read_text())['cases'][-2:]
    selection_path=output/'selection.json'
    reviewed=json.loads(selection_path.read_text()) if selection_path.exists() else {}
    model=CLIPModel.from_pretrained(SCORER,revision=SCORER_REVISION).float().eval()
    processor=CLIPProcessor.from_pretrained(SCORER,revision=SCORER_REVISION,use_fast=False)
    gallery=Image.new('RGB',(1200,1400),BG)
    draw=ImageDraw.Draw(gallery)
    selection,rows={},[]
    with torch.inference_mode():
        for case_index,case in enumerate(cases):
            results=[]
            prompts=[f"{case['style']} {p}" for p in case['prompts']]
            for mode_index,source_root in enumerate(['results/showcase_v3_candidates_20261025','results/showcase_v3_refinements_20261025']):
                folder=ROOT/source_root/case['id']/f"{case['seeds'][0]:04}"
                run=json.loads((folder/'run.json').read_text())
                with Image.open(folder/'sample_1024.views.png') as raw:
                    sheet=raw.convert('RGB')
                endpoints=[sheet.crop((1024*i,0,1024*(i+1),1024)) for i in range(2)]
                result=model(**processor(text=prompts,images=endpoints,padding=True,return_tensors='pt'))
                matrix=F.normalize(result.image_embeds,dim=-1)@F.normalize(result.text_embeds,dim=-1).T
                scores=matrix.diag().tolist()
                row=dict(case=case['id'],mode=run['mode'],steps=run['steps'],guidance=run['guidance'],
                    source_root=source_root,scores=scores,weaker_view=min(scores),similarity_matrix=matrix.tolist(),
                    image_sha256=run['image_sha256'],generation_seconds=run['seconds'])
                results.append(row)
                y=(case_index*2+mode_index)*350
                draw.text((16,y+8),f"{case['id']} / {run['mode']} / {run['steps']} steps / {scores[0]:.3f}, {scores[1]:.3f}",font=font(25,True),fill=INK)
                gallery.paste(sheet.resize((600,300)),(16,y+46))
                for i,label in enumerate(case['labels']):draw.text((650,y+100+i*70),label,font=font(29),fill=INK)
            chosen=max(results,key=lambda r:r['weaker_view'])
            prior=reviewed.get(case['id'],{})
            if prior.get('visual_review'):
                chosen=next(row for row in results if row['source_root']==prior['source_root'])
            selection[case['id']]=dict(source_root=chosen['source_root'],
                settings={key:chosen[key] for key in ['mode','steps','guidance']},
                visual_review=bool(prior.get('visual_review')),
                reason=prior['reason'] if prior.get('visual_review') else
                    'Higher weaker-view CLIP alignment within this two-candidate comparison; visual review also required.')
            rows.extend(results)
            print('Selected',case['id'],chosen['mode'],chosen['steps'],chosen['scores'],flush=True)
    gallery.save(output/'puzzle-comparison.jpg',quality=93)
    write_json(output/'puzzle-comparison.json',dict(scorer=SCORER,revision=SCORER_REVISION,
        interpretation='Prompt alignment is not a human illusion rating. Same prompts and seed, different sampler/step profile.',candidates=rows))
    write_json(output/'selection.json',selection)


if __name__=='__main__':main()
