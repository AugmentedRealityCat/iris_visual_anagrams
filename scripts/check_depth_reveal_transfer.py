"""Evaluate existing depth reveals at a different inference resolution and RGB semantics."""
import gc
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    import torch
    from PIL import Image
    from visual_anagrams.iris_tasks import resolve_task_model,load_task,save_depth
    from visual_anagrams.depth_anagram import correlation
    from visual_anagrams.balancing import CleanPreviewScorer,CLIP_ID,CLIP_REVISION
    torch.set_num_threads(6)
    out=ROOT/'results/depth-lab-20261026'
    runner=load_task('depth',resolve_task_model('depth'),'cuda')
    report={}
    for name in ('depth-anagram-256-0.12','depth-anagram-384-0.06'):
        folder=out/name
        images={key:Image.open(folder/(key+'.png')).convert('RGB') for key in ('input','result','target')}
        target=runner(images['target'].resize((512,512),Image.Resampling.BICUBIC),max_side=512)
        records={}
        for key in ('input','result'):
            predicted=runner(images[key].resize((512,512),Image.Resampling.BICUBIC),max_side=512)
            records[key]=dict(target_correlation_at_512=correlation(predicted,target))
            export=folder/(key+'-depth-512');export.mkdir(exist_ok=True);save_depth(predicted,export)
        report[name]=records
    del runner;gc.collect();torch.cuda.empty_cache()
    scorer=CleanPreviewScorer(['a comic book portrait of the Joker with green hair and a red smile',
        'a comic book portrait of Batman wearing a black pointed-eared cowl'])
    for name in report:
        folder=out/name
        images=[Image.open(folder/(key+'.png')).convert('RGB') for key in ('input','result')]
        scores=scorer.score_images(images)
        report[name]['rgb_clip_scores']=dict(initial=scores[0],optimized=scores[1],
            order=['Joker','Batman'],model=CLIP_ID,revision=CLIP_REVISION)
    (out/'transfer-checks.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
