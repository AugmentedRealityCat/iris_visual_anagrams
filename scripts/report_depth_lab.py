"""Publish the completed depth pilot, including unsuccessful tradeoffs."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image,ImageDraw
    from scripts.build_showcase_v2 import font,BG,INK,MUTED
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=ROOT/'results/depth-lab-20261026')
    parser.add_argument('--output',type=Path,default=ROOT/'examples/depth-lab')
    args=parser.parse_args();source=args.input;out=args.output;out.mkdir(parents=True,exist_ok=True)
    transfers=json.loads((source/'transfer-checks.json').read_text())
    measurements={};artifacts={};video_checks={}
    def copy(original,name):
        shutil.copyfile(original,out/name)
        digest=hashlib.sha256(original.read_bytes()).hexdigest()
        assert digest==hashlib.sha256((out/name).read_bytes()).hexdigest()
        artifacts[name]=dict(sha256=digest,bytes=(out/name).stat().st_size)
    def contact(name,title,tiles):
        width=320;header=75;row=380;columns=len(tiles)
        canvas=Image.new('RGB',(width*columns,header+row),BG);draw=ImageDraw.Draw(canvas)
        draw.text((16,16),title,font=font(25,True),fill=INK)
        for i,(label,image) in enumerate(tiles):
            if image.mode in ('I;16','I;16L','I;16B','I'):
                image=Image.fromarray(np.uint8(np.clip(np.asarray(image,dtype=np.float32)/257,0,255)))
            preview=image.convert('RGB').resize((300,300),Image.Resampling.LANCZOS)
            canvas.paste(preview,(width*i+10,header+38))
            draw.text((width*i+10,header),label,font=font(18),fill=MUTED)
        canvas.save(out/name,quality=92)
    for label in ('joker','batman'):copy(source/(label+'.png'),label+'.png')
    for name in ('parallax-batman','stereogram-batman','stereogram-dots','focus-hybrid'):
        folder=source/'render-final'/name/'0000'
        measurements[name]=json.loads((folder/'run.json').read_text())
        for file in ('result.png','depth16.png','decoded-depth.png','result.mp4','near-focus.png'):
            if (folder/file).exists():copy(folder/file,name+'-'+file)
        if (folder/'result.mp4').exists():
            reader=imageio.get_reader(folder/'result.mp4',format='FFMPEG')
            frames=[];total=0
            try:
                for index,frame in enumerate(reader):
                    assert frame.shape==(512,512,3)
                    if index in (0,18,36):frames.append((f'Frame {index}',Image.fromarray(frame)))
                    total+=1
            finally:reader.close()
            assert total==72
            video_checks[name]=dict(frames_decoded=total,size=[512,512],fps=24,seconds=3.)
            contact(name+'.jpg',name.replace('-',' ').title(),frames)
    for name in ('depth-anagram-256-0.12','depth-anagram-384-0.06'):
        folder=source/name
        record=json.loads((folder/'run.json').read_text());record['transfer']=transfers[name]
        measurements[name]=record
        tiles=[]
        for label,path in [('Original RGB','input.png'),('Optimized RGB','result.png'),
            ('Original depth','initial-depth/result.png'),('Optimized depth','depth-preview.png'),('Target depth','target-depth/result.png')]:
            with Image.open(folder/path) as image:tiles.append((label,image.copy()))
        contact(name+'.jpg',f"Joker colour / Batman depth: {record['work_size'][0]}px, budget {record['budget']}",tiles)
        copy(folder/'result.png',name+'.png')
        copy(folder/'depth16.png',name+'-depth16.png')
        for label in ('input','result'):copy(folder/f'{label}-depth-512/result.png',name+f'-{label}-depth-512.png')
    stereo=source/'render-final/stereogram-dots/0000'
    texture=source/'render-final/stereogram-batman/0000'
    contact('stereograms.jpg','Batman relief encoded in stereo correspondence',[
        ('Random dots',Image.open(stereo/'result.png')),('Joker texture',Image.open(texture/'result.png')),
        ('Decoded correspondence',Image.open(stereo/'decoded-depth.png')),('Target relief',Image.open(stereo/'depth16.png'))])
    data=dict(reference_generation=json.loads((source/'references.json').read_text()),
        environment=json.loads((source/'environment.json').read_text()),
        experiments=measurements,video_validation=video_checks,artifacts=artifacts,
        code_sha256={path:hashlib.sha256((ROOT/path).read_bytes()).hexdigest() for path in [
            'depth_lab.py','visual_anagrams/depth_effects.py','visual_anagrams/depth_anagram.py']})
    (out/'results.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
    preferred=measurements['depth-anagram-384-0.06'];old=measurements['depth-anagram-256-0.12']
    checks=preferred['robustness'];transfer=transfers['depth-anagram-384-0.06']
    lines=['# Iris depth experiments','',
        'Four experimental tasks implemented in Iris Tools and tried on RTX 3060 12 GB. '+
        'Two ordinary 1024-pixel character references were generated with the existing Turbo r2 LoRA at four steps. No rotations were generated. '+
        'The generator was released before loading depth; no additional model or LLM was trained.','',
        '## Colour / depth anagram','',
        'Bounded edits to a Joker portrait steer the frozen Iris depth prediction toward the depth of a separate Batman reference. '+
        'The target is a depth relief, not a second ordinary RGB portrait. Pixel editing uses checkpointed input gradients, with all model weights frozen.','',
        '| Profile | Initial correlation | Saved RGB correlation | Resize 75% | JPEG 85 | PSNR | Seconds | Peak GiB |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name in ('depth-anagram-256-0.12','depth-anagram-384-0.06'):
        r=measurements[name];c=r['robustness']
        lines.append(f"| {r['work_size'][0]}px / {r['steps']} edits / bound {r['budget']} | {c['initial']['correlation']:.3f} | {c['clean']['correlation']:.3f} | {c['resize_75']['correlation']:.3f} | {c['jpeg_85']['correlation']:.3f} | {r['psnr_db']:.2f} dB | {r['seconds']:.1f} | {r['peak_allocated_gib']:.2f} |")
    lines += ['',
        '![Higher-edit trial](depth-anagram-256-0.12.jpg)','',
        '![Gentler, higher-resolution trial](depth-anagram-384-0.06.jpg)','',
        'The 256px trial has obvious coloured block artifacts. The gentler 384px profile preserves the character better, '+
        'but still has visible colour/texture changes. Resolution, iteration count and edit bound all changed together: '+
        'this is a practical profile comparison, not an isolated edit-bound ablation. '+
        'The new experimental task starts at 384px, 24 iterations and bound 0.06; ordinary anagram defaults are unchanged. '+
        'The saved RGB rounding can differ from the continuous bound by one 8-bit quantization step.','',
        f"At a separate 512px depth inference resolution, the gentler trial's target correlation changed from {transfer['input']['target_correlation_at_512']:.3f} (original) to {transfer['result']['target_correlation_at_512']:.3f} (edited). "+
        'Both images and the target were resized and depth was re-estimated. '+
        'Most of the optimization gain is lost at the new resolution, so the effect is resolution-dependent. '+
        'This check differs from reducing then restoring the image at the training resolution. '+
        'CLIP RGB scores and the complete per-iteration history are in results.json.','',
        '**Result: measurable Iris-depth steering, with a visible image-quality tradeoff; a convincing Batman reveal was not achieved.** '+
        'The relief is a coarse bust/helmet shape, not a clean detailed Batman portrait. '+
        'Correlation is normalized shape agreement, not human recognition or calibrated geometry. '+
        'There is no independent depth-model or human-perception validation, so this can still be estimator-specific.','',
        '## Magic Eye stereograms','',
        '![Dots, texture, decoded depth and target](stereograms.jpg)','',
        f"Random-dot depth correlation decoded from local stereo matching: {measurements['stereogram-dots']['decoded_depth_correlation']:.3f}; "+
        f"Joker-textured version: {measurements['stereogram-batman']['decoded_depth_correlation']:.3f}. "+
        'These are correspondence diagnostics, not human viewing scores. The repeat period is 46px at 512px output width. '+
        'The symmetric constraints include a hidden-surface visibility test. Texture colours free points with added speckle; '+
        'it does not preserve an independently readable full Joker portrait.','',
        '[Random-dot PNG](stereogram-dots-result.png) · [Textured PNG](stereogram-batman-result.png)','',
        'View the PNG at a comfortable size and let the eyes converge behind the image, keeping the head level. '+
        'The intended relief is Batman. Human stereo fusion still needs checking.','',
        '## Parallax','',
        '[Watch the 3-second Batman camera loop](parallax-batman-result.mp4)','',
        '![Three parallax frames](parallax-batman.jpg)','',
        f"Bilinear forward splatting uses a near-surface z-buffer; the largest uncovered fraction after cropping was {100*measurements['parallax-batman']['max_uncovered_fraction']:.2f}%. "+
        'Local colour diffusion fills disocclusions and a fixed crop removes outer edges. '+
        'This gives a useful small 2.5D movement, but fine hair/cowl boundaries can stretch or acquire a halo. '+
        'Hidden backgrounds are not reconstructed with a learned inpainting model; larger moves expose the limitation.','',
        '## Focus sweep','',
        '[Watch the 3-second frequency-hybrid focus loop](focus-hybrid-result.mp4)','',
        '![Three focus frames](focus-hybrid.jpg)','',
        'One fixed Vader/castle hybrid is split into 12 depth bands and rendered using premultiplied-alpha blur '+
        'from far to near focus and back. No subject images are crossfaded. '+
        'The depth-dependent focus change works, but this example does not establish a convincing focus-triggered switch '+
        'between two readable subjects. Large blur also softens the boundaries. Keep it experimental.','',
        '## Usage and reproduction','',
        'Open Iris Tools and select any of the four tasks marked experimental. The same tasks are available via iris_tools.py:','',
        '```powershell',
        'python iris_tools.py --task parallax --input batman.png --depth_formats gif mp4',
        'python iris_tools.py --task stereogram --input batman.png --texture joker.png',
        'python iris_tools.py --task focus --input hybrid.png',
        'python iris_tools.py --task depth_anagram --input joker.png --target batman.png',
        '```','',
        'Renderers can reuse an existing raw relative-log depth.npy via --depth_map (matching the original input dimensions). '+
        'Cached renders load no model weights. Videos use H.264, 24 FPS, without outer white borders. '+
        'Parallax and focus now offer MP4, GIF, or both via Animation output / --depth_formats. '+
        'Their animations play and loop inside the GUI Result preview with Play/Pause. '+
        'Load result accepts earlier MP4/GIF files, or discovers an animation beside its result PNG. '+
        'All tasks save settings, source hashes and result measurements. RGB optimization centre-fits the target to the input aspect ratio.','',
        '[Playback and GIF export validation](media-validation.json).','',
        'The pilot script is `python scripts/experiment_depth_lab.py`; its cached prompt embeddings from the earlier flow-shift study are required. '+
        'It preserves completed runs. To repeat the gentler trial use `--skip_generation --optimization_side 384 --iterations 24 --budget .06`. '+
        'Run `python scripts/render_depth_pilot.py` for the final cached renders and '+
        '`python scripts/check_depth_reveal_transfer.py` for the 512px/CLIP checks, then '+
        '`python scripts/report_depth_lab.py` to publish the saved results. '+
        'Times exclude loading and initial reference generation. Render-only times exclude the cached depth inference.','',
        'The first optimization attempt failed because inference-mode rotary caches could not be used for input gradients. '+
        'The optimizer now clears those caches before differentiation; a real tiny Iris depth regression test covers this transition.','',
        '[Full measurements, hashes, model provenance and decoded-video checks](results.json). '+
        '[Final automated validation](validation.json).','',
        '## Research basis','',
        '[Iris depth implementation](https://github.com/speridlabs/iris-3b/blob/a8d15239dea469aba042cfa56ca3bb4e450d5ebc/src/iris3b/downstream/depth.py) supplies relative log depth, not metres. '+
        '[Thimbleby, Inglis and Witten](https://harold.thimbleby.net/sirds/ieee3d.pdf) supply the stereo geometry and visibility principle. '+
        '[3D Photography](https://shihmengli.github.io/3D-Photo-Inpainting/) motivates handling disocclusions; '+
        'our simple local filling is not its learned layered-inpainting method. '+
        'Colour/depth optimization and focus rendering here are exploratory adaptations, not claimed novel research or verified reproductions.','']
    (out/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    print('Published',out,flush=True)


if __name__=='__main__':main()
