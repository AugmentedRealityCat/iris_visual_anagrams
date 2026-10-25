"""Build the extended showcase, including filtered component presentations.

Component endpoints are derived from one saved image: blur/full detail,
grayscale/full colour, or diagonal motion blur/full detail. They are not
independently generated pictures. Geometric methods retain saved permutations.
"""

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from scripts.build_showcase_v2 import BG, INK, MUTED, ACCENT, center, font, write_json
from visual_anagrams.export import AnimationOptions, animation_timeline, prompt_overlay, render_frame


def settings(case):
    mode = 'Standard Iris' if case['mode'] == 'standard' else 'Turbo r2'
    method = 'sum / Classic' if case['reduction'] == 'sum' else 'adaptive'
    return f"{mode} / {case['steps']} steps / {method}"


def gaussian_component(pixels, sigma, kernel_size):
    """The same reflect-padded Gaussian as the sampler, evaluated separably."""
    import torch
    import torch.nn.functional as F
    coords = torch.arange(kernel_size, dtype=pixels.dtype) - (kernel_size-1)/2
    kernel = torch.exp(-.5*(coords/sigma)**2)
    kernel /= kernel.sum()
    radius = kernel_size//2
    x = pixels[None]
    x = F.conv2d(F.pad(x,(radius,radius,0,0),mode='reflect'),
                 kernel.reshape(1,1,1,-1).repeat(3,1,1,1),groups=3)
    x = F.conv2d(F.pad(x,(0,0,radius,radius),mode='reflect'),
                 kernel.reshape(1,1,-1,1).repeat(3,1,1,1),groups=3)
    return x[0]


def motion_component(pixels, kernel_size):
    """Diagonal averaging with the sampler's zero padding in signed RGB space."""
    import torch.nn.functional as F
    radius = kernel_size//2
    padded = F.pad(pixels,(radius,radius,radius,radius))
    result = pixels*0
    height, width = pixels.shape[-2:]
    for offset in range(kernel_size):
        result += padded[:,offset:offset+height,offset:offset+width]
    return result/kernel_size


def component_endpoints(image, case, views):
    import torch
    pixels = torch.from_numpy(np.asarray(image).copy()).permute(2,0,1).float()/127.5-1
    kind = case['presentation']
    if kind == 'frequency':
        factor = image.height//64
        kernel_size = views[0].kernel_size*factor + ((factor+1)%2)
        first = gaussian_component(pixels,views[0].sigma*factor,kernel_size)
    elif kind == 'motion':
        factor = image.height//64
        kernel_size = views[0].size*factor + ((factor+1)%2)
        first = motion_component(pixels,kernel_size)
    elif kind == 'color':
        first = pixels.mean(0,keepdim=True).expand_as(pixels)
    else:
        raise ValueError(kind)
    array = first.clamp(-1,1).add(1).mul(127.5).round().byte().permute(1,2,0).numpy()
    return [Image.fromarray(array),image.copy()]


def scene(item, t, size, *, labels=True):
    case,image,views,endpoints,_record = item
    if case['presentation'] == 'geometric':
        options = AnimationOptions(size=size,borderless=True,labels=labels)
        return render_frame(image,case['views'][1],views[1],t,options,*case['display_labels'])
    # Continuously reveal the existing fine-detail / colour residual of the same image.
    frame = Image.blend(endpoints[0],endpoints[1],t).resize((size,size),Image.Resampling.LANCZOS)
    if labels:
        label = case['display_labels'][0 if t<.5 else 1]
        opacity = max(0,1-t*5) if t<.5 else max(0,(t-.8)*5)
        frame = prompt_overlay(frame,label,opacity=opacity,font_size=max(14,round(size*.03)))
    return frame


def overlay(frame, case, index, count):
    frame = frame.convert('RGBA')
    banner = Image.new('RGBA',frame.size)
    draw = ImageDraw.Draw(banner)
    for y in range(155):
        draw.line((0,y,1080,y),fill=(9,13,22,round(220*(1-y/155))))
    draw.text((26,18),f"{index:02}/{count:02}  {case['method']}",font=font(34,True),fill=INK)
    draw.text((27,66),settings(case),font=font(25),fill=INK)
    draw.text((27,103),case['style_label'],font=font(23),fill=ACCENT)
    return Image.alpha_composite(frame,banner).convert('RGB')


def comparison(item):
    case,_image,_views,endpoints,_record = item
    frame=Image.new('RGB',(1080,1080),BG)
    draw=ImageDraw.Draw(frame)
    center(draw,case['method'].upper(),70,45,bold=True)
    center(draw,case['style_label'],140,31,ACCENT)
    for i,endpoint in enumerate(endpoints):
        tile=endpoint.resize((520,520),Image.Resampling.LANCZOS)
        frame.paste(prompt_overlay(tile,case['display_labels'][i],font_size=25),(10+i*540,280))
    center(draw,settings(case),850,29,MUTED)
    center(draw,'One saved image / transformed or filtered views',927,27)
    return frame


def title(items, outro=False):
    frame=Image.new('RGB',(1080,1080),BG)
    draw=ImageDraw.Draw(frame)
    center(draw,'VISUAL ANAGRAMS',38,60,bold=True)
    center(draw,'Iris 3B INT8 / Standard + Turbo r2',119,31,ACCENT)
    for i,item in enumerate(items):
        case,_image,_views,endpoints,_record=item
        tile=endpoints[0].resize((244,244),Image.Resampling.LANCZOS)
        frame.paste(prompt_overlay(tile,case['display_labels'][0],font_size=17),(25+(i%4)*263,200+(i//4)*254))
    if outro:
        center(draw,'RTX 3060 12 GB / generated locally',974,28)
        center(draw,'Experimental results. Labels show the intended views.',1022,24,MUTED)
    else:
        center(draw,'12 methods / puzzles, warps and component hybrids',974,30,bold=True)
        center(draw,'30-step Standard + four-step Turbo / flow shift 8',1022,26,MUTED)
    return frame


def gallery(items, path, title_text='IRIS VISUAL ANAGRAMS / EXTENDED SHOWCASE'):
    canvas=Image.new('RGB',(1200,100+460*((len(items)+1)//2)),BG)
    draw=ImageDraw.Draw(canvas)
    center(draw,title_text,22,32,bold=True,width=1200)
    for i,item in enumerate(items):
        case,_image,_views,endpoints,_record=item
        x,y=(i%2)*600,100+(i//2)*460
        draw.text((x+16,y),case['method'],font=font(27,True),fill=INK)
        draw.text((x+16,y+42),case['style_label'],font=font(23),fill=ACCENT)
        for j,endpoint in enumerate(endpoints):
            tile=endpoint.resize((280,280),Image.Resampling.LANCZOS)
            canvas.paste(prompt_overlay(tile,case['display_labels'][j],font_size=17),(x+16+j*288,y+90))
        draw.text((x+16,y+389),settings(case),font=font(19),fill=MUTED)
    canvas.save(path,quality=92)


def main():
    import torch
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'results/showcase_v3_candidates_20261025')
    parser.add_argument('--output',type=Path,default=ROOT/'examples/showcase-v3')
    parser.add_argument('--assets-only',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(6)
    plan=json.loads((args.output/'plan.json').read_text())
    selection_path=args.output/'selection.json'
    selection=json.loads(selection_path.read_text()) if selection_path.exists() else {}
    items,validation=[],[]
    for raw_case in plan['cases']:
        case=dict(raw_case)
        choice=selection.get(case['id'],{})
        overrides=choice.get('settings',{})
        assert set(overrides)<=set(('mode','steps','guidance'))
        case.update(overrides)
        case.setdefault('display_labels',case['labels'])
        seed=case['seeds'][0]
        source_root=ROOT/choice['source_root'] if choice.get('source_root') else args.source
        folder=source_root/case['id']/f'{seed:04}'
        case['candidate_source']=str(folder.relative_to(ROOT)).replace('\\','/')
        source=folder/'sample_1024.png'
        record=json.loads((folder/'run.json').read_text())
        assert record['steps']==case['steps'] and record['shift']==plan['shift']
        assert record.get('mode','standard')==case['mode']
        assert record.get('quality_method',record.get('method'))==case['quality_method']
        assert hashlib.sha256(source.read_bytes()).hexdigest()==record['image_sha256']
        with (folder.parent/'metadata.pkl').open('rb') as stream:
            views=pickle.load(stream)['views']
        with Image.open(source) as raw:
            image=raw.convert('RGB')
        assert image.size==(1024,1024)
        if case['presentation']=='geometric':
            with Image.open(folder/'sample_1024.views.png') as sheet:
                assert sheet.size==(2048,1024)
                endpoints=[sheet.crop((i*1024,0,(i+1)*1024,1024)).convert('RGB') for i in range(2)]
        else:
            assert case['reduction']=='sum' and case['quality_method']=='plain'
            endpoints=component_endpoints(image,case,views)
        item=(case,image,views,endpoints,record)
        errors=[float(np.abs(np.asarray(scene(item,t,1024,labels=False)).astype(float)-np.asarray(endpoints[i])).mean())
                for i,t in enumerate([0.,1.])]
        assert max(errors)<1,(case['id'],errors)
        shutil.copyfile(source,args.output/f"{case['id']}.png")
        preview=Image.new('RGB',(1024,512))
        for i,endpoint in enumerate(endpoints):
            preview.paste(prompt_overlay(endpoint.resize((512,512),Image.Resampling.LANCZOS),
                case['display_labels'][i],font_size=23),(i*512,0))
        preview.save(args.output/f"{case['id']}.views.jpg",quality=92)
        validation.append(dict(case=case['id'],seed=seed,steps=case['steps'],mode=case['mode'],
            method=case['quality_method'],reduction=case['reduction'],presentation=case['presentation'],
            image_sha256=record['image_sha256'],endpoint_mean_absolute_pixel_error=errors,
            endpoint_rgb_sha256=[hashlib.sha256(np.asarray(e).tobytes()).hexdigest() for e in endpoints]))
        items.append(item)
        print('Prepared',case['id'],flush=True)
    gallery(items,args.output/'gallery.jpg')
    gallery(items[6:],args.output/'new-methods.jpg','ADDED METHODS / NO NEW ROTATION RUNS')
    title(items).save(args.output/'reddit-thumbnail.jpg',quality=93)
    write_json(args.output/'validation.json',dict(images=validation))
    if args.assets_only:return
    fps,frames,chapters=24,0,[]
    timeline=animation_timeline(AnimationOptions(fps=fps,size=1024,transition=2.,hold=2.))
    video=args.output/'showcase4.mp4'
    temporary=args.output/'.showcase4.tmp.mp4'
    with imageio.get_writer(temporary,format='FFMPEG',fps=fps,codec='libx264',quality=None,
        pixelformat='yuv420p',macro_block_size=1,ffmpeg_log_level='error',
        output_params=['-crf','30','-preset','slow','-threads','6','-movflags','+faststart']) as writer:
        def write(frame,count=1):
            nonlocal frames
            pixels=np.asarray(frame)
            for _ in range(count):writer.append_data(pixels)
            frames+=count
        write(title(items),fps*3)
        for i,item in enumerate(items,1):
            case=item[0]
            chapters.append(dict(case=case['id'],start_seconds=frames/fps,mode=case['mode'],steps=case['steps']))
            write(overlay(scene(item,0,1080),case,i,len(items)),fps)
            cache={}
            for t in timeline:
                key=round(t,12)
                if key not in cache:cache[key]=overlay(scene(item,t,1080),case,i,len(items))
                write(cache[key])
            write(comparison(item),fps)
            print(f"Encoded chapter {i}/{len(items)}: {case['id']}",flush=True)
        write(title(items,outro=True),fps*3)
    temporary.replace(video)
    with imageio.get_reader(video) as reader:
        meta=reader.get_meta_data()
        assert tuple(meta['size'])==(1080,1080) and meta['fps']==fps
        assert abs(meta['duration']-frames/fps)<.1
    environment=json.loads((args.source/'environment.json').read_text())
    manifest=dict(version=3,model=environment['model'],model_revision=environment['model_revision'],
        code_revision=environment['code_revision'],text_revision=environment['text_revision'],
        turbo_repo=environment['turbo_repo'],turbo_revision=environment['turbo_revision'],
        turbo_adapter='r2-1000 (four-step examples only)',gpu=environment['gpu'],
        resolution=plan['resolution'],shift=plan['shift'],
        new_images=6+(len(list((ROOT/'results/showcase_v3_refinements_20261025').glob('*/*/run.json')))),
        reused_images=6,new_rotation_runs=0,
        examples=[dict(case,selected_seed=case['seeds'][0],full_prompts=[f"{case['style']} {p}" for p in case['prompts']],
            generation_seconds=record['seconds'],image_sha256=record['image_sha256'],
            peak_allocated_gib=record['peak_allocated_gib']) for case,_im,_v,_e,record in items],
        video=dict(file=video.name,bytes=video.stat().st_size,duration_seconds=frames/fps,fps=fps,
            size=[1080,1080],audio=False,borderless=True,labels=True,crf=30,preset='slow',chapters=chapters))
    write_json(args.output/'manifest.json',manifest)
    print(f'Saved {video}: {frames/fps:.1f}s, {video.stat().st_size/1e6:.2f}MB',flush=True)


if __name__=='__main__':main()
