"""Experimental Iris depth reveals, stereograms, parallax and rack-focus exports."""

from pathlib import Path
import math

DEPTH_TASKS={
    'Depth parallax (experimental)':'parallax',
    'Video parallax (experimental)':'parallax_video',
    'Magic Eye stereogram (experimental)':'stereogram',
    'Depth focus sweep (experimental)':'focus',
    'Colour / depth anagram (experimental)':'depth_anagram',
}
DEPTH_FORMATS={'MP4 video':['mp4'],'Animated GIF':['gif'],'GIF + MP4':['gif','mp4']}


def add_depth_args(parser):
    parser.add_argument('--target',type=Path,help='Reference whose predicted depth the RGB image should reveal')
    parser.add_argument('--texture',type=Path,help='Optional colour texture for a stereogram')
    parser.add_argument('--depth_map',type=Path,help='Optional cached relative-log-depth NPY for rendering; never used for RGB optimization')
    parser.add_argument('--render_side',type=int,default=768)
    parser.add_argument('--duration',type=float,default=3.)
    parser.add_argument('--fps',type=int,default=24)
    parser.add_argument('--depth_formats',nargs='+',choices=['mp4','gif'],default=['mp4'],
        help='Parallax/focus animation output: mp4, gif, or both')
    parser.add_argument('--motion',type=float,default=12.,help='Parallax displacement range in output pixels')
    parser.add_argument('--video_start',type=float,default=0.,help='Video clip start time in seconds')
    parser.add_argument('--video_seconds',type=float,default=0.,help='Video clip length in seconds; 0 = remaining video')
    parser.add_argument('--video_fps',type=float,default=0.,help='Video output FPS; 0 = source FPS, capped at 60')
    parser.add_argument('--depth_smoothing',type=float,default=.25,help='Video depth history weight, 0..0.95; lower for fast motion')
    parser.add_argument('--no_audio',action='store_true',help='Omit source audio from video-parallax MP4')
    parser.add_argument('--aperture',type=float,default=16.,help='Maximum artistic defocus radius in output pixels')
    parser.add_argument('--period',type=int,default=None,help='Stereogram far-plane repeat period in pixels; default about 9 percent of width')
    parser.add_argument('--iterations',type=int,default=24)
    parser.add_argument('--optimization_side',type=int,default=384)
    parser.add_argument('--edit_budget',type=float,default=.06,help='Continuous RGB edit bound in 0..1 units; saving rounds to 8-bit')
    return parser


def validate_depth_args(args):
    if args.task not in DEPTH_TASKS.values():return
    if not 128<=args.render_side<=2048:raise ValueError('Render side must be between 128 and 2048')
    if not math.isfinite(args.duration) or not .5<=args.duration<=30 or not 1<=args.fps<=60:
        raise ValueError('Use a duration of 0.5–30 seconds and 1–60 FPS')
    if args.task!='parallax_video' and args.duration*args.fps>900:raise ValueError('Keep experimental exports to at most 900 frames')
    if not all(math.isfinite(v) and 0<=v<=100 for v in (args.motion,args.aperture)):
        raise ValueError('Motion and aperture must be finite, between 0 and 100 pixels')
    if args.period is not None and args.period<8:raise ValueError('Stereo period must be at least 8 pixels')
    for name in ('target','texture','depth_map'):
        path=getattr(args,name)
        if path is not None and not path.is_file():raise ValueError(f'Choose an existing {name.replace("_"," ")} file')
    if args.task=='parallax_video':
        if args.depth_map is not None:raise ValueError('Video parallax requires live per-frame depth, not a single cached depth map')
        if not all(math.isfinite(v) and v>=0 for v in (args.video_start,args.video_seconds,args.video_fps)) or args.video_fps>60 or 0<args.video_fps<1:
            raise ValueError('Video start/length must be nonnegative; output FPS must be 0 (source) or 1–60')
        if not math.isfinite(args.depth_smoothing) or not 0<=args.depth_smoothing<=.95:
            raise ValueError('Depth smoothing must be between 0 and 0.95')
    if args.task=='stereogram' and args.input is not None and args.input.is_file():
        from PIL import Image,ImageOps
        with Image.open(args.input) as image:
            width,height=ImageOps.exif_transpose(image).size
        output_width=max(2,round(width*min(1,args.render_side/max(width,height))/2)*2)
        if output_width<24 or (args.period is not None and args.period>output_width//3):
            raise ValueError('Stereo period must be at most one third of the rendered image width')
    if args.task=='depth_anagram':
        if args.target is None:raise ValueError('Colour / depth anagrams need a target reference image')
        if args.depth_map is not None:raise ValueError('RGB optimization requires the live depth model, not a cached depth map')
        if not 1<=args.iterations<=200 or not 64<=args.optimization_side<=512 or not math.isfinite(args.edit_budget) or not 0<args.edit_budget<=.25:
            raise ValueError('Use 1–200 iterations, a 64–512 optimization side and an RGB edit budget in (0, 0.25]')
    if args.num_samples!=1:raise ValueError('Depth experiments process one input image per run')


def run_experiment(runner,args,source,destination):
    import json
    import time
    import numpy as np
    from PIL import Image,ImageOps
    from visual_anagrams.depth_effects import (fit_image,near_map,resize_map,parallax_frame,
        parallax_frames,focus_frame,focus_frames,stereogram,decode_stereogram,write_animations)
    from visual_anagrams.iris_tasks import save_depth

    destination=Path(destination);destination.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();details=dict(task=args.task)
    if args.task=='depth_anagram':
        from visual_anagrams.depth_anagram import optimize_depth_anagram
        with Image.open(args.target) as image:target=ImageOps.exif_transpose(image).convert('RGB')
        result,source,target,target_depth,record=optimize_depth_anagram(runner,source,target,
            steps=args.iterations,max_side=args.optimization_side,budget=args.edit_budget)
        details.update(record)
        target.save(destination/'target.png')
        target_dir=destination/'target-depth';target_dir.mkdir();save_depth(target_depth,target_dir)
        depth=runner(result,max_side=args.optimization_side)
        initial_dir=destination/'initial-depth';initial_dir.mkdir()
        save_depth(runner(source,max_side=args.optimization_side),initial_dir)
        result.save(destination/'result.png')
    else:
        source=fit_image(source,args.render_side)
        if args.depth_map is not None:
            depth=np.load(args.depth_map,allow_pickle=False)
            near_map(depth) # reject nonfinite / incorrectly shaped caches before resize
            with Image.open(args.input) as original:expected=ImageOps.exif_transpose(original).size
            if depth.shape!=(expected[1],expected[0]):raise ValueError('Cached depth shape must match the original input image')
            depth=resize_map(depth,source.size)
        else:depth=runner(source,max_side=args.max_side)
        near=near_map(depth)
        frames=max(2,round(args.duration*args.fps))
        if args.task=='parallax':
            if args.motion+8>=min(source.size):raise ValueError('Motion is too large for this image')
            result,_=parallax_frame(source,near,args.motion,0,crop=math.ceil(args.motion*.5)+2)
            result.save(destination/'result.png')
            formats=list(dict.fromkeys(args.depth_formats))
            records=write_animations([destination/f'result.{ext}' for ext in formats],parallax_frames(source,near,frames,args.motion),args.fps)
            details.update(max_uncovered_fraction=max(row['uncovered_fraction'] for row in records),
                fill='local colour diffusion, no learned hidden-background reconstruction',frames=frames,fps=args.fps,
                animation_files=[f'result.{ext}' for ext in formats])
        elif args.task=='focus':
            result=focus_frame(source,near,0,args.aperture);result.save(destination/'result.png')
            focus_frame(source,near,1,args.aperture).save(destination/'near-focus.png')
            formats=list(dict.fromkeys(args.depth_formats))
            records=write_animations([destination/f'result.{ext}' for ext in formats],focus_frames(source,near,frames,args.aperture),args.fps)
            details.update(frames=frames,fps=args.fps,
                animation_files=[f'result.{ext}' for ext in formats],
                limitation='Artistic depth-layer blur of one fixed RGB image; a second hidden subject is not guaranteed.')
        else:
            texture=None
            if args.texture is not None:
                with Image.open(args.texture) as image:texture=ImageOps.exif_transpose(image).convert('RGB')
            period=args.period if args.period is not None else max(8,min(source.width//3,round(source.width*.09)))
            result,record=stereogram(near,period,args.seed,texture)
            result.save(destination/'result.png');details.update(record)
            decoded=decode_stereogram(result,period)
            Image.fromarray(np.uint8(decoded*255)).save(destination/'decoded-depth.png')
            margin=period
            valid=(slice(None),slice(margin,-margin))
            from visual_anagrams.depth_anagram import correlation
            details['decoded_depth_correlation']=correlation(decoded[valid],near[valid])
            details['limitation']='Depth constraints are encoded; human stereo fusion still needs visual checking. Texture is repeated, not an independent complete colour subject.'
    source.save(destination/'input.png')
    save_depth(depth,destination)
    # save_depth writes its colour preview to result.png; retain the actual effect.
    (destination/'result.png').replace(destination/'depth-preview.png')
    result.save(destination/'result.png')
    details.update(seconds=time.perf_counter()-started,output_size=list(result.size))
    (destination/'run.json').write_text(json.dumps(details,indent=2),encoding='utf-8')
    print(f'Saved {destination/"result.png"}',flush=True)
    return details


def run_depth_task(args,source):
    from datetime import datetime
    import hashlib
    import json
    import torch
    from visual_anagrams.iris_tasks import DOWNSTREAM_REVISION,resolve_task_model,load_task
    name=args.name or f'{args.task}-{datetime.now():%Y%m%d-%H%M%S-%f}'
    output=Path(args.save_dir)/name
    if output.exists():raise ValueError(f'Output folder already exists: {output}; choose a new name')
    if args.device=='cuda' and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
        raise ValueError('Iris depth on CUDA requires a BF16-capable GPU')
    torch.set_num_threads(min(6,torch.get_num_threads()))
    directory=None;runner=None
    if args.depth_map is None:
        print('Loading Iris INT8 depth (no text encoder)',flush=True)
        directory=resolve_task_model('depth',args.model_path)
        runner=load_task('depth',directory,args.device)
    output.mkdir(parents=True,exist_ok=False)
    metadata=dict(settings={key:str(value) if isinstance(value,Path) else value for key,value in vars(args).items()},
        depth_revision=DOWNSTREAM_REVISION if directory is not None and args.model_path is None else None,
        model_manifest=json.loads((directory/'manifest.json').read_text()) if directory is not None else None,
        inputs={key:dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            for key in ('input','target','texture','depth_map') if (path:=getattr(args,key)) is not None})
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    if args.device=='cuda':torch.cuda.reset_peak_memory_stats()
    if args.task=='parallax_video':
        from visual_anagrams.video_parallax import render_video_parallax
        details=render_video_parallax(runner,args,source,output/f'{args.seed:04}')
    else:details=run_experiment(runner,args,source,output/f'{args.seed:04}')
    details['peak_allocated_gib']=torch.cuda.max_memory_allocated()/2**30 if args.device=='cuda' else None
    (output/f'{args.seed:04}'/'run.json').write_text(json.dumps(details,indent=2),encoding='utf-8')

