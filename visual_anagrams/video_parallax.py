"""Stream a source video through Iris depth and a small artistic camera movement.

Temporal smoothing is a change-gated heuristic, not video-trained depth or optical flow.
"""

import math
from pathlib import Path

import numpy as np
from PIL import Image

from .depth_effects import parallax_frame,write_animations


def probe_video(path):
    import imageio_ffmpeg
    reader=imageio_ffmpeg.read_frames(str(path))
    try:
        meta=next(reader)
        fps=float(meta.get('fps') or 0)
        if not math.isfinite(fps) or fps<=0:raise ValueError('Cannot determine the input video frame rate')
        return dict(size=list(meta['size']),fps=fps,duration=float(meta.get('duration') or 0),
            audio_codec=meta.get('audio_codec'))
    except (OSError,RuntimeError,StopIteration) as exc:
        raise ValueError('Cannot decode the input video: '+str(exc).splitlines()[0]) from exc
    finally:reader.close()


def video_dimensions(size,max_side):
    width,height=size;scale=min(1,max_side/max(size))
    return tuple(max(2,round(side*scale/2)*2) for side in (width,height))


def validate_video_info(args,info):
    if info['fps']<1:raise ValueError('Video parallax requires an input frame rate of at least 1 FPS')
    if info['duration'] and args.video_start>=info['duration']:raise ValueError('Start time is beyond the end of the video')
    if args.motion+8>=min(video_dimensions(info['size'],args.render_side)):
        raise ValueError('Camera motion is too large for this video size')


def read_video_clip(path,info,*,start=0.,seconds=0.,fps=0.,max_side=768):
    """Decode a time-limited clip at a constant output FPS; retain source time order."""
    import imageio_ffmpeg
    rate=min(info['fps'],fps or info['fps'],60.)
    size=video_dimensions(info['size'],max_side)
    filters=f'fps={rate:.8f},scale={size[0]}:{size[1]}:flags=lanczos'
    options=['-vf',filters]
    if seconds:options+=['-t',str(seconds)]
    reader=imageio_ffmpeg.read_frames(str(path),input_params=['-ss',str(start)],output_params=options)
    try:
        metadata=next(reader)
        for index,pixels in enumerate(reader):
            yield index,Image.frombytes('RGB',metadata['size'],pixels),rate
    finally:reader.close()


class TemporalDepth:
    """Stabilize percentile range; reject pixel history where RGB changed or at cuts."""
    def __init__(self,smoothing=.25):
        if not math.isfinite(smoothing) or not 0<=smoothing<=.95:raise ValueError('Depth smoothing must be in [0, 0.95]')
        self.smoothing=smoothing;self.previous=None;self.rgb=None;self.bounds=None;self.resets=0

    def update(self,depth,image):
        depth=np.asarray(depth,dtype=np.float32)
        if depth.shape!=(image.height,image.width) or not np.isfinite(depth).all():
            raise ValueError('Video depth must be finite and match the frame')
        rgb=np.asarray(image.convert('RGB'),dtype=np.float32)/255
        bounds=np.percentile(depth,[2,98]).astype(np.float32)
        changed_size=self.rgb is not None and self.rgb.shape!=rgb.shape
        difference=None if self.rgb is None or changed_size else np.abs(rgb-self.rgb).mean(-1)
        cut=changed_size or (difference is not None and difference.mean()>.20)
        if cut:self.previous=None;self.bounds=None;self.resets+=1
        if self.bounds is not None:bounds=(1-self.smoothing)*bounds+self.smoothing*self.bounds
        low,high=bounds
        near=np.full_like(depth,.5) if high-low<1e-6 else np.clip((high-depth)/(high-low),0,1)
        if self.previous is not None and self.smoothing:
            confidence=np.exp(-difference/.06)
            weight=self.smoothing*confidence
            near=(1-weight)*near+weight*self.previous
        self.previous=near;self.rgb=rgb;self.bounds=bounds
        return near.astype(np.float32)


def mux_clip_audio(video,source,*,start,duration):
    """Preserve the selected clip's first audio track, re-encoded to AAC."""
    import subprocess
    import imageio_ffmpeg
    video=Path(video);temporary=video.with_name(video.stem+'.audio.partial.mp4')
    command=[imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-loglevel','error','-y',
        '-i',str(video),'-ss',str(start),'-i',str(source),'-map','0:v:0','-map','1:a:0?',
        '-c:v','copy','-c:a','aac','-b:a','192k','-t',str(duration),'-movflags','+faststart',str(temporary)]
    try:
        result=subprocess.run(command,capture_output=True,text=True,check=False)
        if result.returncode:raise RuntimeError('Audio export failed: '+result.stderr.strip()[:500])
        temporary.replace(video)
    finally:temporary.unlink(missing_ok=True)


def render_video_parallax(runner,args,info,destination):
    import json
    import time
    from .iris_tasks import save_depth
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=False)
    size=video_dimensions(info['size'],args.render_side)
    if args.motion+8>=min(size):raise ValueError('Camera motion is too large for this video size')
    if info['duration'] and args.video_start>=info['duration']:raise ValueError('Start time is beyond the end of the video')
    rate=min(info['fps'],args.video_fps or info['fps'],60.)
    remaining=max(0,info['duration']-args.video_start)
    selected=min(remaining,args.video_seconds) if args.video_seconds and remaining else args.video_seconds or remaining
    expected=round(selected*rate) if selected else 0
    tracker=TemporalDepth(args.depth_smoothing);crop=math.ceil(args.motion*.5)+2 if args.motion else 0
    started=time.perf_counter()
    def frames():
        for index,image,fps in read_video_clip(args.input,info,start=args.video_start,seconds=args.video_seconds,
                fps=args.video_fps,max_side=args.render_side):
            depth=runner(image,max_side=args.max_side)
            near=tracker.update(depth,image)
            # Camera motion is local to the selected clip, independent of its start time.
            phase=2*math.pi*(index/fps)/args.duration
            result,record=parallax_frame(image,near,args.motion*math.sin(phase),args.motion*.3*math.sin(2*phase),crop)
            if index==0:
                image.save(destination/'input.png');save_depth(depth,destination)
                (destination/'result.png').replace(destination/'depth-preview.png')
                result.save(destination/'result.png')
            print(f'Parallax video frame {index+1}'+(f'/{expected}' if expected else ''),flush=True)
            yield result,dict(frame=index,clip_time=index/fps,source_time=args.video_start+index/fps,**record)
    formats=list(dict.fromkeys(args.depth_formats))
    records=write_animations([destination/f'result.{ext}' for ext in formats],frames(),rate)
    audio=bool(info['audio_codec'] and not args.no_audio and 'mp4' in formats)
    if audio:mux_clip_audio(destination/'result.mp4',args.input,start=args.video_start,duration=len(records)/rate)
    details=dict(task='parallax_video',source=info,clip_start=args.video_start,requested_seconds=args.video_seconds,
        seconds_exported=len(records)/rate,frames=len(records),fps=rate,output_size=list(size),
        animation_files=[f'result.{ext}' for ext in formats],audio_preserved=audio,
        depth_smoothing=args.depth_smoothing,scene_resets=tracker.resets,
        max_uncovered_fraction=max(record['uncovered_fraction'] for record in records),
        seconds=time.perf_counter()-started,frame_records=records,
        limitation='Per-frame Iris depth with change-gated smoothing; not a video-trained or motion-compensated depth model. Large motion can expose halos and stretched background.')
    (destination/'run.json').write_text(json.dumps(details,indent=2),encoding='utf-8')
    print(f'Saved {destination/"result.png"}',flush=True)
    return details
