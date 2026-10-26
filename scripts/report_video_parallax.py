"""Validate and publish the one-second video-input parallax check."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image,ImageDraw
    from scripts.build_showcase_v2 import BG,INK,MUTED,font
    source=ROOT/'results/video-parallax-check-20261026/first-second/0000'
    out=ROOT/'examples/video-parallax';out.mkdir(parents=True,exist_ok=True)
    record=json.loads((source/'run.json').read_text())
    assert record['frames']==24 and record['seconds_exported']==1 and record['fps']==24
    hashes={}
    for name in ('result.mp4','result.gif','result.png','input.png','depth-preview.png'):
        digest=hashlib.sha256((source/name).read_bytes()).hexdigest()
        shutil.copyfile(source/name,out/name)
        assert hashlib.sha256((out/name).read_bytes()).hexdigest()==digest
        hashes[name]=dict(sha256=digest,bytes=(out/name).stat().st_size)
    with Image.open(out/'result.gif') as gif:
        assert gif.n_frames==24 and gif.size==(512,512) and gif.info['loop']==0
        duration=0
        for i in range(gif.n_frames):gif.seek(i);gif.load();duration+=gif.info['duration']
        assert duration==1000
    tiles=[]
    with imageio.get_reader(out/'result.mp4',format='FFMPEG') as reader:
        count=0
        for i,array in enumerate(reader):
            assert array.shape==(512,512,3)
            if i in (0,12,23):tiles.append((f'{i/24:.2f} seconds',Image.fromarray(array)))
            count+=1
        assert count==24
    canvas=Image.new('RGB',(960,410),BG);draw=ImageDraw.Draw(canvas)
    draw.text((14,12),'Video input: first second with added parallax',font=font(25,True),fill=INK)
    for i,(label,image) in enumerate(tiles):
        draw.text((i*320+10,62),label,font=font(19),fill=MUTED)
        canvas.paste(image.resize((300,300),Image.Resampling.LANCZOS),(i*320+10,95))
    canvas.save(out/'frames.jpg',quality=92)
    log_bytes=(ROOT/'.research/video-parallax-full-tests.log').read_bytes()
    log=log_bytes.decode('utf-16' if log_bytes.startswith(b'\xff\xfe') else 'utf-8')
    tests=re.search(r'(\d+) passed in ([\d.]+)s',log)
    assert tests and 'FAILED ' not in log,log[-1500:]
    data=dict(status='passed',real_model_run=record,
        metadata=json.loads((source.parent/'metadata.json').read_text()),artifacts=hashes,
        decoded_video_frames=count,decoded_gif_frames=24,gif_duration_ms=duration,
        tests_passed=int(tests[1]),test_seconds=float(tests[2]),
        code_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in [
            'visual_anagrams/video_parallax.py','depth_lab.py','visual_anagrams/tools_gui.py','iris_tools.py']})
    (out/'results.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
    lines=['# Parallax applied to an existing video','',
        '[Play the one-second MP4](result.mp4) · [Animated GIF](result.gif)','',
        '![Three output frames](frames.jpg)','',
        'The input is the earlier [three-second Batman parallax clip](../depth-lab/parallax-batman-result.mp4). '+
        'This new run decodes its first second, estimates Iris depth independently for every output frame, '+
        'and adds an 8px camera movement. It is a video-input/clip-timing check; it does not establish robustness '+
        'on arbitrary real-world subject motion. No new image generator or extra depth model was loaded.','',
        f"RTX 3060 12 GB: **{record['frames']} frames, {record['fps']:g} FPS, {record['seconds_exported']:g} second, 512 × 512**. "+
        f"Processing took {record['seconds']:.1f} seconds, excluding model loading, with {record['peak_allocated_gib']:.2f} GiB peak allocated VRAM. "+
        f"GIF: {hashes['result.gif']['bytes']/1e6:.2f} MB; MP4: {hashes['result.mp4']['bytes']/1e6:.2f} MB. "+
        'Both outputs were decoded completely and have the requested frame count/duration.','',
        '## GUI','',
        'In **Iris Tools → Task**, choose **Video parallax (experimental)** and browse for a video. '+
        'Set **Start time = 0**, **Clip length = 1** to process only the first second. '+
        'Start 10 and length 2 selects seconds 10–12; length 0 means the rest of the video. '+
        'Times accept fractional seconds. Choose MP4, GIF or both; the Result preview plays the animation with Play/Pause.','',
        '**Video output FPS** follows the source by default, capped at 60. A lower FPS reduces inference work. '+
        '**Camera loop** sets the added movement period, independently of the selected clip length. '+
        '**Keep source audio in MP4** retains the selected section of the first audio track as AAC; GIF and embedded playback are silent.','',
        'Depth smoothing uses recent estimates only where RGB changes are small and resets at detected scene cuts. '+
        'It is a heuristic, not optical-flow alignment or a video-trained depth model. '+
        'Lower smoothing for fast motion. Small camera moves are more reliable; newly exposed areas use local colour filling '+
        'and can show stretched edges or halos. Single-frame cached depth maps are disabled for video input.','',
        '## CLI','',
        '```powershell',
        'python iris_tools.py --task parallax_video --input video.mp4 --video_start 0 --video_seconds 1 --depth_formats mp4 gif',
        '```','',
        'Use `--video_fps 8` for a cheaper preview, `--depth_smoothing 0` to disable depth history, '+
        'and `--no_audio` to omit audio. `--duration` is the camera-loop period.','',
        f"**{int(tests[1])} automated tests passed.** Coverage includes first/later one-second clip selection, "+
        'source-time ordering, processing the remaining video, FPS reduction, actual AAC retention, '+
        'scene-cut resets, GUI clip controls, GIF/MP4 export, and embedded playback. '+
        'The source of this real-model example has no audio; audio retention is checked separately using a synthetic clip with an audio track.','',
        '[Complete settings, per-frame records, model provenance and validation](results.json). '+
        'Reproduce the real-model check with `python scripts/check_video_parallax.py`, then '+
        '`python scripts/report_video_parallax.py` to publish it.','']
    (out/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    print('Published',out,flush=True)


if __name__=='__main__':main()
