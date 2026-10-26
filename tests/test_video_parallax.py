"""Frame-accurate clip selection, audio, per-frame depth, temporal reset and GUI routing."""

import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image
import pytest

from visual_anagrams.depth_effects import near_map,write_video
from visual_anagrams.video_parallax import TemporalDepth,probe_video,read_video_clip


@pytest.fixture
def source_video(tmp_path):
    import imageio_ffmpeg
    silent=tmp_path/'silent.mp4';source=tmp_path/'source.mp4'
    def frames():
        for colour in ('red','green','blue','yellow'):
            for _ in range(4):yield Image.new('RGB',(96,64),colour),{}
    write_video(silent,frames(),fps=4)
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-loglevel','error','-y',
        '-i',str(silent),'-f','lavfi','-i','sine=frequency=440:duration=4',
        '-map','0:v:0','-map','1:a:0','-c:v','copy','-c:a','aac','-shortest',str(source)],check=True)
    return source


class Depth:
    def __init__(self):self.calls=0
    def __call__(self,image,max_side):
        self.calls+=1
        return np.indices((image.height,image.width))[1].astype(np.float32)


@pytest.mark.parametrize('start,channel',[(0,0),(1,1)])
def test_cli_selects_one_second_and_keeps_audio(tmp_path,monkeypatch,source_video,start,channel):
    from iris_tools import main
    import visual_anagrams.iris_tasks as tasks
    model=tmp_path/'model';model.mkdir();(model/'manifest.json').write_text('{}')
    runner=Depth()
    monkeypatch.setattr(tasks,'resolve_task_model',lambda *a:model)
    monkeypatch.setattr(tasks,'load_task',lambda *a:runner)
    main(['--task','parallax_video','--input',str(source_video),'--video_start',str(start),'--video_seconds','1',
        '--name','clip','--save_dir',str(tmp_path),'--device','cpu','--motion','0','--render_side','128',
        '--depth_formats','gif','mp4'])
    folder=tmp_path/'clip/0000';record=json.loads((folder/'run.json').read_text())
    assert record['frames']==runner.calls==4 and record['seconds_exported']==1
    assert record['clip_start']==start and record['audio_preserved']
    assert [r['source_time'] for r in record['frame_records']]==[start+i/4 for i in range(4)]
    info=probe_video(folder/'result.mp4')
    assert info['audio_codec'] and abs(info['duration']-1)<.05
    output=[image for _,image,_ in read_video_clip(folder/'result.mp4',info)]
    assert len(output)==4
    for image in output:assert np.asarray(image).mean((0,1)).argmax()==channel
    with Image.open(folder/'result.gif') as image:
        assert image.n_frames==4
        assert image.info['loop']==0


def test_clip_length_zero_processes_the_rest_and_fps_reduction_uses_source_time(source_video):
    info=probe_video(source_video)
    frames=list(read_video_clip(source_video,info,start=2,seconds=0,fps=2,max_side=128))
    assert len(frames)==4 and all(fps==2 for _,_,fps in frames)
    assert np.asarray(frames[0][1]).mean((0,1)).argmax()==2 # blue at t=2
    assert np.asarray(frames[-1][1]).mean((0,1))[0]>100 # yellow at t=3.5


def test_temporal_depth_smooths_small_changes_and_resets_at_scene_cuts():
    rng=np.random.default_rng(3)
    depth=np.tile(np.linspace(-1,1,64,dtype=np.float32),(32,1))
    noisy=depth+rng.normal(0,.12,depth.shape).astype(np.float32)
    gray=Image.new('RGB',(64,32),'gray')
    tracker=TemporalDepth(.8);initial=tracker.update(depth,gray)
    smoothed=tracker.update(noisy,gray)
    assert np.abs(smoothed-initial).mean()<np.abs(near_map(noisy)-initial).mean()
    cut=tracker.update(noisy,Image.new('RGB',(64,32),'blue'))
    assert tracker.resets==1
    np.testing.assert_allclose(cut,near_map(noisy),atol=1e-6)


@pytest.mark.parametrize('flags',[
    ['--video_start','-1'],['--video_seconds','-1'],['--video_seconds','nan'],
    ['--video_fps','100'],['--depth_smoothing','1'],
])
def test_invalid_clip_options_stop_before_models(tmp_path,flags):
    from iris_tools import main
    fake=tmp_path/'video.mp4';fake.write_text('not a video')
    with pytest.raises(SystemExit) as exc:main(['--task','parallax_video','--input',str(fake),*flags])
    assert exc.value.code==2


def test_start_beyond_end_stops_before_loading_model(monkeypatch,source_video):
    from iris_tools import main
    import visual_anagrams.iris_tasks as tasks
    monkeypatch.setattr(tasks,'load_task',lambda *a:pytest.fail('invalid clip must not load weights'))
    with pytest.raises(SystemExit) as exc:
        main(['--task','parallax_video','--input',str(source_video),'--video_start','9'])
    assert exc.value.code==2


def test_gui_routes_first_second_and_hides_static_depth_cache(tmp_path):
    from iris_tools import build_parser
    from visual_anagrams.gui import AnagramApp
    from visual_anagrams.tools_gui import build_tools_command
    fake=tmp_path/'video.mp4';fake.write_text('GUI validates file paths; the worker probes the video')
    app=AnagramApp(settings_path=tmp_path/'ui.json');app.withdraw()
    try:
        page=app.tools_page
        for key,value in dict(task='Video parallax (experimental)',input=str(fake),video_start='0',video_seconds='1',
            video_fps='Source',keep_audio='no',output=str(tmp_path),depth_map='stale-cache.npy',depth_format='GIF + MP4').items():page.form[key].set(value)
        page.task_changed()
        command,_=build_tools_command({key:value.get() for key,value in page.form.items()})
        parsed=build_parser().parse_args(command[3:])
        assert parsed.task=='parallax_video' and parsed.video_start==0 and parsed.video_seconds==1
        assert parsed.no_audio and parsed.video_fps==0 and parsed.depth_map is None
        assert parsed.depth_formats==['gif','mp4']
        assert page.fields['video_start'].grid_info() and page.fields['video_seconds'].grid_info()
        assert not page.fields['depth_map'].grid_info() and not page.fields['fps'].grid_info()
        assert page.field_labels['input']['text']=='Input video'
        page.handle_line('Parallax video frame 3/12')
        assert page.progress['value']==3 and page.progress['maximum']==12
        page.form['task'].set('Depth parallax (experimental)');page.task_changed()
        assert not page.fields['video_seconds'].grid_info() and page.fields['fps'].grid_info()
    finally:app.destroy()
