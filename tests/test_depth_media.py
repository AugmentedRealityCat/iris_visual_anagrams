"""Actual GIF/MP4 exports and embedded Tk playback, with no model downloads."""

import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import pytest

from visual_anagrams.depth_effects import write_animations


@pytest.fixture
def animations(tmp_path):
    frames=[(Image.new('RGB',(128,96),colour),{'index':index})
        for index,colour in enumerate(('red','green','blue'))]
    paths=[tmp_path/'result.mp4',tmp_path/'result.gif']
    write_animations(paths,iter(frames),fps=10)
    frames[0][0].save(tmp_path/'result.png')
    Image.new('RGB',(128,96),'white').save(tmp_path/'input.png')
    return {path.suffix:path for path in paths}


@pytest.mark.parametrize('task',['parallax','focus'])
@pytest.mark.parametrize('formats',[['gif'],['gif','mp4']])
def test_cached_depth_exports_gif_or_both_with_correct_timing(tmp_path,monkeypatch,task,formats):
    from iris_tools import main
    import visual_anagrams.iris_tasks as tasks
    monkeypatch.setattr(tasks,'load_task',lambda *a,**kw:pytest.fail('cached renders must not load weights'))
    y,x=np.indices((128,128));pixels=np.stack([x*2,y*2,(x+y)%256],-1).astype('uint8')
    source=tmp_path/'input.png';Image.fromarray(pixels).save(source)
    depth=tmp_path/'depth.npy';np.save(depth,x.astype(np.float32))
    main(['--task',task,'--input',str(source),'--depth_map',str(depth),'--device','cpu',
        '--save_dir',str(tmp_path),'--name','animation','--render_side','128','--duration','1','--fps','4',
        '--depth_formats',*formats])
    folder=tmp_path/'animation/0000'
    with Image.open(folder/'result.gif') as image:
        assert image.size==(128,128) and image.n_frames==4 and image.info['loop']==0
        durations=[]
        for index in range(image.n_frames):
            image.seek(index);image.load();durations.append(image.info['duration'])
        assert sum(durations)==1000
    assert (folder/'result.mp4').exists()==('mp4' in formats)
    if 'mp4' in formats:
        import imageio.v2 as imageio
        with imageio.get_reader(folder/'result.mp4',format='FFMPEG') as reader:
            assert sum(1 for _ in reader)==4
    record=json.loads((folder/'run.json').read_text())
    assert record['animation_files']==[f'result.{ext}' for ext in formats]
    assert not list(folder.glob('*.partial.*'))


def test_interrupted_export_preserves_existing_files_and_removes_temporaries(tmp_path):
    paths=[tmp_path/'result.gif',tmp_path/'result.mp4']
    for path in paths:path.write_bytes(b'previous output')
    def frames():
        yield Image.new('RGB',(32,32),'red'),{}
        raise RuntimeError('render stopped')
    with pytest.raises(RuntimeError,match='render stopped'):write_animations(paths,frames(),fps=10)
    assert all(path.read_bytes()==b'previous output' for path in paths)
    assert not list(tmp_path.glob('*.partial.*'))


def pump(app,condition,timeout=6):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        app.update()
        if condition():return
        time.sleep(.01)
    pytest.fail('Embedded animation did not reach the expected state')


@pytest.mark.parametrize('extension',['.mp4','.gif'])
def test_gui_loads_plays_loops_pauses_and_releases_animation(tmp_path,monkeypatch,animations,extension):
    from visual_anagrams.gui import AnagramApp
    from visual_anagrams import tools_gui
    app=AnagramApp(settings_path=tmp_path/'ui.json');app.withdraw()
    try:
        page=app.tools_page
        media=animations[extension]
        if extension=='.mp4':
            import shutil
            media=tmp_path/'video-without-poster.mp4';shutil.copyfile(animations[extension],media)
        monkeypatch.setattr(tools_gui.filedialog,'askopenfilename',lambda **kw:str(media))
        page.load_result()
        assert page.animation_path==media
        assert page.player.playing and page.play_button['text']=='Pause'
        pump(app,lambda:page.player.displayed_frames>=5) # longer than the three-frame clip: exercises looping
        assert page.preview_images[1].size==(128,96)
        assert page.preview_images[0].getpixel((0,0))==(255,255,255)
        page.toggle_playback()
        paused=page.player.displayed_frames
        assert not page.player.playing and page.play_button['text']=='Play'
        deadline=time.monotonic()+.18
        while time.monotonic()<deadline:app.update();time.sleep(.01)
        assert page.player.displayed_frames==paused
        page.toggle_playback();pump(app,lambda:page.player.displayed_frames>paused)
        session=page.player.session
        still=tmp_path/'still.png';Image.new('RGB',(128,96),'yellow').save(still)
        page.images=[still];page.picker.configure(values=['Still']);page.picker.current(0);page.show_result()
        assert page.player.session is None and not page.player.playing
        assert page.play_button.instate(['disabled'])
        assert page.preview_images[1].getpixel((0,0))==(255,255,0)
        session.thread.join(timeout=3)
        assert not session.thread.is_alive()
    finally:app.destroy()


def test_gui_discovers_generated_animation_and_stops_on_destroy(tmp_path,animations):
    from visual_anagrams.gui import AnagramApp
    # The result is in a sample subfolder, just like the generation worker writes.
    app=AnagramApp(settings_path=tmp_path/'ui.json');app.withdraw()
    try:
        page=app.tools_page;page.output_dir=tmp_path.parent
        page.images=[tmp_path/'result.png'];page.picker.configure(values=['Result']);page.picker.current(0)
        page.show_result()
        assert page.animation_path==animations['.mp4']
        pump(app,lambda:page.player.displayed_frames>0)
        session=page.player.session
    finally:app.destroy()
    session.thread.join(timeout=3)
    assert not session.thread.is_alive()


@pytest.mark.parametrize('label,formats',[('MP4 video',['mp4']),('Animated GIF',['gif']),('GIF + MP4',['gif','mp4'])])
def test_gui_output_format_routes_to_cli(tmp_path,label,formats):
    from visual_anagrams.tools_gui import build_tools_command
    from iris_tools import build_parser
    source=tmp_path/'source.png';Image.new('RGB',(128,128)).save(source)
    settings=dict(task='Depth parallax (experimental)',input=str(source),output=str(tmp_path),name='output',depth_format=label)
    command,_=build_tools_command(settings)
    assert build_parser().parse_args(command[3:]).depth_formats==formats
