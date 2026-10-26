"""Renderer geometry, depth optimization gradients, and GUI/CLI contracts."""

from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn

from visual_anagrams.depth_effects import (near_map,parallax_frame,focus_frame,
    stereogram,decode_stereogram,write_video)


def test_relative_depth_normalization_is_affine_invariant_and_handles_flat_depth():
    depth=np.arange(64,dtype=np.float32).reshape(8,8)
    np.testing.assert_allclose(near_map(depth),near_map(depth*3+11),atol=1e-6)
    assert near_map(depth)[0,0]==1 and near_map(depth)[-1,-1]==0
    assert np.all(near_map(np.zeros((8,8)))==.5)
    with pytest.raises(ValueError):near_map(np.full((8,8),np.nan))


def test_parallax_preserves_identity_and_resolves_near_surface_occlusion():
    pixels=np.zeros((32,64,3),dtype=np.uint8);pixels[:,:32,0]=255;pixels[:,32:,2]=255
    image=Image.fromarray(pixels);near=np.zeros((32,64),dtype=np.float32);near[:,:32]=1
    still,record=parallax_frame(image,near)
    np.testing.assert_array_equal(np.asarray(still),pixels)
    assert record['uncovered_fraction']==0
    moved,_=parallax_frame(image,near,dx=8)
    # The red foreground moves right and covers the left-moving blue background.
    np.testing.assert_array_equal(np.asarray(moved)[16,33],[255,0,0])
    assert np.asarray(moved).shape==pixels.shape
    with pytest.raises(ValueError):parallax_frame(image,near[:,:30])


def test_focus_uses_one_fixed_image_and_zero_aperture_is_identity():
    rng=np.random.default_rng(5)
    pixels=rng.integers(0,256,(48,64,3),dtype=np.uint8);image=Image.fromarray(pixels)
    near=np.ones((48,64),dtype=np.float32)
    np.testing.assert_array_equal(np.asarray(focus_frame(image,near,aperture=0)),pixels)
    np.testing.assert_array_equal(np.asarray(focus_frame(image,near,focus=1)),pixels)
    blurred=np.asarray(focus_frame(image,near,focus=0,aperture=8))
    assert blurred.std()<pixels.std()/2


def test_stereogram_encodes_two_depths_with_symmetric_correspondence():
    near=np.zeros((60,240),dtype=np.float32);near[12:48,75:165]=1
    image,record=stereogram(near,period=40,seed=7)
    assert record['near_separation']==32 and record['far_separation']==40
    pixels=np.asarray(image)
    assert np.array_equal(pixels[30,104],pixels[30,136])
    assert np.array_equal(pixels[5,100],pixels[5,140])
    decoded=decode_stereogram(image,40)
    assert decoded[20:40,95:145].mean()>.8
    assert decoded[3:9,70:170].mean()<.15
    same,_=stereogram(near,period=40,seed=7)
    np.testing.assert_array_equal(np.asarray(same),pixels)


class AnalyticDepth(nn.Module):
    def __init__(self):
        super().__init__();self.anchor=nn.Parameter(torch.zeros(()),requires_grad=False)
        self.pixel=nn.Linear(1,1,bias=False).requires_grad_(False)
    def forward(self,rgb,*args):return rgb[:,:1]*.8+rgb[:,1:2]*.2+self.anchor


class Runner:
    def __init__(self):self.model=AnalyticDepth();self.device=torch.device('cpu');self.patch=1;self.embeddings=None;self.mask=None
    def __call__(self,image,max_side=256):
        array=np.asarray(image,dtype=np.float32)/127.5-1
        return .8*array[:,:,0]+.2*array[:,:,1]


def test_depth_optimization_improves_target_shape_with_frozen_weights_and_bounded_pixels():
    from visual_anagrams.depth_anagram import optimize_depth_anagram
    y,x=np.indices((64,64));a=np.uint8(80+x);b=np.uint8(80+y)
    source=Image.fromarray(np.stack([a,a,a],-1));target=Image.fromarray(np.stack([b,b,b],-1))
    runner=Runner()
    result,_,_,_,details=optimize_depth_anagram(runner,source,target,steps=5,max_side=64,budget=.12)
    assert details['robustness']['clean']['correlation']>details['robustness']['initial']['correlation']+.1
    assert details['max_rgb_change']<=.12+1/255
    assert not runner.model.training or runner.model.anchor.grad is None
    assert runner.model.anchor.grad is None
    assert result.size==(64,64)


def test_real_tiny_iris_depth_switches_from_inference_to_input_gradients():
    from iris3b.config import ModelConfig,PixelStageConfig
    from iris3b.downstream.depth import IrisDepth,DepthPredictor
    from visual_anagrams.depth_anagram import optimize_depth_anagram
    cfg=ModelConfig(hidden_size=32,depth=2,dual_depth=1,num_heads=4,num_kv_heads=2,
        patch_size=4,text_dim=16,text_len=4,text_adapter='linear',
        pixel=PixelStageConfig(depth=1,hidden_size=4,attn_hidden_size=32,num_heads=4))
    torch.manual_seed(9)
    runner=DepthPredictor.__new__(DepthPredictor)
    runner.model=IrisDepth(cfg).eval().requires_grad_(False)
    nn.init.normal_(runner.model.pixel.final_layer.linear.weight,std=.1)
    runner.device=torch.device('cpu');runner.patch=4
    runner.embeddings=torch.randn(1,4,16);runner.mask=torch.ones(1,4,dtype=torch.bool)
    pixels=np.random.default_rng(3).integers(40,210,(64,64,3),dtype=np.uint8)
    source=Image.fromarray(pixels);target=Image.fromarray(pixels[:,::-1].copy())
    runner(source,max_side=64) # primes rotary tables under inference_mode
    _,_,_,_,details=optimize_depth_anagram(runner,source,target,steps=1,max_side=64,budget=.06)
    assert len(details['history'])==2
    assert not runner.model.pixel.training and all(p.grad is None for p in runner.model.parameters())


def test_cached_depth_video_cli_does_not_load_a_model(tmp_path,monkeypatch):
    from iris_tools import main
    import visual_anagrams.iris_tasks as tasks
    monkeypatch.setattr(tasks,'load_task',lambda *a,**kw:pytest.fail('cached rendering must not load a model'))
    image=tmp_path/'input.png';Image.new('RGB',(128,128),'green').save(image)
    depth=tmp_path/'depth.npy';np.save(depth,np.indices((128,128))[1].astype(np.float32))
    main(['--task','parallax','--input',str(image),'--depth_map',str(depth),'--name','cached',
        '--save_dir',str(tmp_path),'--device','cpu','--render_side','128','--duration','.5','--fps','4'])
    folder=tmp_path/'cached/0000'
    assert (folder/'result.mp4').stat().st_size>500
    import imageio.v2 as imageio
    frames=imageio.mimread(folder/'result.mp4',format='FFMPEG')
    assert len(frames)==2 and frames[0].shape[:2]==(128,128)
    assert json.loads((folder/'run.json').read_text())['frames']==2


@pytest.mark.parametrize('flags',[
    ['--task','depth_anagram'],['--task','parallax','--motion','nan'],
    ['--task','focus','--duration','999'],['--task','stereogram','--period','2'],
])
def test_invalid_depth_options_stop_before_model_loading(tmp_path,flags):
    from iris_tools import main
    image=tmp_path/'input.png';Image.new('RGB',(128,128)).save(image)
    with pytest.raises(SystemExit) as exc:main(['--input',str(image),*flags])
    assert exc.value.code==2


def test_depth_gui_routes_all_experiments_and_keeps_generation_defaults(tmp_path):
    from visual_anagrams.gui import AnagramApp
    from visual_anagrams.tools_gui import build_tools_command
    from depth_lab import DEPTH_TASKS
    from iris_tools import build_parser,validate_args
    image=tmp_path/'input.png';Image.new('RGB',(256,256)).save(image)
    app=AnagramApp(settings_path=tmp_path/'ui.json');app.withdraw()
    try:
        page=app.tools_page;original={key:page.form[key].get() for key in ('steps','guidance','mode')}
        for key in ('input','target','texture'):page.form[key].set(str(image))
        cached=tmp_path/'depth.npy';np.save(cached,np.zeros((256,256),dtype=np.float32))
        page.form['depth_map'].set(str(cached))
        page.form['output'].set(str(tmp_path))
        for label,task in DEPTH_TASKS.items():
            page.form['task'].set(label);page.task_changed()
            command,_=build_tools_command({key:value.get() for key,value in page.form.items()})
            parsed=validate_args(build_parser().parse_args(command[3:]));assert parsed.task==task
            if task=='depth_anagram':assert parsed.depth_map is None
            assert not page.fields['prompt'].grid_info()
            assert bool(page.fields['target'].grid_info())==(task=='depth_anagram')
        assert {key:page.form[key].get() for key in original}==original
    finally:app.destroy()
