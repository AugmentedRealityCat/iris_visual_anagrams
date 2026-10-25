"""Production balancing must preserve the pilot algorithm and normal sampler schedules."""

import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from scripts.quality_methods import ResearchMixer, sample_research
from visual_anagrams.balancing import AdaptiveBalancer
from visual_anagrams.iris import PromptEncoding, sample_iris
from visual_anagrams.turbo import sample_turbo
from visual_anagrams.views import get_views


class SmallVelocity(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor=nn.Parameter(torch.zeros(()))
        self.cfg=SimpleNamespace(in_channels=3,patch_size=1)

    def forward(self,x,t,y,y_mask=None):
        return SimpleNamespace(x=.2*x+.001*y.mean()+.0001*t[:,None,None,None])


@pytest.mark.parametrize("turbo", [False, True])
@pytest.mark.parametrize("steps", [4, 8])
def test_production_balancing_matches_research_sampler(turbo, steps):
    positive = PromptEncoding(torch.linspace(.01,.9,24).reshape(2,4,3),torch.ones(2,4))
    negative = PromptEncoding(torch.zeros(1,4,3),torch.ones(1,4))
    views = get_views(["identity","rotate_180"])
    scorer = lambda *_: [.2,.6]
    expected = sample_research(SmallVelocity(),positive,negative,views,ResearchMixer("preview-balanced",views,scorer),
                              turbo=turbo,steps=steps,seed=42,device="cpu",size=8)
    mixer = AdaptiveBalancer(views,scorer)
    options = dict(size=8,num_inference_steps=steps,device="cpu",progress=False,prediction_mixer=mixer,
                   generator=torch.Generator().manual_seed(42))
    actual = (sample_turbo(SmallVelocity(),positive,views,**options) if turbo else
              sample_iris(SmallVelocity(),positive,negative,views,guidance_scale=3,**options))
    torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-5)
    assert len(mixer.history)==steps
    assert mixer.history[-1]["weights"][0] > .5


def test_adaptive_bounds_and_invalid_scores():
    views = get_views(["identity","rotate_180"])
    image = torch.zeros(1,3,8,8)
    mixer = AdaptiveBalancer(views,lambda *_: [-1,1])
    for step in range(30):
        mixer(image,[image+.1,image-.1],.5,step)
    assert all(.25 <= w <= .75 for w in mixer.weights)
    assert mixer.weights.sum()==pytest.approx(1)
    with pytest.raises(FloatingPointError):
        AdaptiveBalancer(views,lambda *_:[float("nan"),.3])(image,[image,image],.5,0)


@pytest.mark.parametrize("turbo", [False, True])
@pytest.mark.parametrize("method", ["adaptive", None, "plain"])
def test_cli_routes_balancing_resets_each_sample_and_saves_diagnostics(tmp_path, monkeypatch, turbo, method):
    from generate import main
    import visual_anagrams.balancing as balancing
    import visual_anagrams.iris as iris
    import visual_anagrams.turbo as turbo_module

    (tmp_path/"manifest.json").write_text("{}",encoding="utf-8")
    monkeypatch.setattr(iris,"resolve_model",lambda _:tmp_path)
    monkeypatch.setattr(iris,"read_config",lambda _:SimpleNamespace(flow=SimpleNamespace(prediction="v",num_train_timesteps=1000)))
    positive=PromptEncoding(torch.randn(2,4,3),torch.ones(2,4))
    negative=PromptEncoding(torch.zeros(1,4,3),torch.ones(1,4))
    monkeypatch.setattr(iris,"encode_prompts",lambda *args,**kwargs:(positive,None if turbo else negative))
    monkeypatch.setattr(iris,"load_model",lambda *args,**kwargs:SmallVelocity())
    calls=[]
    def scorer(prompts):
        calls.append(prompts)
        return lambda *_:[.1,.5]
    monkeypatch.setattr(balancing,"CleanPreviewScorer",scorer)
    normal,fast=iris.sample_iris,turbo_module.sample_turbo
    monkeypatch.setattr(iris,"sample_iris",lambda *args,**kwargs:normal(*args,size=8,progress=False,**kwargs))
    monkeypatch.setattr(turbo_module,"sample_turbo",lambda *args,**kwargs:fast(*args,size=8,progress=False,**kwargs))
    if turbo:
        monkeypatch.setattr(turbo_module,"resolve_turbo_adapter",lambda _:tmp_path)
        monkeypatch.setattr(turbo_module,"read_adapter_metadata",lambda _:dict(rank=64,alpha=64))
    main(["--name","adaptive","--save_dir",str(tmp_path),"--prompts","owl","dog","--views","identity","rotate_180",
          "--num_samples","2","--num_inference_steps","4","--device","cpu",
          *(["--quality_method",method] if method else []),
          "--text_encoder_device","cpu",*(["--turbo"] if turbo else [])])
    adaptive = method in ("adaptive", None)
    assert len(calls)==int(adaptive)
    for seed in [0,1]:
        path=tmp_path/f"adaptive/{seed:04}/quality.json"
        if not adaptive:
            assert not path.exists()
            continue
        diagnostic=json.loads(path.read_text())
        assert diagnostic["method"]=="adaptive"
        assert diagnostic["steps"][0]["weights"]==[.5,.5]
        assert diagnostic["steps"][-1]["weights"][0]>.5


def test_gui_adaptive_selection_survives_turbo_toggle_and_hides_gradient_controls(tmp_path):
    from generate import build_parser
    from visual_anagrams.gui import AnagramApp,GENERATION_MODES,QUALITY_METHODS,build_command

    app=AnagramApp(settings_path=tmp_path/"ui.json")
    app.withdraw()
    label=next(label for label,method in QUALITY_METHODS.items() if method=="adaptive")
    try:
        app.form["quality_method"].set(label)
        app.quality_changed()
        assert all(not entry.grid_info() for entry in app.method_entries)
        app.form["mode"].set(next(label for label,adapter in GENERATION_MODES.items() if adapter=="r2-1000"))
        app.mode_changed()
        assert app.form["quality_method"].get()==label
        assert app.quality_combo.instate(["readonly","!disabled"])
        command,_=build_command({k:v.get() for k,v in app.form.items()},[row.values() for row in app.rows])
        args=build_parser().parse_args(command[3:])
        assert args.turbo and args.quality_method=="adaptive" and args.num_inference_steps==4
        assert "--optimization_interval" not in command
        app.form["mode"].set("Standard Iris")
        app.mode_changed()
        assert app.form["quality_method"].get()==label
    finally:
        app.destroy()


def test_gui_scoped_turbo_default_and_explicit_classic_override(tmp_path):
    from visual_anagrams.gui import AnagramApp,GENERATION_MODES,QUALITY_METHODS,build_command
    from generate import build_parser

    app=AnagramApp(settings_path=tmp_path/"ui.json")
    app.withdraw()
    labels={method:label for label,method in QUALITY_METHODS.items()}
    turbo=next(label for label,adapter in GENERATION_MODES.items() if adapter=="r2-1000")
    try:
        assert app.form["quality_method"].get()==labels["adaptive"]
        app.form["mode"].set(turbo); app.mode_changed()
        assert app.form["quality_method"].get()==labels["adaptive"]
        app.form["quality_method"].set(labels["plain"]); app.quality_changed()
        command,_=build_command({k:v.get() for k,v in app.form.items()},[row.values() for row in app.rows])
        assert build_parser().parse_args(command[3:]).quality_method=="plain"
        app.form["mode"].set("Standard Iris"); app.mode_changed()
        app.form["mode"].set(turbo); app.mode_changed()
        assert app.form["quality_method"].get()==labels["plain"]
        app.form["mode"].set("Standard Iris"); app.mode_changed()
        app.turbo_quality_method=None
        app.add_view("a third prompt")
        app.form["mode"].set(turbo); app.mode_changed()
        assert app.form["quality_method"].get()==labels["plain"]
        app.form["mode"].set("Standard Iris"); app.mode_changed()
        app.remove_view(app.rows[-1]); app.turbo_quality_method=None
        for row,label in zip(app.rows,["Low frequencies","High frequencies"]):
            row.view.set(label); row._changed()
        app.form["mode"].set(turbo); app.mode_changed()
        assert app.form["quality_method"].get()==labels["plain"]
    finally:
        app.destroy()
