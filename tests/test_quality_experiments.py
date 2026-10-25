"""Math, routing and UI checks for opt-in experiments; no image-model downloads."""

import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from visual_anagrams.balancing import AdaptiveBalancer
from visual_anagrams.experiment_options import ExperimentOptions, GUI_DEFAULTS, options_from_args
from visual_anagrams.iris import PromptEncoding, sample_iris
from visual_anagrams.negative_planner import NegativeRequest, parse_subjects, plan_negatives
from visual_anagrams.quality_experiments import ExperimentalFlow, component_previews, project_conflicts, scheduled_guidance
from visual_anagrams.views import get_views


@pytest.fixture(scope='module',autouse=True)
def limit_threads():
    previous=torch.get_num_threads();torch.set_num_threads(4)
    yield
    torch.set_num_threads(previous)


class Scores:
    def __init__(self,matrix=None):self.matrix=matrix or [[.6,.61],[.1,.5]]
    def score_matrix(self,*_):return self.matrix
    def score_images(self,*_):return self.matrix
    def __call__(self,*_):return [self.matrix[i][i] for i in range(2)]


class ConditioningModel(nn.Module):
    def __init__(self):super().__init__();self.anchor=nn.Parameter(torch.zeros(()));self.cfg=SimpleNamespace(in_channels=3,patch_size=1);self.calls=0
    def forward(self,x,t,y,y_mask=None):
        self.calls+=1
        value=y if y.ndim==4 else y.flatten(1).mean(1)[:,None,None,None].expand_as(x)
        return SimpleNamespace(x=value+.01*x)


def encoding(values):
    return PromptEncoding(torch.as_tensor(values,dtype=torch.float32).reshape(-1,1,1),torch.ones(len(values),1))


def test_neutral_options_and_explicit_overrides_keep_the_old_sampler():
    from generate import build_parser
    args=build_parser().parse_args(['--name','test','--prompts','cat','dog','--views','identity','flip','--cross_negative','off'])
    assert not options_from_args(args).enabled
    assert ExperimentOptions().cli_args()==[]
    assert all(GUI_DEFAULTS[key]=='no' for key in ('exp_contrastive','exp_component','exp_conflict'))
    common=dict(size=8,num_inference_steps=4,device='cpu',progress=False,guidance_scale=3)
    def run(options=None):
        return sample_iris(ConditioningModel(),encoding([.1,.2]),encoding([0]),get_views(['identity','flip']),
            generator=torch.Generator().manual_seed(7),experiments=options,**common)
    torch.testing.assert_close(run(),run(ExperimentOptions()),rtol=0,atol=0)


@pytest.mark.parametrize('options,names,reduction,method,turbo',[
    (ExperimentOptions(contrastive=True),['identity','flip'],'mean','plain',False),
    (ExperimentOptions(cross_negative='qwen'),['identity','flip'],'mean','adaptive',True),
    (ExperimentOptions(component_guidance=True),['identity','flip'],'sum','plain',False),
    (ExperimentOptions(conflict_projection=True),['grayscale','color'],'sum','plain',False),
    (ExperimentOptions(guidance_schedule='middle'),['identity','flip'],'mean','syncvc',False),
    (ExperimentOptions(freeze_below=.2),['identity','flip'],'mean','plain',False),
])
def test_incompatible_experiments_are_rejected(options,names,reduction,method,turbo):
    with pytest.raises(ValueError):options.validate(names,reduction=reduction,quality_method=method,turbo=turbo)


def test_contrastive_balancing_detects_wrong_subject_and_freezes():
    views=get_views(['identity','flip']);x=torch.zeros(1,3,8,8)
    normal=AdaptiveBalancer(views,Scores())
    margin=AdaptiveBalancer(views,Scores(),contrastive=True,freeze_below=.1)
    normal(x,[x,x],.5,0);margin(x,[x,x],.5,0)
    assert normal.weights[0]<.5 and margin.weights[0]>.5
    assert margin.history[0]['margins']==pytest.approx([-.01,.4])
    before=margin.weights.clone();margin(x,[x,x],.05,2)
    torch.testing.assert_close(margin.weights,before)
    assert margin.history[-1]['scores'] is None


def test_conflict_projection_is_symmetric_bounded_and_does_not_edit_agreement():
    left=torch.tensor([1.,0.]);right=torch.tensor([-1.,1.])
    projected,record=project_conflicts([left,right],1.)
    assert record['projected'] and torch.dot(*projected)>=0
    reversed_values,_=project_conflicts([right,left],1.)
    torch.testing.assert_close(projected[0],reversed_values[1])
    for before,after in zip([left,right],projected):assert after.norm()<=before.norm()+1e-7
    unchanged,_=project_conflicts([left,left],1.)
    torch.testing.assert_close(unchanged[0],left,rtol=0,atol=0)
    zero,_=project_conflicts([left*0,right*0],1.)
    assert all(torch.isfinite(v).all() for v in zero)


def test_conflict_projection_keeps_the_shared_reference_prediction():
    model=ConditioningModel();x=torch.zeros(1,3,8,8)
    flow=ExperimentalFlow(ExperimentOptions(conflict_projection=True,conflict_strength=1),
        get_views(['identity','flip']),'mean',3)
    # Guided fields are -8 and 16, with shared reference 10. Projecting the
    # entire fields would erase the reference; projecting residuals returns 10.
    result=flow.predict(model,x,torch.tensor([500.]),encoding([4.,12.]),encoding([10.]),None,None,0,1000)
    torch.testing.assert_close(result,torch.full_like(x,10),atol=1e-5,rtol=0)


def test_guidance_schedule_is_smooth_in_flow_noise_units():
    options=ExperimentOptions(guidance_schedule='middle')
    assert scheduled_guidance(3,1,options)==1 and scheduled_guidance(3,0,options)==1
    assert scheduled_guidance(3,.5,options)==3
    assert scheduled_guidance(3,.125,options)==pytest.approx(2)
    assert scheduled_guidance(3,.875,options)==pytest.approx(2)
    assert scheduled_guidance(3,.100001,options)==pytest.approx(1,abs=1e-7)


@pytest.mark.parametrize('strength',[0.,1.,.5])
def test_cross_negative_reference_has_the_expected_affine_coefficients(strength):
    # Two conditional predictions + shared negatives; full-strength cross mode can skip the shared negatives.
    model=ConditioningModel();x=torch.zeros(1,3,8,8)
    options=ExperimentOptions(cross_negative='manual',cross_strength=strength)
    flow=ExperimentalFlow(options,get_views(['identity','flip']),'mean',3)
    result=flow.predict(model,x,torch.tensor([500.]),encoding([1.,2.]),encoding([.25]),encoding([2.,1.]),None,0,1000)
    references=[(1-strength)*.25+strength*other for other in [2.,1.]]
    expected=sum(ref+3*(pos-ref) for ref,pos in zip(references,[1.,2.]))/2
    torch.testing.assert_close(result,torch.full_like(x,expected))
    assert model.calls==(6 if 0<strength<1 else 4)


def test_component_guidance_preserves_shared_reference_when_prompts_agree():
    # Even unequal gains must leave the shared prior unchanged when both residuals are zero.
    model=ConditioningModel();x=torch.zeros(1,3,64,64)
    base=torch.randn(1,3,64,64,generator=torch.Generator().manual_seed(2))*.1
    positive=PromptEncoding(base.repeat(2,1,1,1),torch.ones(2,1))
    negative=PromptEncoding(base,torch.ones(1,1))
    flow=ExperimentalFlow(ExperimentOptions(component_guidance=True),get_views(['grayscale','color']),'sum',3,
        Scores([[.1,0],[0,.8]]))
    result=flow.predict(model,x,torch.tensor([500.]),positive,negative,None,None,0,1000)
    torch.testing.assert_close(result,base,rtol=0,atol=0)
    assert model.calls==3 and flow.component.gains[0]>1 and flow.component.gains[1]<1


@pytest.mark.parametrize('names',[['low_pass','high_pass'],['grayscale','color'],['motion','motion_res']])
def test_component_scorer_sees_correct_observable_filters(names):
    x=torch.randn(3,64,64,generator=torch.Generator().manual_seed(3))*.1
    views=get_views(names)
    expected=views[0].inverse_view(x.clone())
    images=component_previews(x,views)
    import numpy as np
    expected=expected.clamp(-1,1).add(1).mul(127.5).round().byte().permute(1,2,0).numpy()
    assert np.abs(np.asarray(images[0]).astype(float)-expected).max()<=1
    original=x.add(1).mul(127.5).round().byte().permute(1,2,0).numpy()
    assert np.array_equal(np.asarray(images[1]),original)


def test_negative_subject_extraction_removes_shared_context_and_rejects_invention():
    prompts=['a portrait of a cat in moonlight','a portrait of a dog in moonlight']
    plan=plan_negatives(prompts,'subjects')
    assert plan['view_negative_prompts']==['dog','cat']
    assert parse_subjects('```json\n{"subjects":["cat","dog"]}\n```',prompts)==['cat','dog']
    for text in ['{"subjects":["horse","dog"]}','{"subjects":["portrait","dog"]}']:
        with pytest.raises(ValueError):parse_subjects(text,prompts)


def test_qwen_planning_and_encoding_share_one_encoder(monkeypatch):
    import iris3b.text.qwen3_vl as qwen
    import huggingface_hub
    import visual_anagrams.negative_planner as planner
    from visual_anagrams.iris import encode_prompts
    calls=[]
    class Encoder:
        def __init__(self,*args,**kwargs):calls.append('load')
        def to(self,device):return self
        def encode(self,prompts):
            calls.append(prompts[0]);return SimpleNamespace(embeddings=torch.full((1,4,3),float(len(prompts[0]))),mask=torch.ones(1,4))
    monkeypatch.setattr(qwen,'Qwen3VLTextEncoder',Encoder)
    monkeypatch.setattr(huggingface_hub,'snapshot_download',lambda *a,**kw:'cached-qwen')
    monkeypatch.setattr(planner,'greedy_decode',lambda *a,**kw:'{"subjects":["Joker","Batman"]}')
    prompts=['Joker smiling','Batman in a black cowl']
    request=NegativeRequest('qwen',prompts)
    positive,negative=encode_prompts(SimpleNamespace(text_encoder=SimpleNamespace()),prompts,device='cpu',negative_request=request)
    assert calls==['load',*prompts,'','Batman','Joker']
    assert negative.embeddings.sum()==0 and positive.embeddings.shape[0]==2
    assert request.encoding.embeddings[:,0,0].tolist()==[6,5]


def test_gui_defaults_routing_and_mode_compatibility(tmp_path):
    from generate import build_parser
    from visual_anagrams.gui import AnagramApp,build_command
    app=AnagramApp(settings_path=tmp_path/'ui.json');app.withdraw()
    try:
        from visual_anagrams.experiment_options import context_options,gui_values
        expected=gui_values(context_options(turbo=False,views=['identity','rotate_cw'],reduction='mean',quality_method='adaptive'))
        assert all(app.form[k].get()==v for k,v in expected.items())
        app.form['output'].set(str(tmp_path))
        default_command,_=build_command({k:v.get() for k,v in app.form.items()},[r.values() for r in app.rows])
        assert '--negative_fallback' in default_command
        for key,value in dict(exp_contrastive='yes',exp_negative='Manual per-view negatives',
                exp_manual_negatives='dog | cat',exp_schedule='middle',exp_conflict='yes',exp_freeze='.1').items():app.form[key].set(value)
        app._update_experiments()
        command,_=build_command({k:v.get() for k,v in app.form.items()},[r.values() for r in app.rows])
        args=build_parser().parse_args(command[3:])
        assert args.contrastive_balancing and args.conflict_projection and args.guidance_schedule=='middle'
        assert args.view_negative_prompts==['dog','cat']
        app.form['mode'].set('Turbo · r2-1000 (fast)');app.mode_changed()
        assert app.form['exp_negative'].get()=='Off' and app.form['exp_schedule'].get()=='constant'
        assert app.experiment_widgets['exp_conflict'].instate(['disabled'])
        app.form['mode'].set('Standard Iris');app.mode_changed()
        assert app.form['exp_negative'].get()=='Manual per-view negatives'
        for row,label in zip(app.rows,['Grayscale component','Color component']):row.view.set(label);row._changed()
        assert app.form['exp_negative'].get()=='Off' and app.form['exp_contrastive'].get()=='no'
        assert not app.experiment_widgets['exp_component'].instate(['disabled'])
        app.form['exp_component'].set('yes');app._update_experiments()
        command,_=build_command({k:v.get() for k,v in app.form.items()},[r.values() for r in app.rows])
        args=build_parser().parse_args(command[3:])
        assert args.component_guidance and args.reduction=='sum' and args.quality_method=='plain'
    finally:app.destroy()


def test_context_defaults_can_be_overridden_in_cli_and_gui(tmp_path,monkeypatch):
    from generate import build_parser
    from visual_anagrams.gui import AnagramApp,build_command
    import visual_anagrams.quality_defaults as policy
    monkeypatch.setattr(policy,'STANDARD_GEOMETRIC',dict(cross_negative='qwen',cross_strength=.25,
        guidance_schedule='middle',guidance_low=.3))
    base=['--name','test','--prompts','cat','dog','--views','identity','flip']
    assert options_from_args(build_parser().parse_args(base)).cross_negative=='qwen'
    assert options_from_args(build_parser().parse_args(base)).cross_strength==.25
    assert options_from_args(build_parser().parse_args(base)).guidance_low==.3
    explicit=options_from_args(build_parser().parse_args(base+['--cross_negative','off','--guidance_schedule','constant']))
    assert explicit.cross_negative=='off' and explicit.guidance_schedule=='constant'
    assert not options_from_args(build_parser().parse_args(base+['--turbo'])).enabled
    app=AnagramApp(settings_path=tmp_path/'ui.json');app.withdraw()
    try:
        assert app.form['exp_negative'].get()=='Qwen subject extraction'
        app.form['output'].set(str(tmp_path))
        app.form['exp_negative'].set('Off');app.form['exp_schedule'].set('constant');app._update_experiments()
        command,_=build_command({k:v.get() for k,v in app.form.items()},[r.values() for r in app.rows])
        parsed=build_parser().parse_args(command[3:])
        assert not options_from_args(parsed).enabled
        app.form['mode'].set('Turbo · r2-1000 (fast)');app.mode_changed()
        assert app.form['exp_negative'].get()=='Off'
        app.form['mode'].set('Standard Iris');app.mode_changed()
        assert app.form['exp_negative'].get()=='Off' and app.form['exp_schedule'].get()=='constant'
    finally:app.destroy()


def test_tested_component_defaults_are_scoped_and_user_can_turn_them_off(tmp_path):
    from generate import build_parser
    from visual_anagrams.gui import AnagramApp,build_command
    parser=build_parser()
    base=['--name','test','--prompts','skull','rose','--views','grayscale','color']
    assert options_from_args(parser.parse_args(base)).component_guidance
    assert not options_from_args(parser.parse_args(base+['--no_component_guidance'])).enabled
    assert not options_from_args(parser.parse_args(base+['--turbo'])).enabled
    motion=parser.parse_args(['--name','test','--prompts','ship','moon','--views','motion','motion_res'])
    assert not options_from_args(motion).component_guidance
    app=AnagramApp(settings_path=tmp_path/'ui.json');app.withdraw()
    try:
        app.form['output'].set(str(tmp_path))
        for row,label in zip(app.rows,['Grayscale component','Color component']):row.view.set(label);row._changed()
        assert app.form['exp_component'].get()=='yes'
        app.form['exp_component'].set('no');app._update_experiments()
        command,_=build_command({k:v.get() for k,v in app.form.items()},[r.values() for r in app.rows])
        assert not options_from_args(parser.parse_args(command[3:])).component_guidance
    finally:app.destroy()


def test_cli_saves_resolved_negatives_and_per_sample_experiment_diagnostics(tmp_path,monkeypatch):
    from generate import main
    import visual_anagrams.iris as iris
    import visual_anagrams.balancing as balancing
    (tmp_path/'manifest.json').write_text('{}')
    monkeypatch.setattr(iris,'resolve_model',lambda _:tmp_path)
    monkeypatch.setattr(iris,'read_config',lambda _:SimpleNamespace(flow=SimpleNamespace(prediction='v',num_train_timesteps=1000)))
    def encode(*a,negative_request=None,**kw):
        negative_request.result=dict(mode='qwen',subjects=['cat','dog'],view_negative_prompts=['dog','cat'])
        negative_request.encoding=encoding([.2,.1])
        return encoding([.1,.2]),encoding([0])
    monkeypatch.setattr(iris,'encode_prompts',encode)
    monkeypatch.setattr(iris,'load_model',lambda *a,**kw:ConditioningModel())
    monkeypatch.setattr(balancing,'CleanPreviewScorer',lambda _:Scores())
    sample=iris.sample_iris
    monkeypatch.setattr(iris,'sample_iris',lambda *a,**kw:sample(*a,size=8,progress=False,**kw))
    main(['--name','test','--save_dir',str(tmp_path),'--prompts','cat','dog','--views','identity','flip',
        '--num_samples','2','--num_inference_steps','4','--device','cpu','--text_encoder_device','cpu',
        '--contrastive_balancing','--cross_negative','qwen','--guidance_schedule','middle','--conflict_projection'])
    meta=json.loads((tmp_path/'test/metadata.json').read_text())
    assert meta['resolved_negative_plan']['view_negative_prompts']==['dog','cat']
    for seed in [0,1]:
        data=json.loads((tmp_path/f'test/{seed:04}/quality.json').read_text())
        assert data['balancing']['contrastive']
        assert data['balancing']['steps'][0]['weights']==[.5,.5]
        assert len(data['experiments']['steps'])==4


@pytest.mark.parametrize('flags,message,recover',[
    ([], 'Qwen subject extraction failed: invalid subject', True),
    (['--cross_negative','qwen'], 'Qwen subject extraction failed: invalid subject', False),
    (['--cross_negative','qwen','--negative_fallback'], 'Qwen subject extraction failed: invalid subject', True),
    ([], 'Embedding dimensions do not match', False),
])
def test_cli_planning_fallback_is_recorded_and_does_not_hide_encoding_errors(tmp_path,monkeypatch,flags,message,recover):
    from generate import main
    import visual_anagrams.iris as iris
    import visual_anagrams.balancing as balancing
    (tmp_path/'manifest.json').write_text('{}')
    monkeypatch.setattr(iris,'resolve_model',lambda _:tmp_path)
    monkeypatch.setattr(iris,'read_config',lambda _:SimpleNamespace(flow=SimpleNamespace(prediction='v',num_train_timesteps=1000)))
    calls=[]
    def encode(*a,negative_request=None,**kw):
        calls.append('plan' if negative_request else 'encode')
        if negative_request is not None:raise ValueError(message)
        return encoding([.1,.2]),encoding([0])
    def load(*a,**kw):
        calls.append('load')
        return ConditioningModel()
    monkeypatch.setattr(iris,'encode_prompts',encode)
    monkeypatch.setattr(iris,'load_model',load)
    monkeypatch.setattr(balancing,'CleanPreviewScorer',lambda _:Scores())
    sample=iris.sample_iris
    monkeypatch.setattr(iris,'sample_iris',lambda *a,**kw:sample(*a,size=8,progress=False,**kw))
    arguments=['--name','test','--save_dir',str(tmp_path),'--prompts','cat','dog','--views','identity','flip',
        '--num_inference_steps','4','--device','cpu','--text_encoder_device','cpu',*flags]
    if not recover:
        with pytest.raises(ValueError,match=message):main(arguments)
        assert calls==['plan']
        return
    main(arguments)
    assert calls==['plan','encode','load']
    meta=json.loads((tmp_path/'test/metadata.json').read_text())
    assert meta['cross_negative']==meta['quality_experiments']['cross_negative']=='off'
    assert meta['resolved_negative_plan']==dict(mode='off',requested_mode='qwen',fallback=True,reason=message)
    assert (tmp_path/'test/0000/sample_8.png').exists()


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
@pytest.mark.parametrize('component',[False,True])
def test_real_tiny_iris_cuda_runs_geometric_and_component_experiments(component):
    from iris3b.config import ModelConfig,PixelStageConfig
    from iris3b.models.dit import IrisDiT
    config=ModelConfig(hidden_size=64,depth=2,dual_depth=1,num_heads=4,num_kv_heads=2,
        patch_size=4,text_dim=32,text_len=4,text_adapter='linear',
        pixel=PixelStageConfig(depth=1,hidden_size=4,attn_hidden_size=32,num_heads=4))
    torch.manual_seed(19)
    model=IrisDiT(config).eval().requires_grad_(False).cuda()
    nn.init.normal_(model.final_layer.linear.weight,std=.01)
    positive=PromptEncoding(torch.randn(2,4,32),torch.ones(2,4,dtype=torch.int64))
    negative=PromptEncoding(torch.zeros(1,4,32),torch.ones(1,4,dtype=torch.int64))
    names=['grayscale','color'] if component else ['identity','flip']
    views=get_views(names)
    options=(ExperimentOptions(component_guidance=True,guidance_schedule='middle') if component else
        ExperimentOptions(contrastive=True,cross_negative='manual',guidance_schedule='middle',conflict_projection=True))
    mixer=None if component else AdaptiveBalancer(views,Scores())
    diagnostics={}
    output=sample_iris(model,positive,negative,views,size=16,num_inference_steps=4,device='cuda',progress=False,
        experiments=options,prediction_mixer=mixer,component_scorer=Scores() if component else None,
        view_negative=positive if not component else None,experiment_diagnostics=diagnostics,
        generator=torch.Generator(device='cuda').manual_seed(19))
    assert output.shape==(1,3,16,16) and torch.isfinite(output).all()
    assert len(diagnostics['steps'])==4
