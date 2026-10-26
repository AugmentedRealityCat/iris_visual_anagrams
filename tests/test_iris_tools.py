"""Task-specific INT8 loaders, flow img2img and the independent desktop page."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn

from iris_tools import build_parser, main, validate_args
from visual_anagrams.iris import CODE_REVISION, Int8Linear, PromptEncoding, sample_iris
from visual_anagrams.iris_tasks import load_task, resolve_task_model, sample_img2img, save_depth
from visual_anagrams.tools_gui import TASKS, build_tools_command
from visual_anagrams.views import get_views


def tiny_export(tmp_path, task):
    from iris3b.config import ModelConfig, PixelStageConfig
    from iris3b.downstream.depth import IrisDepth
    from iris3b.models.dit import IrisDiT
    from omegaconf import OmegaConf
    from safetensors.torch import save_file

    cfg = ModelConfig(hidden_size=32, depth=2, dual_depth=0, num_heads=4, num_kv_heads=2,
                      patch_size=4, text_dim=16, text_len=4, text_adapter="linear",
                      pixel=PixelStageConfig(enabled=True, depth=1, hidden_size=8, attn_hidden_size=32, num_heads=4))
    model = (IrisDepth(cfg) if task == "depth" else IrisDiT(cfg)).eval()
    state = model.state_dict()
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Linear) and "modulation_cores." not in name:
            weight = state.pop(name + ".weight")
            scale = weight.abs().amax(1, keepdim=True).clamp_min(1e-8) / 127
            state[name + ".qweight"] = (weight / scale).round().clamp(-127,127).to(torch.int8)
            state[name + ".scale"] = scale.float()
    save_file(state, tmp_path / "model.safetensors")
    save_file(dict(embeddings=torch.randn(1,4,16), mask=torch.ones(1,4,dtype=torch.bool)), tmp_path / "empty_prompt.safetensors")
    settings = dict(name="depth") if task == "depth" else dict(name="restoration", tile=16, sigma=.5)
    OmegaConf.save(OmegaConf.create(dict(model=asdict(cfg), task=settings)), tmp_path / "config.yaml")
    files = [dict(path=p.name, size=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
             for p in tmp_path.iterdir()]
    (tmp_path / "manifest.json").write_text(json.dumps(dict(format="iris-rowwise-int8-v1", task="depth" if task=="depth" else "upscale",
                                              code_revision=CODE_REVISION, files=files)), encoding="utf-8")


@pytest.mark.parametrize("task", ["depth", "restore"])
def test_task_int8_loader_runs_real_upstream_recipe(tmp_path, task):
    tiny_export(tmp_path, task)
    folder = resolve_task_model(task, tmp_path)
    runner = load_task(task, folder, "cpu")
    assert any(isinstance(layer, Int8Linear) for layer in runner.model.modules())
    assert all(not p.requires_grad for p in runner.model.parameters())
    if task == "depth":
        output = runner(Image.new("RGB", (30,18), "gray"), max_side=32)
        assert output.shape == (18,30) and np.isfinite(output).all()
    else:
        output = runner(Image.new("RGB", (6,4), "gray"))
        assert output.size == (24,16)
        assert runner.completed_tiles == 2
    with pytest.raises(ValueError, match="checkpoint"):
        resolve_task_model("restore" if task == "depth" else "depth", tmp_path)
    with (tmp_path / "model.safetensors").open("ab") as stream:
        stream.write(b"bad")
    with pytest.raises(ValueError, match="Incomplete"):
        resolve_task_model(task, tmp_path)


def test_depth_exports_preserve_values_and_near_white_convention(tmp_path):
    depth = np.linspace(-1, 1, 96).reshape(8,12).astype(np.float32)
    details = save_depth(depth, tmp_path)
    np.testing.assert_array_equal(np.load(tmp_path / "depth.npy", allow_pickle=False), depth)
    with Image.open(tmp_path / "depth16.png") as image:
        array = np.array(image)
        assert image.size == (12,8) and array[0,0] == 65535 and array[-1,-1] == 0
    assert "not metres" in details["units"]
    with pytest.raises(FloatingPointError):
        save_depth(np.full((2,2),np.nan), tmp_path)


class ZeroVelocity(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()), requires_grad=False)
        self.cfg = type("Config", (), dict(patch_size=4, in_channels=3))()

    def forward(self, image, *args, **kwargs):
        return type("Output", (), dict(x=torch.zeros_like(image)))()


def test_img2img_strength_controls_initial_signal_and_noise():
    from iris3b.flow.solver import FlowDPMSolver

    model = ZeroVelocity()
    encoding = PromptEncoding(torch.zeros(1,4,16), torch.ones(1,4))
    initial = torch.full((1,3,8,12), .2)
    strength = .25
    t = strength * FlowDPMSolver.time_grid(3,4)[0]
    noise = torch.randn(initial.shape, generator=torch.Generator().manual_seed(17))
    output = sample_img2img(model, encoding, encoding, initial, strength=strength, steps=3,
               generator=torch.Generator().manual_seed(17), device="cpu", progress=False)
    torch.testing.assert_close(output, ((1-t)*initial+t*noise).clamp(-1,1))


def test_classic_and_turbo_support_rectangular_images():
    from visual_anagrams.turbo import sample_turbo
    model = ZeroVelocity()
    encoding = PromptEncoding(torch.zeros(1,4,16), torch.ones(1,4))
    for sampler in [lambda: sample_iris(model, encoding, encoding, get_views(["identity"]), size=(16,32),
                       num_inference_steps=2, device="cpu", progress=False),
                    lambda: sample_turbo(model, encoding, get_views(["identity"]), size=(16,32),
                       num_inference_steps=2, device="cpu", progress=False)]:
        output = sampler()
        assert output.shape == (1,3,16,32) and torch.isfinite(output).all()


@pytest.mark.parametrize("extra", [["--task","generate"], ["--task","depth"],
    ["--task","generate","--prompt","owl","--steps","0"],
    ["--task","generate","--prompt","owl","--turbo","--guidance","3"]])
def test_invalid_tool_arguments_stop_before_loading_models(extra):
    with pytest.raises(SystemExit) as exc:
        main(extra)
    assert exc.value.code == 2


def test_gui_pages_do_not_import_models():
    script = "from visual_anagrams.gui import AnagramApp; import sys; app=AnagramApp(); app.withdraw(); assert 'torch' not in sys.modules; assert len(app.pages.tabs())==2; app.destroy()"
    subprocess.run([sys.executable,"-c",script], check=True, timeout=20, capture_output=True)


def test_tools_page_builds_each_task_and_preserves_anagram_settings(tmp_path):
    from visual_anagrams.gui import AnagramApp
    app = AnagramApp()
    app.withdraw()
    try:
        page = app.tools_page
        original = {key: value.get() for key,value in app.form.items()}
        input_path = tmp_path / "input image.png"
        Image.new("RGB", (256,192)).save(input_path)
        page.form["input"].set(str(input_path))
        page.form['target'].set(str(input_path))
        page.form["output"].set(str(tmp_path))
        for label, task in TASKS.items():
            page.form["task"].set(label)
            page.task_changed()
            command, _ = build_tools_command({k:v.get() for k,v in page.form.items()})
            args = validate_args(build_parser().parse_args(command[3:]))
            assert args.task == task
            if task not in ('generate','img2img'):
                assert not page.fields["prompt"].grid_info()
            else:
                assert page.fields["prompt"].grid_info()
        page.form["task"].set("Image generation")
        page.task_changed()
        page.form["mode"].set("Turbo r2-1000 (fast)")
        page.mode_changed()
        assert page.form["steps"].get()=="4"
        page.form["task"].set("Image-to-image (experimental)")
        page.task_changed()
        assert page.form["steps"].get()=="30" and page.form["guidance"].get()=="3"
        assert {k:v.get() for k,v in app.form.items()} == original
    finally:
        app.destroy()


def test_tools_worker_displays_saved_results(tmp_path, monkeypatch):
    from visual_anagrams.gui import AnagramApp
    app = AnagramApp()
    app.withdraw()
    try:
        page = app.tools_page
        page.form["output"].set(str(tmp_path))
        destination = tmp_path / "iris-image/0000"
        script = tmp_path / "fake_tool.py"
        script.write_text("from pathlib import Path\nfrom PIL import Image\n"
            f"folder=Path({str(destination)!r}); folder.mkdir(parents=True)\n"
            "Image.new('RGB',(128,96),'blue').save(folder/'result.png')\n"
            "print('Saved '+str(folder/'result.png'),flush=True)\n", encoding="utf-8")
        start = page.job.start
        monkeypatch.setattr(page.job,"start",lambda _: start([sys.executable,"-u",str(script)]))
        page.run()
        assert app.generate_button.instate(["disabled"])
        deadline = time.monotonic()+10
        while page.running and time.monotonic()<deadline:
            app.update()
            time.sleep(.02)
        assert not page.running
        assert page.preview_images[1].size == (128,96)
        assert page.status.get()=="Finished · 1 result(s) saved"
        assert app.generate_button.instate(["!disabled"])
    finally:
        app.tools_page.job.cancel()
        app.destroy()


@pytest.mark.skipif(sys.platform != "win32", reason="Windows folder-opening integration")
def test_open_results_during_download_does_not_create_reserved_run_folder(tmp_path, monkeypatch):
    import os
    from visual_anagrams.gui import AnagramApp

    app = AnagramApp()
    app.withdraw()
    try:
        page = app.tools_page
        page.form["output"].set(str(tmp_path))
        pending = tmp_path / "pending-model-download"
        page.output_dir = pending
        opened = []
        monkeypatch.setattr(os, "startfile", lambda path: opened.append(Path(path)))
        page.open_results()
        assert opened == [tmp_path]
        assert not pending.exists()
    finally:
        app.destroy()
