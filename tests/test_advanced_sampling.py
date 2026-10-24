from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from generate import build_parser, main
from visual_anagrams.advanced_sampling import flow_euler, flow_prior_penalty, optimize_overlap, sample_advanced
from visual_anagrams.iris import Int8Linear, PromptEncoding
from visual_anagrams.method_options import MethodOptions
from visual_anagrams.optimization import attention_map, gradient_memory, subject_token_indices
from visual_anagrams.views import get_views


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_frozen_int8_input_backward_matches_dequantized_linear(dtype):
    layer = Int8Linear(8, 6, device="cpu")
    layer.qweight.copy_(torch.arange(48, dtype=torch.int8).reshape(6, 8) - 24)
    layer.scale.fill_(0.002)
    layer.bias.data.fill_(0.03)
    x = torch.randn(2, 3, 8, dtype=dtype).requires_grad_(True)
    reference = x.detach().clone().requires_grad_(True)
    weight = (layer.qweight.float() * layer.scale).to(dtype)
    expected = F.linear(reference, weight, layer.bias.to(dtype))
    actual = layer(x)
    torch.testing.assert_close(actual, expected)
    grad = torch.randn_like(actual)
    torch.testing.assert_close(torch.autograd.grad(actual, x, grad)[0],
                               torch.autograd.grad(expected, reference, grad)[0])


def test_flow_prior_and_update_are_schedule_consistent():
    image = torch.randn(1, 3, 8, 8)
    velocity = torch.randn_like(image)
    torch.testing.assert_close(flow_euler(image, velocity, .8, .6), image - .2 * velocity)
    control = torch.ones_like(image, requires_grad=True)
    loss = flow_prior_penalty(control, control * 2, .8, .6)
    assert loss > 0
    assert torch.isfinite(torch.autograd.grad(loss, control)[0]).all()
    assert flow_prior_penalty(control * 0, control * 0, .8, .6) == 0


def tiny_iris():
    from iris3b.config import ModelConfig, PixelStageConfig
    from iris3b.models.dit import IrisDiT
    cfg = ModelConfig(hidden_size=32, depth=2, dual_depth=0, num_heads=4, num_kv_heads=2,
                      patch_size=4, text_dim=16, text_len=4, text_adapter="linear",
                      pixel=PixelStageConfig(enabled=False))
    return IrisDiT(cfg).eval().requires_grad_(False)


def test_real_joint_attention_map_is_differentiable_with_checkpointing():
    model = tiny_iris()
    image = torch.randn(1, 3, 16, 16, requires_grad=True)
    with gradient_memory(model, False):
        value = attention_map(model, image, torch.tensor([800.]), torch.randn(1, 4, 16),
                              torch.ones(1, 4), [1], resolution=4)
        assert value.shape == (1, 1, 4, 4)
        gradient, = torch.autograd.grad(value.square().mean(), image)
        assert torch.isfinite(gradient).all() and gradient.abs().max() > 0
    assert not model.training and model.activation_checkpointing == "none"


def test_attention_update_respects_signal_scaled_trust_region():
    torch.manual_seed(7)
    model = tiny_iris()
    image = torch.randn(1, 3, 16, 16)
    positive = PromptEncoding(torch.randn(2, 4, 16), torch.ones(2, 4))
    result, record = optimize_overlap(model, image, .8, positive,
                    get_views(["identity", "rotate_180"]), [[1], [1]],
                    MethodOptions("anagram_mtl", cpu_offload=False))
    rms = (result - image).square().mean().sqrt()
    assert 0 < rms <= .05 * .8 * .2 + 1e-6
    assert record["attention_noise_step"] == pytest.approx(.008)


@pytest.mark.parametrize("method", ["anagram_mtl", "syncvc"])
def test_advanced_methods_execute_actual_optimization_and_record_diagnostics(method):
    model = tiny_iris()
    positive = PromptEncoding(torch.randn(2, 4, 16), torch.ones(2, 4))
    negative = PromptEncoding(torch.zeros(1, 4, 16), torch.ones(1, 4))
    diagnostics = {}
    image = sample_advanced(model, positive, negative, get_views(["identity", "rotate_180"]),
             options=MethodOptions(method, interval=1, cpu_offload=False),
             subject_positions=[[1], [1]], scorer=lambda *_: [.2, .3],
             num_inference_steps=3, size=16, device="cpu", progress=False, diagnostics=diagnostics,
             generator=torch.Generator().manual_seed(5))
    assert image.shape == (1, 3, 16, 16) and torch.isfinite(image).all()
    key = "gradient_rms" if method == "anagram_mtl" else "agreement_loss"
    assert any(key in row for row in diagnostics["steps"])
    assert not model.training and model.activation_checkpointing == "none"


@pytest.mark.parametrize("extra", [
    ["--turbo"], ["--reduction", "sum"], ["--optimization_interval", "0"],
    ["--method_strength", "nan"], ["--optimization_steps", "0"],
    ["--subjects", "owl"],
])
def test_invalid_advanced_options_fail_before_download(extra):
    with pytest.raises(SystemExit) as exc:
        main(["--name", "test", "--quality_method", "anagram_mtl", "--prompts", "owl", "dog",
              "--views", "identity", "rotate_180", *extra])
    assert exc.value.code == 2


def test_gui_forwards_method_settings_and_rejects_turbo(tmp_path):
    from visual_anagrams.gui import build_command
    settings = dict(name="test", output=str(tmp_path), style="a watercolor painting of", negative="",
                    steps="30", samples="1", seed="0", guidance="3", shift="4", order="2", reduction="auto",
                    device="cuda", text_device="cpu", quality_method="Anagram-MTL · Iris (experimental)",
                    subjects="owl | dog", optimization_interval="5", optimization_steps="1", method_strength=".05")
    rows = [dict(prompt="an owl", view="identity"), dict(prompt="a dog", view="rotate_180")]
    command, _ = build_command(settings, rows)
    args = build_parser().parse_args(command[3:])
    assert args.quality_method == "anagram_mtl" and args.subjects == ["owl", "dog"]
    assert args.optimization_interval == 5 and args.optimization_steps == 1
    settings.update(mode="Turbo · r2-1000 (fast)", guidance="1")
    with pytest.raises(ValueError, match="require Standard"):
        build_command(settings, rows)


def test_subject_spans_use_caption_offsets():
    class Tokenizer:
        def __call__(self, text, **kwargs):
            return dict(offset_mapping=[(m.start(), m.end()) for m in __import__('re').finditer(r'\S+', text)])
    assert subject_token_indices(Tokenizer(), ["a painting of an owl"], ["owl"]) == [[4]]
    with pytest.raises(ValueError):
        subject_token_indices(Tokenizer(), ["an owl"], ["dog"])


def test_gui_method_controls_and_turbo_restore():
    from visual_anagrams.gui import AnagramApp, GENERATION_MODES
    from visual_anagrams.method_options import QUALITY_METHODS

    app = AnagramApp()
    app.withdraw()
    labels = {method: label for label, method in QUALITY_METHODS.items()}
    try:
        assert app.form["quality_method"].get() == labels["plain"]
        assert all(not entry.grid_info() for entry in app.method_entries)
        app.form["quality_method"].set(labels["syncvc"])
        app.quality_changed()
        assert app.order_combo.instate(["disabled"])
        assert app.method_entries[3].instate(["disabled"])
        assert all(entry.grid_info() for entry in app.method_entries)
        app.form["quality_method"].set(labels["anagram_mtl"])
        app.quality_changed()
        assert app.method_entries[3].instate(["!disabled"])
        assert app.order_combo.instate(["readonly", "!disabled"])
        app.form["mode"].set(next(label for label, adapter in GENERATION_MODES.items() if adapter == "r2-1000"))
        app.mode_changed()
        assert app.form["quality_method"].get() == labels["plain"]
        assert app.quality_combo.instate(["disabled"])
        app.form["mode"].set("Standard Iris")
        app.mode_changed()
        assert app.form["quality_method"].get() == labels["anagram_mtl"]
        assert app.quality_combo.instate(["readonly", "!disabled"])
        for method in ("anagram_mtl", "syncvc"):
            app.handle_line(f"Iris {method}: 50%|#####| 15/30 [05:00<05:00]")
            assert app.progress.cget("maximum") == 30
            assert app.progress.cget("value") == 15
            assert "Denoising 15/30" in app.status.get()
    finally:
        app.destroy()
