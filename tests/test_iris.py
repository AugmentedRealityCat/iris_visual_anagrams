"""Sampler/quantization regressions without downloading model weights."""

from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from torch import nn
from torch.nn import functional as F

from generate import main
from visual_anagrams.iris import Int8Linear, PromptEncoding, load_int8, sample_iris
from visual_anagrams.views import get_views


class TinyModel(nn.Module):
    def __init__(self, image_prediction=False):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.cfg = SimpleNamespace(in_channels=3, patch_size=1)
        self.image_prediction = image_prediction
        self.calls = []

    def forward(self, x, t, y, y_mask=None):
        assert y_mask is not None
        self.calls.append((x.shape[0], y_mask.clone()))
        if self.image_prediction:
            result = y.clone()
        else:
            text = y.flatten(1).mean(1)[:, None, None, None]
            result = 0.25 * x + 0.001 * text + 0.00001 * t[:, None, None, None]
        return SimpleNamespace(x=result)


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("guidance", [1.0, 3.0])
def test_identity_matches_upstream_sampler(order, guidance):
    from iris3b.sampling import generate

    positive = PromptEncoding(torch.ones(1, 4, 3), torch.tensor([[1, 1, 0, 0]]))
    negative = PromptEncoding(torch.zeros(1, 4, 3), torch.tensor([[1, 0, 0, 0]]))
    encoder = SimpleNamespace(encode=lambda _: positive, null=lambda _: negative)
    expected = generate(
        TinyModel(), encoder, ["test"], 8, 8, steps=5, order=order,
        cfg_scale=guidance, generator=torch.Generator().manual_seed(42), device="cpu",
    )
    model = TinyModel()
    actual = sample_iris(
        model, positive, negative, get_views(["identity"]), size=8,
        num_inference_steps=5, order=order, guidance_scale=guidance,
        generator=torch.Generator().manual_seed(42), device="cpu", progress=False,
    )
    torch.testing.assert_close(actual, expected)
    assert all(batch == 1 for batch, _ in model.calls)
    assert any(torch.equal(mask, positive.mask) for _, mask in model.calls)
    if guidance != 1:
        assert any(torch.equal(mask, negative.mask) for _, mask in model.calls)


@pytest.mark.parametrize("reduction", ["mean", "sum", "alternate"])
def test_views_align_predictions_before_integration(reduction):
    views = get_views(["identity", "rotate_cw", "negate"])
    target = torch.linspace(-0.05, 0.05, 3 * 8 * 8).reshape(3, 8, 8)
    positive = PromptEncoding(torch.stack([view.view(target) for view in views]), torch.ones(3, 1))
    negative = PromptEncoding(torch.zeros(1, 3, 8, 8), torch.ones(1, 1))
    model = TinyModel(image_prediction=True)
    actual = sample_iris(
        model, positive, negative, views, size=8, prediction="x", reduction=reduction,
        num_inference_steps=4, guidance_scale=3.0, device="cpu", progress=False,
    )
    expected = target * 3 * (len(views) if reduction == "sum" else 1)
    torch.testing.assert_close(actual[0], expected)
    assert all(batch == 1 for batch, _ in model.calls)
    assert len(model.calls) == 8 * (1 if reduction == "alternate" else len(views))


def test_seeded_sampling_is_reproducible():
    encoding = PromptEncoding(torch.ones(1, 4, 3), torch.ones(1, 4))
    kwargs = dict(size=8, num_inference_steps=4, device="cpu", progress=False)
    first = sample_iris(TinyModel(), encoding, encoding, get_views(["identity"]),
                        generator=torch.Generator().manual_seed(7), **kwargs)
    second = sample_iris(TinyModel(), encoding, encoding, get_views(["identity"]),
                         generator=torch.Generator().manual_seed(7), **kwargs)
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_int8_meta_loading_preserves_values_and_shared_core(tmp_path):
    qweight = torch.tensor([[1, -2, 3], [-4, 5, -6]], dtype=torch.int8)
    scales = torch.tensor([[0.1], [0.2]])
    bias = torch.tensor([0.3, -0.4])
    with torch.device("meta"):
        model = nn.Module()
        model.proj = nn.Linear(3, 2)
        model.modulation_cores = nn.ModuleDict({"img": nn.Linear(2, 2)})
    shared_core = model.modulation_cores["img"]
    state = {
        "proj.qweight": qweight, "proj.scale": scales, "proj.bias": bias,
        "modulation_cores.img.weight": torch.eye(2), "modulation_cores.img.bias": torch.zeros(2),
    }
    path = tmp_path / "model.safetensors"
    save_file(state, path)
    load_int8(model, path)
    assert isinstance(model.proj, Int8Linear)
    assert model.modulation_cores["img"] is shared_core
    assert model.proj.qweight.dtype == torch.int8
    assert model.proj.scale.dtype == torch.float32
    x = torch.tensor([[0.7, 0.2, -0.1]])
    torch.testing.assert_close(model.proj(x), F.linear(x, qweight.float() * scales, bias))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_real_iris_architecture_with_packed_linears_on_cuda(tmp_path):
    from iris3b.config import ModelConfig, PixelStageConfig
    from iris3b.models.dit import IrisDiT

    cfg = ModelConfig(
        hidden_size=64, depth=2, dual_depth=1, num_heads=4, num_kv_heads=2,
        patch_size=4, text_dim=32, text_len=4, text_adapter="linear",
        pixel=PixelStageConfig(depth=1, hidden_size=4, attn_hidden_size=32, num_heads=4),
    )
    torch.manual_seed(4)
    original = IrisDiT(cfg).eval()
    nn.init.normal_(original.final_layer.linear.weight, std=0.01)
    state = dict(original.state_dict())
    for name, module in original.named_modules():
        if isinstance(module, nn.Linear) and module.weight.numel() >= 4096 and "modulation_cores." not in name:
            weight = state.pop(name + ".weight").float()
            scale = (weight.abs().amax(1, keepdim=True) / 127).clamp(min=1e-8)
            state[name + ".qweight"] = (weight / scale).round().clamp(-127, 127).to(torch.int8)
            state[name + ".scale"] = scale
    path = tmp_path / "tiny_iris.safetensors"
    save_file(state, path)
    with torch.device("meta"):
        packed = IrisDiT(cfg)
    load_int8(packed, path).cuda()
    encoding = PromptEncoding(torch.randn(2, 4, 32), torch.ones(2, 4, dtype=torch.int64))
    negative = PromptEncoding(torch.zeros(1, 4, 32), torch.ones(1, 4, dtype=torch.int64))
    result = sample_iris(
        packed, encoding, negative, get_views(["identity", "rotate_cw"]), size=16,
        num_inference_steps=3, device="cuda", progress=False,
        generator=torch.Generator(device="cuda").manual_seed(4),
    )
    assert result.shape == (1, 3, 16, 16)
    assert torch.isfinite(result).all()
    assert any(isinstance(module, Int8Linear) for module in packed.modules())


@pytest.mark.parametrize("bad_scale", [0.0, -1.0, float("nan")])
def test_invalid_quantization_scales_are_rejected(tmp_path, bad_scale):
    model = nn.Sequential(nn.Linear(3, 2, device="meta"))
    path = tmp_path / "model.safetensors"
    save_file({"0.qweight": torch.ones(2, 3, dtype=torch.int8),
               "0.scale": torch.full((2, 1), bad_scale), "0.bias": torch.zeros(2)}, path)
    with pytest.raises(ValueError, match="Invalid Iris INT8"):
        load_int8(model, path)


@pytest.mark.parametrize("name", [
    "identity", "flip", "rotate_cw", "rotate_ccw", "rotate_180", "negate",
    "skew", "patch_permute", "inner_circle", "square_hinge", "jigsaw",
])
def test_1024_views_round_trip(name):
    torch.manual_seed(0)
    view = get_views([name])[0]
    pixels = torch.arange(3 * 1024 * 1024, dtype=torch.float32).reshape(3, 1024, 1024)
    torch.testing.assert_close(view.inverse_view(view.view(pixels)), pixels, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("name,turns", [("rotate_cw", -1), ("rotate_ccw", 1), ("rotate_180", 2)])
@pytest.mark.parametrize("channels", [3, 6])
def test_1024_rotations_are_exact_under_cuda_bf16_autocast(name, turns, channels):
    # The model-wide autocast context also encloses view transforms. The old
    # affine grid rounded border coordinates at 1024px; small-image and CPU
    # round-trip tests did not expose the noisy-border failure.
    pixels = torch.randn(channels, 1024, 1024, device="cuda", dtype=torch.float32)
    expected = torch.rot90(pixels, k=turns, dims=(-2, -1))
    view = get_views([name])[0]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        transformed = view.view(pixels)
        restored = view.inverse_view(transformed)
    torch.testing.assert_close(transformed, expected, rtol=0, atol=0)
    torch.testing.assert_close(restored, pixels, rtol=0, atol=0)


@pytest.mark.parametrize("argv", [
    ["--prompts", "one", "--views", "identity", "flip"],
    ["--prompts", "one", "--views", "identity", "--num_inference_steps", "0"],
    ["--prompts", "one", "--views", "identity", "--shift", "nan"],
])
def test_invalid_cli_fails_before_model_download(argv):
    with pytest.raises(SystemExit) as exc:
        main(["--name", "invalid", *argv])
    assert exc.value.code == 2
