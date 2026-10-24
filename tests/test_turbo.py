"""LoRA loading, Turbo schedule and multi-view regressions without downloads."""

import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from torch import nn
from torch.nn import functional as F

from generate import build_parser, main
from visual_anagrams.iris import Int8Linear, PromptEncoding, load_int8
from visual_anagrams.turbo import (
    TARGETS, TURBO_ID, TURBO_REVISION, Int8LoRALinear,
    apply_turbo_lora, resolve_turbo_adapter, sample_turbo,
)
from visual_anagrams.views import get_views


def test_cli_defaults_to_standard_and_retains_optional_turbo_r2():
    args = build_parser().parse_args(["--name", "test", "--prompts", "one", "--views", "identity"])
    assert not args.turbo and args.reduction == "auto"
    args = build_parser().parse_args(["--name", "test", "--prompts", "one", "--views", "identity", "--turbo"])
    assert args.turbo and args.turbo_adapter == "r2-1000"


def adapter_file(model, path, *, corrupt=False):
    torch.manual_seed(5)
    state = {}
    for name, module in model.named_modules():
        if name.split(".")[0] in TARGETS and isinstance(module, (nn.Linear, Int8Linear)):
            state[name + ".lora_down"] = torch.randn(2, module.in_features) * 0.05
            state[name + ".lora_up"] = torch.randn(module.out_features, 2) * 0.05
    if corrupt:
        state[next(reversed(state))] = torch.ones(1)
    metadata = {"rank": "2", "alpha": "4.0", "extra": json.dumps(
        dict(turbo_steps=4, shift=4.0, cfg_scale=3.0, resolution=1024))}
    save_file(state, path, metadata=metadata)
    return state


def mixed_model():
    model = nn.Module()
    packed = Int8Linear(3, 4, device="cpu")
    packed.qweight.fill_(3)
    packed.scale.fill_(0.1)
    with torch.no_grad():
        packed.bias.zero_()
    model.blocks = nn.ModuleDict({"packed": packed})
    model.modulation_cores = nn.ModuleDict({"core": nn.Linear(3, 4)})
    return model


def test_lora_preserves_int8_and_shared_modulation_identity(tmp_path):
    model = mixed_model()
    packed = model.blocks["packed"]
    core_reference = (model.modulation_cores["core"],)
    x = torch.randn(2, 3)
    base_output, core_output = packed(x).detach(), core_reference[0](x).detach()
    stored = packed.qweight.clone()
    path = tmp_path / "adapter.safetensors"
    state = adapter_file(model, path)
    metadata = apply_turbo_lora(model, path)
    assert metadata["applied_layers"] == 2
    assert isinstance(model.blocks["packed"], Int8LoRALinear)
    assert model.blocks["packed"].base is packed
    assert packed.qweight.dtype == torch.int8
    torch.testing.assert_close(packed.qweight, stored)
    for name, base in (("blocks.packed", base_output), ("modulation_cores.core", core_output)):
        residual = 2 * F.linear(F.linear(x, state[name + ".lora_down"]), state[name + ".lora_up"])
        torch.testing.assert_close(model.get_submodule(name)(x), base + residual)
    assert model.modulation_cores["core"] is core_reference[0]


def test_invalid_adapter_is_rejected_before_any_layer_is_modified(tmp_path):
    model = mixed_model()
    core_weight = model.modulation_cores["core"].weight.detach().clone()
    path = tmp_path / "bad.safetensors"
    adapter_file(model, path, corrupt=True)
    with pytest.raises(ValueError, match="does not match"):
        apply_turbo_lora(model, path)
    assert isinstance(model.blocks["packed"], Int8Linear)
    torch.testing.assert_close(model.modulation_cores["core"].weight, core_weight)


def test_resolver_pins_revision_and_fetches_only_chosen_adapter(tmp_path, monkeypatch):
    import huggingface_hub

    manifest = tmp_path / "config.json"
    manifest.write_text(json.dumps({"base_model": "speridlabs/iris-3b", "adapters": {
        "r1-1500": {"file": "r1-1500/student.safetensors"}}}), encoding="utf-8")
    calls = []

    def download(repo, filename, revision):
        calls.append((repo, filename, revision))
        return str(manifest if filename == "config.json" else tmp_path / "student.safetensors")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    assert resolve_turbo_adapter("r1-1500").name == "student.safetensors"
    assert calls == [(TURBO_ID, "config.json", TURBO_REVISION),
                     (TURBO_ID, "r1-1500/student.safetensors", TURBO_REVISION)]


class TargetVelocity(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.cfg = SimpleNamespace(in_channels=3, patch_size=1)
        self.inputs, self.times = [], []

    def forward(self, x, t, y, y_mask=None):
        assert x.shape[0] == 1 and y_mask is not None
        self.inputs.append(x.clone())
        self.times.append(float(t[0]))
        return SimpleNamespace(x=(x - y) / (t[:, None, None, None] / 1000))


@pytest.mark.parametrize("reduction", ["mean", "alternate"])
def test_turbo_aligns_views_uses_four_calls_per_view_and_shared_fresh_noise(reduction):
    target = torch.linspace(-0.2, 0.2, 3 * 8 * 8).reshape(3, 8, 8)
    views = get_views(["identity", "rotate_cw", "negate"])
    encoding = PromptEncoding(torch.stack([view.view(target) for view in views]), torch.ones(3, 1))
    model = TargetVelocity()
    generator = torch.Generator().manual_seed(23)
    result = sample_turbo(model, encoding, views, size=8, device="cpu", generator=generator,
                          reduction=reduction, progress=False)
    torch.testing.assert_close(result[0], target, atol=1e-6, rtol=1e-5)
    calls_per_step = 1 if reduction == "alternate" else 3
    assert len(model.inputs) == 4 * calls_per_step
    torch.testing.assert_close(torch.tensor(model.times[::calls_per_step]),
                               torch.tensor([1000, 12000/13, 800, 4000/7]))
    reference = torch.Generator().manual_seed(23)
    initial = torch.randn((1, 3, 8, 8), generator=reference)
    torch.testing.assert_close(model.inputs[0], initial)
    for _ in range(3):
        torch.randn((1, 3, 8, 8), generator=reference)
    torch.testing.assert_close(torch.randn(5, generator=generator), torch.randn(5, generator=reference))
    if reduction == "mean":
        for step in range(4):
            canonical = model.inputs[step * 3][0]
            for index, view in enumerate(views):
                torch.testing.assert_close(model.inputs[step * 3 + index][0], view.view(canonical))


def test_turbo_sum_combines_partitioned_components():
    class ChannelView:
        def __init__(self, channel):
            self.channel = channel

        def view(self, image):
            return image

        def inverse_view(self, velocity):
            result = torch.zeros_like(velocity)
            result[self.channel] = velocity[self.channel]
            return result

    target = torch.full((3, 8, 8), 0.1)
    encoding = PromptEncoding(torch.stack([target] * 3), torch.ones(3, 1))
    result = sample_turbo(TargetVelocity(), encoding, [ChannelView(i) for i in range(3)],
                          size=8, reduction="sum", device="cpu", progress=False)
    torch.testing.assert_close(result[0], target)


@pytest.mark.parametrize("extra", [["--guidance_scale", "3"], ["--negative_prompt", "blurry"]])
def test_turbo_cli_rejects_extra_guidance_before_downloading(extra):
    with pytest.raises(SystemExit) as exc:
        main(["--name", "test", "--prompts", "one", "--views", "identity", "--turbo", *extra])
    assert exc.value.code == 2


@pytest.mark.parametrize("mode", [None, "--standard", "--turbo"])
def test_cli_selects_correct_sampler_and_records_adapter(tmp_path, monkeypatch, mode):
    turbo = mode == "--turbo"
    import visual_anagrams.iris as iris
    import visual_anagrams.turbo as turbo_module
    import visual_anagrams.utils as utils

    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")
    path = tmp_path / "adapter.safetensors"
    adapter_file(mixed_model(), path)
    calls = {}
    monkeypatch.setattr(iris, "resolve_model", lambda _: tmp_path)
    monkeypatch.setattr(iris, "read_config", lambda _: SimpleNamespace(
        flow=SimpleNamespace(prediction="v", num_train_timesteps=1000)))

    def encode(*args, **kwargs):
        calls["encode_negative"] = kwargs["encode_negative"]
        return "positive", "negative" if kwargs["encode_negative"] else None

    def load(*args, **kwargs):
        calls["adapter_path"] = kwargs["turbo_path"]
        return "model"

    def sample(*args, **kwargs):
        calls["steps"] = kwargs["num_inference_steps"]
        calls["guidance"] = kwargs.get("guidance_scale")
        calls["sampler"] = "standard"
        return torch.zeros(1, 3, 4, 4)

    def sample_fast(*args, **kwargs):
        image = sample(*args, **kwargs)
        calls["sampler"] = "turbo"
        return image

    monkeypatch.setattr(iris, "encode_prompts", encode)
    monkeypatch.setattr(iris, "load_model", load)
    monkeypatch.setattr(iris, "sample_iris", sample)
    monkeypatch.setattr(turbo_module, "resolve_turbo_adapter", lambda _: path)
    monkeypatch.setattr(turbo_module, "sample_turbo", sample_fast)
    monkeypatch.setattr(utils, "save_illusion", lambda *args: None)
    main(["--name", "test", "--save_dir", str(tmp_path), "--prompts", "one", "--views", "identity",
          "--device", "cpu", "--text_encoder_device", "cpu", *([mode] if mode else [])])
    metadata = json.loads((tmp_path / "test" / "metadata.json").read_text())
    assert calls["encode_negative"] is (not turbo)
    assert calls["sampler"] == ("turbo" if turbo else "standard")
    assert calls["steps"] == (4 if turbo else 30)
    assert calls["guidance"] == (None if turbo else 3.0)
    assert calls["adapter_path"] == (path if turbo else None)
    if turbo:
        assert metadata["turbo_lora"]["revision"] == TURBO_REVISION
        assert metadata["guidance_scale"] == 1
    else:
        assert metadata["turbo_lora"] is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_real_iris_with_packed_lora_and_turbo_sampler_on_cuda(tmp_path):
    from iris3b.config import ModelConfig, PixelStageConfig
    from iris3b.models.dit import IrisDiT

    cfg = ModelConfig(hidden_size=64, depth=2, dual_depth=1, num_heads=4, num_kv_heads=2,
                      patch_size=4, text_dim=32, text_len=4, text_adapter="linear",
                      pixel=PixelStageConfig(depth=1, hidden_size=4, attn_hidden_size=32, num_heads=4))
    original = IrisDiT(cfg).eval()
    state = dict(original.state_dict())
    for name, module in original.named_modules():
        if isinstance(module, nn.Linear) and module.weight.numel() >= 4096 and "modulation_cores." not in name:
            weight = state.pop(name + ".weight").float()
            scale = (weight.abs().amax(1, keepdim=True) / 127).clamp(min=1e-8)
            state[name + ".qweight"] = (weight / scale).round().clamp(-127, 127).to(torch.int8)
            state[name + ".scale"] = scale
    base_path = tmp_path / "base.safetensors"
    save_file(state, base_path)
    with torch.device("meta"):
        model = IrisDiT(cfg)
    load_int8(model, base_path)
    path = tmp_path / "adapter.safetensors"
    adapter_file(model, path)
    apply_turbo_lora(model, path)
    model.cuda()
    encoding = PromptEncoding(torch.randn(2, 4, 32), torch.ones(2, 4, dtype=torch.int64))
    result = sample_turbo(model, encoding, get_views(["identity", "rotate_cw"]),
                          size=16, device="cuda", progress=False)
    assert result.shape == (1, 3, 16, 16) and torch.isfinite(result).all()
    assert any(isinstance(layer, Int8LoRALinear) for layer in model.modules())
