"""Community Iris Turbo LoRA support with a shared multi-view noise state.

Adapter format and re-noising schedule follow pentaCoxian/iris-3b-turbo,
commit d6316790813ee9a531e595d9a54500f468be223e (Apache-2.0).
"""

from __future__ import annotations

from contextlib import nullcontext
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from tqdm.auto import tqdm

from .iris import Int8Linear

TURBO_ID = "pentacoxian-dev/iris-3b-turbo-lora"
TURBO_REVISION = "fe16b3496eb440585993c92d95641c08c6b08263"
TURBO_ADAPTERS = ("r2-1000", "r1-1500")
TURBO_CODE_REVISION = "d6316790813ee9a531e595d9a54500f468be223e"
TARGETS = {"s_embedder", "t_embedder", "modulation_cores", "blocks", "pixel_blocks", "final_layer"}


def resolve_turbo_adapter(adapter="r2-1000"):
    """Fetch the pinned manifest and only the selected adapter."""
    from huggingface_hub import hf_hub_download

    if adapter not in TURBO_ADAPTERS:
        raise ValueError(f"Unknown Turbo adapter: {adapter}")
    manifest_path = hf_hub_download(TURBO_ID, "config.json", revision=TURBO_REVISION)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    entry = manifest["adapters"][adapter]
    if manifest.get("base_model") != "speridlabs/iris-3b" or entry["file"] != f"{adapter}/student.safetensors":
        raise ValueError("Unexpected Iris Turbo adapter manifest")
    return Path(hf_hub_download(TURBO_ID, entry["file"], revision=TURBO_REVISION))


def read_adapter_metadata(path):
    from safetensors import safe_open

    with safe_open(str(path), framework="pt") as stream:
        metadata = stream.metadata() or {}
    try:
        extra = json.loads(metadata["extra"])
        rank, alpha = int(metadata["rank"]), float(metadata["alpha"])
        if rank <= 0 or not math.isfinite(alpha) or alpha <= 0:
            raise ValueError
        if (extra["turbo_steps"], extra["shift"], extra["cfg_scale"], extra["resolution"]) != (4, 4.0, 3.0, 1024):
            raise ValueError
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Expected an Iris 4-step Turbo LoRA with shift 4 and built-in CFG 3") from exc
    return dict(extra, rank=rank, alpha=alpha)


class Int8LoRALinear(nn.Module):
    """Keep the base packed; evaluate its low-rank residual under autocast."""

    def __init__(self, base, down, up, scale):
        super().__init__()
        self.base = base
        self.in_features, self.out_features = base.in_features, base.out_features
        self.register_buffer("lora_down", down)
        self.register_buffer("lora_up", up)
        self.scale = scale

    def forward(self, inputs):
        dtype = (torch.get_autocast_dtype(inputs.device.type)
                 if torch.is_autocast_enabled(inputs.device.type) else inputs.dtype)
        residual = F.linear(F.linear(inputs.to(dtype), self.lora_down.to(dtype)), self.lora_up.to(dtype))
        return self.base(inputs) + self.scale * residual


@torch.no_grad()
def apply_turbo_lora(model, path):
    """Validate all pairs before applying them; keep shared core objects intact."""
    from safetensors.torch import load_file

    metadata = read_adapter_metadata(path)
    tensors = load_file(str(path), device="cpu")
    if not tensors or any(not key.endswith((".lora_down", ".lora_up")) for key in tensors):
        raise ValueError("Unexpected tensors in Iris Turbo LoRA")
    prefixes = sorted({key.rsplit(".", 1)[0] for key in tensors})
    expected = {name for name, layer in model.named_modules()
                if name.split(".")[0] in TARGETS and isinstance(layer, (nn.Linear, Int8Linear))}
    if set(prefixes) != expected:
        raise ValueError("Turbo LoRA layer coverage does not match the Iris architecture")
    layers = []
    for prefix in prefixes:
        try:
            linear = model.get_submodule(prefix)
            down, up = tensors[prefix + ".lora_down"], tensors[prefix + ".lora_up"]
            if prefix.split(".")[0] not in TARGETS or not isinstance(linear, (nn.Linear, Int8Linear)):
                raise ValueError
            if down.shape != (metadata["rank"], linear.in_features) or up.shape != (linear.out_features, metadata["rank"]):
                raise ValueError
            if not down.is_floating_point() or not up.is_floating_point() or not torch.isfinite(down).all() or not torch.isfinite(up).all():
                raise ValueError
        except (AttributeError, KeyError, ValueError) as exc:
            raise ValueError(f"Turbo LoRA does not match the Iris layer: {prefix}") from exc
        layers.append((prefix, linear, down.float(), up.float()))
    scale = metadata["alpha"] / metadata["rank"]
    for prefix, linear, down, up in layers:
        if isinstance(linear, Int8Linear):
            device = linear.qweight.device
            model.set_submodule(prefix, Int8LoRALinear(linear, down.to(device), up.to(device), scale))
        else:
            # Modulation cores are also held in unregistered tuples inside
            # blocks. Merging in place preserves those references, including
            # the released adapters' zero-valued modulation-core updates.
            delta = up.to(linear.weight.device) @ down.to(linear.weight.device)
            linear.weight.add_((scale * delta).to(linear.weight.dtype))
    return dict(metadata, applied_layers=len(layers))


@torch.inference_mode()
def sample_turbo(
    model, positive, views, *, size=1024, num_inference_steps=4,
    reduction="auto", generator=None, device="cuda", shift=4.0, progress=True,
):
    """Predict aligned velocities, then re-noise one shared clean-image estimate.

    Guidance is distilled into the adapter: one conditional forward per view
    per step, with no negative prompt or unconditional network evaluation.
    """
    from iris3b.flow.schedule import shift_sigma

    count = len(views)
    if not count or positive.embeddings.shape[0] != count or positive.mask.shape[0] != count:
        raise ValueError("Number of prompts must match number of views")
    from visual_anagrams.composition import resolve_reduction

    reduction = resolve_reduction(reduction, views)
    if num_inference_steps < 1 or not math.isfinite(shift) or shift <= 0:
        raise ValueError("Turbo needs positive steps and a positive finite shift")
    from .iris import image_dimensions

    height, width = image_dimensions(size, model.cfg.patch_size)
    device = torch.device(device)
    dtype = next(model.parameters()).dtype
    embeddings = positive.embeddings.to(device=device, dtype=dtype)
    mask = positive.mask.to(device)
    times = shift_sigma(1.0 - torch.arange(num_inference_steps, dtype=torch.float64) / num_inference_steps, shift).tolist()
    image = torch.randn((1, model.cfg.in_channels, height, width), generator=generator, device=device)
    context = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
    with context:
        for step, time in enumerate(tqdm(times, desc="Iris multi-view denoising", disable=not progress)):
            t_model = torch.full((1,), time, device=device) * 1000.0
            combined = torch.zeros_like(image)
            indices = [step % count] if reduction == "alternate" else range(count)
            for index in indices:
                view = views[index]
                viewed = view.view(image[0].clone()).unsqueeze(0).to(dtype)
                predicted = model(viewed, t_model, embeddings[index:index + 1], y_mask=mask[index:index + 1]).x.float()
                combined += view.inverse_view(predicted[0]).unsqueeze(0)
            if reduction == "mean":
                combined /= count
            clean = image - time * combined
            if not torch.isfinite(clean).all():
                raise FloatingPointError("Iris Turbo produced a non-finite prediction")
            if step + 1 == len(times):
                return clean.clamp(-1, 1)
            fresh = torch.randn(image.shape, generator=generator, device=device)
            image = (1.0 - times[step + 1]) * clean + times[step + 1] * fresh
