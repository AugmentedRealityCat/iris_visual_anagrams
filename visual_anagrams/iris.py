"""Iris INT8 loading and pixel-space multi-view flow sampling.

Linear weights use Aikimi's iris-rowwise-int8-v1 layout: INT8 storage,
FP32 row scales, and floating point GEMMs under BF16 CUDA autocast.
Shared modulation cores retain their precision and module identity.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from tqdm.auto import tqdm

MODEL_ID = "Aikimi/iris-3b-int8"
MODEL_REVISION = "6231647d02bb538b401a675061c11c7a64bba81f"
CODE_REVISION = "a8d15239dea469aba042cfa56ca3bb4e450d5ebc"
TEXT_ID = "Qwen/Qwen3-VL-4B-Instruct"
TEXT_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"


class Int8Linear(nn.Module):
    """Rowwise symmetric INT8 storage, FP32 scales, autocast computation."""

    def __init__(self, in_features, out_features, bias=True, device="meta"):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.register_buffer("qweight", torch.empty(out_features, in_features, dtype=torch.int8, device=device))
        self.register_buffer("scale", torch.empty(out_features, 1, dtype=torch.float32, device=device))
        self.bias = nn.Parameter(torch.empty(out_features, device=device), requires_grad=False) if bias else None

    def forward(self, inputs):
        dtype = (
            torch.get_autocast_dtype(inputs.device.type)
            if torch.is_autocast_enabled(inputs.device.type)
            else inputs.dtype
        )
        weight = (self.qweight.float() * self.scale).to(dtype)
        bias = self.bias.to(dtype) if self.bias is not None else None
        return F.linear(inputs.to(dtype), weight, bias)


def load_int8(model, path):
    """Replace packed linears on a meta model before assigning checkpoint data."""
    from safetensors.torch import load_file

    state = load_file(str(path), device="cpu")
    modules = dict(model.named_modules())
    count = 0
    for key, weight in state.items():
        if not key.endswith(".qweight"):
            continue
        name = key.removesuffix(".qweight")
        module = modules.get(name)
        scale = state.get(name + ".scale")
        if (
            not isinstance(module, nn.Linear)
            or "modulation_cores." in name
            or weight.dtype != torch.int8
            or tuple(weight.shape) != (module.out_features, module.in_features)
            or scale is None
            or scale.dtype != torch.float32
            or tuple(scale.shape) != (module.out_features, 1)
            or not torch.isfinite(scale).all()
            or not (scale > 0).all()
        ):
            raise ValueError(f"Invalid Iris INT8 weights or scales: {name}")
        model.set_submodule(name, Int8Linear(module.in_features, module.out_features, module.bias is not None))
        count += 1
    if not count:
        raise ValueError("Expected an iris-rowwise-int8-v1 checkpoint containing .qweight tensors")
    model.load_state_dict(state, strict=True, assign=True)
    return model.float().eval().requires_grad_(False)


def resolve_model(model_path=None):
    """Download only generation assets, or verify a local INT8 export."""
    from huggingface_hub import snapshot_download

    directory = Path(model_path) if model_path else Path(snapshot_download(
        MODEL_ID, revision=MODEL_REVISION,
        allow_patterns=["config.yaml", "model.safetensors", "manifest.json", "LICENSE", "NOTICE"],
    ))
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") != "iris-rowwise-int8-v1" or manifest.get("task") != "generate":
        raise ValueError("Use the Iris INT8 text-to-image checkpoint, not depth/upscaler weights")
    if manifest.get("code_revision") != CODE_REVISION:
        raise ValueError(f"Iris checkpoint requires different inference code: {manifest.get('code_revision')}")
    records = {record["path"]: record for record in manifest["files"]}
    for name in ("config.yaml", "model.safetensors"):
        record = records[name]
        path = directory / name
        if path.stat().st_size != record["size"]:
            raise ValueError(f"Incomplete Iris checkpoint: {path}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024**2), b""):
                digest.update(block)
        if digest.hexdigest() != record["sha256"]:
            raise ValueError(f"Iris checkpoint SHA-256 mismatch: {path}")
    return directory


def read_config(directory):
    from iris3b.config import inference_config
    from omegaconf import OmegaConf

    raw = OmegaConf.to_container(OmegaConf.load(Path(directory) / "config.yaml"))
    return inference_config(raw, [])


def load_model(directory, cfg, device="cuda", turbo_path=None):
    from iris3b.models.dit import IrisDiT

    with torch.device("meta"):
        model = IrisDiT(cfg.model)
    model = load_int8(model, Path(directory) / "model.safetensors")
    if turbo_path is not None:
        from .turbo import apply_turbo_lora

        apply_turbo_lora(model, turbo_path)
    return model.to(device)


@dataclass
class PromptEncoding:
    embeddings: torch.Tensor
    mask: torch.Tensor


@torch.inference_mode()
def encode_prompts(cfg, prompts, negative_prompt="", device="cuda", text_encoder_path=None, *, encode_negative=True):
    """Encode one prompt at a time; release Qwen before loading Iris on GPU."""
    from huggingface_hub import snapshot_download
    from iris3b.text.qwen3_vl import Qwen3VLTextEncoder

    folder = text_encoder_path or snapshot_download(
        TEXT_ID, revision=TEXT_REVISION,
        allow_patterns=["*.json", "*.safetensors", "merges.txt", "vocab.json"],
    )
    cfg.text_encoder.pretrained = str(folder)
    encoder = Qwen3VLTextEncoder(cfg.text_encoder, device="cpu")
    try:
        encoder.to(device)
        encodings = []
        texts = [*prompts, negative_prompt] if encode_negative else prompts
        for prompt in tqdm(texts, desc="Encoding prompts"):
            # Avoid upstream null-cache filenames containing absolute Windows paths.
            value = encoder.encode([prompt])
            encodings.append(PromptEncoding(value.embeddings.cpu(), value.mask.cpu()))
        positive = PromptEncoding(
            torch.cat([value.embeddings for value in encodings[:len(prompts)]]),
            torch.cat([value.mask for value in encodings[:len(prompts)]]),
        )
        return positive, encodings[-1] if encode_negative else None
    finally:
        del encoder
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@torch.inference_mode()
def sample_iris(
    model, positive, negative, views, *, size=1024, num_inference_steps=30,
    guidance_scale=3.0, reduction="auto", generator=None, device="cuda",
    shift=4.0, order=2, num_train_timesteps=1000, prediction="v", progress=True,
):
    """Align per-view predictions and integrate one shared RGB image.

    View/CFG branches are evaluated separately so the model batch stays at
    one. The upstream solver retains its FP32 state, shifted time grid and
    lower-order final projection.
    """
    from iris3b.flow.solver import FlowDPMSolver

    num_views = len(views)
    if not num_views or positive.embeddings.shape[0] != num_views or positive.mask.shape[0] != num_views:
        raise ValueError("Number of prompts must match number of views")
    if negative.embeddings.shape[0] != 1 or negative.mask.shape[0] != 1:
        raise ValueError("Expected one shared negative prompt")
    from visual_anagrams.composition import resolve_reduction

    reduction = resolve_reduction(reduction, views)
    if num_inference_steps < 1 or order not in (1, 2):
        raise ValueError("Use at least one step and solver order 1 or 2")
    if not math.isfinite(shift) or shift <= 0 or not math.isfinite(guidance_scale):
        raise ValueError("Shift must be positive and guidance must be finite")
    if size <= 0 or size % model.cfg.patch_size:
        raise ValueError("Image size must be a positive multiple of the model patch size")

    device = torch.device(device)
    positive = PromptEncoding(positive.embeddings.to(device), positive.mask.to(device))
    negative = PromptEncoding(negative.embeddings.to(device), negative.mask.to(device))
    dtype = next(model.parameters()).dtype
    noise = torch.randn((1, model.cfg.in_channels, size, size), generator=generator, device=device, dtype=torch.float32)
    step = 0
    bar = tqdm(total=num_inference_steps, desc="Iris multi-view denoising", disable=not progress)

    def model_fn(x, t_model, _conditioning):
        nonlocal step
        indices = [step % num_views] if reduction == "alternate" else range(num_views)
        combined = torch.zeros_like(x)
        for index in indices:
            view = views[index]
            viewed = view.view(x[0].clone()).unsqueeze(0).to(dtype)
            predicted = model(
                viewed, t_model, positive.embeddings[index:index + 1],
                y_mask=positive.mask[index:index + 1],
            ).x.float()
            if guidance_scale != 1.0 and 0 < float(t_model[0]) < num_train_timesteps:
                uncond = model(viewed, t_model, negative.embeddings, y_mask=negative.mask).x.float()
                predicted = uncond + guidance_scale * (predicted - uncond)
            combined += view.inverse_view(predicted[0]).unsqueeze(0)
        if reduction == "mean":
            combined /= num_views
        if not torch.isfinite(combined).all():
            raise FloatingPointError("Iris produced non-finite predictions")
        step += 1
        bar.update(1)
        return combined

    # CFG is applied per view above. The solver integrates one shared image
    # and only needs a dummy conditioning tensor.
    solver = FlowDPMSolver(model_fn, num_timesteps=num_train_timesteps, prediction=prediction)
    context = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
    try:
        with context:
            result = solver.sample(
                noise, torch.empty((1, 0), device=device),
                steps=num_inference_steps, order=order, shift=shift,
            )
        if not torch.isfinite(result).all():
            raise FloatingPointError("Iris produced a non-finite image")
        return result.clamp(-1, 1)
    finally:
        bar.close()
