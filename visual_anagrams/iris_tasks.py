"""Standalone Iris tasks using verified INT8 exports and pinned upstream recipes."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from tqdm.auto import tqdm

from .iris import CODE_REVISION, MODEL_ID, PromptEncoding, load_int8

DOWNSTREAM_REVISION = "1ff39deeb474f4c88e7d04e8c186cbcc57368aac"
TASK_FOLDERS = {"depth": "depth", "restore": "upscaler"}


def resolve_task_model(task, model_path=None):
    """Only fetch the chosen task; reject generation weights in a downstream slot."""
    from huggingface_hub import snapshot_download

    if task not in TASK_FOLDERS:
        raise ValueError("Expected depth or restore")
    folder = TASK_FOLDERS[task]
    directory = Path(model_path) if model_path else Path(snapshot_download(
        MODEL_ID, revision=DOWNSTREAM_REVISION,
        allow_patterns=[f"{folder}/*", "LICENSE", "NOTICE"],
    )) / folder
    if model_path and (directory / folder / "manifest.json").is_file():
        directory = directory / folder
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    expected = "depth" if task == "depth" else "upscale"
    if manifest.get("format") != "iris-rowwise-int8-v1" or manifest.get("task") != expected:
        raise ValueError(f"Select the {folder} INT8 checkpoint, not generation weights")
    if manifest.get("code_revision") != CODE_REVISION:
        raise ValueError("Downstream checkpoint requires a different Iris inference revision")
    records = {item["path"]: item for item in manifest["files"]}
    for name in ("model.safetensors", "config.yaml", "empty_prompt.safetensors"):
        path, record = directory / name, records[name]
        if path.stat().st_size != record["size"]:
            raise ValueError(f"Incomplete Iris checkpoint: {path}")
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != record["sha256"]:
            raise ValueError(f"Iris checkpoint SHA-256 mismatch: {path}")
    return directory


def load_task(task, directory, device="cuda"):
    """Reuse upstream preprocessing, tiled blending and colour fix with INT8 weights."""
    from iris3b.config import inference_config
    from iris3b.downstream.depth import DepthPredictor, IrisDepth
    from iris3b.models.dit import IrisDiT
    from omegaconf import OmegaConf
    from safetensors.torch import load_file

    if task not in TASK_FOLDERS:
        raise ValueError("Expected depth or restore")
    raw = OmegaConf.to_container(OmegaConf.load(Path(directory) / "config.yaml"))
    settings = raw.pop("task")
    expected = "depth" if task == "depth" else "restoration"
    if settings["name"] != expected:
        raise ValueError(f"Expected a {expected} export")
    cfg = inference_config(raw, [])
    with torch.device("meta"):
        model = IrisDepth(cfg.model) if task == "depth" else IrisDiT(cfg.model)
    model = load_int8(model, Path(directory) / "model.safetensors").to(device)
    prompt = load_file(str(Path(directory) / "empty_prompt.safetensors"))
    runner_type = DepthPredictor if task == "depth" else CheckedRestorer
    runner = runner_type.__new__(runner_type)
    runner.model, runner.device = model, torch.device(device)
    runner.embeddings = prompt["embeddings"].to(device, torch.float32)
    runner.mask = prompt["mask"].to(device, torch.bool)
    runner.patch = cfg.model.patch_size
    if task == "depth":
        model.num_train_timesteps = cfg.flow.num_train_timesteps
    else:
        if cfg.flow.prediction != "v":
            raise ValueError("The restorer requires velocity prediction")
        runner.tile, runner.sigma = settings["tile"], settings["sigma"]
        runner.time = runner.sigma * cfg.flow.num_train_timesteps
    return runner


# Imported only in the worker, never when the desktop page opens.
from iris3b.downstream.restoration import Restorer


class CheckedRestorer(Restorer):
    def restore_tile(self, image):
        result = super().restore_tile(image)
        if not torch.isfinite(result).all():
            raise FloatingPointError("Iris restoration produced non-finite pixels")
        self.completed_tiles += 1
        print(f"Restoration tile {self.completed_tiles}", flush=True)
        return result

    def __call__(self, *args, **kwargs):
        self.completed_tiles = 0
        return super().__call__(*args, **kwargs)


def save_depth(depth, directory):
    """Raw relative log depth, near-white 16-bit PNG and a colour preview."""
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim != 2 or not np.isfinite(depth).all():
        raise FloatingPointError("Iris depth output must be a finite 2D array")
    directory = Path(directory)
    low, high = (float(x) for x in np.percentile(depth, [2, 98]))
    near = np.clip((high - depth) / max(high - low, 1e-6), 0, 1)
    np.save(directory / "depth.npy", depth, allow_pickle=False)
    Image.fromarray((near * 65535).round().astype(np.uint16)).save(directory / "depth16.png")
    # Fixed palette: far blue -> cyan -> yellow -> near white.
    stops = np.array([[20, 24, 70], [18, 130, 175], [245, 193, 68], [255, 255, 235]])
    colors = np.stack([np.interp(near, np.linspace(0, 1, 4), stops[:, c]) for c in range(3)], axis=-1)
    Image.fromarray(colors.round().astype(np.uint8)).save(directory / "result.png")
    return dict(units="affine-invariant relative log depth, not metres", near="bright", percentiles=[low, high])


@torch.inference_mode()
def sample_img2img(model, positive, negative, initial, *, strength=.5, steps=30, guidance=3.,
                   shift=4., order=2, generator=None, device="cuda", progress=True):
    """Experimental flow noise-and-denoise; no trained editing/image conditioning."""
    from iris3b.flow.solver import FlowDPMSolver

    if not math.isfinite(strength) or not 0 < strength <= 1:
        raise ValueError("Image-to-image strength must be in (0, 1]")
    if steps < 1 or order not in (1, 2) or not math.isfinite(shift) or shift <= 0:
        raise ValueError("Use positive steps/shift and solver order 1 or 2")
    positive = PromptEncoding(positive.embeddings.to(device), positive.mask.to(device))
    negative = PromptEncoding(negative.embeddings.to(device), negative.mask.to(device))
    image = initial.to(device=device, dtype=torch.float32)
    times = [strength * t for t in FlowDPMSolver.time_grid(steps, shift)]
    noise = torch.randn(image.shape, generator=generator, device=device)
    image = (1-times[0]) * image + times[0] * noise
    history = []
    with torch.autocast(torch.device(device).type, dtype=torch.bfloat16, enabled=torch.device(device).type == "cuda"):
        for index, (current, following) in enumerate(tqdm(list(zip(times, times[1:])),
                desc="Iris image-to-image", disable=not progress)):
            time = torch.full((1,), current * 1000, device=device)
            velocity = model(image, time, positive.embeddings, y_mask=positive.mask).x.float()
            if guidance != 1 and 0 < current < 1:
                null = model(image, time, negative.embeddings, y_mask=negative.mask).x.float()
                velocity = null + guidance * (velocity - null)
            clean = image - current * velocity
            history.append((current, clean))
            if order == 2 and len(history) > 1 and index < steps - 1:
                image = FlowDPMSolver._second_order(image, history[-2], history[-1], following)
            else:
                image = FlowDPMSolver._first_order(image, current, following, clean)
            history = history[-2:]
            if not torch.isfinite(image).all():
                raise FloatingPointError("Iris image-to-image produced non-finite pixels")
    return image.clamp(-1, 1)
