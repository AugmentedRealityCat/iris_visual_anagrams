"""Isolated research samplers; these do not change generation or GUI defaults.

Inspired by Anagram-MTL, not a reproduction of its DeepFloyd pipeline. Preview
balancing scores predicted clean images with ordinary CLIP. Variance correction
converts flow velocity to noise, uses empirical Pearson correlations, limits
the gain, and skips the near-pure-noise endpoint. Both are pilot hypotheses.
"""

from contextlib import nullcontext
import torch
import torch.nn.functional as F


class ResearchMixer:
    def __init__(self, method, views, scorer=None, *, interval=2, temperature=10.0):
        if method not in ("mean", "preview-balanced", "bounded-variance"):
            raise ValueError("Unknown research combination")
        if method == "preview-balanced" and scorer is None:
            raise ValueError("Preview balancing requires a clean-image scoring function")
        self.method, self.views, self.scorer = method, views, scorer
        self.interval, self.temperature = interval, temperature
        self.weights = None
        self.history = []

    def __call__(self, image, velocities, time, step):
        predictions = torch.stack(velocities)
        count = len(velocities)
        if self.weights is None:
            self.weights = torch.full((count,), 1 / count, device=image.device)
        scores = None
        if self.method == "preview-balanced" and time < 0.95 and step % self.interval == 0:
            preview = image - time * predictions.mean(0)
            scores = self.scorer(preview[0], self.views)
            values = torch.tensor(scores, device=image.device, dtype=torch.float32)
            desired = torch.softmax(-self.temperature * values, dim=0)
            # The pilot has two views. Bounds and smoothing limit feedback
            # from noisy or poorly calibrated image/text similarity scores.
            desired = desired.clamp(0.25, 0.75)
            desired /= desired.sum()
            self.weights = 0.5 * self.weights + 0.5 * desired
        weights = self.weights.reshape(count, *([1] * (predictions.ndim - 1)))
        combined = (predictions * weights).sum(0)
        gain = 1.0
        if self.method == "bounded-variance" and 0.05 < time < 0.95:
            noises = image[None] + (1 - time) * predictions
            flat = noises.flatten(1)
            centered = flat - flat.mean(1, keepdim=True)
            # Degenerate noise estimates have no meaningful correlation.
            # Keep these and the small covariance calculation in FP32.
            with torch.autocast(image.device.type, enabled=False):
                if (centered.norm(dim=1) > 1e-6).all():
                    normalized = F.normalize(centered.float(), dim=1)
                    correlations = normalized @ normalized.T
                    variance = self.weights @ correlations @ self.weights
                    raw_gain = variance.clamp_min(1e-6).rsqrt().clamp(1, 1.2)
                    gain = float(1 + 0.5 * (raw_gain - 1))
            if gain != 1:
                combined_noise = image + (1 - time) * combined
                combined = (gain * combined_noise - image) / (1 - time)
        self.history.append(dict(step=step, time=time, scores=scores,
                                 weights=self.weights.tolist(), gain=gain))
        if not torch.isfinite(combined).all():
            raise FloatingPointError("Research combiner produced a non-finite velocity")
        return combined


@torch.inference_mode()
def sample_research(model, positive, negative, views, mixer, *, turbo, steps,
                    guidance=3.0, seed=0, device="cuda", size=1024):
    """Match the production sampler, replacing only prediction combination."""
    from iris3b.flow.schedule import shift_sigma
    from iris3b.flow.solver import FlowDPMSolver

    device = torch.device(device)
    dtype = next(model.parameters()).dtype
    generator = torch.Generator(device=device).manual_seed(seed)
    embeddings = positive.embeddings.to(device=device, dtype=dtype) if turbo else positive.embeddings.to(device)
    mask = positive.mask.to(device)
    if negative is not None:
        neg, neg_mask = negative.embeddings.to(device), negative.mask.to(device)
    image = torch.randn((1, 3, size, size), device=device, generator=generator)
    step = 0

    def predict(x, t_model, _conditioning=None):
        nonlocal step
        predictions = []
        for index, view in enumerate(views):
            viewed = view.view(x[0].clone())[None].to(dtype)
            velocity = model(viewed, t_model, embeddings[index:index + 1], y_mask=mask[index:index + 1]).x.float()
            if not turbo and guidance != 1 and 0 < float(t_model[0]) < 1000:
                uncond = model(viewed, t_model, neg, y_mask=neg_mask).x.float()
                velocity = uncond + guidance * (velocity - uncond)
            predictions.append(view.inverse_view(velocity[0])[None])
        combined = mixer(x, predictions, float(t_model[0]) / 1000, step)
        step += 1
        return combined

    context = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
    with context:
        if turbo:
            times = shift_sigma(1 - torch.arange(steps, dtype=torch.float64) / steps, 4.0).tolist()
            for index, time in enumerate(times):
                t_model = torch.full((1,), time, device=device) * 1000
                clean = image - time * predict(image, t_model)
                if not torch.isfinite(clean).all():
                    raise FloatingPointError("Research sampler produced a non-finite image")
                if index == len(times) - 1:
                    return clean.clamp(-1, 1)
                noise = torch.randn(image.shape, device=device, generator=generator)
                image = (1 - times[index + 1]) * clean + times[index + 1] * noise
        else:
            solver = FlowDPMSolver(predict, num_timesteps=1000, prediction="v")
            return solver.sample(image, torch.empty((1, 0), device=device),
                                 steps=steps, order=2, shift=4.0).clamp(-1, 1)


class CleanPreviewCLIP:
    def __init__(self, model_id, revision):
        from transformers import CLIPModel, CLIPProcessor

        self.model = CLIPModel.from_pretrained(model_id, revision=revision).float().eval()
        self.processor = CLIPProcessor.from_pretrained(model_id, revision=revision, use_fast=False)
        self.text = None

    def set_prompts(self, prompts):
        tokens = self.processor(text=prompts, return_tensors="pt", padding=True)
        with torch.inference_mode():
            self.text = F.normalize(self.model.get_text_features(**tokens), dim=-1)

    def __call__(self, preview, views):
        from PIL import Image

        images = []
        for view in views:
            pixels = view.view(preview.clone()).clamp(-1, 1).add(1).mul(127.5)
            array = pixels.permute(1, 2, 0).round().byte().cpu().numpy()
            images.append(Image.fromarray(array))
        inputs = self.processor(images=images, return_tensors="pt")
        with torch.inference_mode():
            encoded = F.normalize(self.model.get_image_features(**inputs), dim=-1)
            return (encoded @ self.text.T).diag().tolist()
