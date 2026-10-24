"""Optional Anagram-MTL and SyncVC adaptations for Iris's rectified flow.

MTL: clean-preview CLIP balancing, bounded noise-space variance correction,
and gradient optimization of real joint-attention overlap. SyncVC: independent
view trajectories, optimized variational controls, clean-image agreement and
the flow-schedule analogue of the reference DDIM prior regularizer.

These are Iris adaptations, not claims of reproducing DeepFloyd results.
"""

from contextlib import nullcontext

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .method_options import MethodOptions
from .optimization import align_attention, attention_map, gradient_memory, overlap_loss

GUIDE_ID = "openai/clip-vit-base-patch32"
GUIDE_REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"


class PreviewScorer:
    """CPU CLIP scoring of clean previews; no interpretation of flow noise as DDPM noise."""
    def __init__(self, prompts):
        from transformers import CLIPModel, CLIPProcessor

        self.model = CLIPModel.from_pretrained(GUIDE_ID, revision=GUIDE_REVISION).float().eval()
        self.processor = CLIPProcessor.from_pretrained(GUIDE_ID, revision=GUIDE_REVISION, use_fast=False)
        with torch.no_grad():
            encoded = self.processor(text=prompts, return_tensors="pt", padding=True)
            self.text = F.normalize(self.model.get_text_features(**encoded), dim=-1)

    def __call__(self, image, views):
        from PIL import Image

        images = []
        for view in views:
            pixels = ((view.view(image[0].clone()).clamp(-1, 1) + 1) * 127.5).round().byte()
            images.append(Image.fromarray(pixels.permute(1, 2, 0).cpu().numpy()))
        with torch.no_grad():
            inputs = self.processor(images=images, return_tensors="pt")
            embeddings = F.normalize(self.model.get_image_features(**inputs), dim=-1)
            return (embeddings @ self.text.T).diag().tolist()


def flow_euler(image, velocity, current, following):
    return image + (following - current) * velocity


def flow_prior_penalty(control, epsilon_change, current, following, weight=0.2):
    """Reference SyncVC DDIM penalty with alpha=1-t and sigma=t."""
    alpha_ratio = (1 - following) / (1 - current)
    gamma_t = alpha_ratio - following / current
    return weight * ((alpha_ratio * control).square().sum()
                     + (current * gamma_t * epsilon_change).square().sum())


def _remap(image, source, destination):
    return destination.view(source.inverse_view(image[0].clone()))[None]


def optimize_overlap(model, image, time, positive, views, positions, options):
    """Differentiate each view separately, summing the exact two loss partials."""
    t_model = torch.full((1,), time * 1000, device=image.device)
    with torch.no_grad():
        references = [align_attention(attention_map(model, view.view(image[0].clone())[None], t_model,
                      positive.embeddings[i:i + 1], positive.mask[i:i + 1], positions[i]), view, image.shape[-1])
                      for i, view in enumerate(views)]
        before = float(overlap_loss(*references))
    gradient = torch.zeros_like(image)
    for i, view in enumerate(views):
        with torch.enable_grad(), gradient_memory(model, options.cpu_offload):
            candidate = image.detach().clone().requires_grad_(True)
            own = attention_map(model, view.view(candidate[0].clone())[None], t_model,
                                positive.embeddings[i:i + 1], positive.mask[i:i + 1], positions[i])
            own = align_attention(own, view, image.shape[-1])
            loss = overlap_loss(own, references[1 - i])
            part, = torch.autograd.grad(loss, candidate)
            gradient += part.detach()
    if not torch.isfinite(gradient).all():
        raise FloatingPointError("Attention-overlap gradient is non-finite")
    rms = gradient.square().mean().sqrt()
    direction = (gradient / rms.clamp_min(1e-8)).clamp(-3, 3)
    # The signal coefficient is 1-t. A fixed normalized edit to x_t near
    # pure noise can become a much larger clean-image edit. Bound the update
    # in signal units and fade it toward the clean endpoint as well.
    noise_step = options.strength * time * (1 - time)
    result = image - noise_step * direction
    return result.detach(), dict(overlap_loss=before, gradient_rms=float(rms),
                                 attention_noise_step=noise_step)


@torch.no_grad()
def sample_advanced(model, positive, negative, views, *, options, prompts=None, subject_positions=None,
                    num_inference_steps=30, guidance_scale=3.0, generator=None, device="cuda", size=1024,
                    shift=4.0, order=2, progress=True, diagnostics=None, scorer=None):
    from iris3b.flow.solver import FlowDPMSolver
    from .composition import resolve_reduction
    from .iris import PromptEncoding

    options.validate(views, reduction=resolve_reduction("auto", views), steps=num_inference_steps)
    if options.method == "plain":
        raise ValueError("Use sample_iris for the Classic method")
    if positive.embeddings.shape[0] != 2 or negative is None:
        raise ValueError("Advanced methods require two prompt encodings and a negative encoding")
    if options.method == "anagram_mtl" and (not subject_positions or len(subject_positions) != 2):
        raise ValueError("Anagram-MTL requires subject token positions for both prompts")
    device = torch.device(device)
    positive = PromptEncoding(positive.embeddings.to(device).clone(), positive.mask.to(device).clone())
    negative = PromptEncoding(negative.embeddings.to(device).clone(), negative.mask.to(device).clone())
    dtype = next(model.parameters()).dtype
    times = FlowDPMSolver.time_grid(num_inference_steps, shift)
    if options.method == "anagram_mtl" and not any(i % options.interval == 0 and 0.2 < t < 0.95
                                                   for i, t in enumerate(times[1:])):
        raise ValueError("This budget has no attention-optimization steps; use more generation steps or a smaller interval")
    image = torch.randn((1, 3, size, size), device=device, generator=generator)
    records = []
    if options.method == "anagram_mtl":
        scorer = scorer or PreviewScorer(prompts)
    weights = torch.full((2,), 0.5, device=device)
    states = [views[0].view(image[0].clone())[None],
              torch.randn(image.shape, device=device, generator=generator)]
    history = []

    def predict(x, time, index):
        t_model = torch.full((1,), time * 1000, device=device)
        output = model(x.to(dtype), t_model, positive.embeddings[index:index + 1],
                       y_mask=positive.mask[index:index + 1]).x.float()
        if guidance_scale != 1 and 0 < time < 1:
            null = model(x.to(dtype), t_model, negative.embeddings, y_mask=negative.mask).x.float()
            output = null + guidance_scale * (output - null)
        return output

    context = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
    with context:
        for step, (current, following) in enumerate(tqdm(list(zip(times, times[1:])),
                 desc=f"Iris {options.method}", disable=not progress)):
            record = dict(step=step, time=current)
            if options.method == "anagram_mtl":
                velocities = [view.inverse_view(predict(view.view(image[0].clone())[None], current, i)[0])[None]
                              for i, view in enumerate(views)]
                if step % options.interval == 0 and current < 0.95:
                    scores = scorer(image - current * torch.stack(velocities).mean(0), views)
                    values = torch.tensor(scores, device=device).clamp_min(1e-4).pow(current - 2)
                    desired = (values / values.sum()).clamp(0.25, 0.75)
                    weights = 0.5 * weights + 0.5 * desired / desired.sum()
                    record["preview_scores"] = scores
                velocity = sum(w * v for w, v in zip(weights, velocities))
                if 0.05 < current < 0.95:
                    noises = torch.stack([image + (1 - current) * v for v in velocities]).flatten(1)
                    centered = noises - noises.mean(1, keepdim=True)
                    with torch.autocast(device.type, enabled=False):
                        normalized = F.normalize(centered.float(), dim=1)
                        variance = weights @ (normalized @ normalized.T) @ weights
                        gain = 1 + 0.5 * (variance.clamp_min(1e-6).rsqrt().clamp(1, 1.2) - 1)
                    velocity = (gain * (image + (1 - current) * velocity) - image) / (1 - current)
                    record["variance_gain"] = float(gain)
                clean = image - current * velocity
                history.append((current, clean))
                if order == 2 and len(history) >= 2 and step < num_inference_steps - 1:
                    image = FlowDPMSolver._second_order(image, history[-2], history[-1], following)
                else:
                    image = FlowDPMSolver._first_order(image, current, following, clean)
                history = history[-2:]
                if step % options.interval == 0 and 0.2 < following < 0.95:
                    for _ in range(options.iterations):
                        image, extra = optimize_overlap(model, image, following, positive, views,
                                                        subject_positions, options)
                        record.update(extra)
                    history.clear()  # restart multistep history after a gradient correction
                record["weights"] = weights.tolist()
            else:
                left, right = states
                left_velocity = predict(left, current, 0)
                left_clean = left - current * left_velocity
                reference = _remap(left_clean, views[0], views[1]).detach()
                left_in_right = _remap(left, views[0], views[1]).detach()
                base_velocity = predict(right, current, 1)
                base_epsilon = right + (1 - current) * base_velocity
                control = torch.zeros_like(right)
                optimize = step % options.interval == 0 and 0.05 < current < 0.95
                if optimize:
                    with torch.enable_grad(), gradient_memory(model, options.cpu_offload):
                        control.requires_grad_(True)
                        optimizer = torch.optim.Adam([control], lr=0.01)
                        for _ in range(options.iterations):
                            optimizer.zero_grad()
                            candidate = right + 2 * control
                            raw = predict(candidate, current, 1)
                            clean = candidate - current * raw
                            epsilon = candidate + (1 - current) * raw
                            guided_epsilon = epsilon - current * options.strength * (left_in_right - candidate)
                            agreement = 0.5 * options.strength * (clean - reference).square().sum()
                            prior = flow_prior_penalty(control, guided_epsilon - base_epsilon,
                                                       current, following)
                            loss = agreement + prior
                            loss.backward()
                            if not torch.isfinite(control.grad).all():
                                raise FloatingPointError("SyncVC control gradient is non-finite")
                            optimizer.step()
                        record.update(agreement_loss=float(agreement.detach()), prior_loss=float(prior.detach()))
                    control = control.detach()
                candidate = right + 2 * control
                raw = predict(candidate, current, 1) if optimize else base_velocity
                if 0.05 < current < 0.95:
                    raw = raw - current / (1 - current) * options.strength * (left_in_right - candidate)
                states = [flow_euler(left, left_velocity, current, following),
                          flow_euler(candidate, raw, current, following)]
                image = views[1].inverse_view(states[1][0].clone())[None]
                record["control_rms"] = float(control.square().mean().sqrt())
            if not torch.isfinite(image).all():
                raise FloatingPointError("Advanced sampler produced a non-finite image")
            records.append(record)
    if diagnostics is not None:
        diagnostics.update(method=options.method, steps=records,
                           attention_step_schedule="strength*t*(1-t)" if options.method == "anagram_mtl" else None,
                           actual_solver="flow_euler" if options.method == "syncvc" else f"dpm_order_{order}_with_restarts")
    return image.clamp(-1, 1)
