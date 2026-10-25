"""Lightweight clean-preview balancing, promoted from the tested research method."""

import torch
import torch.nn.functional as F

CLIP_ID = "openai/clip-vit-base-patch32"
CLIP_REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"


class CleanPreviewScorer:
    """Score each transformed clean preview against its own prompt on CPU."""
    def __init__(self, prompts):
        from transformers import CLIPModel, CLIPProcessor

        self.model = CLIPModel.from_pretrained(CLIP_ID, revision=CLIP_REVISION).float().eval()
        self.processor = CLIPProcessor.from_pretrained(CLIP_ID, revision=CLIP_REVISION, use_fast=False)
        with torch.inference_mode():
            tokens = self.processor(text=prompts, return_tensors="pt", padding=True)
            self.text = F.normalize(self.model.get_text_features(**tokens), dim=-1)

    def __call__(self, preview, views):
        from PIL import Image

        images = []
        for view in views:
            pixels = view.view(preview.clone()).clamp(-1, 1).add(1).mul(127.5)
            array = pixels.permute(1, 2, 0).round().byte().cpu().numpy()
            images.append(Image.fromarray(array))
        with torch.inference_mode():
            inputs = self.processor(images=images, return_tensors="pt")
            encoded = F.normalize(self.model.get_image_features(**inputs), dim=-1)
            return (encoded @ self.text.T).diag().tolist()


class AdaptiveBalancer:
    """Give weaker views more weight, with bounds and smoothing to limit feedback.

    Same fixed settings as the earlier Turbo pilot: score every two steps below
    t=.95, softmax(-10*score), clamp desired weights to [.25,.75], 50% smoothing.
    A new instance is required for each generated sample.
    """
    def __init__(self, views, scorer):
        if len(views) != 2:
            raise ValueError("Adaptive balancing currently requires exactly two views")
        if scorer is None:
            raise ValueError("Adaptive balancing requires a preview scorer")
        self.views, self.scorer = views, scorer
        self.weights = None
        self.history = []

    def __call__(self, image, velocities, time, step):
        predictions = torch.stack(velocities)
        if self.weights is None:
            self.weights = torch.full((2,), .5, device=image.device)
        scores = None
        if time < .95 and step % 2 == 0:
            preview = image - time * predictions.mean(0)
            scores = self.scorer(preview[0], self.views)
            values = torch.tensor(scores, device=image.device, dtype=torch.float32)
            if values.shape != (2,) or not torch.isfinite(values).all():
                raise FloatingPointError("Adaptive balancing requires two finite preview scores")
            desired = torch.softmax(-10. * values, dim=0).clamp(.25, .75)
            desired /= desired.sum()
            self.weights = .5 * self.weights + .5 * desired
        weights = self.weights.reshape(2, *([1] * (predictions.ndim - 1)))
        combined = (predictions * weights).sum(0)
        self.history.append(dict(step=step, time=time, scores=scores, weights=self.weights.tolist()))
        if not torch.isfinite(combined).all():
            raise FloatingPointError("Adaptive balancing produced non-finite predictions")
        return combined

    def diagnostics(self):
        return dict(method="adaptive", scorer=dict(model=CLIP_ID, revision=CLIP_REVISION, device="cpu"),
                    interval=2, temperature=10., desired_weight_bounds=[.25,.75], smoothing=.5, steps=self.history)
