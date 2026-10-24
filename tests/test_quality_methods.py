"""Research sampler controls must reproduce the production integration."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from scripts.quality_methods import ResearchMixer, sample_research
from visual_anagrams.iris import PromptEncoding, sample_iris
from visual_anagrams.turbo import sample_turbo
from visual_anagrams.views import get_views


class SmallVelocity(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.cfg = SimpleNamespace(in_channels=3, patch_size=1)

    def forward(self, x, t, y, y_mask=None):
        return SimpleNamespace(x=0.2 * x + 0.001 * y.mean() + 0.0001 * t[:, None, None, None])


@pytest.mark.parametrize("turbo", [False, True])
@pytest.mark.parametrize("steps", [4, 8, 16])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_research_mean_control_matches_real_sampler(turbo, steps, dtype):
    positive = PromptEncoding(torch.linspace(0.01, 0.9, 24).reshape(2, 4, 3).to(dtype), torch.ones(2, 4))
    negative = PromptEncoding(torch.zeros(1, 4, 3), torch.ones(1, 4))
    views = get_views(["identity", "rotate_cw"])
    actual = sample_research(SmallVelocity(), positive, negative, views, ResearchMixer("mean", views),
                             turbo=turbo, steps=steps, seed=42, device="cpu", size=8)
    options = dict(size=8, num_inference_steps=steps, device="cpu", progress=False,
                   generator=torch.Generator().manual_seed(42))
    expected = (sample_turbo(SmallVelocity(), positive, views, **options) if turbo else
                sample_iris(SmallVelocity(), positive, negative, views, guidance_scale=3, **options))
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


def test_preview_balancing_favors_weak_view_and_bounds_weights():
    views = get_views(["identity", "rotate_180"])
    mixer = ResearchMixer("preview-balanced", views, scorer=lambda *_: [0.05, 0.8])
    image = torch.zeros(1, 3, 8, 8)
    mixer(image, [image + 0.1, image - 0.1], 0.7, 0)
    assert 0.5 < mixer.weights[0] <= 0.75
    assert 0.25 <= mixer.weights[1] < 0.5
    torch.testing.assert_close(mixer.weights.sum(), torch.tensor(1.0))


@pytest.mark.parametrize("time", [1.0, 0.999, 0.7, 0.01])
def test_variance_correction_preserves_identical_predictions(time):
    torch.manual_seed(3)
    image, velocity = torch.randn(2, 1, 3, 8, 8)
    mixer = ResearchMixer("bounded-variance", [None, None])
    actual = mixer(image, [velocity, velocity], time, 0)
    torch.testing.assert_close(actual, velocity, atol=1e-6, rtol=1e-5)


def test_variance_gain_is_bounded_for_conflicting_predictions():
    torch.manual_seed(4)
    image, velocity = torch.randn(2, 1, 3, 8, 8)
    mixer = ResearchMixer("bounded-variance", [None, None])
    actual = mixer(image, [velocity * 10, velocity * -10], 0.7, 0)
    assert torch.isfinite(actual).all()
    assert 1 <= mixer.history[-1]["gain"] <= 1.100001


def test_constant_identical_noise_does_not_trigger_variance_gain():
    image = torch.zeros(1, 3, 8, 8)
    velocity = image + 0.1
    mixer = ResearchMixer("bounded-variance", [None, None])
    actual = mixer(image, [velocity, velocity], 0.7, 0)
    torch.testing.assert_close(actual, velocity)
    assert mixer.history[-1]["gain"] == 1
