"""Verify component partitions and their use with Iris velocity predictions."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from generate import main
from visual_anagrams.composition import resolve_reduction
from visual_anagrams.iris import PromptEncoding, sample_iris
from visual_anagrams.turbo import sample_turbo
from visual_anagrams.views import get_views


GROUPS = [
    ["low_pass", "high_pass"],
    ["triple_low_pass", "triple_medium_pass", "triple_high_pass"],
    ["grayscale", "color"],
    ["motion", "motion_res"],
    ["scale", "scale"],
]


@pytest.fixture(scope="module", autouse=True)
def small_convolution_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(4)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("names", GROUPS)
def test_components_are_linear_and_reconstruct_prediction(names):
    generator = torch.Generator().manual_seed(123)
    x, y = torch.randn((2, 3, 64, 64), generator=generator)
    views = get_views(names)
    assert resolve_reduction("auto", views) == "sum"
    assert resolve_reduction("auto", names) == "sum"
    parts = []
    for view in views:
        torch.testing.assert_close(view.view(x.clone()), x)
        parts.append(view.inverse_view(x.clone()))
        actual = view.inverse_view((0.4 * x - 0.7 * y).clone())
        expected = 0.4 * view.inverse_view(x.clone()) - 0.7 * view.inverse_view(y.clone())
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(torch.stack(parts).sum(0), x, atol=1e-6, rtol=1e-5)


class OracleVelocity(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.cfg = SimpleNamespace(in_channels=3, patch_size=1)

    def forward(self, x, t, y, y_mask=None):
        return SimpleNamespace(x=(x - y) / (t[:, None, None, None] / 1000))


@pytest.mark.parametrize("turbo", [False, True])
@pytest.mark.parametrize("names", GROUPS)
def test_flow_sampler_recovers_sum_of_differently_prompted_components(names, turbo):
    generator = torch.Generator().manual_seed(12)
    targets = 0.06 * torch.randn((len(names), 3, 64, 64), generator=generator)
    views = get_views(names)
    expected = sum(view.inverse_view(target.clone()) for view, target in zip(views, targets))
    positive = PromptEncoding(targets, torch.ones(len(names), 1))
    options = dict(size=64, num_inference_steps=4, device="cpu", progress=False,
                   generator=torch.Generator().manual_seed(45))
    if turbo:
        result = sample_turbo(OracleVelocity(), positive, views, **options)
    else:
        negative = PromptEncoding(torch.zeros(1, 3, 64, 64), torch.ones(1, 1))
        result = sample_iris(OracleVelocity(), positive, negative, views,
                             guidance_scale=1, **options)
    torch.testing.assert_close(result[0], expected, atol=2e-6, rtol=1e-4)


@pytest.mark.parametrize("reduction", ["mean", "alternate"])
def test_components_reject_invalid_reduction(reduction):
    with pytest.raises(ValueError, match="require sum"):
        resolve_reduction(reduction, ["grayscale", "color"])


@pytest.mark.parametrize("names,args", [
    (["low_pass"], None),
    (["grayscale", "identity"], None),
    (["low_pass", "color"], None),
    (["grayscale", "color", "color"], None),
    (["low_pass", "high_pass"], ["2", "3"]),
    (["scale", "scale"], ["0.7", "0.5"]),
    (["scale"], ["nan"]),
    (["low_pass", "high_pass"], ["0", "0"]),
])
def test_invalid_component_configuration_is_rejected_before_download(names, args):
    command = ["--name", "invalid", "--views", *names, "--prompts", *["test"] * len(names)]
    if args:
        command += ["--view_args", *args]
    with pytest.raises(SystemExit) as exc:
        main(command)
    assert exc.value.code == 2


def test_instance_validation_checks_filter_parameters_and_signed_scale_weights():
    views = get_views(["low_pass", "high_pass"])
    views[1].kernel_size = 31
    with pytest.raises(ValueError, match="matching filter"):
        resolve_reduction("auto", views)
    assert resolve_reduction("auto", get_views(["scale", "scale"], [-6.5, 7.5])) == "sum"


def test_auto_preserves_geometric_view_averaging():
    assert resolve_reduction("auto", get_views(["identity", "rotate_cw"])) == "mean"
    assert resolve_reduction("alternate", ["identity", "flip"]) == "alternate"
