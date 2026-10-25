"""Showcase filters must display the projections actually used by the sampler."""

import pytest
import torch

from scripts.build_showcase_v3 import gaussian_component, motion_component
from visual_anagrams.views import get_views


@pytest.fixture(scope='module', autouse=True)
def limit_threads():
    previous=torch.get_num_threads()
    torch.set_num_threads(4)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize('side',[64,256])
def test_separable_showcase_gaussian_matches_sampler_projection(side):
    x=torch.randn(3,side,side,generator=torch.Generator().manual_seed(1))
    view=get_views(['low_pass'])[0]
    factor=side//64
    kernel=view.kernel_size*factor+((factor+1)%2)
    expected=view.inverse_view(x.clone())
    actual=gaussian_component(x,view.sigma*factor,kernel)
    torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-5)


@pytest.mark.parametrize('side',[64,256])
def test_showcase_diagonal_average_matches_sampler_including_edges(side):
    x=torch.randn(3,side,side,generator=torch.Generator().manual_seed(2))
    view=get_views(['motion'])[0]
    factor=side//64
    kernel=view.size*factor+((factor+1)%2)
    expected=view.save_view(x.clone())
    actual=motion_component(x,kernel)
    torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-5)
