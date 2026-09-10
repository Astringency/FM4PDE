import numpy as np
import pytest
import torch

from scripts.train.evaluate_resume_study import field_scores


@pytest.mark.parametrize('pde', ['poisson', 'nsnonbounded', 'burger'])
def test_equal_power_opposite_modes_are_reported_as_inaccurate(pde):
    axis = torch.arange(32, dtype=torch.float64) / 32
    field = (torch.sin(2 * torch.pi * axis)[None, :] +
             0.3 * torch.cos(4 * torch.pi * axis)[:, None])[None, None]
    score = field_scores(-field, field, pde)
    np.testing.assert_allclose(score['relative_l2'], 2)
    np.testing.assert_allclose(score['low_relative_l2'], 2)
    np.testing.assert_allclose(score['low_power_ratio'], 1)
    np.testing.assert_allclose(score['low_alignment'], -1)
    exact = field_scores(field, field, pde)
    np.testing.assert_allclose(exact['relative_l2'], 0)
    np.testing.assert_allclose(exact['low_relative_l2'], 0)
    np.testing.assert_allclose(exact['low_alignment'], 1)
