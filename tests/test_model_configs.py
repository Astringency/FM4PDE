import gc

import pytest

torch = pytest.importorskip("torch")

from models.model_configs import MODEL_CONFIGS, instantiate_model


EXPECTED_PDES = {
    "darcy",
    "poisson",
    "helmholtz",
    "nsnonbounded",
    "burger",
    "reaction_diffusion",
    "shallow_water",
    "heat",
    "wave",
    "advection_diffusion",
    "steady_heat_conduction",
}


def test_all_pdes_have_model_configs():
    assert EXPECTED_PDES.issubset(MODEL_CONFIGS)


@pytest.mark.parametrize("pde", sorted(EXPECTED_PDES))
def test_instantiate_model_with_channel_override(pde):
    model = instantiate_model(pde, use_ema=False, in_channels=3, out_channels=3)
    assert model.in_channels == 3
    assert model.out_channels == 3
    del model
    gc.collect()
