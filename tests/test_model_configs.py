import gc

import pytest

torch = pytest.importorskip("torch")

from models.model_configs import (
    MODEL_CONFIGS,
    get_model_config,
    get_model_config_metadata,
    instantiate_model,
)
from models.unet import fourier_feature_channel_count


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


def test_recommended_model_configs_cover_all_pdes():
    assert EXPECTED_PDES.issubset(MODEL_CONFIGS)
    for pde in EXPECTED_PDES:
        cfg = get_model_config(pde, profile="recommended")
        metadata = get_model_config_metadata(pde, profile="recommended")
        assert cfg["architecture_profile"] == "recommended"
        assert metadata["architecture_profile"] == "recommended"
        assert metadata["in_channels"] == cfg["in_channels"]
        assert metadata["out_channels"] == cfg["out_channels"]


def test_model_config_metadata_does_not_pass_to_unet(monkeypatch):
    tiny = {
        "in_channels": 2,
        "model_channels": 32,
        "out_channels": 2,
        "num_res_blocks": 1,
        "attention_resolutions": (),
        "dropout": 0.0,
        "channel_mult": (1,),
        "conv_resample": True,
        "dims": 2,
        "num_classes": None,
        "use_checkpoint": False,
        "num_heads": 1,
        "num_head_channels": -1,
        "num_heads_upsample": -1,
        "use_scale_shift_norm": False,
        "resblock_updown": False,
        "use_new_attention_order": False,
        "with_fourier_features": False,
        "architecture_family": "unit_test_family",
        "architecture_profile": "recommended",
        "axis_semantics": "spatial_2d",
        "scalar_conditioning": False,
        "scalar_conditioning_params": ["alpha"],
        "notes": "metadata-only keys should be filtered",
    }
    monkeypatch.setitem(MODEL_CONFIGS, "heat", tiny)

    model = instantiate_model("heat", use_ema=False, in_channels=2, out_channels=2)

    assert model.out_channels == 2
    del model
    gc.collect()


@pytest.mark.parametrize("channels", [1, 2, 4])
def test_value_fourier_feature_channel_count_matches_forward(channels):
    cfg = {
        "in_channels": channels,
        "model_channels": 32,
        "out_channels": channels,
        "num_res_blocks": 1,
        "attention_resolutions": (),
        "dropout": 0.0,
        "channel_mult": (1,),
        "conv_resample": True,
        "dims": 2,
        "num_classes": None,
        "use_checkpoint": False,
        "num_heads": 1,
        "num_head_channels": -1,
        "num_heads_upsample": -1,
        "use_scale_shift_norm": False,
        "resblock_updown": False,
        "use_new_attention_order": False,
        "with_value_fourier_features": True,
    }
    model = instantiate_model("heat", use_ema=False, model_config=cfg)
    expected_input_channels = channels + fourier_feature_channel_count(
        channels,
        start=cfg.get("fourier_feature_start", 6),
        stop=cfg.get("fourier_feature_stop", 8),
        step=cfg.get("fourier_feature_step", 1),
    )
    assert model.input_blocks[0][0].weight.shape[1] == expected_input_channels

    x = torch.randn(1, channels, 8, 8)
    t = torch.tensor([0.5])
    out = model(x, t, extra={})

    assert out.shape == x.shape
    del model
    gc.collect()


def test_attention_resolution_not_high_res_for_recommended():
    for pde in EXPECTED_PDES:
        metadata = get_model_config_metadata(pde, profile="recommended")
        assert tuple(metadata["attention_resolutions"]) != (2,)


def test_burger_axis_semantics():
    metadata = get_model_config_metadata("burger", profile="recommended")
    assert metadata["architecture_family"] == "full_time_space"
    assert metadata["axis_semantics"] == "BCHW_as_time_space_H_time_W_space"
    assert metadata["with_fourier_features"] is True
    assert metadata["with_value_fourier_features"] is True
    assert metadata["with_coordinate_fourier_features"] is True


def test_burger_recommended_uses_coordinate_fourier():
    metadata = get_model_config_metadata("burger", profile="recommended")
    assert metadata["with_coordinate_fourier_features"] is True
    assert metadata["axis_semantics"] == "BCHW_as_time_space_H_time_W_space"


@pytest.mark.parametrize("pde", ["helmholtz", "wave", "steady_heat_conduction"])
def test_helmholtz_wave_steady_heat_coordinate_fourier(pde):
    metadata = get_model_config_metadata(pde, profile="recommended")
    assert metadata["with_coordinate_fourier_features"] is True


def test_ns_coordinate_fourier_default_false():
    metadata = get_model_config_metadata("nsnonbounded", profile="recommended")
    assert metadata["with_coordinate_fourier_features"] is False
    assert "periodic" in metadata["notes"]
    assert "translation-equivariance" in metadata["notes"]


@pytest.mark.parametrize("pde", ["wave", "shallow_water", "nsnonbounded"])
def test_temporal_heavy_pdes_are_heavier(pde):
    metadata = get_model_config_metadata(pde, profile="recommended")
    assert metadata["architecture_family"] == "temporal_endpoint_heavy"
    assert metadata["model_channels"] >= 160


@pytest.mark.parametrize("pde", ["poisson"])
def test_light_pdes_are_not_heavy(pde):
    metadata = get_model_config_metadata(pde, profile="recommended")
    assert metadata["architecture_family"] == "light_smooth"
    assert metadata["model_channels"] <= 128


def test_heat_recommended_uses_base_architecture():
    metadata = get_model_config_metadata("heat", profile="recommended")
    base_metadata = get_model_config_metadata("heat", profile="base")

    assert metadata["architecture_family"] == "temporal_endpoint_base"
    assert metadata["model_channels"] == base_metadata["model_channels"] == 128
    assert metadata["num_res_blocks"] == base_metadata["num_res_blocks"] == 4
    assert tuple(metadata["channel_mult"]) == tuple(base_metadata["channel_mult"]) == (1, 2, 4)
    assert tuple(metadata["attention_resolutions"]) == tuple(base_metadata["attention_resolutions"]) == (16,)
    assert metadata["with_value_fourier_features"] is False
    assert metadata["with_coordinate_fourier_features"] is False
