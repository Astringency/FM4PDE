from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from data.transform import PDEStandardizer
from models.model_configs import instantiate_model
from sampling.model_io import WrappedModel, load_fm4pde_checkpoint_bundle


def _tiny_model_config() -> dict:
    return {
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
        "use_scale_shift_norm": False,
        "resblock_updown": False,
        "with_value_fourier_features": False,
        "architecture_family": "unit_test_family",
        "architecture_profile": "unit_test_profile",
        "axis_semantics": "spatial_2d",
        "scalar_conditioning": False,
        "scalar_conditioning_params": ["alpha"],
    }


def test_checkpoint_model_config_is_used_for_sampling(tmp_path):
    cfg = _tiny_model_config()
    model = instantiate_model("heat", use_ema=False, model_config=cfg)
    checkpoint_path = tmp_path / "fm4heat.pth"
    torch.save(
        {
            "model": model.state_dict(),
            "normalizer": PDEStandardizer.identity(2, channel_names=["u0", "uT"]).state_dict(),
            "num_channels": 2,
            "model_profile": "unit_test_profile",
            "model_config": cfg,
            "model_config_metadata": cfg,
            "checkpoint_schema_version": 3,
        },
        checkpoint_path,
    )

    wrapped, _, payload = load_fm4pde_checkpoint_bundle(
        str(checkpoint_path),
        "heat",
        device=torch.device("cpu"),
        wrap=True,
        model_profile="unit_test_profile",
    )

    assert isinstance(wrapped, WrappedModel)
    assert payload["selected_model_profile"] == "unit_test_profile"
    assert payload["selected_architecture_family"] == "unit_test_family"
    assert payload["runtime_requested_model_profile"] == "unit_test_profile"
    assert all(not parameter.requires_grad for parameter in wrapped.model.parameters())


def test_sampling_model_io_derives_canonical_fourier_metadata(tmp_path):
    cfg = _tiny_model_config()
    cfg.update(
        {
            "with_value_fourier_features": True,
            "with_coordinate_fourier_features": True,
        }
    )
    model = instantiate_model("heat", use_ema=False, model_config=cfg)
    checkpoint_path = tmp_path / "fm4heat_fourier_metadata_missing.pth"
    torch.save(
        {
            "model": model.state_dict(),
            "normalizer": PDEStandardizer.identity(2, channel_names=["u0", "uT"]).state_dict(),
            "num_channels": 2,
            "model_profile": "unit_test_profile",
            "model_config": cfg,
            "model_config_metadata": cfg,
            "checkpoint_schema_version": 3,
        },
        checkpoint_path,
    )

    _, _, payload = load_fm4pde_checkpoint_bundle(
        str(checkpoint_path),
        "heat",
        device=torch.device("cpu"),
        wrap=True,
        model_profile="unit_test_profile",
    )

    metadata = payload["selected_model_config_metadata"]
    assert payload["selected_model_profile"] == "unit_test_profile"
    assert payload["runtime_requested_model_profile"] == "unit_test_profile"
    assert metadata["with_value_fourier_features"] is True
    assert metadata["with_coordinate_fourier_features"] is True
    assert metadata["value_fourier_feature_channels"] == 8
    assert metadata["coordinate_fourier_feature_channels"] == 18
    assert metadata["fourier_feature_channels"] == 26
    assert metadata["effective_in_channels"] == 28
    assert metadata["coordinate_fourier_coord_range"] == "unit"
    assert metadata["coordinate_fourier_include_raw_coords"] is True


def test_checkpoint_without_model_config_is_rejected(tmp_path):
    cfg = _tiny_model_config()
    model = instantiate_model("heat", use_ema=False, model_config=cfg)
    checkpoint_path = tmp_path / "fm4heat_invalid.pth"
    torch.save(
        {
            "model": model.state_dict(),
            "normalizer": PDEStandardizer.identity(2, channel_names=["u0", "uT"]).state_dict(),
            "num_channels": 2,
            "checkpoint_schema_version": 3,
        },
        checkpoint_path,
    )

    with pytest.raises(ValueError, match="Checkpoint has no model_config"):
        load_fm4pde_checkpoint_bundle(
            str(checkpoint_path),
            "heat",
            device=torch.device("cpu"),
            wrap=True,
        )
