from __future__ import annotations

import argparse

import pytest

torch = pytest.importorskip("torch")

from models.model_configs import get_model_config, get_model_config_metadata
from training.load_and_save import save_model


class _DummyScaler:
    def state_dict(self):
        return {}


def test_training_metadata_records_model_config():
    from train import _build_data_metadata

    args = argparse.Namespace(
        dataset="heat",
        data_path="/tmp/PDEdata",
        data_size=1,
        max_train_samples=2,
        normalization_eps=1e-6,
        model_profile="recommended",
    )
    model_metadata = get_model_config_metadata(
        "heat",
        profile="recommended",
        in_channels=2,
        out_channels=2,
    )

    metadata = _build_data_metadata(
        args=args,
        pde_names=["heat"],
        data=torch.zeros(2, 2, 4, 4),
        label=torch.full((2,), 7, dtype=torch.long),
        loader_metadata={},
        model_config_metadata=model_metadata,
    )

    assert metadata["model_profile"] == "recommended"
    assert metadata["model_config_metadata"]["architecture_family"] == "light_smooth"
    assert metadata["architecture_family"] == "light_smooth"
    assert metadata["attention_resolutions"] == [16]
    assert metadata["channel_mult"] == [1, 2, 4]
    assert metadata["with_fourier_features"] is False
    assert metadata["with_value_fourier_features"] is False
    assert metadata["with_coordinate_fourier_features"] is False
    assert metadata["scalar_conditioning"] is False
    assert "alpha" in metadata["scalar_conditioning_params"]


def test_checkpoint_payload_records_model_config(tmp_path):
    model = torch.nn.Linear(1, 1)
    model_config = get_model_config("heat", profile="recommended", in_channels=2, out_channels=2)
    model_metadata = get_model_config_metadata("heat", profile="recommended", in_channels=2, out_channels=2)
    args = argparse.Namespace(output_dir=str(tmp_path), dataset="heat", use_ema=False, model_profile="recommended")

    save_model(
        args=args,
        epoch=0,
        model=model,
        model_without_ddp=model,
        optimizer=None,
        lr_schedule=None,
        loss_scaler=_DummyScaler(),
        final=True,
        data_shape=(1, 2, 4, 4),
        num_channels=2,
        model_profile="recommended",
        requested_model_profile="auto",
        resolved_model_profile="recommended",
        resume_architecture_metadata={"resume": False},
        model_config=model_config,
        model_config_metadata=model_metadata,
    )

    checkpoint = torch.load(tmp_path / "fm4heat.pth", map_location="cpu", weights_only=False)
    assert checkpoint["checkpoint_schema_version"] == 3
    assert checkpoint["model_profile"] == "recommended"
    assert checkpoint["requested_model_profile"] == "auto"
    assert checkpoint["resolved_model_profile"] == "recommended"
    assert checkpoint["resume_architecture_metadata"] == {"resume": False}
    assert checkpoint["model_config"]["in_channels"] == 2
    assert checkpoint["model_config_metadata"]["architecture_family"] == "light_smooth"
    assert checkpoint["model_config_metadata"]["attention_resolutions"] == [16]
    assert checkpoint["model_config_metadata"]["channel_mult"] == [1, 2, 4]
    assert checkpoint["model_config_metadata"]["with_fourier_features"] is False
    assert checkpoint["model_config_metadata"]["with_value_fourier_features"] is False
    assert checkpoint["model_config_metadata"]["with_coordinate_fourier_features"] is False
