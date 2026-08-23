from __future__ import annotations

import argparse

import pytest

torch = pytest.importorskip("torch")

from sampling.model_io import WrappedModel, load_fm4pde_checkpoint_bundle
from data.transform import PDEStandardizer
from models.model_configs import MODEL_CONFIGS_RECOMMENDED, get_model_config_metadata, instantiate_model
from training.load_and_save import save_model


class _DummyScaler:
    def state_dict(self):
        return {}


def _tiny_heat_config() -> dict:
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
    }


@pytest.fixture(autouse=True)
def tiny_heat_model(monkeypatch):
    monkeypatch.setitem(MODEL_CONFIGS_RECOMMENDED, "heat", _tiny_heat_config())


def _checkpoint_args(tmp_path, use_ema: bool) -> argparse.Namespace:
    return argparse.Namespace(output_dir=str(tmp_path), dataset="heat", use_ema=use_ema)


def _save_heat_checkpoint(tmp_path, model, use_ema: bool):
    normalizer = PDEStandardizer.identity(2, channel_names=["u0", "uT"], pde="heat")
    save_model(
        args=_checkpoint_args(tmp_path, use_ema=use_ema),
        epoch=0,
        model=model,
        model_without_ddp=model,
        optimizer=None,
        lr_schedule=None,
        loss_scaler=_DummyScaler(),
        final=True,
        normalizer=normalizer,
        data_shape=(1, 2, 4, 4),
        num_channels=2,
        model_profile="recommended",
        model_config=MODEL_CONFIGS_RECOMMENDED["heat"],
        model_config_metadata=get_model_config_metadata("heat", profile="recommended", in_channels=2, out_channels=2),
    )
    return tmp_path / "fm4heat.pth"


def _first_different_key(left: dict, right: dict) -> str:
    for key, value in left.items():
        if key in right and torch.is_tensor(value) and not torch.allclose(value, right[key]):
            return key
    raise AssertionError("Expected at least one differing tensor key")


def test_non_ema_checkpoint_loads_for_sampling(tmp_path):
    model = instantiate_model("heat", use_ema=False, in_channels=2, out_channels=2)
    checkpoint_path = _save_heat_checkpoint(tmp_path, model, use_ema=False)

    wrapped, _, payload = load_fm4pde_checkpoint_bundle(
        str(checkpoint_path),
        "heat",
        device=torch.device("cpu"),
        wrap=True,
        prefer_ema=True,
        model_profile="recommended",
    )

    assert isinstance(wrapped, WrappedModel)
    assert payload["checkpoint_schema_version"] == 3
    assert payload["selected_inference_weight"] == "raw"
    assert payload["has_ema"] is False


def test_ema_checkpoint_loads_ema_or_raw_plain_weights(tmp_path):
    model = instantiate_model("heat", use_ema=True, in_channels=2, out_channels=2)
    with torch.no_grad():
        for param in model.model.parameters():
            param.add_(1.0)
            break
    model.update_ema()
    checkpoint_path = _save_heat_checkpoint(tmp_path, model, use_ema=True)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    assert payload["has_ema"] is True
    assert payload["model_ema"] is not None
    assert any(key.startswith("model.") for key in payload["model_for_resume"])
    assert not any(key.startswith("model.") for key in payload["model_ema"])
    assert not any(key.startswith("shadow_params.") for key in payload["model_ema"])

    differing_key = _first_different_key(payload["model"], payload["model_ema"])

    wrapped_ema, _, ema_payload = load_fm4pde_checkpoint_bundle(
        str(checkpoint_path),
        "heat",
        device=torch.device("cpu"),
        wrap=True,
        prefer_ema=True,
        model_profile="recommended",
    )
    assert isinstance(wrapped_ema, WrappedModel)
    assert ema_payload["selected_inference_weight"] == "ema"
    assert torch.allclose(wrapped_ema.model.state_dict()[differing_key], payload["model_ema"][differing_key])

    wrapped_raw, _, raw_payload = load_fm4pde_checkpoint_bundle(
        str(checkpoint_path),
        "heat",
        device=torch.device("cpu"),
        wrap=True,
        prefer_ema=False,
    )
    assert raw_payload["selected_inference_weight"] == "raw"
    assert torch.allclose(wrapped_raw.model.state_dict()[differing_key], payload["model"][differing_key])
