from __future__ import annotations

import argparse

import pytest

torch = pytest.importorskip("torch")

from models.model_configs import get_model_config
from train import resolve_training_model_profile, _resolve_training_model_config
from training.load_and_save import inspect_checkpoint_architecture


def _args(path: str = "", model_profile: str = "auto", allow_override: bool = False):
    return argparse.Namespace(
        resume=path,
        model_profile=model_profile,
        allow_model_profile_override=allow_override,
    )


def _write_checkpoint(
    tmp_path,
    *,
    model_profile: str | None = "heavy",
    include_model_config: bool = True,
    in_channels: int = 2,
    out_channels: int = 2,
):
    checkpoint = {
        "num_channels": in_channels,
        "checkpoint_schema_version": 3,
    }
    if model_profile is not None:
        checkpoint["model_profile"] = model_profile
    if include_model_config:
        cfg_profile = model_profile if model_profile in {"recommended", "light", "base", "heavy", "legacy_base"} else "recommended"
        cfg = get_model_config(
            "heat",
            profile=cfg_profile,
            in_channels=in_channels,
            out_channels=out_channels,
        )
        if model_profile is None:
            cfg.pop("architecture_profile", None)
        checkpoint["model_config"] = cfg
        checkpoint["model_config_metadata"] = dict(cfg)
    path = tmp_path / "checkpoint.pth"
    torch.save(checkpoint, path)
    return path


def test_non_resume_auto_resolves_recommended():
    resolved, metadata = resolve_training_model_profile(_args())

    assert resolved == "recommended"
    assert metadata["resume"] is False
    assert metadata["requested_model_profile"] == "auto"


def test_inspect_checkpoint_architecture_missing_path_raises_clear_file_not_found(tmp_path):
    missing_path = tmp_path / "does-not-exist.pth"

    with pytest.raises(FileNotFoundError, match="resume checkpoint not found"):
        inspect_checkpoint_architecture(missing_path)


def test_resume_auto_uses_checkpoint_profile(tmp_path):
    path = _write_checkpoint(tmp_path, model_profile="heavy")

    resolved, metadata = resolve_training_model_profile(_args(str(path), model_profile="auto"))

    assert resolved == "heavy"
    assert metadata["checkpoint_model_profile"] == "heavy"
    assert metadata["has_model_config"] is True


def test_resume_profile_mismatch_raises_clear_error(tmp_path):
    path = _write_checkpoint(tmp_path, model_profile="heavy")

    with pytest.raises(ValueError) as excinfo:
        resolve_training_model_profile(_args(str(path), model_profile="base"))

    message = str(excinfo.value)
    assert "resume checkpoint architecture/profile does not match requested model_profile" in message
    assert "checkpoint_model_profile=heavy" in message
    assert "requested_model_profile=base" in message
    assert "Use --model_profile heavy or retrain." in message
    assert "To intentionally override, pass --allow_model_profile_override." in message


def test_resume_profile_override_requires_flag(tmp_path):
    path = _write_checkpoint(tmp_path, model_profile="heavy")

    with pytest.raises(ValueError):
        resolve_training_model_profile(_args(str(path), model_profile="base"))

    with pytest.warns(RuntimeWarning, match="allow_model_profile_override"):
        resolved, metadata = resolve_training_model_profile(
            _args(str(path), model_profile="base", allow_override=True)
        )

    assert resolved == "base"
    assert metadata["override"] is True


def test_resume_legacy_checkpoint_requires_explicit_profile(tmp_path):
    path = tmp_path / "legacy.pth"
    torch.save({"model": {}, "num_channels": 2}, path)

    with pytest.raises(ValueError, match="lacks architecture metadata"):
        resolve_training_model_profile(_args(str(path), model_profile="auto"))

    resolved, metadata = resolve_training_model_profile(
        _args(str(path), model_profile="legacy_base")
    )

    assert resolved == "legacy_base"
    assert metadata["checkpoint_lacks_architecture_metadata"] is True


def test_resume_checkpoint_model_config_channel_mismatch_raises(tmp_path):
    path = _write_checkpoint(tmp_path, model_profile="heavy", in_channels=3, out_channels=3)
    resolved, metadata = resolve_training_model_profile(_args(str(path), model_profile="auto"))

    with pytest.raises(ValueError, match="checkpoint architecture channel count does not match current training data"):
        _resolve_training_model_config(
            model_arch="heat",
            pde_names=["heat"],
            resolved_model_profile=resolved,
            resume_arch_meta=metadata,
            num_channels=2,
        )
