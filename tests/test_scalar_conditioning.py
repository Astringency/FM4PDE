from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from data.scalar_conditioning import (
    fit_scalar_conditioning,
    scalar_conditioning_metadata_from_checkpoint_payload,
    standardize_scalar_conditioning,
)


def test_fit_and_checkpoint_standardize_scalar_conditioning():
    train_meta = {
        "heat": {
            "pde_params": {
                "alpha": torch.tensor([1.0, 3.0]),
                "T": torch.tensor([2.0, 2.0]),
            }
        }
    }
    val_meta = {
        "heat": {
            "pde_params": {
                "alpha": torch.tensor([5.0]),
                "T": torch.tensor([2.0]),
            }
        }
    }
    model_config = {
        "scalar_conditioning": True,
        "scalar_conditioning_dim": 2,
        "scalar_conditioning_params": ["alpha", "T"],
    }

    train_scalar, val_scalar, metadata = fit_scalar_conditioning(
        model_config=model_config,
        pde_names=["heat"],
        train_loader_metadata=train_meta,
        val_loader_metadata=val_meta,
        train_sample_count=2,
        val_sample_count=1,
        eps=1e-6,
    )
    checkpoint_meta = scalar_conditioning_metadata_from_checkpoint_payload(
        {
            "model_config": model_config,
            "data_metadata": {
                "scalar_conditioning_enabled": True,
                "scalar_conditioning_dim": metadata["dim"],
                "scalar_conditioning_params": metadata["params"],
                "scalar_conditioning_mean": metadata["mean"],
                "scalar_conditioning_std": metadata["std"],
                "scalar_conditioning_normalization": metadata["normalization"],
            },
        }
    )
    runtime = standardize_scalar_conditioning(
        pde_params={"alpha": torch.tensor([5.0]), "T": torch.tensor([2.0])},
        metadata=checkpoint_meta,
        expected_sample_count=1,
        source_name="test ground truth",
    )

    assert torch.allclose(train_scalar, torch.tensor([[-1.0, 0.0], [1.0, 0.0]]))
    assert torch.allclose(val_scalar, torch.tensor([[3.0, 0.0]]))
    assert torch.allclose(runtime, val_scalar)
    assert checkpoint_meta["normalization"] == "train_mean_std"


def test_checkpoint_scalar_conditioning_requires_saved_stats():
    with pytest.raises(ValueError, match="mean/std"):
        scalar_conditioning_metadata_from_checkpoint_payload(
            {
                "model_config": {
                    "scalar_conditioning": True,
                    "scalar_conditioning_dim": 1,
                    "scalar_conditioning_params": ["alpha"],
                },
                "data_metadata": {
                    "scalar_conditioning_enabled": True,
                    "scalar_conditioning_dim": 1,
                    "scalar_conditioning_params": ["alpha"],
                },
            }
        )


def test_standardize_scalar_conditioning_fails_for_missing_param_and_batch_mismatch():
    metadata = {
        "enabled": True,
        "params": ["alpha", "T"],
        "dim": 2,
        "mean": [2.0, 2.0],
        "std": [1.0, 1.0],
    }

    with pytest.raises(ValueError, match="missing"):
        standardize_scalar_conditioning(
            pde_params={"alpha": torch.tensor([1.0])},
            metadata=metadata,
            expected_sample_count=1,
            source_name="test ground truth",
        )

    with pytest.raises(ValueError, match="expected 2"):
        standardize_scalar_conditioning(
            pde_params={"alpha": torch.tensor([1.0]), "T": torch.tensor([2.0])},
            metadata=metadata,
            expected_sample_count=2,
            source_name="test ground truth",
        )


def test_periodic_eval_uses_same_validation_rows_for_scalar_and_residual(monkeypatch, tmp_path):
    import train
    from data.transform import PDEStandardizer

    captured: dict[str, object] = {}

    def fake_sample(**kwargs):
        captured["model_extra"] = kwargs["model_extra"]
        return torch.zeros(
            kwargs["batch_size"],
            kwargs["num_channels"],
            kwargs["resolution"],
            kwargs["resolution"],
            device=kwargs["device"],
            dtype=kwargs["dtype"],
        )

    def fake_residual(pde_name, coef, sol, *, pde_params, residual_mode):
        captured["residual_pde_params"] = pde_params
        return SimpleNamespace(
            residual=torch.zeros_like(sol),
            status="ok",
            metadata={"resolved_residual_mode": residual_mode},
        )

    monkeypatch.setattr(train, "_euler_flow_sample", fake_sample)
    monkeypatch.setattr(train, "compute_pde_residual", fake_residual)
    monkeypatch.setattr(train, "_save_eval_figure", lambda *args, **kwargs: None)

    args = argparse.Namespace(
        eval_num_samples=2,
        eval_num_steps=1,
        eval_residual_mode="auto",
        output_dir=str(tmp_path),
        seed=0,
    )
    normalizer = PDEStandardizer.identity(2, channel_names=["u0", "uT"], pde="heat")
    scalar_meta = {
        "enabled": True,
        "params": ["alpha", "T"],
        "dim": 2,
        "mean": [2.0, 2.0],
        "std": [1.0, 1.0],
        "normalization": "train_mean_std",
    }
    val_metadata = {
        "heat": {
            "pde_params": {
                "alpha": torch.tensor([1.0, 3.0, 5.0]),
                "T": torch.tensor([2.0, 2.0, 2.0]),
            }
        }
    }

    stats = train._run_periodic_flow_eval(
        args=args,
        model=torch.nn.Identity(),
        normalizer=normalizer,
        pde_name="heat",
        epoch=0,
        num_channels=2,
        resolution=4,
        device=torch.device("cpu"),
        val_loader_metadata=val_metadata,
        scalar_conditioning_metadata=scalar_meta,
    )

    scalar = captured["model_extra"]["scalar_conditioning"]
    pde_params = captured["residual_pde_params"]
    assert torch.allclose(scalar, torch.tensor([[-1.0, 0.0], [1.0, 0.0]]))
    assert torch.allclose(pde_params["alpha"], torch.tensor([1.0, 3.0]))
    assert torch.allclose(pde_params["T"], torch.tensor([2.0, 2.0]))
    assert stats["eval_pde_residual_status"] == "ok"


def test_sampling_runner_builds_scalar_extra_from_checkpoint_and_ground_truth():
    from sampling.config import AblationConfig
    from sampling.runner import _scalar_conditioning_for_sampling

    config = AblationConfig(pde="heat", batch_size=2, offset=4)
    gt = SimpleNamespace(
        pair=torch.zeros(2, 2, 4, 4, dtype=torch.float32),
        pde_params={
            "alpha": torch.tensor([1.0, 3.0]),
            "T": torch.tensor([2.0, 2.0]),
        },
    )
    checkpoint_payload = {
        "model_config": {
            "scalar_conditioning": True,
            "scalar_conditioning_dim": 2,
            "scalar_conditioning_params": ["alpha", "T"],
        },
        "data_metadata": {
            "scalar_conditioning_enabled": True,
            "scalar_conditioning_dim": 2,
            "scalar_conditioning_params": ["alpha", "T"],
            "scalar_conditioning_mean": [2.0, 2.0],
            "scalar_conditioning_std": [1.0, 1.0],
            "scalar_conditioning_normalization": "train_mean_std",
        },
    }

    model_extra, metadata = _scalar_conditioning_for_sampling(
        checkpoint_payload=checkpoint_payload,
        gt=gt,
        config=config,
        device=torch.device("cpu"),
    )

    assert metadata["enabled"] is True
    assert metadata["source"] == "data_aligned_ground_truth"
    assert metadata["standardization"] == "train_mean_std"
    assert metadata["params"] == ["alpha", "T"]
    assert torch.allclose(
        model_extra["scalar_conditioning"],
        torch.tensor([[-1.0, 0.0], [1.0, 0.0]]),
    )


def test_sampling_runner_non_scalar_checkpoint_keeps_old_behavior():
    from sampling.config import AblationConfig
    from sampling.runner import _scalar_conditioning_for_sampling

    config = AblationConfig(pde="heat", batch_size=1, offset=0)
    gt = SimpleNamespace(
        pair=torch.zeros(1, 2, 4, 4, dtype=torch.float32),
        pde_params={"alpha": torch.tensor([1.0])},
    )

    model_extra, metadata = _scalar_conditioning_for_sampling(
        checkpoint_payload={"model_config": {"scalar_conditioning": False}},
        gt=gt,
        config=config,
        device=torch.device("cpu"),
    )

    assert model_extra is None
    assert metadata["enabled"] is False
