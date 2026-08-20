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
        sample = torch.zeros(
            kwargs["batch_size"],
            kwargs["num_channels"],
            kwargs["resolution"],
            kwargs["resolution"],
            device=kwargs["device"],
            dtype=kwargs["dtype"],
        )
        sample[:, 0] = 3.0
        sample[:, 1] = 7.0
        return sample

    def fake_residual(pde_name, coef, sol, *, pde_params, residual_mode):
        captured["residual_pde_params"] = pde_params
        captured["residual_coef"] = coef
        captured["residual_sol"] = sol
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
                "observed_initial": torch.full((3, 1, 4, 4), 101.0),
                "trajectory": torch.full((3, 5, 1, 4, 4), 202.0),
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
    assert "observed_initial" not in pde_params
    assert "trajectory" not in pde_params
    assert torch.allclose(captured["residual_coef"], torch.full((2, 1, 4, 4), 3.0))
    assert torch.allclose(captured["residual_sol"], torch.full((2, 1, 4, 4), 7.0))
    assert stats["eval_pde_residual_status"] == "ok"
    assert stats["eval_pde_uses_ground_truth_fields"] is False
    assert stats["eval_pde_excluded_ground_truth_field_params"] == ["observed_initial", "trajectory"]


def test_periodic_training_eval_rejects_near_endpoint_observations(monkeypatch, tmp_path):
    import train
    from data.transform import PDEStandardizer

    captured = {}

    def fake_sample(**kwargs):
        return torch.zeros(
            kwargs["batch_size"],
            kwargs["num_channels"],
            kwargs["resolution"],
            kwargs["resolution"],
            device=kwargs["device"],
            dtype=kwargs["dtype"],
        )

    def fake_residual(pde_name, coef, sol, *, pde_params, residual_mode):
        captured["pde_params"] = pde_params
        return SimpleNamespace(
            residual=torch.zeros_like(sol),
            status="approximate",
            metadata={"resolved_residual_mode": residual_mode},
        )

    monkeypatch.setattr(train, "_euler_flow_sample", fake_sample)
    monkeypatch.setattr(train, "compute_pde_residual", fake_residual)
    monkeypatch.setattr(train, "_save_eval_figure", lambda *args, **kwargs: None)

    q_dt = torch.full((3, 1, 4, 4), 100.0)
    q_tm = torch.full((3, 1, 4, 4), -100.0)
    mask_0 = torch.zeros_like(q_dt)
    mask_T = torch.zeros_like(q_tm)
    mask_0[:, :, 1, 2] = 1.0
    mask_T[:, :, 2, 1] = 1.0
    q_dt[:, :, 1, 2] = torch.tensor([1.0, 2.0, 3.0]).reshape(3, 1)
    q_tm[:, :, 2, 1] = torch.tensor([4.0, 5.0, 6.0]).reshape(3, 1)
    val_metadata = {
        "heat": {
            "pde_params": {
                "alpha": torch.tensor([0.1, 0.2, 0.3]),
                "near_endpoint_temporal": {
                    "q_dt": q_dt,
                    "q_T_minus_dt": q_tm,
                    "dt": torch.tensor([0.1, 0.2, 0.3]),
                    "mask_0": mask_0,
                    "mask_T": mask_T,
                },
            }
        }
    }
    args = argparse.Namespace(
        eval_num_samples=2,
        eval_num_steps=1,
        eval_residual_mode="near_endpoint_temporal",
        output_dir=str(tmp_path),
        seed=0,
    )

    with pytest.raises(ValueError, match="unconditional"):
        train._run_periodic_flow_eval(
            args=args,
            model=torch.nn.Identity(),
            normalizer=PDEStandardizer.identity(2, channel_names=["u0", "uT"], pde="heat"),
            pde_name="heat",
            epoch=0,
            num_channels=2,
            resolution=4,
            device=torch.device("cpu"),
            val_loader_metadata=val_metadata,
        )


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
