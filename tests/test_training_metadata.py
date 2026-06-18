from __future__ import annotations

import argparse
import json

import pytest

torch = pytest.importorskip("torch")

from data.metadata import detach_pde_params, summarize_pde_params
from training.load_and_save import save_model


class _DummyScaler:
    def state_dict(self):
        return {}


def test_summarize_pde_params_is_json_safe():
    loader_metadata = {
        "heat": {"alpha": torch.tensor([0.1, 0.2, 0.3])},
        "advection_diffusion": {
            "b_x": torch.tensor([1.0, 2.0]),
            "b_y": torch.tensor([3.0, 4.0]),
            "kappa": torch.tensor([0.01, 0.02]),
        },
    }

    summary = summarize_pde_params(loader_metadata)

    assert summary["heat"]["alpha"]["count"] == 3
    assert summary["heat"]["alpha"]["mean"] == pytest.approx(0.2)
    assert summary["heat"]["alpha"]["min"] == pytest.approx(0.1)
    assert summary["heat"]["alpha"]["max"] == pytest.approx(0.3)
    assert summary["advection_diffusion"]["kappa"]["std"] == pytest.approx(0.005)
    for pde_summary in summary.values():
        for stats in pde_summary.values():
            assert isinstance(stats["count"], int)
            for key in ("mean", "std", "min", "max", "first", "last"):
                assert isinstance(stats[key], float)
    json.dumps(summary)


def test_summarize_pde_params_empty_metadata():
    assert summarize_pde_params({}) == {}


def test_summarize_pde_params_accepts_loader_metadata_shape():
    loader_metadata = {
        "reaction_diffusion": {
            "pde_params": {
                "T": torch.tensor([1.0, 1.0]),
                "D_u": torch.tensor([2e-3, 2e-3]),
                "sample_seed": torch.tensor([10, 11]),
            },
            "extra_metadata": {
                "selected_file_format": "new_gen_rd",
                "init_mode": ["grf", "grf"],
            },
        }
    }

    summary = summarize_pde_params(loader_metadata)
    detached = detach_pde_params(loader_metadata)

    assert summary["reaction_diffusion"]["T"]["count"] == 2
    assert summary["reaction_diffusion"]["D_u"]["mean"] == pytest.approx(2e-3)
    assert torch.equal(detached["reaction_diffusion"]["sample_seed"], torch.tensor([10, 11]))


def test_train_data_metadata_keeps_loader_metadata():
    from train import _build_data_metadata

    args = argparse.Namespace(
        dataset="reaction_diffusion",
        data_path="/tmp/PDEdata",
        data_size=1,
        max_train_samples=2,
        normalization_eps=1e-6,
    )
    loader_metadata = {
        "reaction_diffusion": {
            "pde_params": {
                "T": torch.tensor([1.0, 1.0]),
                "D_u": torch.tensor([2e-3, 2e-3]),
                "D_v": torch.tensor([4e-3, 4e-3]),
                "k": torch.tensor([3e-3, 3e-3]),
            },
            "pde_param_sources": {"T": "root_attr:T"},
            "selected_file_format": "new_gen_rd",
            "selected_files": ["/tmp/PDEdata/reaction_diffusion/reaction_diffusion_grf_2-4-4-T1-steps2.h5"],
            "num_loaded_samples": 2,
            "extra_metadata": {"init_mode": ["grf", "grf"]},
            "channel_names": ["u0", "v0", "uT", "vT"],
            "scalar_params_loaded": True,
        }
    }

    metadata = _build_data_metadata(
        args=args,
        pde_names=["reaction_diffusion"],
        data=torch.zeros(2, 4, 4, 4),
        label=torch.full((2,), 5, dtype=torch.long),
        loader_metadata=loader_metadata,
    )

    assert metadata["pde_param_summary"]["reaction_diffusion"]["D_u"]["mean"] == pytest.approx(2e-3)
    rd_meta = metadata["loader_metadata"]["reaction_diffusion"]
    assert rd_meta["selected_file_format"] == "new_gen_rd"
    assert rd_meta["num_loaded_samples"] == 2
    assert rd_meta["extra_metadata"]["init_mode"] == ["grf", "grf"]


def test_save_model_includes_data_metadata(tmp_path):
    model = torch.nn.Linear(1, 1)
    data_metadata = {
        "dataset": "heat",
        "pde_param_summary": {"heat": {"alpha": {"count": 1, "mean": 0.1}}},
    }
    args = argparse.Namespace(output_dir=str(tmp_path), dataset="heat", use_ema=False)

    save_model(
        args=args,
        epoch=0,
        model=model,
        model_without_ddp=model,
        optimizer=None,
        lr_schedule=None,
        loss_scaler=_DummyScaler(),
        final=True,
        data_shape=(1, 1),
        num_channels=1,
        data_metadata=data_metadata,
    )

    checkpoint = torch.load(tmp_path / "fm4heat.pth", map_location="cpu", weights_only=False)
    assert checkpoint["data_metadata"] == data_metadata
