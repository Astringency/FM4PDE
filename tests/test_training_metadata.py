from __future__ import annotations

import argparse
import json

import pytest

torch = pytest.importorskip("torch")

from data.metadata import summarize_pde_params
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
