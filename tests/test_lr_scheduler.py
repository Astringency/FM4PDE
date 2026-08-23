from __future__ import annotations

import argparse

import pytest

torch = pytest.importorskip("torch")

from train import _build_lr_scheduler, _resolve_lr_scheduler_name, _step_lr_scheduler


def _args(**overrides):
    values = {
        "lr": 1e-4,
        "min_lr": 1e-6,
        "epochs": 20,
        "lr_scheduler": "warmup_cosine",
        "warmup_epochs": 5,
        "warmup_start_factor": 0.1,
        "plateau_factor": 0.5,
        "plateau_patience": 0,
        "plateau_threshold": 0.0,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _optimizer(lr: float = 1e-4):
    model = torch.nn.Linear(1, 1)
    return torch.optim.AdamW(model.parameters(), lr=lr)


def test_default_lr_scheduler_resolves_to_warmup_cosine():
    assert _resolve_lr_scheduler_name(_args()) == "warmup_cosine"


def test_warmup_cosine_warms_up_then_decays_to_min_lr():
    args = _args()
    optimizer = _optimizer(args.lr)
    scheduler = _build_lr_scheduler(optimizer, args, "warmup_cosine")

    assert optimizer.param_groups[0]["lr"] == pytest.approx(args.lr * args.warmup_start_factor)

    lrs = []
    for _ in range(args.epochs):
        optimizer.step()
        _step_lr_scheduler(scheduler, "warmup_cosine", val_loss=0.0)
        lrs.append(optimizer.param_groups[0]["lr"])

    assert max(lrs) == pytest.approx(args.lr)
    assert lrs[0] < lrs[3] < max(lrs)
    assert lrs[-1] == pytest.approx(args.min_lr)


def test_plateau_scheduler_uses_validation_loss():
    args = _args(lr_scheduler="plateau")
    optimizer = _optimizer(args.lr)
    scheduler = _build_lr_scheduler(optimizer, args, "plateau")

    _step_lr_scheduler(scheduler, "plateau", val_loss=1.0)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(args.lr)

    _step_lr_scheduler(scheduler, "plateau", val_loss=1.1)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(args.lr * args.plateau_factor)
