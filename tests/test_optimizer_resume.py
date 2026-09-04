from __future__ import annotations

import argparse

import pytest

torch = pytest.importorskip("torch")

from training.load_and_save import (
    CHECKPOINT_SCHEMA_VERSION,
    _load_optimizer_state_preserving_runtime_options,
    load_model,
)


class RecordingScaler:
    def __init__(self):
        self.loaded_state = None

    def load_state_dict(self, state_dict):
        self.loaded_state = state_dict


def _take_adamw_step(parameter, optimizer):
    optimizer.zero_grad(set_to_none=True)
    parameter.square().sum().backward()
    optimizer.step()


def test_old_adamw_state_keeps_new_fused_runtime_choice():
    old_parameter = torch.nn.Parameter(torch.tensor([1.0]))
    old_optimizer = torch.optim.AdamW([old_parameter], lr=0.01)
    _take_adamw_step(old_parameter, old_optimizer)
    old_state = old_optimizer.state_dict()
    assert old_state["param_groups"][0]["fused"] is None

    new_parameter = torch.nn.Parameter(torch.tensor([1.0]))
    new_optimizer = torch.optim.AdamW([new_parameter], lr=0.5, fused=True)
    _load_optimizer_state_preserving_runtime_options(new_optimizer, old_state)

    assert new_optimizer.param_groups[0]["lr"] == pytest.approx(0.01)
    assert new_optimizer.param_groups[0]["fused"] is True
    assert new_optimizer.param_groups[0]["foreach"] is None
    _take_adamw_step(new_parameter, new_optimizer)


def test_load_model_restores_old_checkpoint_and_preserves_runtime_options(tmp_path):
    old_model = torch.nn.Linear(2, 1)
    old_optimizer = torch.optim.AdamW(old_model.parameters(), lr=0.02)
    old_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(old_optimizer, T_max=8)
    old_optimizer.zero_grad(set_to_none=True)
    old_model(torch.ones(1, 2)).sum().backward()
    old_optimizer.step()
    old_scheduler.step()

    checkpoint_path = tmp_path / "old-checkpoint.pth"
    torch.save(
        {
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "model_for_resume": old_model.state_dict(),
            "optimizer": old_optimizer.state_dict(),
            "lr_schedule": old_scheduler.state_dict(),
            "scaler": {"resume_marker": 7},
            "normalizer": {"mean": torch.zeros(1), "std": torch.ones(1)},
            "epoch": 3,
        },
        checkpoint_path,
    )

    resumed_model = torch.nn.Linear(2, 1)
    resumed_optimizer = torch.optim.AdamW(
        resumed_model.parameters(), lr=0.5, fused=True
    )
    resumed_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        resumed_optimizer, T_max=8
    )
    resumed_scaler = RecordingScaler()
    args = argparse.Namespace(
        resume=str(checkpoint_path),
        dataset="nsnonbounded",
        start_epoch=0,
        resolved_lr_scheduler="warmup_cosine",
        lr_scheduler="warmup_cosine",
    )

    checkpoint = load_model(
        args=args,
        model_without_ddp=resumed_model,
        optimizer=resumed_optimizer,
        loss_scaler=resumed_scaler,
        lr_schedule=resumed_scheduler,
    )

    assert checkpoint is not None
    assert args.start_epoch == 4
    assert resumed_optimizer.param_groups[0]["fused"] is True
    assert resumed_optimizer.param_groups[0]["lr"] == pytest.approx(
        old_optimizer.param_groups[0]["lr"]
    )
    assert resumed_scheduler.state_dict() == old_scheduler.state_dict()
    assert resumed_scaler.loaded_state == {"resume_marker": 7}
    for expected, actual in zip(old_model.parameters(), resumed_model.parameters()):
        assert torch.equal(expected, actual)

    # A real optimizer step after resume validates that the restored moment and
    # step tensors are accepted by the newly selected fused implementation.
    resumed_optimizer.zero_grad(set_to_none=True)
    resumed_model(torch.ones(1, 2)).sum().backward()
    resumed_optimizer.step()
