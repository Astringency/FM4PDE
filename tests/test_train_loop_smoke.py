import argparse

import pytest

torch = pytest.importorskip("torch")

from training.grad_scaler import NativeScalerWithGradNormCount
from training.train_loop import _conditioning_for_model, train_one_epoch, validate_one_epoch


class TinyVelocityModel(torch.nn.Module):
    num_classes = None

    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.1))

    def forward(self, x, t, extra=None):
        return self.scale * x


class TinyScalarVelocityModel(torch.nn.Module):
    num_classes = None

    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.1))

    def forward(self, x, t, extra=None):
        assert extra is not None
        scalar = extra["scalar_conditioning"]
        assert scalar.shape == (x.shape[0], 2)
        return self.scale * x + scalar.sum() * 0.0


def test_train_one_epoch_updates_optimizer():
    data = torch.randn(4, 2, 8, 8)
    labels = torch.zeros(4, dtype=torch.long)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(data, labels), batch_size=2)
    model = TinyVelocityModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    args = argparse.Namespace(
        accum_iter=1,
        test_run=True,
        class_drop_prob=0.0,
        skewed_timesteps=False,
        sampling_dtype="float32",
        clip_grad=None,
    )

    before = model.scale.detach().clone()
    stats = train_one_epoch(
        model=model,
        data_loader=loader,
        optimizer=optimizer,
        device=torch.device("cpu"),
        epoch=0,
        loss_scaler=NativeScalerWithGradNormCount(),
        args=args,
    )

    assert "loss" in stats
    assert not torch.allclose(before, model.scale.detach())


def test_validate_one_epoch_reports_loss_without_optimizer_update():
    data = torch.randn(4, 2, 8, 8)
    labels = torch.zeros(4, dtype=torch.long)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(data, labels), batch_size=2)
    model = TinyVelocityModel()
    args = argparse.Namespace(
        class_drop_prob=0.0,
        skewed_timesteps=False,
        sampling_dtype="float32",
    )

    before = model.scale.detach().clone()
    stats = validate_one_epoch(
        model=model,
        data_loader=loader,
        device=torch.device("cpu"),
        epoch=0,
        args=args,
    )

    assert "loss" in stats
    assert stats["loss"] > 0
    assert torch.allclose(before, model.scale.detach())
    assert model.training


def test_train_one_epoch_accepts_scalar_conditioning_batch():
    data = torch.randn(4, 2, 8, 8)
    labels = torch.zeros(4, dtype=torch.long)
    scalar = torch.randn(4, 2)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(data, labels, scalar),
        batch_size=2,
    )
    model = TinyScalarVelocityModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    args = argparse.Namespace(
        accum_iter=1,
        test_run=True,
        class_drop_prob=0.0,
        skewed_timesteps=False,
        sampling_dtype="float32",
        clip_grad=None,
    )

    stats = train_one_epoch(
        model=model,
        data_loader=loader,
        optimizer=optimizer,
        device=torch.device("cpu"),
        epoch=0,
        loss_scaler=NativeScalerWithGradNormCount(),
        args=args,
    )

    assert "loss" in stats


def test_class_dropout_is_applied_per_sample(monkeypatch):
    model = TinyVelocityModel()
    model.num_classes = 3
    labels = torch.tensor([0, 1, 2, 0])
    monkeypatch.setattr(
        torch,
        "rand",
        lambda shape, device=None: torch.tensor([0.1, 0.9, 0.2, 0.8], device=device),
    )
    conditioning = _conditioning_for_model(model, labels, class_drop_prob=0.5)
    assert conditioning["label"].tolist() == [3, 1, 3, 0]
