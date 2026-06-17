import argparse

import pytest

torch = pytest.importorskip("torch")

from training.grad_scaler import NativeScalerWithGradNormCount
from training.train_loop import train_one_epoch


class TinyVelocityModel(torch.nn.Module):
    num_classes = None

    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.1))

    def forward(self, x, t, extra=None):
        return self.scale * x


def test_train_one_epoch_updates_optimizer():
    data = torch.randn(4, 2, 8, 8)
    labels = torch.zeros(4, dtype=torch.long)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(data, labels), batch_size=2)
    model = TinyVelocityModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, total_iters=1, factor=1.0)
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
        lr_schedule=scheduler,
        device=torch.device("cpu"),
        epoch=0,
        loss_scaler=NativeScalerWithGradNormCount(),
        args=args,
    )

    assert "loss" in stats
    assert not torch.allclose(before, model.scale.detach())
