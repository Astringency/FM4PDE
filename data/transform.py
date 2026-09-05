from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


class PDEStandardizer:
    """Channel-wise standardization fitted from training-set BCHW tensors."""

    normalization_type = "channelwise_standardization"

    def __init__(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        eps: float = 1e-6,
        channel_names: list[str] | None = None,
        pde: str | None = None,
    ) -> None:
        self.eps = float(eps)
        self.mean = self._as_channel_stats(mean, "mean")
        self.std = torch.where(
            self._as_channel_stats(std, "std") < self.eps,
            torch.ones_like(self._as_channel_stats(std, "std")),
            self._as_channel_stats(std, "std"),
        )
        self.channel_names = list(channel_names) if channel_names is not None else None
        self.pde = pde

    @classmethod
    def fit(
        cls,
        data: torch.Tensor,
        eps: float = 1e-6,
        channel_names: list[str] | None = None,
        pde: str | None = None,
    ) -> "PDEStandardizer":
        if data.ndim != 4:
            raise ValueError(f"PDEStandardizer.fit expects [N,C,H,W], got shape={tuple(data.shape)}")
        mean = data.mean(dim=(0, 2, 3), keepdim=True)
        std = data.std(dim=(0, 2, 3), keepdim=True, unbiased=False)
        std = torch.where(std < eps, torch.ones_like(std), std)
        return cls(mean=mean, std=std, eps=eps, channel_names=channel_names, pde=pde)

    @classmethod
    def identity(
        cls,
        num_channels: int,
        eps: float = 1e-6,
        channel_names: list[str] | None = None,
        pde: str | None = None,
    ) -> "PDEStandardizer":
        if num_channels < 1:
            raise ValueError(f"num_channels must be positive, got {num_channels}")
        mean = torch.zeros(1, num_channels, 1, 1)
        std = torch.ones(1, num_channels, 1, 1)
        return cls(mean=mean, std=std, eps=eps, channel_names=channel_names, pde=pde)

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "PDEStandardizer":
        if state is None:
            raise ValueError("Cannot construct PDEStandardizer from None state")
        if cls is PDEStandardizer and state.get("type") == LegacyAffineNormalizer.normalization_type:
            return LegacyAffineNormalizer.from_state_dict(state)
        if state.get("type", cls.normalization_type) not in {cls.normalization_type, None}:
            raise ValueError(f"Unsupported normalizer type: {state.get('type')!r}")
        return cls(
            mean=torch.as_tensor(state["mean"]),
            std=torch.as_tensor(state["std"]),
            eps=float(state.get("eps", 1e-6)),
            channel_names=state.get("channel_names"),
            pde=state.get("pde"),
        )

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        self._check_channels(x)
        return (x - self.mean.to(device=x.device, dtype=x.dtype)) / self.std.to(device=x.device, dtype=x.dtype)

    def inverse_transform(self, z: torch.Tensor) -> torch.Tensor:
        self._check_channels(z)
        return z * self.std.to(device=z.device, dtype=z.dtype) + self.mean.to(device=z.device, dtype=z.dtype)

    def state_dict(self) -> dict[str, Any]:
        return {
            "type": self.normalization_type,
            "mean": self.mean.detach().cpu(),
            "std": self.std.detach().cpu(),
            "eps": self.eps,
            "channel_names": self.channel_names,
            "pde": self.pde,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        loaded = self.from_state_dict(state)
        self.mean = loaded.mean
        self.std = loaded.std
        self.eps = loaded.eps
        self.channel_names = loaded.channel_names
        self.pde = loaded.pde

    def save(self, path: str | Path) -> Path:
        pt_path, json_path = self._resolve_save_paths(path)
        pt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), pt_path)
        json_path.write_text(json.dumps(self.to_json_dict(), indent=2) + "\n", encoding="utf-8")
        return pt_path

    @classmethod
    def load(cls, path: str | Path, map_location: str | torch.device = "cpu") -> "PDEStandardizer":
        pt_path = Path(path)
        if pt_path.is_dir():
            pt_path = pt_path / "normalizer.pt"
        state = torch.load(pt_path, map_location=map_location, weights_only=False)
        return cls.from_state_dict(state)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "type": self.normalization_type,
            "eps": self.eps,
            "pde": self.pde,
            "channel_names": self.channel_names,
            "mean": self.mean.detach().cpu().view(-1).tolist(),
            "std": self.std.detach().cpu().view(-1).tolist(),
            "shape": list(self.mean.shape),
        }

    @staticmethod
    def _as_channel_stats(value: torch.Tensor, name: str) -> torch.Tensor:
        tensor = torch.as_tensor(value).detach().clone().to(dtype=torch.float32)
        if tensor.ndim == 1:
            tensor = tensor.view(1, -1, 1, 1)
        if tensor.ndim != 4 or tensor.shape[0] != 1 or tensor.shape[2:] != (1, 1):
            raise ValueError(f"{name} must have shape [1,C,1,1] or [C], got {tuple(tensor.shape)}")
        return tensor

    @staticmethod
    def _resolve_save_paths(path: str | Path) -> tuple[Path, Path]:
        target = Path(path)
        if target.suffix:
            pt_path = target
            json_path = target.with_name("normalization.json")
        else:
            pt_path = target / "normalizer.pt"
            json_path = target / "normalization.json"
        return pt_path, json_path

    def _check_channels(self, x: torch.Tensor) -> None:
        if x.ndim != 4:
            raise ValueError(f"Expected [N,C,H,W] tensor, got shape={tuple(x.shape)}")
        expected = int(self.mean.shape[1])
        if int(x.shape[1]) != expected:
            raise ValueError(f"Expected {expected} channels, got {int(x.shape[1])}")


class LegacyAffineNormalizer(PDEStandardizer):
    """Physical affine mapping for legacy latent coordinates in [-1, 1]."""

    normalization_type = "legacy_affine"

    @classmethod
    def fit(cls, data, eps=1e-8, channel_names=None, pde=None):
        return cls.fit_pools([data], eps=eps, channel_names=channel_names, pde=pde)

    @classmethod
    def fit_pools(cls, pools, eps=1e-8, channel_names=None, pde=None):
        if not pools or any(data.ndim != 4 for data in pools):
            raise ValueError("Legacy Min-Max fit expects nonempty [N,C,H,W] pools")
        if len({int(data.shape[1]) for data in pools}) != 1:
            raise ValueError("Legacy training pools must have the same channel count")
        low = torch.stack([data.amin(dim=(0, 2, 3), keepdim=True) for data in pools]).amin(0)
        high = torch.stack([data.amax(dim=(0, 2, 3), keepdim=True) for data in pools]).amax(0)
        scale = (high - low + eps) / 2
        return cls(low + scale, scale, eps=0.0, channel_names=channel_names, pde=pde)
