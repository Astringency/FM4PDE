from __future__ import annotations

from typing import Final


CANONICAL_DATASET_TYPES: Final = ("train", "id", "smooth", "rough")
DATASET_TYPE_ALIASES: Final = {
    "train": "train",
    "id": "id",
    "test": "id",
    "easy": "smooth",
    "easytest": "smooth",
    "smooth": "smooth",
    "ood_smooth": "smooth",
    "hard": "rough",
    "hardtest": "rough",
    "rough": "rough",
    "ood_rough": "rough",
}
SEED_OFFSETS: Final = {
    "train": 0,
    "id": 10_000_000,
    "smooth": 20_000_000,
    "rough": 30_000_000,
}

STATIC_GRF_PROFILES: Final = {
    "train": (2.0, 3.0),
    "id": (2.0, 3.0),
    "smooth": (3.0, 4.0),
    "rough": (1.5, 5.0),
}

TEMPORAL_GRF_PROFILES: Final = {
    "train": (2.5, 7.0),
    "id": (2.5, 7.0),
    "smooth": (3.0, 6.5),
    "rough": (1.5, 5.0),
}

REACTION_DIFFUSION_GRF_PROFILES: Final = {
    "train": (0.15, 2.0),
    "id": (0.15, 2.0),
    "smooth": (0.25, 3.0),
    "rough": (0.07, 1.25),
}

STEADY_HEAT_SOURCE_SIGMA_PROFILES: Final = {
    "train": (0.04, 0.12),
    "id": (0.04, 0.12),
    "smooth": (0.12, 0.20),
    "rough": (0.015, 0.05),
}

SHALLOW_WATER_SPATIAL_PROFILES: Final = {
    "train": {"transition_width": 0.0, "boundary_roughness": 0.0, "boundary_mode": 0},
    "id": {"transition_width": 0.0, "boundary_roughness": 0.0, "boundary_mode": 0},
    "smooth": {"transition_width": 0.12, "boundary_roughness": 0.0, "boundary_mode": 0},
    "rough": {"transition_width": 0.0, "boundary_roughness": 0.15, "boundary_mode": 6},
}


def canonical_dataset_type(dataset_type: str) -> str:
    try:
        return DATASET_TYPE_ALIASES[str(dataset_type).strip().lower()]
    except KeyError as exc:
        raise ValueError(
            "dataset type must be train, id, smooth, or rough "
            "(legacy aliases easytest and hardtest are also accepted)"
        ) from exc


def seed_offset(dataset_type: str) -> int:
    return int(SEED_OFFSETS[canonical_dataset_type(dataset_type)])


def shifted_periodic_grf_parameters(
    dataset_type: str,
    *,
    train_smoothness: float,
    train_tau: float,
) -> tuple[float, float]:
    profile = canonical_dataset_type(dataset_type)
    if profile in {"train", "id"}:
        return float(train_smoothness), float(train_tau)
    if profile == "smooth":
        return float(train_smoothness + 1.0), float(train_tau + 1.0)
    return float(max(1.0, train_smoothness - 1.5)), float(train_tau + 2.0)
