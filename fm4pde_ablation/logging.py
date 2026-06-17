from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from fm4pde_ablation.config import AblationConfig, save_resolved_config


def make_run_dir(config: AblationConfig) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = Path(config.output_dir) / config.pde / config.task / config.resolved_ablation_name()
    run_dir = base / timestamp
    suffix = 1
    while run_dir.exists():
        run_dir = base / f"{timestamp}-{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "figures").mkdir(exist_ok=True)
    return run_dir


def write_run_metadata(config: AblationConfig, run_dir: str | os.PathLike[str]) -> None:
    run_dir = Path(run_dir)
    save_resolved_config(config, run_dir)
    metadata = {
        "git_commit": git_commit_hash(),
        "checkpoint_path": config.checkpoint_path,
        "data_path": config.data_path,
        "data_config_path": config.data_config_path,
        "mask_seed": config.mask_seed,
        "noise_seed": config.noise_seed,
        "sample_seed": config.sample_seed,
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")


def save_torch(path: str | os.PathLike[str], payload: Any) -> None:
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def git_commit_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"
