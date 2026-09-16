#!/usr/bin/env python3
"""Create a schema-compatible inference-only copy of a training checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path


HEAVY_TRAINING_KEYS = {
    "model",
    "model_ema",
    "model_for_resume",
    "optimizer",
    "scaler",
}


def make_inference_checkpoint(
    source: Path,
    output: Path,
    *,
    gradient_checkpointing: bool = False,
    force: bool = False,
) -> None:
    import torch

    source = source.resolve()
    output = output.resolve()
    if output.is_file() and not force:
        print(f"SKIP {output}")
        return
    print(f"READ {source}", flush=True)
    payload = torch.load(source, map_location="cpu", weights_only=False, mmap=True)
    selected = payload.get("model_ema")
    selected_name = "ema"
    if not isinstance(selected, dict) or not selected:
        selected = payload.get("model")
        selected_name = "raw"
    if not isinstance(selected, dict) or not selected:
        raise ValueError(f"No inference state_dict in {source}")

    slim = {key: value for key, value in payload.items() if key not in HEAVY_TRAINING_KEYS}
    slim["model"] = selected
    slim["model_ema"] = None
    slim["has_ema"] = False
    slim["use_ema"] = False
    slim["inference_weight"] = selected_name
    slim["slimmed_from"] = str(source)
    if gradient_checkpointing:
        slim["model_config"] = dict(slim["model_config"])
        slim["model_config"]["use_checkpoint"] = True
        slim["inference_gradient_checkpointing"] = True
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    torch.save(slim, temporary)
    temporary.replace(output)
    print(f"WRITE {output} ({output.stat().st_size} bytes, selected={selected_name})", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    make_inference_checkpoint(
        args.source,
        args.output,
        gradient_checkpointing=args.gradient_checkpointing,
        force=args.force,
    )


if __name__ == "__main__":
    main()
