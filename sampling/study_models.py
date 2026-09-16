"""Select pretrained models consistently for repeated-sampling experiments."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from sampling.config import VALID_MODEL_PROFILES, load_config

ROOT = Path(__file__).resolve().parents[1]


def add_model_arguments(parser):
    parser.add_argument(
        "--config-dir", type=Path, default=ROOT / "configs/ablations/base",
        help="Model defaults from <config-dir>/both/<pde>.yaml; experimental controls stay unchanged.",
    )
    parser.add_argument("--checkpoint", type=Path, help="Override the pretrained checkpoint for one selected PDE.")
    parser.add_argument("--model-profile", choices=sorted(VALID_MODEL_PROFILES),
                        help="Override the configured architecture profile; auto reads checkpoint metadata.")
    parser.add_argument("--plan-only", action="store_true", help="Print model choices without loading data or running inference.")


def model_config(args, pde):
    if args.checkpoint and len(args.pdes) != 1:
        raise ValueError("--checkpoint requires one --pdes value; use CHECKPOINT_<PDE> for multiple equations.")
    overrides = {}
    if args.checkpoint:
        overrides["checkpoint_path"] = str(args.checkpoint.expanduser().resolve())
    if args.model_profile:
        overrides["model_profile"] = args.model_profile
    cfg = load_config(args.config_dir / "both" / f"{pde}.yaml", overrides)
    if cfg.pde != pde:
        raise ValueError(f"Configuration is for {cfg.pde}, expected {pde}")
    return cfg


def print_model_plan(args):
    if not args.plan_only:
        return False
    for pde in args.pdes:
        cfg = model_config(args, pde)
        print(json.dumps(dict(pde=pde, checkpoint_path=cfg.checkpoint_path,
                              model_profile=cfg.model_profile)))
    return True


def bind_model(output, cfg):
    """Prevent a changed checkpoint/profile from silently reusing old predictions."""
    h = hashlib.sha256()
    with Path(cfg.checkpoint_path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    identity = dict(pde=cfg.pde, checkpoint_path=cfg.checkpoint_path,
                    checkpoint_sha256=h.hexdigest(), model_profile=cfg.model_profile)
    output = Path(output)
    target = output / "sampling_model.json"
    if target.exists():
        if json.loads(target.read_text()) != identity:
            raise ValueError(f"Model changed: choose a new output directory instead of {output}")
    else:
        if next(output.rglob("receipt.json"), None) is not None:
            raise ValueError(f"Existing predictions have no model selection record; choose a new output directory: {output}")
        output.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(identity, indent=2) + "\n")
    return identity
