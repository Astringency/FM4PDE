from __future__ import annotations

import argparse
import warnings

from fm4pde_ablation.config import str2bool
from fm4pde_ablation.runner import run_from_legacy_args


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FM4PDE legacy sampling wrapper.")
    parser.add_argument("--pdetype", type=str, default="poisson", help="Which PDE to solve.")
    parser.add_argument(
        "--problem",
        type=str,
        default="both",
        choices=["forward", "inverse", "both", "unconditional"],
        help="Task semantics for the sampler.",
    )
    parser.add_argument("--mode", type=str, default="sparse", choices=["sparse", "full"], help="Legacy observation mode.")
    parser.add_argument("--dt_sampler", type=str, default="old", help="Kept for CLI compatibility; ignored by new runner.")
    parser.add_argument("--config", type=str, default="configs/poisson.yaml", help="Legacy or ablation config.")
    parser.add_argument("--pdeguide", type=str2bool, default=True, help="Whether to use PDE residual guidance.")
    parser.add_argument("--lr_decay", type=str2bool, default=True, help="Kept for compatibility; use guidance_schedule instead.")
    parser.add_argument("--freq_decay", type=int, default=100, help="Kept for compatibility.")
    parser.add_argument("--hybrid", type=str2bool, default=False, help="Map legacy hybrid intent through sampler_phase overrides.")
    parser.add_argument("--remark", type=str, default="", help="Legacy remark; stored in config.extra.")
    parser.add_argument("--guide", type=str, default=None, help="Whether to use guidance.")
    parser.add_argument("--num_steps", type=int, default=0, help="Sampling steps.")
    parser.add_argument("--batch", type=int, default=1, help="Number of legacy repeated samples.")
    parser.add_argument("--k", type=int, default=1, help="Helmholtz k.")
    parser.add_argument("--sampler", type=str, default=None, choices=["deterministic", "stochastic", "hybrid_d2s", "hybrid_s2d"], help="Sampler phase.")
    parser.add_argument("--num_obs", type=int, default=0, help="Number of observation points.")
    parser.add_argument("--perturb", type=str2bool, default=False, help="Legacy perturb flag; stored for compatibility.")
    parser.add_argument("--perturb_rate", type=float, default=0.0, help="Legacy perturb rate; stored for compatibility.")
    parser.add_argument("--dry-run", action="store_true", help="Run without loading the FM4PDE checkpoint.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    message = (
        "sample.py is deprecated and now only maps legacy arguments into "
        "fm4pde_ablation.runner. Prefer `python -m fm4pde_ablation.runner --config ...`."
    )
    warnings.warn(message, DeprecationWarning, stacklevel=2)
    print(f"DeprecationWarning: {message}")
    results = run_from_legacy_args(args)
    for result in results:
        print(result)


if __name__ == "__main__":
    main()
