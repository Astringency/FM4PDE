from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent))
    from common import add_common_arguments, apply_quick_test_defaults, generate_dataset, namespace_to_config
    from generate_advection_diffusion import advection_diffusion_metadata, solve_advection_diffusion_chunk
    from generate_heat import heat_metadata, solve_heat_chunk
    from generate_wave import solve_wave_chunk, wave_metadata
else:  # pragma: no cover
    from .common import add_common_arguments, apply_quick_test_defaults, generate_dataset, namespace_to_config
    from .generate_advection_diffusion import advection_diffusion_metadata, solve_advection_diffusion_chunk
    from .generate_heat import heat_metadata, solve_heat_chunk
    from .generate_wave import solve_wave_chunk, wave_metadata


PDE_TABLE = {
    "heat": (solve_heat_chunk, heat_metadata),
    "wave": (solve_wave_chunk, wave_metadata),
    "advection_diffusion": (solve_advection_diffusion_chunk, advection_diffusion_metadata),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate FM4PDE future PDE HDF5 datasets.")
    add_common_arguments(parser)
    parser.add_argument("--random-v0", action="store_true", help="Use a smooth random initial velocity for wave.")
    parser.add_argument("--variable-c", action="store_true", help="Use smooth random wave speed and FD time stepping.")
    parser.add_argument("--c", type=float, default=1.0, help="Constant wave speed when --variable-c is not set.")
    return parser


def main(argv: list[str] | None = None) -> dict[str, object]:
    parser = build_parser()
    args = parser.parse_args(argv)
    apply_quick_test_defaults(args)
    selected = list(PDE_TABLE) if args.pde == "all" else [args.pde]
    all_written: dict[str, list[str]] = {}
    checks: dict[str, object] = {}
    for pde in selected:
        solver, metadata_fn = PDE_TABLE[pde]
        extra = {}
        if pde == "wave":
            extra = {"random_v0": args.random_v0, "variable_c": args.variable_c, "c": args.c}
        config = namespace_to_config(args, pde, extra=extra)
        metadata = metadata_fn(config)
        written = generate_dataset(config, solver, metadata)
        all_written[pde] = [str(path) for path in written]
        check_path = config.out_root / pde / "no_leakage_check.json"
        if check_path.exists():
            checks[pde] = json.loads(check_path.read_text(encoding="utf-8"))

    summary = {"written": all_written, "no_leakage_checks": checks}
    if len(selected) > 1 and not args.dry_run:
        summary_path = Path(args.out_root) / "no_leakage_check.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
