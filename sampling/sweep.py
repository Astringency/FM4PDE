from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Any

from sampling.config import VALID_PDES, load_config, load_yaml_file, parse_cli_overrides
from sampling.runner import run_single_ablation


def expand_grid(
    grid_path: str,
    selected_groups: set[str] | None = None,
    selected_pdes: set[str] | None = None,
    global_overrides: dict[str, Any] | None = None,
) -> list[tuple[str, dict[str, Any]]]:
    spec = load_yaml_file(grid_path)
    default_base_configs = _base_configs_from_spec(spec, fallback=["configs/ablations/smoke.yaml"])
    groups = spec.get("groups", {})
    if not isinstance(groups, dict):
        raise ValueError("Sweep YAML must contain groups: mapping")
    _validate_selection(groups, selected_groups, selected_pdes)
    jobs: list[tuple[str, dict[str, Any]]] = []
    for group_name, group_spec in groups.items():
        if selected_groups and group_name not in selected_groups:
            continue
        if not isinstance(group_spec, dict):
            continue
        group_base_configs = _base_configs_from_spec(group_spec, fallback=default_base_configs)
        matrix = group_spec.get("matrix", {})
        fixed = group_spec.get("fixed", {})
        product = bool(group_spec.get("product", False))
        expanded = _expand_matrix(matrix, product=product)
        for base_index, base_config in enumerate(group_base_configs):
            base_pde = str(load_yaml_file(base_config).get("pde", ""))
            if selected_pdes and base_pde not in selected_pdes:
                continue
            for index, params in enumerate(expanded):
                overrides = {}
                overrides.update(fixed)
                overrides.update(params)
                overrides.update(global_overrides or {})
                if "test_index" in overrides and "offset" not in overrides:
                    overrides["offset"] = overrides["test_index"]
                stem = Path(str(base_config)).stem
                suffix = f"{base_index:02d}_{index:03d}" if len(group_base_configs) > 1 else f"{index:03d}"
                overrides["ablation_name"] = f"{group_name}_{stem}_{suffix}"
                overrides["ablation_group"] = group_name
                jobs.append((str(base_config), overrides))
    return jobs


def run_grid(
    grid_path: str,
    dry_run: bool = False,
    limit: int | None = None,
    selected_groups: set[str] | None = None,
    selected_pdes: set[str] | None = None,
    global_overrides: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    results = []
    jobs = expand_grid(
        grid_path,
        selected_groups=selected_groups,
        selected_pdes=selected_pdes,
        global_overrides=global_overrides,
    )
    for config_path, overrides in jobs[:limit]:
        if dry_run:
            overrides["dry_run"] = True
        cfg = load_config(config_path, overrides=overrides)
        results.append(run_single_ablation(cfg))
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run grouped FM4PDE ablation sweeps.")
    parser.add_argument("--grid", required=True, help="Sweep grid YAML.")
    parser.add_argument("--dry-run", action="store_true", help="Run with dry_run=true.")
    parser.add_argument("--list", action="store_true", help="Only list expanded jobs.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of jobs.")
    parser.add_argument("--group", action="append", default=[], help="Run or list only one group. Can be repeated.")
    parser.add_argument("--pde", action="append", default=[], help="Run or list only one PDE. Can be repeated.")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Apply a key=value config override to every selected job. Can be repeated.",
    )
    args = parser.parse_args(argv)
    selected_groups = set(args.group) if args.group else None
    selected_pdes = set(args.pde) if args.pde else None
    try:
        global_overrides = parse_cli_overrides(args.override)
        jobs = expand_grid(
            args.grid,
            selected_groups=selected_groups,
            selected_pdes=selected_pdes,
            global_overrides=global_overrides,
        )
    except ValueError as exc:
        parser.error(str(exc))
    if not jobs:
        parser.error("The selected PDE/group filters produced no ablation jobs")
    selected = jobs[: args.limit] if args.limit else jobs
    if args.list:
        for config_path, overrides in selected:
            print(config_path, overrides)
        return
    run_grid(
        args.grid,
        dry_run=args.dry_run,
        limit=args.limit,
        selected_groups=selected_groups,
        selected_pdes=selected_pdes,
        global_overrides=global_overrides,
    )


def _expand_matrix(matrix: dict[str, Any], product: bool) -> list[dict[str, Any]]:
    if not matrix:
        return [{}]
    keys = list(matrix.keys())
    values = [v if isinstance(v, list) else [v] for v in matrix.values()]
    if product:
        return [dict(zip(keys, combo)) for combo in itertools.product(*values)]
    max_len = max(len(v) for v in values)
    rows = []
    for i in range(max_len):
        row = {}
        for key, vals in zip(keys, values):
            row[key] = vals[i] if i < len(vals) else vals[-1]
        rows.append(row)
    return rows


def _base_configs_from_spec(spec: dict[str, Any], fallback: list[str]) -> list[str]:
    if "base_configs" in spec:
        value = spec["base_configs"]
    elif "base_config" in spec:
        value = spec["base_config"]
    else:
        return list(fallback)
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _validate_selection(
    groups: dict[str, Any],
    selected_groups: set[str] | None,
    selected_pdes: set[str] | None,
) -> None:
    unknown_groups = (selected_groups or set()) - set(groups)
    if unknown_groups:
        raise ValueError(f"Unknown ablation group(s): {', '.join(sorted(unknown_groups))}")
    unknown_pdes = (selected_pdes or set()) - VALID_PDES
    if unknown_pdes:
        raise ValueError(f"Unknown PDE(s): {', '.join(sorted(unknown_pdes))}")


if __name__ == "__main__":
    main()
