from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Any

from fm4pde_ablation.config import load_config, load_yaml_file
from fm4pde_ablation.runner import run_single_ablation


def expand_grid(grid_path: str) -> list[tuple[str, dict[str, Any]]]:
    spec = load_yaml_file(grid_path)
    base_config = spec.get("base_config", "configs/ablations/smoke.yaml")
    groups = spec.get("groups", {})
    if not isinstance(groups, dict):
        raise ValueError("Sweep YAML must contain groups: mapping")
    jobs: list[tuple[str, dict[str, Any]]] = []
    for group_name, group_spec in groups.items():
        if not isinstance(group_spec, dict):
            continue
        matrix = group_spec.get("matrix", {})
        fixed = group_spec.get("fixed", {})
        product = bool(group_spec.get("product", False))
        expanded = _expand_matrix(matrix, product=product)
        for index, params in enumerate(expanded):
            overrides = {}
            overrides.update(fixed)
            overrides.update(params)
            overrides["ablation_name"] = f"{group_name}_{index:03d}"
            jobs.append((str(base_config), overrides))
    return jobs


def run_grid(grid_path: str, dry_run: bool = False, limit: int | None = None) -> list[dict[str, Any]]:
    results = []
    jobs = expand_grid(grid_path)
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
    args = parser.parse_args(argv)
    jobs = expand_grid(args.grid)
    selected = jobs[: args.limit] if args.limit else jobs
    if args.list:
        for config_path, overrides in selected:
            print(config_path, overrides)
        return
    run_grid(args.grid, dry_run=args.dry_run, limit=args.limit)


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


if __name__ == "__main__":
    main()
