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
    spec = _load_sweep_spec(grid_path)
    if not spec.get("main_config_root"):
        raise ValueError("Sweep YAML must define main_config_root")
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
        base_items = _pdes_from_spec(group_spec, fallback=_pdes_from_spec(spec, fallback=[]))
        if not base_items:
            raise ValueError(f"Sweep group {group_name!r} selects no PDEs")
        matrix = group_spec.get("matrix", {})
        fixed = group_spec.get("fixed", {})
        product = bool(group_spec.get("product", False))
        expanded = _expand_matrix(matrix, product=product)
        for base_index, base_item in enumerate(base_items):
            base_pde = str(base_item)
            if selected_pdes and base_pde not in selected_pdes:
                continue
            for index, params in enumerate(expanded):
                overrides = _merged_job_overrides(spec, group_spec, base_pde)
                overrides.update(fixed)
                overrides.update(params)
                requested_task = str(
                    (global_overrides or {}).get(
                        "task", overrides.get("task", spec.get("default_task", "both"))
                    )
                )
                config_path = _resolve_main_config(
                    spec,
                    group_spec,
                    pde=base_pde,
                    task=requested_task,
                )
                _apply_conditional_overrides(
                    group_spec.get("conditional_overrides", []),
                    pde=base_pde,
                    overrides=overrides,
                )
                overrides.update(global_overrides or {})
                stem = Path(str(config_path)).stem
                suffix = f"{base_index:02d}_{index:03d}" if len(base_items) > 1 else f"{index:03d}"
                overrides["ablation_name"] = f"{group_name}_{stem}_{suffix}"
                overrides["ablation_group"] = group_name
                jobs.append((str(config_path), overrides))
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


def _load_sweep_spec(grid_path: str) -> dict[str, Any]:
    """Load a grid and its optional shared suite profile.

    Suite profiles contain only source selection and common/per-PDE overrides;
    the concrete grid remains the owner of experiment groups.
    """
    grid = load_yaml_file(grid_path)
    suite_path = grid.get("suite_config")
    if not suite_path:
        return grid
    suite = load_yaml_file(str(suite_path))
    if "groups" in suite:
        raise ValueError("suite_config must not define experiment groups")
    merged = dict(suite)
    for key, value in grid.items():
        if key == "suite_config":
            continue
        if key in {"common_overrides", "pde_overrides", "task_fallbacks"}:
            inherited = merged.get(key, {})
            if not isinstance(inherited, dict) or not isinstance(value, dict):
                raise ValueError(f"{key} must be a mapping")
            merged[key] = {**inherited, **value}
        else:
            merged[key] = value
    return merged


def _pdes_from_spec(spec: dict[str, Any], fallback: list[str]) -> list[str]:
    value = spec.get("pdes", fallback)
    if isinstance(value, list):
        pdes = [str(item) for item in value]
    else:
        pdes = [str(value)]
    unknown = set(pdes) - VALID_PDES
    if unknown:
        raise ValueError(f"Unknown PDE(s) in grid: {', '.join(sorted(unknown))}")
    return pdes


def _mapping(spec: dict[str, Any], key: str) -> dict[str, Any]:
    value = spec.get(key, {})
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping")
    return value


def _merged_job_overrides(
    spec: dict[str, Any], group_spec: dict[str, Any], pde: str
) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    overrides.update(_mapping(spec, "common_overrides"))
    suite_pde_overrides = _mapping(spec, "pde_overrides").get(pde, {})
    if not isinstance(suite_pde_overrides, dict):
        raise ValueError(f"pde_overrides[{pde!r}] must be a mapping")
    overrides.update(suite_pde_overrides)
    overrides.update(_mapping(group_spec, "common_overrides"))
    group_pde_overrides = _mapping(group_spec, "pde_overrides").get(pde, {})
    if not isinstance(group_pde_overrides, dict):
        raise ValueError(f"group pde_overrides[{pde!r}] must be a mapping")
    overrides.update(group_pde_overrides)
    return overrides


def _resolve_main_config(
    spec: dict[str, Any],
    group_spec: dict[str, Any],
    *,
    pde: str,
    task: str,
) -> str:
    root_value = group_spec.get("main_config_root", spec.get("main_config_root"))
    if not root_value:
        raise ValueError("main_config_root is required when a grid uses main configs")
    root = Path(str(root_value))
    config_path = root / task / f"{pde}.yaml"
    if config_path.is_file():
        return str(config_path)

    fallbacks = {
        **_mapping(spec, "task_fallbacks"),
        **_mapping(group_spec, "task_fallbacks"),
    }
    fallback = fallbacks.get(pde)
    if isinstance(fallback, dict):
        fallback = fallback.get(task)
    if fallback:
        fallback_path = root / str(fallback) / f"{pde}.yaml"
        if fallback_path.is_file():
            return str(fallback_path)
    raise ValueError(
        f"Missing main config for pde={pde!r}, task={task!r}: {config_path}"
    )


def _apply_conditional_overrides(
    rules: Any,
    *,
    pde: str,
    overrides: dict[str, Any],
) -> None:
    """Apply declarative per-variant settings before command-line overrides."""
    if rules in (None, [], {}):
        return
    if isinstance(rules, dict):
        named_rules = list(rules.items())
    elif isinstance(rules, list):
        named_rules = [(str(index), rule) for index, rule in enumerate(rules)]
    else:
        raise ValueError("conditional_overrides must be a mapping or list")
    context = {
        "pde": pde,
        **overrides,
    }
    for rule_name, rule in named_rules:
        if not isinstance(rule, dict):
            raise ValueError(f"conditional_overrides[{rule_name!r}] must be a mapping")
        conditions = rule.get("when")
        values = rule.get("set")
        if not isinstance(conditions, dict) or not isinstance(values, dict):
            raise ValueError(
                f"conditional_overrides[{rule_name!r}] requires mapping-valued 'when' and 'set'"
            )
        if all(context.get(key) == value for key, value in conditions.items()):
            overrides.update(values)
            context.update(values)


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
