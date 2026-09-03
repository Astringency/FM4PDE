from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sampling.config import VALID_PDES, load_config, load_yaml_file, parse_cli_overrides
from sampling.data import finalize_ground_truth_config
from sampling.runner import run_single_ablation


_RESUME_IGNORED_CONFIG_FIELDS = {
    "device",
    "empty_cache_each_step",
    "output_dir",
    "save_plots",
}

_BACKWARD_COMPATIBLE_CONFIG_DEFAULTS = {
    "deterministic_endpoint_mode": "single_step",
    "deterministic_rollout_checkpoint": False,
    "deterministic_bt_mode": "legacy",
    "deterministic_guidance_coeff": 1.0,
    "deterministic_bt_max_scale": 0.1,
    "deterministic_guidance_start_ratio": 0.0,
    "deterministic_guidance_ramp_ratio": 0.0,
    "deterministic_correction_max_rms": 0.0,
    "deterministic_numerical_guard": True,
}


@dataclass(frozen=True)
class ParallelJobResult:
    index: int
    label: str
    device: str
    return_code: int
    log_path: Path
    elapsed_seconds: float


class ParallelSweepRunner:
    """Run independent ablation jobs in isolated, device-bound processes."""

    def __init__(
        self,
        jobs: list[tuple[str, dict[str, Any]]],
        *,
        devices: tuple[str, ...],
        max_parallel_tasks: int,
        dry_run: bool,
        resume: bool,
        output_dir: Path,
    ) -> None:
        if not devices:
            raise ValueError("devices must contain at least one device")
        if max_parallel_tasks < 1:
            raise ValueError("max_parallel_tasks must be positive")
        self.jobs = jobs
        self.devices = devices
        self.max_parallel_tasks = min(max_parallel_tasks, len(jobs))
        self.dry_run = dry_run
        self.resume = resume
        self.output_dir = output_dir
        self.run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}"
        self.logs_dir = output_dir / ".ablation_sweeps" / self.run_id / "logs"
        self.stop_event = threading.Event()
        self.processes: set[subprocess.Popen[str]] = set()
        self.process_lock = threading.Lock()
        self.device_slots: queue.Queue[str] = queue.Queue()
        slot_count = max(self.max_parallel_tasks, len(devices))
        for index in range(slot_count):
            self.device_slots.put(devices[index % len(devices)])

    def run(self) -> int:
        if not self.jobs:
            return 0
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        print(
            "FM4PDE parallel ablation sweep: "
            f"jobs={len(self.jobs)} workers={self.max_parallel_tasks} "
            f"devices={', '.join(self.devices)}"
        )
        print(f"Task logs: {self.logs_dir}")
        if self.max_parallel_tasks > len(set(self.devices)):
            print(
                "Warning: concurrent task slots exceed unique devices; "
                "some devices will run more than one task."
            )

        executor = ThreadPoolExecutor(
            max_workers=self.max_parallel_tasks,
            thread_name_prefix="ablation-sweep",
        )
        futures: dict[Future[ParallelJobResult], int] = {}
        interrupted = False
        try:
            for index, (config_path, overrides) in enumerate(self.jobs, start=1):
                future = executor.submit(self._run_job, index, config_path, overrides)
                futures[future] = index
            failures: list[ParallelJobResult] = []
            completed = 0
            for future in as_completed(futures):
                result = future.result()
                completed += 1
                if result.return_code != 0:
                    failures.append(result)
                status = "OK" if result.return_code == 0 else f"FAILED({result.return_code})"
                print(
                    f"[{completed}/{len(self.jobs)}] {status} {result.label} "
                    f"device={result.device} elapsed={result.elapsed_seconds:.1f}s"
                )
                if result.return_code != 0:
                    print(f"  log={result.log_path}")
        except KeyboardInterrupt:
            interrupted = True
            self.stop_event.set()
            for future in futures:
                future.cancel()
            self._terminate_processes()
            print("Interrupted; completed artifacts remain resumable with RESUME=true.")
            return 130
        finally:
            executor.shutdown(wait=True, cancel_futures=interrupted)

        if failures:
            print(
                f"Ablation sweep finished with {len(failures)} failed task(s); "
                f"see {self.logs_dir}."
            )
            return 1
        print(f"All {len(self.jobs)} ablation tasks completed successfully.")
        return 0

    def _run_job(
        self,
        index: int,
        config_path: str,
        overrides: dict[str, Any],
    ) -> ParallelJobResult:
        device = self.device_slots.get()
        label = _job_label(config_path, overrides)
        log_path = self.logs_dir / f"{index:04d}_{_safe_filename(label)}.log"
        start = time.monotonic()
        try:
            job_overrides = dict(overrides)
            job_overrides["device"] = device
            label = _job_label(config_path, job_overrides)
            log_path = self.logs_dir / f"{index:04d}_{_safe_filename(label)}.log"
            if self.stop_event.is_set():
                return ParallelJobResult(index, label, device, 130, log_path, 0.0)
            payload = json.dumps(
                {
                    "config_path": config_path,
                    "overrides": job_overrides,
                    "dry_run": self.dry_run,
                    "resume": self.resume,
                },
                separators=(",", ":"),
            )
            command = [
                sys.executable,
                "-u",
                "-m",
                "sampling.sweep_worker",
                "--payload",
                payload,
            ]
            with log_path.open("w", encoding="utf-8") as log_handle:
                process = subprocess.Popen(
                    command,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                    start_new_session=True,
                )
                with self.process_lock:
                    self.processes.add(process)
                try:
                    return_code = process.wait()
                finally:
                    with self.process_lock:
                        self.processes.discard(process)
            return ParallelJobResult(
                index=index,
                label=label,
                device=device,
                return_code=int(return_code),
                log_path=log_path,
                elapsed_seconds=time.monotonic() - start,
            )
        except Exception as exc:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as log_handle:
                log_handle.write(f"parallel launcher error: {type(exc).__name__}: {exc}\n")
            return ParallelJobResult(
                index=index,
                label=label,
                device=device,
                return_code=1,
                log_path=log_path,
                elapsed_seconds=time.monotonic() - start,
            )
        finally:
            self.device_slots.put(device)

    def _terminate_processes(self) -> None:
        with self.process_lock:
            processes = list(self.processes)
        for process in processes:
            if process.poll() is not None:
                continue
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                continue
        deadline = time.monotonic() + 10.0
        for process in processes:
            try:
                process.wait(timeout=max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass


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
                    allow_missing="task" in params and "task" not in (global_overrides or {}),
                )
                if config_path is None:
                    continue
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
    resume: bool = True,
) -> list[dict[str, Any]]:
    results = []
    jobs = expand_grid(
        grid_path,
        selected_groups=selected_groups,
        selected_pdes=selected_pdes,
        global_overrides=global_overrides,
    )
    for config_path, overrides in jobs[:limit]:
        results.append(
            run_job(
                config_path,
                overrides,
                dry_run=dry_run,
                resume=resume,
            )
        )
    return results


def run_job(
    config_path: str,
    overrides: dict[str, Any],
    *,
    dry_run: bool = False,
    resume: bool = True,
) -> dict[str, Any]:
    job_overrides = dict(overrides)
    if dry_run:
        job_overrides["dry_run"] = True
    cfg = finalize_ground_truth_config(load_config(config_path, overrides=job_overrides))
    if resume:
        completed_run = find_matching_completed_run(cfg)
        if completed_run is not None:
            print(
                f"Skip completed: {cfg.pde}/{cfg.task}/{cfg.resolved_ablation_name()} "
                f"({completed_run})"
            )
            return {
                "status": "skipped_completed",
                "run_dir": str(completed_run),
                "ablation_name": cfg.resolved_ablation_name(),
            }
    return run_single_ablation(cfg)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run grouped FM4PDE ablation sweeps.")
    parser.add_argument("--grid", required=True, help="Sweep grid YAML.")
    parser.add_argument("--dry-run", action="store_true", help="Run with dry_run=true.")
    parser.add_argument("--list", action="store_true", help="Only list expanded jobs.")
    parser.add_argument(
        "--parallel",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run independent ablation jobs concurrently (default: disabled).",
    )
    parser.add_argument(
        "--max-parallel-tasks",
        type=int,
        default=2,
        help="Maximum concurrent ablation jobs (default: 2).",
    )
    parser.add_argument(
        "--devices",
        nargs="+",
        default=None,
        help="Devices assigned round-robin to concurrent jobs.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse a matching successful run (default: enabled).",
    )
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
    if args.max_parallel_tasks < 1:
        parser.error("--max-parallel-tasks must be positive")
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
        return 0
    if args.parallel:
        devices = tuple(args.devices or [str(global_overrides.get("device", "cuda"))])
        if not devices:
            parser.error("--devices must contain at least one device")
        output_dir = Path(str(selected[0][1].get("output_dir", "outputs/ablations")))
        signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)
        return ParallelSweepRunner(
            selected,
            devices=devices,
            max_parallel_tasks=args.max_parallel_tasks,
            dry_run=args.dry_run,
            resume=args.resume,
            output_dir=output_dir,
        ).run()
    run_grid(
        args.grid,
        dry_run=args.dry_run,
        limit=args.limit,
        selected_groups=selected_groups,
        selected_pdes=selected_pdes,
        global_overrides=global_overrides,
        resume=args.resume,
    )
    return 0


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
        if key in {"common_overrides", "pde_overrides"}:
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
    allow_missing: bool = False,
) -> str | None:
    root_value = group_spec.get("main_config_root", spec.get("main_config_root"))
    if not root_value:
        raise ValueError("main_config_root is required when a grid uses main configs")
    root = Path(str(root_value))
    config_path = root / task / f"{pde}.yaml"
    if config_path.is_file():
        return str(config_path)
    if allow_missing:
        return None
    raise ValueError(
        f"Missing main config for pde={pde!r}, task={task!r}: {config_path}"
    )


def find_matching_completed_run(config: Any) -> Path | None:
    """Return the newest complete run with the same experiment-defining config."""
    base = (
        Path(config.output_dir)
        / config.pde
        / config.task
        / config.resolved_ablation_name()
    )
    if not base.is_dir():
        return None
    expected = _resume_config(config.asdict())
    for run_dir in sorted(base.iterdir(), reverse=True):
        if run_dir.is_dir() and _is_matching_successful_run(run_dir, expected, config.batch_size):
            return run_dir
    return None


def _is_matching_successful_run(
    run_dir: Path,
    expected_config: dict[str, Any],
    expected_batch_size: int,
) -> bool:
    config_path = run_dir / "resolved_config.yaml"
    metrics_path = run_dir / "metrics_final.json"
    sample_path = run_dir / "metrics_per_sample.csv"
    result_path = run_dir / "result.pt"
    if not all(path.is_file() for path in (config_path, metrics_path, sample_path, result_path)):
        return False
    try:
        existing_config = _resume_config(load_yaml_file(config_path))
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        with sample_path.open(newline="", encoding="utf-8") as handle:
            sample_count = sum(1 for _ in csv.DictReader(handle))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False
    if existing_config != expected_config:
        return False
    if metrics.get("status") != "ok" or metrics.get("pde_residual_status") == "error":
        return False
    if metrics.get("num_samples") not in (None, expected_batch_size):
        return False
    return sample_count == expected_batch_size


def _resume_config(config: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(config)
    for key, value in _BACKWARD_COMPATIBLE_CONFIG_DEFAULTS.items():
        normalized.setdefault(key, value)
    return {
        key: value
        for key, value in normalized.items()
        if key not in _RESUME_IGNORED_CONFIG_FIELDS
    }


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


def _job_label(config_path: str, overrides: dict[str, Any]) -> str:
    pde = str(overrides.get("pde", Path(config_path).stem))
    task = str(overrides.get("task", "unknown"))
    name = str(overrides.get("ablation_name", "ablation"))
    return f"{pde}/{task}/{name}"


def _safe_filename(value: str) -> str:
    safe = "".join(
        character if character.isalnum() or character in "-_." else "_"
        for character in value
    )
    return safe[:180] or "ablation"


def _raise_keyboard_interrupt(_signum: int, _frame: Any) -> None:
    raise KeyboardInterrupt


if __name__ == "__main__":
    raise SystemExit(main())
