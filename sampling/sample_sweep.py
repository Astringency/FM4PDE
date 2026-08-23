from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.live import Live
from rich.table import Table

from sampling.config import add_bool_arg, load_config, load_yaml_file
from sampling.data import finalize_ground_truth_config


_SIGNATURE_IGNORED_FIELDS = {
    "batch_size",
    "device",
    "offset",
    "output_dir",
    "save_plots",
}

GroupKey = tuple[str, str, str]


@dataclass(frozen=True)
class Experiment:
    pde: str
    task: str
    sampler: str
    sensor_mode: str
    config_path: Path
    config: dict[str, Any]
    signature: str

    @property
    def key(self) -> str:
        return f"{self.pde}/{self.task}/{self.sampler}/{self.sensor_mode}"

    @property
    def safe_key(self) -> str:
        return f"{self.pde}__{self.task}__{self.sampler}__{self.sensor_mode}"


@dataclass(frozen=True)
class Chunk:
    offset: int
    batch_size: int


@dataclass
class TaskState:
    pde: str
    task: str
    device: str
    total_samples: int
    completed_samples: int
    status: str = "queued"
    sampler: str = "-"
    sensor_mode: str = "-"
    offset: int | None = None
    batch_size: int = 0
    step: int = 0
    num_steps: int = 0
    rel_l2_a: float | None = None
    rel_l2_u: float | None = None
    error: str | None = None
    started_at: float | None = None
    finished_at: float | None = None
    progress_file: Path | None = None
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def work_fraction(self) -> float:
        with self.lock:
            partial = 0.0
            if self.status == "running" and self.num_steps > 0:
                partial = self.batch_size * min(1.0, self.step / self.num_steps)
            if self.total_samples <= 0:
                return 1.0
            return min(1.0, (self.completed_samples + partial) / self.total_samples)


@dataclass(frozen=True)
class SweepOptions:
    num_samples: int
    max_batch_size: int
    num_steps: int
    num_obs: int
    output_dir: Path
    config_dir: Path
    sample_script: Path
    parallel: bool
    max_parallel_tasks: int
    resume: bool
    vis: bool
    dry_run: bool
    aggregate: bool
    plan_only: bool
    progress_interval: float
    sample_seed: int | None
    devices: tuple[str, ...]


class SweepRunner:
    def __init__(
        self,
        options: SweepOptions,
        experiments: list[Experiment],
        groups: list[GroupKey],
    ) -> None:
        self.options = options
        self.experiments = experiments
        self.groups = groups
        self.console = Console()
        self.processes: set[subprocess.Popen[str]] = set()
        self.process_lock = threading.Lock()
        self.event_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.sweep_id = _sweep_id(experiments)
        self.state_dir = options.output_dir / ".sample_sweeps" / self.sweep_id
        self.logs_dir = self.state_dir / "logs"
        self.completed_dir = self.state_dir / "completed"
        self.progress_dir = self.state_dir / "progress"
        self.event_log = self.state_dir / "sweep.log"
        self.experiments_by_key = {experiment.key: experiment for experiment in experiments}
        self.completed: dict[str, set[int]] = {experiment.key: set() for experiment in experiments}
        self.states: dict[GroupKey, TaskState] = {}
        self.device_slots: queue.Queue[str] = queue.Queue()
        slot_count = max(self.worker_count, len(self.options.devices))
        for index in range(slot_count):
            self.device_slots.put(self.options.devices[index % len(self.options.devices)])

    @property
    def worker_count(self) -> int:
        return min(
            len(self.groups),
            self.options.max_parallel_tasks if self.options.parallel else 1,
        )

    def prepare(self) -> None:
        if not self.options.plan_only:
            self.logs_dir.mkdir(parents=True, exist_ok=True)
            self.completed_dir.mkdir(parents=True, exist_ok=True)
            self.progress_dir.mkdir(parents=True, exist_ok=True)
            _atomic_write_json(
                self.state_dir / "manifest.json",
                {
                    "sweep_id": self.sweep_id,
                    "output_dir": str(self.options.output_dir),
                    "num_samples": self.options.num_samples,
                    "max_batch_size": self.options.max_batch_size,
                    "num_steps": self.options.num_steps,
                    "num_obs": self.options.num_obs,
                    "experiments": [
                        {
                            "pde": item.pde,
                            "task": item.task,
                            "sampler": item.sampler,
                            "sensor_mode": item.sensor_mode,
                            "signature": item.signature,
                            "config_path": str(item.config_path),
                        }
                        for item in self.experiments
                    ],
                },
            )
        if self.options.resume:
            self._load_completed_markers()
            self._recover_completed_progress_files()
            self._load_or_scan_artifacts()
        for group_index, (pde, task, sensor_mode) in enumerate(self.groups):
            matching = [
                experiment
                for experiment in self.experiments
                if experiment.pde == pde
                and experiment.task == task
                and experiment.sensor_mode == sensor_mode
            ]
            total = self.options.num_samples * len(matching)
            done = sum(len(self.completed[experiment.key]) for experiment in matching)
            self.states[(pde, task, sensor_mode)] = TaskState(
                pde=pde,
                task=task,
                device=self.options.devices[group_index % len(self.options.devices)],
                total_samples=total,
                completed_samples=done,
                status="complete" if done >= total else ("resuming" if done else "queued"),
                sensor_mode=sensor_mode,
            )

    def print_header(self) -> None:
        mode = "parallel" if self.options.parallel else "serial"
        self.console.print("[bold]FM4PDE resumable sampling sweep[/bold]")
        self.console.print(
            f"mode={mode}  workers={self.worker_count}  resume={self.options.resume}  "
            f"samples={self.options.num_samples}/experiment  batch<={self.options.max_batch_size}"
        )
        self.console.print(
            f"output={self.options.output_dir}  state={self.state_dir}  "
            f"devices={', '.join(self.options.devices)}"
        )
        if self.options.parallel and self.worker_count > len(set(self.options.devices)):
            self.console.print(
                "[yellow]Warning: concurrent task slots exceed unique devices; some devices will be shared.[/yellow]"
            )

    def print_plan(self) -> None:
        table = Table(title="Sampling plan")
        table.add_column("PDE / task")
        table.add_column("Sampler")
        table.add_column("Sensor")
        table.add_column("Completed", justify="right")
        table.add_column("Remaining chunks", justify="right")
        table.add_column("Device")
        for experiment in self.experiments:
            chunks = compute_missing_chunks(
                self.options.num_samples,
                self.options.max_batch_size,
                self.completed[experiment.key],
            )
            state = self.states[
                (experiment.pde, experiment.task, experiment.sensor_mode)
            ]
            table.add_row(
                f"{experiment.pde} / {experiment.task}",
                experiment.sampler,
                experiment.sensor_mode,
                f"{len(self.completed[experiment.key])}/{self.options.num_samples}",
                str(len(chunks)),
                state.device,
            )
        self.console.print(table)

    def run(self) -> int:
        self.prepare()
        self.print_header()
        if self.options.plan_only:
            self.print_plan()
            return 0
        if all(state.status == "complete" for state in self.states.values()):
            self.console.print("[green]All matching samples are already complete; nothing to run.[/green]")
            return self._aggregate() if self.options.aggregate else 0

        self._log_event("sweep started")
        executor = ThreadPoolExecutor(max_workers=self.worker_count, thread_name_prefix="sample-sweep")
        futures: dict[Future[bool], GroupKey] = {}
        try:
            for group in self.groups:
                futures[executor.submit(self._run_group_guarded, *group)] = group
            with Live(self._progress_table(), console=self.console, refresh_per_second=4) as live:
                while not all(future.done() for future in futures):
                    self._poll_progress_files()
                    live.update(self._progress_table())
                    time.sleep(self.options.progress_interval)
                self._poll_progress_files()
                live.update(self._progress_table())
        except KeyboardInterrupt:
            self.stop_event.set()
            self._terminate_processes()
            for future in futures:
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)
            self._log_event("sweep interrupted")
            self.console.print(
                "[yellow]Interrupted. Completed chunks were saved; rerun the same command with RESUME=true.[/yellow]"
            )
            return 130
        finally:
            if not self.stop_event.is_set():
                executor.shutdown(wait=True)

        failed = [group for future, group in futures.items() if not future.result()]
        if failed:
            self._log_event(f"sweep failed groups={failed}")
            names = ", ".join("/".join(item) for item in failed)
            self.console.print(f"[red]Failed PDE/task/sensor groups: {names}[/red]")
            self.console.print(f"Logs: {self.logs_dir}")
            return 1
        self._log_event("sweep completed")
        self.console.print("[green]All sampling tasks completed successfully.[/green]")
        return self._aggregate() if self.options.aggregate else 0

    def _run_group_guarded(self, pde: str, task: str, sensor_mode: str) -> bool:
        try:
            return self._run_group(pde, task, sensor_mode)
        except Exception as exc:
            state = self.states[(pde, task, sensor_mode)]
            with state.lock:
                state.status = "failed"
                state.error = f"{type(exc).__name__}: {exc}"
                state.finished_at = time.time()
            self._log_event(
                f"failed {pde}/{task}/{sensor_mode}: {type(exc).__name__}: {exc}"
            )
            return False

    def _run_group(self, pde: str, task: str, sensor_mode: str) -> bool:
        state = self.states[(pde, task, sensor_mode)]
        with state.lock:
            if state.status == "complete":
                return True
            state.status = "running"
            state.started_at = time.time()
        device = self.device_slots.get()
        with state.lock:
            state.device = device
        try:
            return self._run_group_on_device(pde, task, sensor_mode, state)
        finally:
            self.device_slots.put(device)

    def _run_group_on_device(
        self,
        pde: str,
        task: str,
        sensor_mode: str,
        state: TaskState,
    ) -> bool:
        group_experiments = [
            experiment
            for experiment in self.experiments
            if experiment.pde == pde
            and experiment.task == task
            and experiment.sensor_mode == sensor_mode
        ]
        for experiment in group_experiments:
            chunks = compute_missing_chunks(
                self.options.num_samples,
                self.options.max_batch_size,
                self.completed[experiment.key],
            )
            for chunk in chunks:
                if self.stop_event.is_set():
                    return False
                if not self._run_chunk(experiment, chunk, state):
                    return False
        with state.lock:
            state.status = "complete"
            state.sampler = "-"
            state.offset = None
            state.batch_size = 0
            state.step = state.num_steps
            state.finished_at = time.time()
            state.progress_file = None
        return True

    def _run_chunk(self, experiment: Experiment, chunk: Chunk, state: TaskState) -> bool:
        progress_file = self.progress_dir / f"{experiment.safe_key}.json"
        log_dir = (
            self.logs_dir
            / experiment.pde
            / experiment.task
            / experiment.sensor_mode
            / experiment.sampler
        )
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = _unique_log_path(
            log_dir / f"offset_{chunk.offset:06d}_batch_{chunk.batch_size}.log"
        )
        with state.lock:
            state.status = "running"
            state.sampler = experiment.sampler
            state.sensor_mode = experiment.sensor_mode
            state.offset = chunk.offset
            state.batch_size = chunk.batch_size
            state.step = 0
            state.num_steps = self.options.num_steps
            state.progress_file = progress_file
            state.error = None
        progress_file.unlink(missing_ok=True)
        env = os.environ.copy()
        env.update(
            {
                "PDE": experiment.pde,
                "TASK": experiment.task,
                "SAMPLER_PHASE": experiment.sampler,
                "SENSOR_MODE": experiment.sensor_mode,
                "BATCH_SIZE": str(chunk.batch_size),
                "OFFSET": str(chunk.offset),
                "NUM_STEPS": str(self.options.num_steps),
                "NUM_OBS": str(self.options.num_obs),
                "OUTPUT_DIR": str(self.options.output_dir),
                "CONFIG_DIR": str(self.options.config_dir),
                "DEVICE": state.device,
                "VIS": "true" if self.options.vis else "false",
                "DRY_RUN": "true" if self.options.dry_run else "false",
                "FM4PDE_SWEEP_PROGRESS_FILE": str(progress_file),
            }
        )
        if self.options.sample_seed is not None:
            env["SAMPLE_SEED"] = str(self.options.sample_seed)
        self._log_event(
            f"start {experiment.key} offset={chunk.offset} "
            f"batch={chunk.batch_size} device={state.device}"
        )
        with log_path.open("w", encoding="utf-8") as log_handle:
            process = subprocess.Popen(
                ["bash", str(self.options.sample_script)],
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            with self.process_lock:
                self.processes.add(process)
            return_code = process.wait()
            with self.process_lock:
                self.processes.discard(process)
        if return_code != 0:
            with state.lock:
                state.status = "failed"
                state.error = f"exit={return_code}; {log_path}"
                state.finished_at = time.time()
            self._log_event(
                f"failed {experiment.key} offset={chunk.offset} "
                f"batch={chunk.batch_size} exit={return_code}"
            )
            return False
        run_dir, verification_error = verify_completed_chunk(progress_file, experiment, chunk)
        if verification_error is not None:
            with state.lock:
                state.status = "failed"
                state.error = f"invalid artifact; {log_path}"
                state.finished_at = time.time()
            self._log_event(
                f"failed verification {experiment.key} offset={chunk.offset} "
                f"batch={chunk.batch_size}: {verification_error}"
            )
            return False
        assert run_dir is not None
        self._write_completed_marker(experiment, chunk, log_path, run_dir)
        self.completed[experiment.key].update(range(chunk.offset, chunk.offset + chunk.batch_size))
        with state.lock:
            state.completed_samples += chunk.batch_size
            state.step = self.options.num_steps
            state.progress_file = None
        progress_file.unlink(missing_ok=True)
        self._log_event(f"complete {experiment.key} offset={chunk.offset} batch={chunk.batch_size}")
        return True

    def _poll_progress_files(self) -> None:
        for state in self.states.values():
            with state.lock:
                path = state.progress_file
            if path is None or not path.exists():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            with state.lock:
                state.step = int(payload.get("step", state.step))
                state.num_steps = int(payload.get("num_steps", state.num_steps))
                state.rel_l2_a = _optional_float(payload.get("rel_l2_a"))
                state.rel_l2_u = _optional_float(payload.get("rel_l2_u"))

    def _progress_table(self) -> Table:
        table = Table(title="FM4PDE sampling progress")
        table.add_column("PDE / task", no_wrap=True)
        table.add_column("Sampler", no_wrap=True)
        table.add_column("Sensor", no_wrap=True)
        table.add_column("Samples", justify="right")
        table.add_column("Current", no_wrap=True)
        table.add_column("Progress", no_wrap=True)
        table.add_column("State", no_wrap=True)
        table.add_column("rel₂(a/u)", justify="right")
        total = 0
        weighted = 0.0
        for group in self.groups:
            state = self.states[group]
            fraction = state.work_fraction()
            with state.lock:
                total += state.total_samples
                weighted += fraction * state.total_samples
                current = "-"
                if state.offset is not None:
                    current = (
                        f"off={state.offset} b={state.batch_size} "
                        f"step={state.step}/{state.num_steps}"
                    )
                rel = "-"
                if state.rel_l2_a is not None or state.rel_l2_u is not None:
                    rel = f"{_fmt(state.rel_l2_a)}/{_fmt(state.rel_l2_u)}"
                status = state.status
                if state.error:
                    status = f"failed ({state.error.split(';', 1)[0]})"
                table.add_row(
                    f"{state.pde} / {state.task}",
                    state.sampler,
                    state.sensor_mode,
                    f"{state.completed_samples}/{state.total_samples}",
                    current,
                    _bar(fraction),
                    status,
                    rel,
                )
        overall = weighted / total if total else 1.0
        statuses = {state.status for state in self.states.values()}
        overall_status = (
            "failed" if "failed" in statuses else "complete" if statuses == {"complete"} else "running"
        )
        table.add_section()
        table.add_row(
            "OVERALL",
            "-",
            "-",
            f"{int(weighted)}/{total}",
            "-",
            _bar(overall),
            overall_status,
            "-",
        )
        return table

    def _load_completed_markers(self) -> None:
        if not self.completed_dir.exists():
            return
        for marker in self.completed_dir.rglob("*.json"):
            try:
                payload = json.loads(marker.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            key = str(payload.get("experiment", ""))
            experiment = self.experiments_by_key.get(key)
            if experiment is None or payload.get("signature") != experiment.signature:
                continue
            offset = int(payload.get("offset", -1))
            batch_size = int(payload.get("batch_size", 0))
            run_dir = Path(str(payload.get("run_dir", "")))
            if not _successful_run_artifact(run_dir, experiment, offset, batch_size):
                continue
            self.completed[key].update(
                _bounded_offsets(offset, batch_size, self.options.num_samples)
            )

    def _load_or_scan_artifacts(self) -> None:
        index_path = self.state_dir / "artifact_index.json"
        if index_path.exists():
            try:
                payload = json.loads(index_path.read_text(encoding="utf-8"))
                if int(payload.get("scanned_num_samples", 0)) >= self.options.num_samples:
                    for key, offsets in payload.get("completed", {}).items():
                        if key in self.completed:
                            self.completed[key].update(
                                int(offset)
                                for offset in offsets
                                if 0 <= int(offset) < self.options.num_samples
                            )
                    return
            except (OSError, ValueError, json.JSONDecodeError):
                pass
        discovered = discover_completed_artifacts(
            self.options.output_dir,
            self.experiments,
            self.options.num_samples,
        )
        for key, offsets in discovered.items():
            self.completed[key].update(offsets)
        if not self.options.plan_only:
            _atomic_write_json(
                index_path,
                {
                    "scanned_num_samples": self.options.num_samples,
                    "completed": {key: sorted(offsets) for key, offsets in discovered.items()},
                },
            )

    def _recover_completed_progress_files(self) -> None:
        if not self.progress_dir.exists():
            return
        for progress_file in self.progress_dir.glob("*.json"):
            try:
                payload = json.loads(progress_file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            key = "/".join(
                str(payload.get(name, ""))
                for name in ("pde", "task", "sampler", "sensor_mode")
            )
            experiment = self.experiments_by_key.get(key)
            if experiment is None:
                continue
            chunk = Chunk(
                offset=int(payload.get("offset", -1)),
                batch_size=int(payload.get("batch_size", 0)),
            )
            run_dir, error = verify_completed_chunk(progress_file, experiment, chunk)
            if error is not None or run_dir is None:
                continue
            self.completed[key].update(
                _bounded_offsets(chunk.offset, chunk.batch_size, self.options.num_samples)
            )
            if not self.options.plan_only:
                log_path = (
                    self.logs_dir
                    / experiment.pde
                    / experiment.task
                    / experiment.sensor_mode
                    / experiment.sampler
                    / "recovered.log"
                )
                self._write_completed_marker(experiment, chunk, log_path, run_dir)
                progress_file.unlink(missing_ok=True)

    def _write_completed_marker(
        self,
        experiment: Experiment,
        chunk: Chunk,
        log_path: Path,
        run_dir: Path,
    ) -> None:
        marker_dir = self.completed_dir / experiment.safe_key
        marker_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(
            marker_dir / f"offset_{chunk.offset:06d}_batch_{chunk.batch_size}.json",
            {
                "experiment": experiment.key,
                "signature": experiment.signature,
                "offset": chunk.offset,
                "batch_size": chunk.batch_size,
                "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "log": str(log_path),
                "run_dir": str(run_dir),
            },
        )

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
        deadline = time.time() + 10
        for process in processes:
            timeout = max(0.0, deadline - time.time())
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

    def _aggregate(self) -> int:
        self.console.print("Aggregating metrics...")
        result = subprocess.run(
            [
                sys.executable,
                "-u",
                "-m",
                "sampling.aggregate",
                str(self.options.output_dir),
                "--output-dir",
                str(self.options.output_dir),
            ],
            check=False,
        )
        return int(result.returncode)

    def _log_event(self, message: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}\n"
        with self.event_lock:
            self.event_log.parent.mkdir(parents=True, exist_ok=True)
            with self.event_log.open("a", encoding="utf-8") as handle:
                handle.write(line)


def compute_missing_chunks(total: int, max_batch: int, completed: set[int]) -> list[Chunk]:
    chunks: list[Chunk] = []
    offset = 0
    while offset < total:
        if offset in completed:
            offset += 1
            continue
        start = offset
        while offset < total and offset not in completed and offset - start < max_batch:
            offset += 1
        chunks.append(Chunk(offset=start, batch_size=offset - start))
    return chunks


def discover_completed_artifacts(
    output_dir: Path,
    experiments: list[Experiment],
    num_samples: int,
) -> dict[str, set[int]]:
    completed = {experiment.key: set() for experiment in experiments}
    by_group: dict[tuple[str, str], dict[tuple[str, str], Experiment]] = {}
    for experiment in experiments:
        by_group.setdefault((experiment.pde, experiment.task), {})[
            (experiment.sampler, experiment.sensor_mode)
        ] = experiment
    for (pde, task), variants in by_group.items():
        group_root = output_dir / pde / task
        if not group_root.exists():
            continue
        for metrics_path in group_root.rglob("metrics_final.json"):
            run_dir = metrics_path.parent
            config_path = run_dir / "resolved_config.yaml"
            if not config_path.exists() or not (run_dir / "result.pt").exists():
                continue
            try:
                stored = load_yaml_file(config_path)
            except (OSError, ValueError):
                continue
            sampler = str(stored.get("sampler_phase", ""))
            sensor_mode = str(stored.get("sensor_mode", ""))
            experiment = variants.get((sampler, sensor_mode))
            if experiment is None:
                continue
            offset = int(stored.get("offset", -1))
            batch_size = int(stored.get("batch_size", 0))
            if not _successful_run_artifact(
                run_dir,
                experiment,
                offset,
                batch_size,
                stored_config=stored,
            ):
                continue
            completed[experiment.key].update(
                _bounded_offsets(offset, batch_size, num_samples)
            )
    return completed


def build_experiments(args: argparse.Namespace) -> tuple[list[Experiment], list[GroupKey]]:
    experiments: list[Experiment] = []
    groups: list[GroupKey] = []
    seen_groups: set[GroupKey] = set()
    requested_sensor_modes: list[str | None] = args.sensor_modes or [None]
    for pde in args.pdes:
        for task in args.tasks:
            config_path = Path(args.config_dir) / task / f"{pde}.yaml"
            if not config_path.is_file():
                raise FileNotFoundError(
                    f"Configuration file does not exist for pde={pde}, task={task}: {config_path}"
                )
            base_config = load_config(config_path)
            if base_config.pde != pde or base_config.task != task:
                raise ValueError(
                    f"Configuration identity mismatch in {config_path}: "
                    f"expected pde={pde}, task={task}; "
                    f"found pde={base_config.pde}, task={base_config.task}"
                )
            for sampler in args.samplers:
                for requested_sensor_mode in requested_sensor_modes:
                    overrides: dict[str, Any] = {
                        "pde": pde,
                        "task": task,
                        "sampler_phase": sampler,
                        "batch_size": 1,
                        "offset": 0,
                        "num_steps": args.num_steps,
                        "num_obs": args.num_obs,
                        "output_dir": args.output_dir,
                        "device": args.devices[0],
                        "save_plots": args.vis,
                        "dry_run": args.dry_run,
                    }
                    if requested_sensor_mode is not None:
                        overrides["sensor_mode"] = requested_sensor_mode
                    if args.sample_seed is not None:
                        overrides["sample_seed"] = args.sample_seed
                    config = finalize_ground_truth_config(
                        load_config(config_path, overrides=overrides)
                    )
                    config.validate()
                    config_dict = config.asdict()
                    group = (pde, task, config.sensor_mode)
                    if group not in seen_groups:
                        groups.append(group)
                        seen_groups.add(group)
                    experiments.append(
                        Experiment(
                            pde=pde,
                            task=task,
                            sampler=sampler,
                            sensor_mode=config.sensor_mode,
                            config_path=config_path,
                            config=config_dict,
                            signature=_config_signature(config_dict),
                        )
                    )
    return experiments, groups


def verify_completed_chunk(
    progress_file: Path,
    experiment: Experiment,
    chunk: Chunk,
) -> tuple[Path | None, str | None]:
    try:
        payload = json.loads(progress_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"missing or invalid runner progress file: {exc}"
    raw_run_dir = payload.get("run_dir")
    if not raw_run_dir:
        return None, "runner progress did not report run_dir"
    run_dir = Path(str(raw_run_dir))
    if not _successful_run_artifact(run_dir, experiment, chunk.offset, chunk.batch_size):
        return None, f"run directory is incomplete or incompatible: {run_dir}"
    return run_dir, None


def _successful_run_artifact(
    run_dir: Path,
    experiment: Experiment,
    offset: int,
    batch_size: int,
    *,
    stored_config: dict[str, Any] | None = None,
) -> bool:
    config_path = run_dir / "resolved_config.yaml"
    metrics_path = run_dir / "metrics_final.json"
    if not config_path.is_file() or not metrics_path.is_file() or not (run_dir / "result.pt").is_file():
        return False
    try:
        stored = stored_config if stored_config is not None else load_yaml_file(config_path)
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    if not _stored_config_matches(stored, experiment.config):
        return False
    if int(stored.get("offset", -1)) != offset or int(stored.get("batch_size", 0)) != batch_size:
        return False
    if metrics.get("status", "ok") != "ok":
        return False
    reported = metrics.get("num_samples")
    if reported is not None and int(reported) != batch_size:
        return False
    per_sample_path = run_dir / "metrics_per_sample.csv"
    return not per_sample_path.exists() or _csv_data_row_count(per_sample_path) == batch_size


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run resumable FM4PDE sampling tasks.")
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--max-batch-size", type=int, default=50)
    parser.add_argument("--pdes", nargs="+", required=True)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--samplers", nargs="+", required=True)
    parser.add_argument(
        "--sensor-modes",
        nargs="+",
        default=None,
        help="sensor modes to sweep; omit to use each PDE config's sensor_mode",
    )
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--num-obs", type=int, default=500)
    parser.add_argument("--output-dir", default="outputs/samples")
    parser.add_argument(
        "--config-dir",
        default="configs/main",
        help="task config root containing both/, forward/, and inverse/",
    )
    parser.add_argument("--sample-script", default="scripts/sample/run_sample.sh")
    parser.add_argument("--devices", nargs="+", default=["cuda"])
    parser.add_argument("--max-parallel-tasks", type=int, default=2)
    parser.add_argument("--progress-interval", type=float, default=1.0)
    parser.add_argument("--sample-seed", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    add_bool_arg(parser, "parallel", False, "run PDE/task/sensor groups concurrently")
    add_bool_arg(parser, "resume", True, "reuse matching successful sample artifacts")
    add_bool_arg(parser, "vis", False, "save a plot for each completed batch")
    add_bool_arg(parser, "aggregate", True, "aggregate metrics after successful completion")
    return parser


def main(argv: list[str] | None = None) -> int:
    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.num_samples < 1:
        parser.error("--num-samples must be positive")
    if args.max_batch_size < 1:
        parser.error("--max-batch-size must be positive")
    if args.max_parallel_tasks < 1:
        parser.error("--max-parallel-tasks must be positive")
    if args.progress_interval <= 0:
        parser.error("--progress-interval must be positive")
    if not args.devices:
        parser.error("--devices must contain at least one device")
    try:
        experiments, groups = build_experiments(args)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    options = SweepOptions(
        num_samples=args.num_samples,
        max_batch_size=args.max_batch_size,
        num_steps=args.num_steps,
        num_obs=args.num_obs,
        output_dir=Path(args.output_dir),
        config_dir=Path(args.config_dir),
        sample_script=Path(args.sample_script),
        parallel=args.parallel,
        max_parallel_tasks=args.max_parallel_tasks,
        resume=args.resume,
        vis=args.vis,
        dry_run=args.dry_run,
        aggregate=args.aggregate,
        plan_only=args.plan_only,
        progress_interval=args.progress_interval,
        sample_seed=args.sample_seed,
        devices=tuple(args.devices),
    )
    runner = SweepRunner(options, experiments, groups)
    try:
        return runner.run()
    except KeyboardInterrupt:
        runner.stop_event.set()
        runner._terminate_processes()
        runner.console.print(
            "[yellow]Interrupted. Rerun the same command with RESUME=true to continue.[/yellow]"
        )
        return 130


def _config_signature(config: dict[str, Any]) -> str:
    payload = {
        key: value for key, value in config.items() if key not in _SIGNATURE_IGNORED_FIELDS
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _stored_config_matches(stored: dict[str, Any], desired: dict[str, Any]) -> bool:
    for key, value in desired.items():
        if key in _SIGNATURE_IGNORED_FIELDS:
            continue
        if key not in stored or stored[key] != value:
            return False
    return True


def _sweep_id(experiments: list[Experiment]) -> str:
    payload = [(item.key, item.signature) for item in experiments]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _bounded_offsets(offset: int, batch_size: int, total: int) -> range:
    if offset < 0 or batch_size < 1 or offset >= total:
        return range(0)
    return range(offset, min(total, offset + batch_size))


def _csv_data_row_count(path: Path) -> int:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return sum(1 for _ in csv.DictReader(handle))
    except OSError:
        return -1


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _unique_log_path(path: Path) -> Path:
    if not path.exists():
        return path
    attempt = 2
    while True:
        candidate = path.with_name(f"{path.stem}.attempt{attempt}{path.suffix}")
        if not candidate.exists():
            return candidate
        attempt += 1


def _bar(fraction: float, width: int = 18) -> str:
    fraction = min(1.0, max(0.0, fraction))
    filled = round(width * fraction)
    return f"{'█' * filled}{'░' * (width - filled)} {fraction * 100:5.1f}%"


def _optional_float(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.3g}"


def _raise_keyboard_interrupt(_signum: int, _frame: Any) -> None:
    raise KeyboardInterrupt


if __name__ == "__main__":
    raise SystemExit(main())
