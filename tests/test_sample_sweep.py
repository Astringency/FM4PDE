from __future__ import annotations

import json
from pathlib import Path

from sampling.config import dump_yaml
from sampling.sample_sweep import (
    Chunk,
    Experiment,
    SweepOptions,
    SweepRunner,
    _config_signature,
    build_arg_parser,
    build_experiments,
    compute_missing_chunks,
    discover_completed_artifacts,
)


def test_compute_missing_chunks_skips_completed_offsets_and_preserves_gaps():
    chunks = compute_missing_chunks(total=10, max_batch=4, completed={0, 1, 5, 8, 9})

    assert chunks == [Chunk(offset=2, batch_size=3), Chunk(offset=6, batch_size=2)]


def test_discover_completed_artifacts_accepts_only_matching_successful_runs(tmp_path):
    random = _experiment("poisson", "forward", "random")
    columns = _experiment("poisson", "forward", "sensor_column")
    _write_artifact(tmp_path, random, offset=0, batch_size=2, status="ok")
    _write_artifact(tmp_path, random, offset=2, batch_size=1, status="failed")
    _write_artifact(tmp_path, columns, offset=2, batch_size=1, status="ok")

    completed = discover_completed_artifacts(tmp_path, [random, columns], num_samples=4)

    assert completed[random.key] == {0, 1}
    assert completed[columns.key] == {2}


def test_sweep_runs_sensor_groups_in_parallel_and_resumes_complete_artifacts(tmp_path):
    sample_script = tmp_path / "fake_sample.sh"
    sample_script.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
python - <<'PY'
import csv
import json
import os
import time
from pathlib import Path

output = Path(os.environ["OUTPUT_DIR"])
pde = os.environ["PDE"]
task = os.environ["TASK"]
sampler = os.environ["SAMPLER_PHASE"]
sensor_mode = os.environ["SENSOR_MODE"]
device = os.environ["DEVICE"].replace(":", "_")
offset = int(os.environ["OFFSET"])
batch = int(os.environ["BATCH_SIZE"])
steps = int(os.environ["NUM_STEPS"])
obs = int(os.environ["NUM_OBS"])
run_dir = output / pde / task / "fake" / f"{sampler}-{sensor_mode}-{offset}-{batch}"
run_dir.mkdir(parents=True, exist_ok=True)
calls = output / "calls"
calls.mkdir(parents=True, exist_ok=True)
(calls / f"{pde}-{task}-{sampler}-{sensor_mode}-{device}-{offset}-{time.time_ns()}").touch()
config = {
    "pde": pde,
    "task": task,
    "sampler_phase": sampler,
    "sensor_mode": sensor_mode,
    "num_steps": steps,
    "num_obs": obs,
    "offset": offset,
    "batch_size": batch,
}
(run_dir / "resolved_config.yaml").write_text(
    "\\n".join(f"{key}: {value}" for key, value in config.items()) + "\\n",
    encoding="utf-8",
)
(run_dir / "metrics_final.json").write_text(
    json.dumps({"status": "ok", "num_samples": batch}), encoding="utf-8"
)
(run_dir / "result.pt").write_bytes(b"fake")
with (run_dir / "metrics_per_sample.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=["sample_index"])
    writer.writeheader()
    for index in range(batch):
        writer.writerow({"sample_index": index})
progress = Path(os.environ["FM4PDE_SWEEP_PROGRESS_FILE"])
progress.parent.mkdir(parents=True, exist_ok=True)
progress.write_text(
    json.dumps({
        "status": "complete",
        "step": steps,
        "num_steps": steps,
        "run_dir": str(run_dir),
        "rel_l2_a": 0.1,
        "rel_l2_u": 0.2,
    }),
    encoding="utf-8",
)
PY
""",
        encoding="utf-8",
    )
    experiments = [
        _experiment("poisson", "forward", "random"),
        _experiment("poisson", "forward", "sensor_column"),
    ]
    groups = [(item.pde, item.task, item.sensor_mode) for item in experiments]
    options = SweepOptions(
        num_samples=4,
        max_batch_size=2,
        num_steps=3,
        num_obs=5,
        output_dir=tmp_path / "outputs",
        config_dir=tmp_path,
        sample_script=sample_script,
        parallel=True,
        max_parallel_tasks=2,
        resume=True,
        vis=False,
        dry_run=False,
        aggregate=False,
        plan_only=False,
        progress_interval=0.01,
        sample_seed=None,
        devices=("cuda:0", "cuda:1"),
    )

    first = SweepRunner(options, experiments, groups)
    assert first.worker_count == 2
    assert first.run() == 0
    first_calls = list((options.output_dir / "calls").iterdir())
    assert len(first_calls) == 4
    assert any("cuda_0" in path.name for path in first_calls)
    assert any("cuda_1" in path.name for path in first_calls)

    resumed = SweepRunner(options, experiments, groups)
    assert resumed.run() == 0
    assert len(list((options.output_dir / "calls").iterdir())) == 4

    missing_result = (
        options.output_dir
        / "poisson"
        / "forward"
        / "fake"
        / "stochastic-random-0-2"
        / "result.pt"
    )
    missing_result.unlink()
    repaired = SweepRunner(options, experiments, groups)
    assert repaired.run() == 0
    assert len(list((options.output_dir / "calls").iterdir())) == 5


def test_build_experiments_expands_sensor_modes_into_parallel_groups(tmp_path):
    (tmp_path / "both").mkdir()
    (tmp_path / "both" / "burger.yaml").write_text(
        "pde: burger\ntask: both\nsensor_mode: random\nnum_sensor_columns: 5\n",
        encoding="utf-8",
    )
    args = build_arg_parser().parse_args(
        [
            "--pdes",
            "burger",
            "--tasks",
            "both",
            "--samplers",
            "stochastic",
            "--sensor-modes",
            "random",
            "sensor_column",
            "--config-dir",
            str(tmp_path),
        ]
    )

    experiments, groups = build_experiments(args)

    assert groups == [
        ("burger", "both", "random"),
        ("burger", "both", "sensor_column"),
    ]
    assert [experiment.sensor_mode for experiment in experiments] == [
        "random",
        "sensor_column",
    ]
    assert len({experiment.key for experiment in experiments}) == 2
    assert all(experiment.config_path == tmp_path / "both" / "burger.yaml" for experiment in experiments)


def test_build_experiments_rejects_missing_task_specific_config(tmp_path):
    (tmp_path / "both").mkdir()
    (tmp_path / "both" / "burger.yaml").write_text(
        "pde: burger\ntask: both\nsensor_mode: random\nnum_sensor_columns: 5\n",
        encoding="utf-8",
    )
    args = build_arg_parser().parse_args(
        [
            "--pdes",
            "burger",
            "--tasks",
            "inverse",
            "--samplers",
            "stochastic",
            "--config-dir",
            str(tmp_path),
        ]
    )

    try:
        build_experiments(args)
    except FileNotFoundError as exc:
        assert "pde=burger, task=inverse" in str(exc)
        assert str(tmp_path / "inverse" / "burger.yaml") in str(exc)
    else:
        raise AssertionError("missing Burger inverse config should be rejected")


def _experiment(pde: str, task: str, sensor_mode: str = "random") -> Experiment:
    config = {
        "pde": pde,
        "task": task,
        "sampler_phase": "stochastic",
        "sensor_mode": sensor_mode,
        "num_steps": 3,
        "num_obs": 5,
    }
    return Experiment(
        pde=pde,
        task=task,
        sampler="stochastic",
        sensor_mode=sensor_mode,
        config_path=Path(f"{pde}.yaml"),
        config=config,
        signature=_config_signature(config),
    )


def _write_artifact(
    root: Path,
    experiment: Experiment,
    *,
    offset: int,
    batch_size: int,
    status: str,
) -> None:
    run_dir = (
        root
        / experiment.pde
        / experiment.task
        / "fake"
        / f"{experiment.sensor_mode}-run-{offset}"
    )
    run_dir.mkdir(parents=True)
    config = {**experiment.config, "offset": offset, "batch_size": batch_size}
    (run_dir / "resolved_config.yaml").write_text(dump_yaml(config), encoding="utf-8")
    (run_dir / "metrics_final.json").write_text(
        json.dumps({"status": status, "num_samples": batch_size}), encoding="utf-8"
    )
    (run_dir / "result.pt").write_bytes(b"fake")
