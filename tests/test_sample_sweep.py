from __future__ import annotations

import json
from pathlib import Path

from sampling.sample_sweep import (
    Chunk,
    Experiment,
    SweepOptions,
    SweepRunner,
    _config_signature,
    compute_missing_chunks,
    discover_completed_artifacts,
)


def test_compute_missing_chunks_skips_completed_offsets_and_preserves_gaps():
    chunks = compute_missing_chunks(total=10, max_batch=4, completed={0, 1, 5, 8, 9})

    assert chunks == [Chunk(offset=2, batch_size=3), Chunk(offset=6, batch_size=2)]


def test_discover_completed_artifacts_accepts_only_matching_successful_runs(tmp_path):
    experiment = _experiment("poisson", "forward")
    _write_artifact(tmp_path, experiment, offset=0, batch_size=2, status="ok")
    _write_artifact(tmp_path, experiment, offset=2, batch_size=1, status="failed")

    completed = discover_completed_artifacts(tmp_path, [experiment], num_samples=4)

    assert completed[experiment.key] == {0, 1}


def test_sweep_runs_pde_task_groups_and_resumes_only_complete_artifacts(tmp_path):
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
device = os.environ["DEVICE"].replace(":", "_")
offset = int(os.environ["OFFSET"])
batch = int(os.environ["BATCH_SIZE"])
steps = int(os.environ["NUM_STEPS"])
obs = int(os.environ["NUM_OBS"])
run_dir = output / pde / task / "fake" / f"{sampler}-{offset}-{batch}"
run_dir.mkdir(parents=True, exist_ok=True)
calls = output / "calls"
calls.mkdir(parents=True, exist_ok=True)
(calls / f"{pde}-{task}-{sampler}-{device}-{offset}-{time.time_ns()}").touch()
config = {
    "pde": pde,
    "task": task,
    "sampler_phase": sampler,
    "num_steps": steps,
    "num_obs": obs,
    "offset": offset,
    "batch_size": batch,
}
(run_dir / "resolved_config.yaml").write_text(json.dumps(config), encoding="utf-8")
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
        _experiment("poisson", "forward"),
        _experiment("darcy", "inverse"),
    ]
    groups = [(item.pde, item.task) for item in experiments]
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

    missing_result = options.output_dir / "poisson" / "forward" / "fake" / "stochastic-0-2" / "result.pt"
    missing_result.unlink()
    repaired = SweepRunner(options, experiments, groups)
    assert repaired.run() == 0
    assert len(list((options.output_dir / "calls").iterdir())) == 5


def _experiment(pde: str, task: str) -> Experiment:
    config = {
        "pde": pde,
        "task": task,
        "sampler_phase": "stochastic",
        "num_steps": 3,
        "num_obs": 5,
    }
    return Experiment(
        pde=pde,
        task=task,
        sampler="stochastic",
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
    run_dir = root / experiment.pde / experiment.task / "fake" / f"run-{offset}"
    run_dir.mkdir(parents=True)
    config = {**experiment.config, "offset": offset, "batch_size": batch_size}
    (run_dir / "resolved_config.yaml").write_text(json.dumps(config), encoding="utf-8")
    (run_dir / "metrics_final.json").write_text(
        json.dumps({"status": status, "num_samples": batch_size}), encoding="utf-8"
    )
    (run_dir / "result.pt").write_bytes(b"fake")
