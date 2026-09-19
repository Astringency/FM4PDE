"""Freeze source identities and prepare the cross-PDE first-100 experiments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from experiments.rough_stress.run import sha256, write_json

STUDY = "observation_stress_20260919"
STORAGE = Path("/large_storage/zhangxf")
FM = STORAGE / "outputs/FM4PDE" / STUDY
BASE = STORAGE / "outputs/FM4PDEbaseline" / STUDY
PROTOCOL = STORAGE / "outputs/FM4PDE/rough_stress_20260918/setup/reference_protocol.json"
NOISE = PROTOCOL.parent / "noise_a100_seed0"
PDES = ["helmholtz", "darcy", "nsnonbounded"]
SETTINGS = [("rough2", 500), ("rough3", 500), ("id", 500),
            ("smooth", 500), ("rough", 500), ("id", 400), ("rough", 400)]


def mounted(path, host):
    value = str(path)
    if host == "193":
        value = value.replace("/large_storage/zhangxf/", "/home/zhangxf/share/zhangxfA100/large_storage/")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--prepare-data", action="store_true")
    args = parser.parse_args()
    (FM / "setup").mkdir(parents=True, exist_ok=True)
    (FM / "logs").mkdir(exist_ok=True)
    BASE.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(PROTOCOL.read_text())
    checkpoints = {}
    for pde in PDES:
        cell = next(c for c in protocol["cells"] if c["pde"] == pde and
                    c["cohort"] == "supervised" and c["setting"] == "sparse_joint")
        frozen = cell["checkpoint"]
        path = Path(frozen["source"].replace("/home/tat512/C01Python/audit/",
                    "/large_storage/zhangxf/outputs/FM4PDE/audit/"))
        if sha256(path) != frozen["sha256"]:
            raise ValueError(f"Checkpoint differs from the paper protocol: {path}")
        checkpoints[pde] = dict(path=str(path), sha256=frozen["sha256"])
    write_json(FM / "setup/fm_checkpoints.json", checkpoints)
    write_json(FM / "setup/scope.json", dict(
        distributions=dict(pdes=PDES, count=100, settings=SETTINGS,
                           tasks=["forward", "inverse", "both"]),
        layout=dict(pde="poisson", distributions=["id", "rough"], count=100,
                    layouts=["random", "grid", "columns", "hole10", "hole25", "strip25"]),
        noise=dict(pde="poisson", distributions=["id", "rough"], count=100,
                   levels=[0, .01, .05, .1], tasks=["forward", "inverse", "both"]),
        averages=dict(pde="poisson", distributions=["id", "rough"], count=100,
                      windows=[1, 4, 8, 16], readings_per_active_field=500,
                      tasks=["forward", "inverse", "both"]),
        training="none; frozen pretrained checkpoints", protocol_sha256=sha256(PROTOCOL)))
    for host in ["197", "193"]:
        jobs = []
        for dist, num_obs in SETTINGS:
            for pde in PDES:
                for task in ["forward", "inverse", "both"]:
                    for method in ["recfno", "senseiver", "voronoicnn", "fm4pde"]:
                        group = "fm" if method == "fm4pde" else "baseline"
                        if (host == "197") != (group == "fm"):
                            continue
                        out = FM if group == "fm" else BASE
                        argv = ["run", "--output-root", mounted(out / "distribution", host),
                            "--input-root", mounted(BASE, host), "--pde", pde,
                            "--method", method, "--task", task, "--distribution", dist,
                            "--num-obs", str(num_obs), "--batch-size", "16", "--stop", "100",
                            "--device", "cuda:0"]
                        if group == "fm":
                            argv += ["--fm-protocol", str(PROTOCOL), "--fm-checkpoint",
                                     checkpoints[pde]["path"], "--noise-root", str(NOISE)]
                        else:
                            argv += ["--baseline-root", "/home/zhangxf/C01Python/FM4PDEbaseline",
                                "--baseline-output-root", mounted(STORAGE / "outputs/FM4PDEbaseline", host)]
                        jobs.append(dict(key=f"{pde}_{method}_{task}_{dist}_obs{num_obs}", group=group,
                            module="experiments.rough_stress.run", argv=argv))
        write_json(FM / f"setup/distribution_{host}.json", dict(jobs=jobs))
    if args.prepare_data:
        for pde in PDES:
            command = [sys.executable, "-u", "-m", "experiments.rough_stress.run", "prepare",
                "--output-root", str(BASE), "--baseline-root", args.baseline_root,
                "--baseline-output-root", str(STORAGE / "outputs/FM4PDEbaseline"),
                "--matrix", str(STORAGE / "outputs/FM4PDEbaseline/matrices/main_results.jsonl"),
                "--data-root", str(STORAGE / "PDEdata"), "--pde", pde, "--count", "100",
                "--distributions", "rough2", "rough3", "id", "smooth", "rough"]
            print("PREPARING", pde, flush=True)
            subprocess.run(command, check=True)
        write_json(FM / "setup/distribution_inputs_complete.json", dict(pdes=PDES, count=100))


if __name__ == "__main__":
    main()
