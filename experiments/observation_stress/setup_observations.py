"""Build the Poisson observation-study manifests without touching prior results."""
from __future__ import annotations

import argparse
import subprocess
import sys

from experiments.observation_stress.measurements import CASES
from experiments.observation_stress.setup import FM, BASE, STORAGE, PROTOCOL, NOISE, mounted
from experiments.rough_stress.run import write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepare-data", action="store_true")
    args = p.parse_args()
    inputs = BASE/"measurements"
    for host in ["197", "193"]:
        jobs = []
        for case in CASES:
            for distribution in ["id", "rough"]:
                for task in ["forward", "inverse", "both"]:
                    for method in ["recfno", "senseiver", "voronoicnn", "fm4pde"]:
                        group = "fm" if method == "fm4pde" else "baseline"
                        if (host == "197") != (group == "fm"):
                            continue
                        root = FM if group == "fm" else BASE
                        argv = ["--output-root", mounted(root/"observations", host),
                            "--input-root", mounted(inputs, host), "--distribution", distribution,
                            "--case", case, "--method", method, "--task", task,
                            "--batch-size", "16", "--stop", "100", "--device", "cuda:0"]
                        if group == "fm":
                            argv += ["--fm-protocol", str(PROTOCOL), "--fm-checkpoint",
                                str(PROTOCOL.parent/"fm4poisson.pth"), "--noise-root", str(NOISE)]
                        else:
                            argv += ["--baseline-root", "/home/zhangxf/C01Python/FM4PDEbaseline",
                                "--baseline-output-root", mounted(STORAGE/"outputs/FM4PDEbaseline", host)]
                        jobs.append(dict(key=f"poisson_{method}_{task}_{distribution}_{case}",
                            group=group, module="experiments.observation_stress.run", argv=argv))
        write_json(FM/f"setup/observations_{host}.json", dict(jobs=jobs))
    if args.prepare_data:
        subprocess.run([sys.executable, "-u", "-m", "experiments.observation_stress.measurements",
            "--source-root", str(STORAGE/"outputs/FM4PDEbaseline/rough_stress_20260918"),
            "--output-root", str(inputs)], check=True)
        write_json(FM/"setup/observation_inputs_complete.json", dict(cases=list(CASES),
            distributions=["id", "rough"], count=100))


if __name__ == "__main__":
    main()
