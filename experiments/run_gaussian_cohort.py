"""Bounded CPU worker pool for the predeclared independent-input Gaussian control."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    target = args.output / "gaussian_tilt/cohort"
    target.mkdir(parents=True, exist_ok=True)
    ids = list(range(1500, 1532))
    protocol = dict(input_ids=ids, seeds=[0, 1, 2], workers=args.workers, start_unix=time.time(),
        primary_comparison="exact_flow100 vs plugin_stochastic100; same full Gaussian prior, masks and weights",
        primary_metrics=["inputwise seed-mean relative L2(a)", "inputwise seed-mean relative L2(u)"],
        statistical_unit="physical input, n=32; three seeds are repeated measurements, not 96 independent inputs",
        inference="paired bootstrap of physical inputs, 25000 resamples; 97.5% intervals per field for two primary field comparisons",
        practical_threshold="at least 5% relative decrease in inputwise mean error; significance and practical magnitude reported separately",
        secondary_controls=["exact_flow200 (same-noise integration refinement)", "direct exact Gaussian posterior draws", "posterior mean"],
        scope="Gaussian-prior guidance/sampler ablation. This does not identify the effect of replacing guidance for a fixed trained FM prior.",
        cohort_provenance="Previously declared 32-ID cache, disjoint from input 0 pilot and IDs 1100..1103 used for prior calibration.")
    path = target / "cohort_protocol.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(protocol, indent=2) + "\n")
    children, lock, stopping = {}, threading.Lock(), threading.Event()
    def stop(signum, frame):
        stopping.set()
        with lock:
            for child in children.values():
                if child.poll() is None:
                    child.terminate()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    def one(i):
        if stopping.is_set():
            return dict(input_id=i, status="not_started")
        command = [sys.executable, str(root / "experiments/run_gaussian_tilt.py"),
            "--inputs", str(args.inputs), "--plan", str(args.plan), "--output", str(args.output),
            "--name", f"cohort/input{i}", "--ids", str(i), "--seeds", "0", "1", "2"]
        receipt = dict(input_id=i, command=command, start_unix=time.time())
        with (target / f"input{i}.log").open("w") as stream:
            child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2"))
            with lock:
                children[i] = child
            receipt["pid"] = child.pid
            (target / f"input{i}_execution.json").write_text(json.dumps(receipt, indent=2) + "\n")
            receipt["exit_code"] = child.wait()
        receipt["end_unix"] = time.time()
        (target / f"input{i}_execution.json").write_text(json.dumps(receipt, indent=2) + "\n")
        return receipt
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(one, i) for i in ids]):
            row = future.result()
            results.append(row)
            print("INPUT_COMPLETE", row, flush=True)
            (target / "progress.json").write_text(json.dumps(results, indent=2) + "\n")
    if any(r.get("exit_code") != 0 for r in results):
        raise RuntimeError("Cohort has failed/stopped workers; inspect per-input logs")
    (target / "complete.json").write_text(json.dumps(dict(input_ids=ids, end_unix=time.time()), indent=2) + "\n")


if __name__ == "__main__":
    main()
