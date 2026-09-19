"""Check the promising forward reconstruction at approximately matched runtime.

The original sampler's step count is fixed using measured seed-0 run times,
without selecting a step count by reconstruction error. Its original guidance
coefficients are retained; this is a compute control, not a tuned optimum.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.proposal_guidance.study import prepared, config_batch, tracked_infer, evaluate
from experiments.aligned_sampling.run_inference import digest, runtime_environment
from sampling.metrics import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    p, data, bundle = prepared(args)
    task = "sparse_forward"
    out = args.root / "budget_control"
    out.mkdir(exist_ok=True)
    source = args.root / "runs" / task
    source_records = {v: [json.loads(f.read_text()) for f in sorted(source.glob(f"{v}_seed0_*.json"))]
                      for v in ["original", "proposal"]}
    assert all(len(rows) == len(p["indices"]) // p["batch_size"] for rows in source_records.values())
    time_ratio = (sum(r["runtime"]["seconds"] for r in source_records["proposal"])
                  / sum(r["runtime"]["seconds"] for r in source_records["original"]))
    steps = round(100 * time_ratio / 5) * 5
    plan = dict(task=task, steps=steps, seed0_time_ratio=time_ratio, seeds=p["seeds"], indices=p["indices"],
        selection="Round 100 × measured seed-0 proposal/original time ratio to nearest 5 steps; no accuracy-based selection",
        environment=runtime_environment("cuda:0", False))
    write_json(out / "protocol.json", plan)
    print("BUDGET_PLAN", plan, flush=True)
    for seed in p["seeds"]:
        for start in range(0, len(p["indices"]), p["batch_size"]):
            indices = p["indices"][start:start+p["batch_size"]]
            cfg, gt, masks, hashes = config_batch(p, data, task, indices, seed, "original", args.root, steps=steps)
            pred, metadata = tracked_infer(cfg, bundle, gt, masks, indices)
            rows = evaluate(pred, data, indices, cfg, gt, masks)
            comparison = json.loads((source / f"proposal_seed{seed}_batch{start:04d}.json").read_text())
            assert hashes == comparison["input_hashes"]
            assert metadata["initial_noise_sha256"] == comparison["runtime"]["initial_noise_sha256"]
            assert metadata["bridge_noise_sha256"][:100] == comparison["runtime"]["bridge_noise_sha256"]
            path = out / f"original_seed{seed}_batch{start:04d}.pt"
            if path.exists():
                raise FileExistsError(path)
            torch.save(dict(indices=indices, seed=seed, prediction=pred, rows=rows, runtime=metadata,
                            config=cfg.asdict(), input_hashes=hashes), path)
            write_json(path.with_suffix(".json"), dict(indices=indices, seed=seed, rows=rows,
                runtime=metadata, input_hashes=hashes, sha256=digest(path)))
            print("BUDGET_BATCH", seed, start, steps, metadata["seconds"],
                  float(np.mean([r["rel_l2_u"] for r in rows])), flush=True)
    write_json(out / "complete.json", dict(steps=steps, cases=len(p["indices"]), seeds=p["seeds"]))


if __name__ == "__main__":
    main()
