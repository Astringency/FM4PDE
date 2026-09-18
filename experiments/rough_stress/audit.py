"""Independently recompute stress-test metrics from saved physical predictions.

Checks all 1000 source indices, frozen masks, receipts, prediction digests and
per-sample metrics. Partial audits explicitly list every missing cell.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from experiments.rough_stress.run import sha256, tensor_hash, write_json

TASK_FIELDS = {"forward": ("u",), "inverse": ("a",), "both": ("a", "u")}
SETTINGS = [(d, 500) for d in ("id", "smooth", "rough", "rough2", "rough3")] + [
    ("id", 400), ("rough", 400)]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def verify_pack(path):
    receipt = json.loads(path.with_suffix(".json").read_text())
    require(sha256(path) == receipt["pack_sha256"], f"Bad pack: {path}")
    data = torch.load(path, map_location="cpu", weights_only=False)
    require(data["count"] == 1000, "Evaluation requires 1000 samples")
    require(data["source_indices"].tolist() == list(range(1000)), "Wrong input indices")
    require(len(set(data["sample_ids"])) == 1000, "Duplicate source samples")
    truth = data["raw"]["full_tensor"]
    require(tuple(truth.shape) == (1000, 2, 128, 128), "Unexpected truth shape")
    require(bool(torch.isfinite(truth).all()), "Nonfinite truth")
    require(tensor_hash(truth) == receipt["physical_tensor_sha256"], "Truth digest mismatch")
    for n, mask in data["masks"].items():
        require(tensor_hash(mask) == receipt["masks"][str(n)], "Mask digest mismatch")
        require(bool(((mask == 0) | (mask == 1)).all()), "Nonbinary mask")
        require(bool((mask.flatten(1).sum(1) == n).all()), "Wrong observation count")
    require(bool((data["masks"][400] <= data["masks"][500]).all()), "Unpaired 400/500 masks")
    # Independently regenerate the baseline contract without calling its adapter.
    for i, sample_id in enumerate(data["sample_ids"]):
        seed = int.from_bytes(hashlib.sha256(f"mask|1|test|{sample_id}|0".encode()).digest()[:8], "big") % (2**63-1)
        order = torch.randperm(128*128, generator=torch.Generator().manual_seed(seed))
        for n in (400, 500):
            expected = torch.zeros(128*128, dtype=torch.uint8)
            expected[order[:n]] = 1
            require(torch.equal(expected, data["masks"][n][i].flatten()), "Mask seed contract mismatch")
    return data, receipt


def field_statistics(data):
    rows = []
    for c, field in enumerate(("a", "u")):
        x = data["raw"]["full_tensor"][:, c].numpy().astype(np.float64)
        rms = np.sqrt(np.mean(x*x, axis=(1, 2)))
        dx, dy = np.diff(x, axis=1), np.diff(x, axis=2)
        gradient = np.sqrt((np.mean(dx*dx, axis=(1, 2)) + np.mean(dy*dy, axis=(1, 2)))/2)
        rows.append(dict(distribution=data["distribution"], field=field, count=len(x),
            mean_rms=float(rms.mean()), mean_neighbor_difference_rms=float(gradient.mean()),
            mean_neighbor_difference_over_rms=float(np.mean(gradient/np.maximum(rms, 1e-12)))))
    return rows


def audit_cell(cell, data, receipt, expected_checkpoint):
    identity = json.loads((cell/"identity.json").read_text())
    require(identity["input_receipt"] == receipt, f"Input changed: {cell}")
    method, task, dist, n = (identity[k] for k in ("method", "task", "distribution", "num_obs"))
    require(cell.parts[-4:] == (method, task, dist, f"obs{n}"), "Identity/path mismatch")
    if identity.get("origin") == "verified_prior_paired_evaluation":
        require(method == "fm4pde" and dist in {"id","rough"} and n == 500, "Invalid reused control")
        proof_path = cell.parents[5]/identity["checkpoint_equivalence_file"]
        require(sha256(proof_path) == identity["checkpoint_equivalence_sha256"], "Equivalence proof changed")
        proof = json.loads(proof_path.read_text())
        require(proof["status"] == "verified" and proof["model_parameters_exact"] and proof["normalizer_exact"], "Checkpoint equivalence failed")
        require(proof["source_checkpoint_sha256"] == identity["checkpoint"]["sha256"], "Wrong archived checkpoint")
        require(proof["equivalent_checkpoint_sha256"] == expected_checkpoint == identity["equivalent_current_checkpoint"]["sha256"], "Wrong equivalent checkpoint")
        match = next(x for x in proof["cells"] if x["task"] == task and x["distribution"] == dist)
        require(match["current_input_receipt"] == receipt and match["count"] == 1000 and match["replay_max_relative_difference"] < 1e-4, "Control reuse was not verified")
    else:
        require(identity["checkpoint"]["sha256"] == expected_checkpoint, "Checkpoint changed")
    require(identity["count"] == 1000 and not identity["tf32"], "Run protocol changed")
    files = sorted(cell.glob("batch_*.json"))
    coverage, per_sample = [], []
    for record_path in files:
        record = json.loads(record_path.read_text())
        lo, hi = record["start"], record["stop"]
        require(0 <= lo < hi <= 1000, "Invalid batch range")
        prediction_path = record_path.with_suffix(".pt")
        require(sha256(prediction_path) == record["prediction_sha256"], "Prediction digest mismatch")
        saved = torch.load(prediction_path, map_location="cpu", weights_only=False)
        require(saved["indices"] == list(range(lo, hi)), "Prediction indices mismatch")
        require(saved["sample_ids"] == data["sample_ids"][lo:hi], "Prediction sample IDs mismatch")
        require(record["observation_mask_sha256"] == tensor_hash(data["masks"][n][lo:hi]), "Consumed masks mismatch")
        prediction = saved["prediction"].numpy().astype(np.float64)
        truth = data["raw"]["full_tensor"][lo:hi].numpy().astype(np.float64)
        channels = 2 if method == "fm4pde" or task == "both" else 1
        require(prediction.shape == (hi-lo, channels, 128, 128), "Invalid prediction shape")
        require(bool(np.isfinite(prediction).all()), "Nonfinite prediction")
        require(len(record["rows"]) == hi-lo, "Missing logged errors")
        metrics = {}
        fields = ("a", "u") if channels == 2 else TASK_FIELDS[task]
        for field in fields:
            tc = 0 if field == "a" else 1
            pc = tc if channels == 2 else 0
            delta = prediction[:, pc] - truth[:, tc]
            rel = np.sqrt(np.sum(delta*delta, axis=(1, 2))) / np.maximum(np.sqrt(np.sum(truth[:, tc]**2, axis=(1, 2))), 1e-12)
            rmse = np.sqrt(np.mean(delta*delta, axis=(1, 2)))
            metrics[field] = (rel, rmse)
            for j, logged in enumerate(record["rows"]):
                require(logged["index"] == lo+j and logged["sample_id"] == saved["sample_ids"][j], "Logged sample mismatch")
                require(math.isclose(logged[f"rel_l2_{field}"], float(rel[j]), rel_tol=1e-10, abs_tol=1e-12), "Relative error mismatch")
                require(math.isclose(logged[f"rmse_{field}"], float(rmse[j]), rel_tol=1e-10, abs_tol=1e-12), "RMSE mismatch")
        if method == "fm4pde":
            require(record["runtime"]["steps"] == 100, "FM step count changed")
            require(record["runtime"]["random_policy"] == "canonical_pool_archived_seed_fixed_input_row_v1", "FM noise changed")
        for field in TASK_FIELDS[task]:
            rel, rmse = metrics[field]
            for j in range(hi-lo):
                per_sample.append(dict(method=method, task=task, distribution=dist, num_obs=n,
                    field=field, index=lo+j, sample_id=saved["sample_ids"][j], rel_l2=float(rel[j]), rmse=float(rmse[j])))
        coverage.extend(range(lo, hi))
    require(sorted(coverage) == list(range(1000)), f"Incomplete/overlapping coverage: {cell} ({len(coverage)})")
    return per_sample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--fm-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    base, fm, output = map(Path, (args.baseline_root, args.fm_root, args.output_root))
    output.mkdir(parents=True, exist_ok=True)
    checkpoints = json.loads((base/"poisson_checkpoints.json").read_text())
    fm_sha = "93e568b9957836776e3e5d17ad59fede6b2de47a3b17b3b8eeaaeadb5c1391f0"
    all_rows, statistics, incomplete, complete = [], [], [], []
    summaries = []
    for dist in ("id", "smooth", "rough", "rough2", "rough3"):
        data, receipt = verify_pack(base/"inputs"/f"poisson_{dist}.pt")
        statistics.extend(field_statistics(data))
        for n in ([500, 400] if dist in {"id", "rough"} else [500]):
            for method in ("recfno", "senseiver", "voronoicnn", "fm4pde"):
                for task in TASK_FIELDS:
                    root = fm if method == "fm4pde" else base
                    cell = root/"results"/"poisson"/method/task/dist/f"obs{n}"
                    job = root/"queue_state"/f"{method}_{task}_{dist}_obs{n}.json"
                    status = json.loads(job.read_text()).get("status") if job.exists() else "unclaimed"
                    if status != "complete":
                        incomplete.append(dict(cell=str(cell), status=status))
                        continue
                    expected = fm_sha if method == "fm4pde" else checkpoints[f"{method}/{task}"]["sha256"]
                    rows = audit_cell(cell, data, receipt, expected)
                    all_rows.extend(rows)
                    complete.append(str(cell))
                    for field in TASK_FIELDS[task]:
                        subset = [row for row in rows if row["field"] == field]
                        values = np.array([row["rel_l2"] for row in subset])
                        summaries.append(dict(method=method, task=task, distribution=dist, num_obs=n,
                            field=field, count=len(values), mean_rel_l2=float(values.mean()),
                            std_rel_l2=float(values.std(ddof=1)), se_rel_l2=float(values.std(ddof=1)/np.sqrt(len(values))),
                            median_rel_l2=float(np.median(values)), p90_rel_l2=float(np.quantile(values, .9)),
                            mean_rmse=float(np.mean([row["rmse"] for row in subset]))))
                    print(f"AUDITED {method}/{task}/{dist}/obs{n}: 1000", flush=True)
        del data
    write_csv(output/"summary.csv", summaries)
    write_csv(output/"per_sample.csv", all_rows)
    write_csv(output/"field_statistics.csv", statistics)
    write_json(output/"audit.json", dict(complete_cells=len(complete), expected_cells=84,
        samples_per_cell=1000, complete=complete, incomplete=incomplete,
        status="complete" if len(complete) == 84 and not incomplete else "partial",
        independent_metric_recomputation=True, independent_mask_regeneration=True))
    if incomplete and not args.allow_incomplete:
        raise SystemExit(f"Missing {len(incomplete)} of 84 cells")
    print(f"AUDIT {len(complete)}/84 complete cells; outputs={output}", flush=True)


if __name__ == "__main__":
    main()
