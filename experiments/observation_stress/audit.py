"""Recompute all requested errors from saved predictions and frozen inputs."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics

import torch

from experiments.observation_stress.measurements import CASES
from experiments.observation_stress.setup import PDES, SETTINGS
from experiments.rough_stress.run import sha256, tensor_hash, write_json

METHODS = ["recfno", "senseiver", "voronoicnn", "fm4pde"]
TASKS = ["forward", "inverse", "both"]


def targets(task):
    return [("a", 0), ("u", 1)] if task == "both" else [("u", 1)] if task == "forward" else [("a", 0)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fm-root", required=True)
    p.add_argument("--baseline-root", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--allow-incomplete", action="store_true")
    args = p.parse_args()
    torch.set_num_threads(2)
    fm, base, out = map(Path, [args.fm_root, args.baseline_root, args.output_root])
    out.mkdir(parents=True, exist_ok=True)
    cache = {}
    def source(pde, dist, case, kind):
        key = (pde, dist, case if kind == "observations" else None)
        if key not in cache:
            path = base/"measurements"/f"{pde}_{dist}_{case}.pt" if kind == "observations" else base/"inputs"/f"{pde}_{dist}.pt"
            receipt = json.loads(path.with_suffix(".json").read_text())
            digest = receipt["sha256"] if kind == "observations" else receipt["pack_sha256"]
            if sha256(path) != digest:
                raise ValueError(f"Input checksum changed: {path}")
            pack = torch.load(path, map_location="cpu", weights_only=False)
            assert pack["count"] == 100 and len(pack["sample_ids"]) == 100
            if kind == "observations":
                for name in ["mask", "clean", "noisy"]:
                    assert tensor_hash(pack[name]) == receipt[name+"_sha256"]
                mask, full = pack["mask"], pack["raw"]["full_tensor"]
                k = int(pack["metadata"]["window"])
                if k == 1:
                    expected = full*mask
                else:
                    expected = torch.zeros_like(full)
                    left, right = k//2, k-1-k//2
                    values = full.unfold(2, k, 1).unfold(3, k, 1).mean((-1, -2))
                    expected[..., left:128-right, left:128-right] = values
                    expected *= mask
                torch.testing.assert_close(pack["clean"], expected, rtol=5e-6, atol=1e-9)
                assert not bool((mask.bool() & pack["excluded"]).any())
                assert bool((mask.flatten(1).sum(1) == (640 if case == "columns" else 500)).all())
                assert not torch.count_nonzero(pack["noisy"]*(1-mask))
            cache[key] = pack
        return cache[key]
    cells = []
    for dist, budget in SETTINGS:
        for pde in PDES:
            for task in TASKS:
                for method in METHODS:
                    cells.append(("distribution", pde, dist, f"obs{budget}", task, method))
    for case in CASES:
        for dist in ["id", "rough"]:
            for task in TASKS:
                for method in METHODS:
                    cells.append(("observations", "poisson", dist, case, task, method))
    rows, completed, incomplete = [], [], []
    for kind, pde, dist, case, task, method in cells:
        root = fm if method == "fm4pde" else base
        folder = root/kind/"results"/pde/method/task/dist/case
        identity_path = folder/"identity.json"
        paths = sorted(folder.glob("batch_*.json"))
        seen = set()
        if not identity_path.exists():
            incomplete.append(dict(path=str(folder), completed_samples=0))
            continue
        identity = json.loads(identity_path.read_text())
        assert (identity["pde"],identity["distribution"],identity["task"],identity["method"]) == (pde,dist,task,method)
        pack = source(pde, dist, case, kind)
        cell_rows = []
        for path in paths:
            record = json.loads(path.read_text())
            predpath = path.with_suffix(".pt")
            if not predpath.exists():
                continue
            if sha256(predpath) != record["prediction_sha256"]:
                raise ValueError(f"Prediction checksum changed: {predpath}")
            saved = torch.load(predpath, map_location="cpu", weights_only=False)
            indices = saved["indices"]
            assert indices == list(range(record["start"], record["stop"]))
            assert not (seen & set(indices))
            seen.update(indices)
            assert saved["sample_ids"] == [pack["sample_ids"][i] for i in indices]
            truth = pack["raw"]["full_tensor"][indices].double()
            pred = saved["prediction"].double()
            assert torch.isfinite(pred).all()
            if kind == "observations":
                mask = pack["mask"][indices]
                active = torch.ones((len(indices), 2, 1, 1))
                if task == "forward": active[:, 1] = 0
                if task == "inverse": active[:, 0] = 0
                assert record["observation_mask_sha256"] == tensor_hash(mask)
                assert record["active_clean_readings_sha256"] == tensor_hash(pack["clean"][indices]*active)
                assert record["active_noisy_readings_sha256"] == tensor_hash(pack["noisy"][indices]*active)
                budget = int(pack["metadata"]["readings_per_field"])
            else:
                budget = int(case[3:])
                assert record["observation_mask_sha256"] == tensor_hash(pack["masks"][budget][indices])
            for field, tc in targets(task):
                pc = tc if method == "fm4pde" or task == "both" else 0
                for offset, i in enumerate(indices):
                    b = truth[offset, tc]
                    a = pred[offset, pc]
                    value = float((a-b).norm()/b.norm().clamp_min(1e-12))
                    rmse = float((a-b).square().mean().sqrt())
                    prior = record["rows"][offset]
                    assert prior["index"] == i and prior["sample_id"] == pack["sample_ids"][i]
                    assert math.isclose(value, prior[f"rel_l2_{field}"], rel_tol=1e-10, abs_tol=1e-12)
                    assert math.isclose(rmse, prior[f"rmse_{field}"], rel_tol=1e-10, abs_tol=1e-12)
                    row = dict(study=kind, pde=pde, distribution=dist, case=case, num_obs=budget,
                        task=task, method=method, field=field, index=i, sample_id=pack["sample_ids"][i],
                        rel_l2=value, rmse=rmse, rel_l2_excluded=None, rel_l2_outside_excluded=None)
                    if kind == "observations" and case in {"hole10", "hole25", "strip25"}:
                        for name, spatial in [("excluded", pack["excluded"][0, 0]),
                                              ("outside_excluded", ~pack["excluded"][0, 0])]:
                            v = float((a[spatial]-b[spatial]).norm()/b[spatial].norm().clamp_min(1e-12))
                            assert math.isclose(v, prior[f"rel_l2_{field}_{name}"], rel_tol=1e-10, abs_tol=1e-12)
                            row[f"rel_l2_{name}"] = v
                    cell_rows.append(row)
        assert seen <= set(range(100))
        if seen == set(range(100)):
            completed.append(str(folder))
            rows.extend(cell_rows)
        else:
            incomplete.append(dict(path=str(folder), completed_samples=len(seen)))
    if rows:
        with (out/"per_sample.csv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        groups = {}
        key_fields = ["study", "pde", "distribution", "case", "num_obs", "task", "method", "field"]
        for row in rows:
            groups.setdefault(tuple(row[k] for k in key_fields), []).append(row)
        summary = []
        for key, group in groups.items():
            row = dict(zip(key_fields, key)); row["count"] = len(group)
            for metric in ["rel_l2", "rmse", "rel_l2_excluded", "rel_l2_outside_excluded"]:
                values = [r[metric] for r in group if r[metric] is not None]
                row["mean_"+metric] = statistics.fmean(values) if values else None
            summary.append(row)
        with (out/"summary.csv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
    write_json(out/"audit.json", dict(expected_cells=len(cells), complete_cells=len(completed),
        status="complete" if not incomplete else "incomplete", samples_per_cell=100,
        metric_rows=len(rows), complete=completed, incomplete=incomplete,
        independent_prediction_metrics=True, frozen_input_and_measurement_hashes_verified=True))
    print(f"AUDIT {len(completed)}/{len(cells)} cells, {len(rows)} metric rows", flush=True)
    if incomplete and not args.allow_incomplete:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
