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
    p.add_argument("--verify-source-files", action="store_true",
                   help="Rehash the original MAT files as well as frozen packs")
    args = p.parse_args()
    torch.set_num_threads(2)
    fm, base, out = map(Path, [args.fm_root, args.baseline_root, args.output_root])
    out.mkdir(parents=True, exist_ok=True)
    cache = {}
    receipts = {}
    checkpoints = {}
    parent_packs = {}
    verified_files = {}
    average_roundoff = []
    initial_noise_hashes = {}
    old_fm = fm.parent/"rough_stress_20260918"
    old_base = base.parent/"rough_stress_20260918"
    protocol_hash = sha256(old_fm/"setup/reference_protocol.json")
    noise_hash = sha256(old_fm/"setup/noise_a100_seed0/manifest.json")
    fm_checkpoints = json.loads((fm/"setup/fm_checkpoints.json").read_text())
    poisson_equivalence = json.loads((fm/"setup/poisson_checkpoint_equivalence.json").read_text())
    assert poisson_equivalence["observed_equality"] is True
    fm_checkpoints["poisson"] = dict(path=str(old_fm/"setup/fm4poisson.pth"),
        sha256=poisson_equivalence["checkpoint_197_sha256"])

    def canonical_path(filename):
        return Path(str(filename).replace(
            "/home/zhangxf/share/zhangxfA100/large_storage/", "/large_storage/zhangxf/"))

    def verify_file(filename, digest):
        filename = canonical_path(filename)
        if str(filename) not in verified_files:
            assert sha256(filename) == digest, f"Source checksum changed: {filename}"
            verified_files[str(filename)] = digest
        assert verified_files[str(filename)] == digest

    for checkpoint in fm_checkpoints.values():
        verify_file(checkpoint["path"], checkpoint["sha256"])

    def source(pde, dist, case, kind):
        key = (pde, dist, case if kind == "observations" else None)
        if key not in cache:
            path = base/"measurements"/f"{pde}_{dist}_{case}.pt" if kind == "observations" else base/"inputs"/f"{pde}_{dist}.pt"
            receipt = json.loads(path.with_suffix(".json").read_text())
            receipts[key] = receipt
            digest = receipt["sha256"] if kind == "observations" else receipt["pack_sha256"]
            if sha256(path) != digest:
                raise ValueError(f"Input checksum changed: {path}")
            pack = torch.load(path, map_location="cpu", weights_only=False)
            assert pack["count"] == 100 and len(pack["sample_ids"]) == 100
            assert len(set(pack["sample_ids"])) == 100
            assert list(pack["source_indices"]) == list(range(100))
            if kind == "distribution":
                assert tensor_hash(pack["raw"]["full_tensor"]) == receipt["physical_tensor_sha256"]
                for budget in [400, 500]:
                    assert tensor_hash(pack["masks"][budget]) == receipt["masks"][str(budget)]
                    assert bool((pack["masks"][budget].flatten(1).sum(1) == budget).all())
                assert bool((pack["masks"][400] <= pack["masks"][500]).all())
                if args.verify_source_files:
                    verify_file(receipt["source"], receipt["source_sha256"])
            if kind == "observations":
                for name in ["mask", "clean", "noisy"]:
                    assert tensor_hash(pack[name]) == receipt[name+"_sha256"]
                mask, full = pack["mask"], pack["raw"]["full_tensor"]
                k = int(pack["metadata"]["window"])
                if k == 1:
                    assert torch.equal(pack["clean"], full*mask)
                else:
                    # Independent double-precision window reduction. Near-zero
                    # means require an absolute forward-error bound based on
                    # the terms being summed, rather than relative tolerance
                    # against a cancellation-dominated mean.
                    exact = full.double()
                    expected = torch.zeros_like(exact)
                    magnitude = torch.zeros_like(exact)
                    left, right = k//2, k-1-k//2
                    values = exact.unfold(2, k, 1).unfold(3, k, 1).mean((-1, -2))
                    abs_values = exact.abs().unfold(2, k, 1).unfold(3, k, 1).mean((-1, -2))
                    expected[..., left:128-right, left:128-right] = values
                    magnitude[..., left:128-right, left:128-right] = abs_values
                    selected = mask.bool().expand_as(full)
                    error = (pack["clean"].double()-expected).abs()[selected]
                    eps = torch.finfo(full.dtype).eps
                    gamma = ((k*k+1)*eps)/(1-(k*k+1)*eps)
                    bound = (gamma*magnitude[selected]).clamp_min(torch.finfo(full.dtype).tiny*eps)
                    assert bool((error <= bound).all()), f"Window average exceeds floating-point error bound: {key}"
                    assert not torch.count_nonzero(pack["clean"]*(1-mask))
                    average_roundoff.append(dict(distribution=dist, case=case,
                        max_absolute_error=float(error.max()),
                        max_bound_fraction=float((error/bound).max()),
                        bound="gamma_(k^2+1) * mean(abs(window)); float32 eps"))
                assert not bool((mask.bool() & pack["excluded"]).any())
                assert bool((mask.flatten(1).sum(1) == (640 if case == "columns" else 500)).all())
                assert not torch.count_nonzero(pack["noisy"]*(1-mask))
                if args.verify_source_files:
                    original = json.loads((old_base/"inputs"/f"poisson_{dist}.json").read_text())
                    assert receipt["source_pack_sha256"] == original["pack_sha256"]
                    verify_file(original["source"], original["source_sha256"])
                    if dist not in parent_packs:
                        parent_path = old_base/"inputs"/f"poisson_{dist}.pt"
                        verify_file(parent_path, original["pack_sha256"])
                        parent_packs[dist] = torch.load(parent_path, map_location="cpu", weights_only=False)
                    parent = parent_packs[dist]
                    assert pack["sample_ids"] == parent["sample_ids"][:100]
                    assert torch.equal(full, parent["raw"]["full_tensor"][:100])
                    if case == "random" or case.startswith("noise"):
                        assert torch.equal(mask, parent["masks"][500][:100].float())
                    if case.startswith("noise"):
                        level = int(case[5:])/100
                        for i in range(100):
                            selected = mask[i, 0].bool()
                            for c in range(2):
                                clean = pack["clean"][i, c][selected].double()
                                scale = clean.std(correction=0).clamp_min(1e-12)
                                eps = torch.randn((1, 1, 128, 128),
                                    generator=torch.Generator().manual_seed(2*i+c), dtype=torch.float32)[0, 0][selected].double()
                                noise = level*scale*eps
                                expected = clean+noise
                                error = (pack["noisy"][i, c][selected].double()-expected).abs()
                                bound = torch.finfo(torch.float32).eps*(4*clean.abs()+32*noise.abs())
                                assert bool((error <= bound).all()), f"Noise reading differs from specified draws: {key}/{i}/{c}"
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
        receipt = receipts[(pde, dist, case if kind == "observations" else None)]
        if kind == "observations":
            assert identity["case"] == case
            assert identity["input_sha256"] == receipt["sha256"]
        else:
            assert identity["num_obs"] == int(case[3:])
            assert identity["input_receipt"] == receipt
        assert identity["batch_size"] == 16 and identity["tf32"] is False
        if method == "fm4pde":
            accepted = {fm_checkpoints[pde]["sha256"]}
            if pde == "poisson":
                accepted.add(poisson_equivalence["checkpoint_216_sha256"])
            assert identity["checkpoint"]["sha256"] in accepted
            assert identity["fm_protocol_sha256"] == protocol_hash
            actual_noise_hash = (identity["noise_manifest_sha256"] if kind == "observations"
                                 else identity["noise_replay"]["manifest_sha256"])
            assert actual_noise_hash == noise_hash
        else:
            if pde not in checkpoints:
                cpfile = base/"measurements/poisson_checkpoints.json" if pde == "poisson" else base/f"{pde}_checkpoints.json"
                checkpoints[pde] = json.loads(cpfile.read_text())
            checkpoint = checkpoints[pde][f"{method}/{task}"]
            actual_checkpoint = dict(identity["checkpoint"], path=str(canonical_path(identity["checkpoint"]["path"])))
            expected_checkpoint = dict(checkpoint, path=str(canonical_path(checkpoint["path"])))
            assert actual_checkpoint == expected_checkpoint
            assert not identity["backend"].get("fallback_used")
            verify_file(checkpoint["path"], checkpoint["sha256"])
            verify_file(checkpoint["source_summary"], checkpoint["source_summary_sha256"])
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
            assert pred.shape == (len(indices), 2 if method == "fm4pde" or task == "both" else 1, 128, 128)
            if method == "fm4pde":
                runtime = record["runtime"]
                assert runtime["steps"] == runtime["nfe"] == 100
                index_key = tuple(indices)
                digest = runtime["initial_noise_sha256"]
                assert initial_noise_hashes.setdefault(index_key, digest) == digest
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
        independent_prediction_metrics=True, frozen_input_and_measurement_hashes_verified=True,
        average_readings_double_precision_checks=average_roundoff,
        verified_checkpoint_and_source_files=verified_files,
        original_mat_files_verified=args.verify_source_files,
        poisson_parent_samples_and_gaussian_noise_verified=args.verify_source_files,
        checkpoint_protocol_and_noise_identities_verified=True,
        initial_noise_hashes_shared_across_cells=True))
    print(f"AUDIT {len(completed)}/{len(cells)} cells, {len(rows)} metric rows", flush=True)
    if incomplete and not args.allow_incomplete:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
