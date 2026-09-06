#!/usr/bin/env python3
"""Paired comparison of late hard, late ramp, and always-on PDE guidance.

Uses unselected ground-truth snapshots from completed real-data runs to avoid
repeated multi-GB reads over SSHFS. Historical predictions are never reused.
All predictions, masks, and initial noise are generated anew. Main YAMLs are
read-only. Sample IDs are chosen before looking at their reconstruction errors.
"""
from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import json
import math
import random
import statistics
import sys
import time
from collections import defaultdict
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

SCHEDULES = {"late_hard": (0.8, 0.0), "late_ramp": (0.8, 0.1), "always_on": (0.0, 0.0)}
PDES = ("poisson", "helmholtz", "darcy", "nsnonbounded", "burger")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def score(row, task, pde):
    a, u = row.get("rel_l2_a"), row.get("rel_l2_u")
    if a is None or u is None or not all(math.isfinite(x) for x in (a, u)):
        return math.inf
    return u if task == "forward" or pde == "burger" else a if task == "inverse" else (a + u) / 2


def slice_params(value, index, size):
    import torch
    if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == size:
        return value[index:index + 1].clone()
    if isinstance(value, dict):
        return {k: slice_params(v, index, size) for k, v in value.items()}
    return copy.deepcopy(value)


def prepare_samples(args, pdes, ids):
    """Trace each reused ground truth to its original real test-data artifact."""
    import torch
    from sampling.data import PDEGroundTruth
    from data.specs import get_pde_spec
    source_root = Path(args.source_root)
    index = {}
    with (source_root / "metrics_per_sample_all.csv").open() as handle:
        for row in csv.DictReader(handle):
            if row["pde"] in pdes and row["task"] == "both" and row["sensor_mode"] == "random":
                sample_id = int(row["sample_id"])
                if sample_id in ids:
                    index[(row["pde"], sample_id)] = row
    manifest = []
    for pde in pdes:
        cache = args.root / "cache" / f"{pde}_ground_truth.pt"
        if cache.exists():
            stored = torch.load(cache, map_location="cpu", weights_only=False)
            if stored["sample_ids"] != ids:
                raise ValueError(f"Cache sample mismatch: {cache}")
            manifest.extend(stored["sources"])
            continue
        truths, sources = {}, []
        loaded_path, payload = None, None
        for sample_id in ids:
            row = index[(pde, sample_id)]
            old = Path(row["run_dir"])
            relative = Path(*old.parts[old.parts.index(source_root.name) + 1:])
            path = source_root / relative / "result.pt"
            if path != loaded_path:
                payload = torch.load(path, map_location="cpu", weights_only=False)
                loaded_path = path
            meta = payload["ground_truth_metadata"]
            if meta.get("synthetic") is not False or payload["metrics"].get("synthetic_data") is not False:
                raise ValueError(f"Not verified real data: {path}")
            if "_id.mat" not in Path(meta["data_path"]).name:
                raise ValueError(f"Not the ID test set: {path}")
            local_index = int(row["sample_index"])
            if int(meta["sample_ids"][local_index]) != sample_id:
                raise ValueError(f"Source row mismatch: {path}")
            coef = payload["coef_ground_truth"][local_index:local_index + 1].clone()
            sol = payload["sol_ground_truth"][local_index:local_index + 1].clone()
            count = len(payload["coef_ground_truth"])
            params = slice_params(payload["pde_params"], local_index, count)
            spec = get_pde_spec(pde)
            sample_meta = copy.deepcopy(meta)
            sample_meta.update(offset=sample_id, sample_offsets=[sample_id], sample_ids=[str(sample_id)], batch_size=1)
            sample_meta["comparison_source_artifact"] = str(path)
            truths[sample_id] = PDEGroundTruth(
                pde, coef, sol, coef if pde == "burger" else torch.cat([coef, sol], dim=1), params,
                list(spec.coef_channel_names), list(spec.sol_channel_names), sample_meta,
            )
            digest = hashlib.sha256(coef.numpy().tobytes() + sol.numpy().tobytes()).hexdigest()
            sources.append(dict(pde=pde, sample_id=sample_id, source_artifact=str(path),
                                data_path=meta["data_path"], tensor_sha256=digest))
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(sample_ids=ids, truths=truths, sources=sources), cache)
        manifest.extend(sources)
        print(f"CACHED {pde}: {len(ids)} real ID test samples", flush=True)
    write_json(args.root / "sample_manifest.json", manifest)


def combine_truths(truths, ids, device):
    import torch
    from sampling.data import PDEGroundTruth
    items = [truths[i] for i in ids]
    def combine(values):
        first = values[0]
        if isinstance(first, torch.Tensor):
            return (torch.cat(values, dim=0) if first.ndim and first.shape[0] == 1 else first.clone()).to(device)
        if isinstance(first, dict):
            return {k: combine([v[k] for v in values]) for k in first}
        if any(v != first for v in values[1:]):
            raise ValueError("Cannot batch differing non-tensor PDE parameters")
        return copy.deepcopy(first)
    coef, sol = torch.cat([x.coef for x in items]).to(device), torch.cat([x.sol for x in items]).to(device)
    meta = copy.deepcopy(items[0].metadata)
    meta.update(sample_ids=[str(i) for i in ids], sample_offsets=ids, batch_size=len(ids), offset=ids[0])
    return PDEGroundTruth(items[0].pde, coef, sol, coef if items[0].pde == "burger" else torch.cat([coef, sol], 1),
                          combine([x.pde_params for x in items]), items[0].channel_names_coef,
                          items[0].channel_names_sol, meta)


def candidates(base):
    levels = sorted({0.1, 1.0, 10.0, float(base.zeta_pde)}, reverse=True)
    return [(name, zeta) for name in ("always_on", "late_hard", "late_ramp") for zeta in levels]


def run_candidate(args, base, bundle, truths, stage, schedule, zeta, ids):
    import torch
    from sampling.config import load_config
    from sampling.runner import run_single_ablation
    all_rows = []
    for begin in range(0, len(ids), args.batch_size):
        batch_ids = ids[begin:begin + args.batch_size]
        label = f"{stage}/{base.pde}/{base.task}/{schedule}_z{zeta:g}/batch{begin}"
        directory = args.root / "runs" / label
        directory.mkdir(parents=True, exist_ok=True)
        receipt = directory / "receipt.json"
        if receipt.exists():
            stored = json.loads(receipt.read_text())
            if stored["sample_ids"] != batch_ids or stored["protocol_hash"] != args.protocol_hash:
                raise ValueError(f"Resume mismatch: {receipt}")
            all_rows.extend(stored["rows"])
            continue
        start, ramp = SCHEDULES[schedule]
        config_path = ROOT / f"configs/main/{base.task}/{base.pde}.yaml"
        cfg = load_config(config_path, overrides=dict(
            output_dir=str(directory), device=args.device, batch_size=len(batch_ids), offset=batch_ids[0],
            sample_seed=args.seed + batch_ids[0], mask_seed=args.seed + batch_ids[0],
            num_steps=100, time_grid="uniform", sampler_phase="stochastic", sensor_mode="random",
            noise_level=0.0, num_obs=500, save_plots=False, save_intermediate=False,
            zeta_pde=zeta, pde_guidance_start_ratio=start, pde_guidance_ramp_ratio=ramp,
            clip_mode="global_norm", clip_threshold=args.clip_threshold,
            pde_guidance_reduction=args.pde_loss,
            ablation_group="pde_guidance_schedule_comparison",
        ))
        ground_truth = combine_truths(truths, batch_ids, args.device)
        print(f"RUN {label} samples={batch_ids}", flush=True)
        began = time.monotonic()
        result = None
        error = ""
        rows = []
        with (directory / "run.log").open("w") as log, redirect_stdout(log), redirect_stderr(log):
            try:
                result = run_single_ablation(cfg, checkpoint_bundle=bundle, ground_truth=ground_truth)
                with (Path(result["run_dir"]) / "metrics_per_sample.csv").open() as handle:
                    metrics = list(csv.DictReader(handle))
                if [int(r["sample_id"]) for r in metrics] != batch_ids:
                    raise ValueError("Output sample IDs do not match input order")
            except torch.OutOfMemoryError:
                raise
            except Exception as exc:
                import traceback
                traceback.print_exc()
                error = f"{type(exc).__name__}: {exc}"
                metrics = [{"sample_id": str(i)} for i in batch_ids]
        for metric in metrics:
            row = dict(stage=stage, pde=base.pde, task=base.task, schedule=schedule, zeta_pde=zeta,
                       sample_id=int(metric["sample_id"]), status=result["status"] if result else "error", error=error,
                       run_dir=result["run_dir"] if result else str(directory))
            for key in ("rel_l2_a", "rel_l2_u", "obs_rel_l2_a", "obs_rel_l2_u", "pde_residual_norm"):
                value = float(metric[key]) if metric.get(key) not in (None, "") else math.inf
                row[key] = value if math.isfinite(value) else None
            primary = score(row, base.task, base.pde)
            row["primary_error"] = primary if math.isfinite(primary) else None
            rows.append(row)
        write_json(receipt, dict(protocol_hash=args.protocol_hash, sample_ids=batch_ids, rows=rows,
                                 seconds=time.monotonic() - began))
        all_rows.extend(rows)
        print(f"DONE {label} seconds={time.monotonic() - began:.1f} scores={[r['primary_error'] for r in rows]}", flush=True)
        del ground_truth
    return all_rows


def summarize(root):
    rows = []
    for receipt in sorted((root / "runs").glob("*/*/*/*/*/receipt.json")):
        rows.extend(json.loads(receipt.read_text())["rows"])
    groups = defaultdict(list)
    for row in rows:
        groups[(row["stage"], row["pde"], row["task"], row["schedule"], row["zeta_pde"])].append(row)
    summaries = []
    for (stage, pde, task, schedule, zeta), group in sorted(groups.items()):
        if len({r["sample_id"] for r in group}) != len(group):
            raise ValueError("Duplicate sample in candidate")
        out = dict(stage=stage, pde=pde, task=task, schedule=schedule, zeta_pde=zeta, n=len(group),
                   failed=sum(r["primary_error"] is None or r["status"] != "ok" for r in group))
        for field in ("primary_error", "rel_l2_a", "rel_l2_u", "pde_residual_norm"):
            values = [r[field] for r in group if r[field] is not None]
            out[field + "_mean"] = statistics.mean(values) if len(values) == len(group) else None
            if field == "primary_error":
                out[field + "_median"] = statistics.median(values) if len(values) == len(group) else None
        summaries.append(out)
    write_csv(root / "per_sample.csv", rows)
    write_csv(root / "summary.csv", summaries)
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-root", default="outputs/main/MAIN1000_100_TEST_id")
    parser.add_argument("--pdes", nargs="+", default=list(PDES), choices=PDES)
    parser.add_argument("--tasks", nargs="+", default=["both", "forward", "inverse"])
    parser.add_argument("--only-cells", nargs="+",
                        help="Execute only these PDE/task cells without changing the shared full experiment protocol; writes partial_complete.json")
    parser.add_argument("--tune-samples", type=int, default=4)
    parser.add_argument("--holdout-samples", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--clip-threshold", type=float, default=50.0)
    parser.add_argument("--pde-loss", choices=["mse", "rms"], default="mse",
                        help="PDE guidance objective only; evaluation always uses the original MSE/residual metrics")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    args.root = args.root.resolve()
    args.root.mkdir(parents=True, exist_ok=True)
    if args.summarize_only:
        summarize(args.root)
        return
    if min(args.tune_samples, args.holdout_samples, args.batch_size) < 1:
        parser.error("Sample counts and batch size must be positive")
    if not math.isfinite(args.clip_threshold) or args.clip_threshold <= 0:
        parser.error("clip-threshold must be positive and finite")
    from sampling.config import load_config
    cells = [(pde, task) for pde in args.pdes for task in args.tasks
             if (ROOT / f"configs/main/{task}/{pde}.yaml").exists()]
    if args.only_cells and set(args.only_cells) - {f"{p}/{t}" for p, t in cells}:
        parser.error("only-cells must name existing cells within --pdes/--tasks")
    ids = random.Random(args.seed).sample(range(1000), args.tune_samples + args.holdout_samples)
    tune, holdout = ids[:args.tune_samples], ids[args.tune_samples:]
    protocol = dict(seed=args.seed, tune_sample_ids=tune, holdout_sample_ids=holdout, cells=cells,
                    batch_size=args.batch_size, steps=100, time_grid="uniform", sampler="stochastic",
                    clip_mode="global_norm", clip_threshold=args.clip_threshold,
                    sensor_mode="random", num_obs=500, test_type="id", schedules=SCHEDULES,
                    configs={f"{p}/{t}": load_config(ROOT / f"configs/main/{t}/{p}.yaml").asdict() for p, t in cells},
                    selection="minimum tune mean primary error among candidates with no failed samples, separately per schedule",
                    primary="forward: rel_l2_u; inverse: rel_l2_a; both: (rel_l2_a+rel_l2_u)/2; burger: rel_l2_u")
    # Keep existing MSE experiment hashes resumable.
    if args.pde_loss != "mse":
        protocol["pde_guidance_reduction"] = args.pde_loss
    args.protocol_hash = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    protocol_path = args.root / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != json.loads(json.dumps(protocol)):
        raise ValueError("Protocol differs from existing experiment; use a new root")
    write_json(protocol_path, protocol)
    if args.only_cells:
        write_json(args.root / "worker_scope.json", dict(cells=args.only_cells, protocol_hash=args.protocol_hash))
    prepare_samples(args, args.pdes, ids)
    if args.prepare_only:
        return
    import torch
    from sampling.model_io import load_fm4pde_checkpoint_bundle
    torch.set_num_threads(2)
    selected = {}
    for pde in args.pdes:
        bases = [load_config(ROOT / f"configs/main/{task}/{pde}.yaml") for p, task in cells
                 if p == pde and (not args.only_cells or f"{p}/{task}" in args.only_cells)]
        if not bases:
            continue
        print(f"LOADING MODEL {pde}", flush=True)
        bundle = load_fm4pde_checkpoint_bundle(bases[0].checkpoint_path, pde, device=args.device,
                                               model_profile=bases[0].model_profile)
        truths = torch.load(args.root / "cache" / f"{pde}_ground_truth.pt", map_location="cpu", weights_only=False)["truths"]
        for base in bases:
            tune_rows = []
            for schedule, zeta in candidates(base):
                tune_rows.extend(run_candidate(args, base, bundle, truths, "tune", schedule, zeta, tune))
                summarize(args.root)
            winners = {}
            for schedule in SCHEDULES:
                options = []
                for name, zeta in candidates(base):
                    if name != schedule:
                        continue
                    group = [r for r in tune_rows if r["schedule"] == name and r["zeta_pde"] == zeta]
                    if len(group) == len(tune) and all(r["primary_error"] is not None and r["status"] == "ok" for r in group):
                        options.append((statistics.mean(r["primary_error"] for r in group), zeta))
                winners[schedule] = min(options)[1] if options else None
            selected[f"{pde}/{base.task}"] = winners
            write_json(args.root / "selected.json", selected)
            validation = {(schedule, zeta) for schedule, zeta in winners.items() if zeta is not None}
            validation.add(("late_hard", float(base.zeta_pde)))
            for schedule, zeta in sorted(validation):
                run_candidate(args, base, bundle, truths, "holdout", schedule, zeta, holdout)
                summarize(args.root)
        del truths, bundle
        gc.collect()
        torch.cuda.empty_cache()
    marker = "partial_complete.json" if args.only_cells else "complete.json"
    write_json(args.root / marker, dict(protocol_hash=args.protocol_hash, status="complete", executed_cells=args.only_cells or cells))
    print(f"COMPLETE: {args.root / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
