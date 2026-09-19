"""Frozen-checkpoint evaluations of layouts, noise, and regional averages."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket
import sys
import time

from experiments.rough_stress.run import (
    ROOT, TASKS, METHODS, baseline_imports, evaluate_errors, git_commit,
    sha256, slice_tree, tensor_hash, write_json,
)
from experiments.observation_stress.measurements import CASES


def baseline_batch(registry, raw, task, mask, readings, method):
    """Use the existing input schema with only the frozen sensor readings."""
    import torch
    from baselines.common.sensors import extract_observations
    from baselines.common.voronoi import voronoi_fill
    from baselines.run import _make_inference_batch

    n = int(mask[0].sum())
    batch = registry.make_task(raw, "poisson", TASKS[task], num_sensors=n,
        sensor_mode="random", sensor_budget_mode="total", seed=1,
        experiment_mode="debug", build_voronoi_grid=False)
    channels = [0, 1] if task == "both" else [0] if task == "forward" else [1]
    values = readings[:, channels].clone()
    active_mask = mask.expand_as(values).clone()
    batch.input_fields = values
    batch.mask = active_mask
    batch.obs_values, batch.obs_coords = extract_observations(values, active_mask)
    batch.metadata.update(masked_grid=values, deferred_dynamic_sensors=False,
        sensor_mode="frozen_external", effective_sensor_mode="frozen_external",
        requested_sensor_mode="frozen_external", mask_id=tensor_hash(active_mask),
        mask_ids=[tensor_hash(m) for m in active_mask], num_sensors=n,
        num_observations_total=n, num_sensors_per_time=n, noise_level=0.,
        input_shape=tuple(values.shape), observation_payload="physical_frozen_measurements")
    batch.metadata.pop("voronoi_grid", None)
    if method in {"recfno", "voronoicnn"}:
        batch.metadata["voronoi_grid"] = voronoi_fill(values, active_mask)
    assert not torch.count_nonzero(values*(1-active_mask))
    view = _make_inference_batch(batch)
    assert not torch.count_nonzero(view.full_tensor)
    assert not torch.count_nonzero(view.target_fields)
    return view


def main():
    import torch
    from experiments.aligned_sampling.run_inference import (
        configure_runtime, effective_config, observation_batch, infer,
    )
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", required=True)
    p.add_argument("--input-root", required=True)
    p.add_argument("--distribution", choices=["id", "rough"], required=True)
    p.add_argument("--case", choices=CASES, required=True)
    p.add_argument("--method", choices=[*METHODS, "fm4pde"], required=True)
    p.add_argument("--task", choices=TASKS, required=True)
    p.add_argument("--baseline-root")
    p.add_argument("--baseline-output-root")
    p.add_argument("--fm-protocol")
    p.add_argument("--fm-checkpoint")
    p.add_argument("--noise-root")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--stop", type=int, default=100)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    configure_runtime(args.device, False, 2)
    path = Path(args.input_root) / f"poisson_{args.distribution}_{args.case}.pt"
    receipt = json.loads(path.with_suffix(".json").read_text())
    if sha256(path) != receipt["sha256"]:
        raise ValueError("Measurement payload checksum mismatch")
    pack = torch.load(path, map_location="cpu", weights_only=False)
    count = pack["count"]
    stop = min(args.stop, count)
    if not 0 < stop <= count:
        raise ValueError("Invalid sample count")
    cell = Path(args.output_root) / "results/poisson" / args.method / args.task / args.distribution / args.case
    cell.mkdir(parents=True, exist_ok=True)
    identity = dict(method=args.method, pde="poisson", task=args.task, distribution=args.distribution,
        case=args.case, count=count, input_sha256=receipt["sha256"], input_metadata=pack["metadata"],
        code_commit=git_commit(ROOT), batch_size=args.batch_size, host=socket.gethostname(),
        torch=torch.__version__, device=args.device, tf32=False,
        baseline_average_interpretation="window reading at anchor; pretrained point-observation interface",
        fm_average_interpretation="exact physical box-average operator in observation likelihood")
    if args.method != "fm4pde":
        registry = baseline_imports(args.baseline_root)
        from baselines.run import load_baseline_checkpoint, _to_device_batch_for_eval
        checkpoints = json.loads((Path(args.input_root)/"poisson_checkpoints.json").read_text())
        checkpoint = checkpoints[f"{args.method}/{args.task}"]
        filename = Path(checkpoint["path"])
        if not filename.exists():
            filename = Path(args.baseline_output_root)/"runs"/str(filename).split("/runs/", 1)[1]
        if sha256(filename) != checkpoint["sha256"]:
            raise ValueError("Checkpoint hash mismatch")
        model = load_baseline_checkpoint(str(filename), map_location=args.device,
            expected=dict(baseline=args.method, pde="poisson", task=TASKS[args.task], seed=1,
                num_sensors=500, source_train_run_id=checkpoint["source_run_id"]),
            require_provenance=True).eval()
        if model.backend_metadata().get("fallback_used"):
            raise ValueError("Fallback models are not permitted")
        identity.update(checkpoint=checkpoint, baseline_commit=git_commit(args.baseline_root),
                        backend=model.backend_metadata())
    else:
        from sampling.model_io import load_fm4pde_checkpoint_bundle
        from sampling.losses import ObservationTargets
        from sampling.observation_operators import BoxAverageObservation
        from experiments.rough_stress.noise import NoiseBank
        protocol = json.loads(Path(args.fm_protocol).read_text())
        setting = "sparse_joint" if args.task == "both" else TASKS[args.task]
        reference = next(c for c in protocol["cells"] if c["cohort"] == "supervised" and
            c["pde"] == "poisson" and c["distribution"] == args.distribution and c["setting"] == setting)
        reference = dict(reference, config=dict(reference["config"]))
        reference["config"].update(num_obs=pack["metadata"]["readings_per_field"], shared_mask=True)
        checkpoint = Path(args.fm_checkpoint)
        identity.update(checkpoint=dict(path=str(checkpoint), sha256=sha256(checkpoint)),
                        fm_config=reference["config"], fm_protocol_sha256=sha256(args.fm_protocol))
        bundle = load_fm4pde_checkpoint_bundle(str(checkpoint), "poisson", args.device,
            model_profile=reference["config"].get("model_profile"))
        bank = NoiseBank(args.noise_root)
        identity["noise_manifest_sha256"] = bank.manifest_sha256
        operator = BoxAverageObservation(pack["metadata"]["window"])
    identity_path = cell/"identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("Run identity changed")
    write_json(identity_path, identity)
    for lo in range(0, stop, args.batch_size):
        hi = min(lo+args.batch_size, stop)
        ids = list(range(lo, hi))
        prediction_path = cell/f"batch_{lo:04d}_{hi:04d}.pt"
        record_path = prediction_path.with_suffix(".json")
        if prediction_path.exists() and record_path.exists():
            if sha256(prediction_path) != json.loads(record_path.read_text())["prediction_sha256"]:
                raise ValueError("Corrupt saved prediction")
            continue
        mask = pack["mask"][lo:hi]
        active = torch.ones((hi-lo, 2, 1, 1))
        if args.task == "forward":
            active[:, 1] = 0
        if args.task == "inverse":
            active[:, 0] = 0
        noisy = pack["noisy"][lo:hi]*active
        clean = pack["clean"][lo:hi]*active
        if args.method != "fm4pde":
            raw = slice_tree(pack["raw"], lo, hi, count)
            view = baseline_batch(registry, raw, args.task, mask, noisy, args.method)
            view = _to_device_batch_for_eval(view, args.device)
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            begin = time.perf_counter()
            with torch.inference_mode():
                pred = model.predict_physical(view)
            torch.cuda.synchronize()
            runtime = dict(seconds=time.perf_counter()-begin,
                           peak_allocated_bytes=torch.cuda.max_memory_allocated())
            pred = pred.detach().cpu()
        else:
            cfg = effective_config(reference, checkpoint, args.device, ids, bank.manifest["count"], cell)
            data = dict(coef_ground_truth=pack["noisy"][:, :1], sol_ground_truth=pack["noisy"][:, 1:2],
                sample_ids=pack["sample_ids"], source_indices=pack["source_indices"],
                pde_params=pack["raw"].get("pde_params", {}), masks=dict(
                    coef=pack["mask"]*(args.task in {"forward", "both"}),
                    sol=pack["mask"]*(args.task in {"inverse", "both"}), metadata=pack["metadata"]))
            gt, masks, hashes = observation_batch(data, cfg, ids, args.device)
            targets = ObservationTargets(clean[:, :1].to(args.device), clean[:, 1:2].to(args.device),
                noisy[:, :1].to(args.device), noisy[:, 1:2].to(args.device), operator)
            operator.validate_mask(masks.coef)
            operator.validate_mask(masks.sol)
            with bank.replay(ids):
                pred, runtime = infer(cfg, bundle, gt, masks, ids, observations=targets)
        if not torch.isfinite(pred).all():
            raise ValueError("Nonfinite predictions")
        truth = pack["raw"]["full_tensor"][lo:hi]
        rows = evaluate_errors(pred, truth, args.task, args.method)
        for j, row in enumerate(rows):
            row.update(index=lo+j, sample_id=pack["sample_ids"][lo+j])
        if args.case in {"hole10", "hole25", "strip25"}:
            for region, spatial in [("excluded", pack["excluded"][0, 0]),
                                    ("outside_excluded", ~pack["excluded"][0, 0])]:
                fields = [("a", 0, 0), ("u", 1, 1)] if args.method == "fm4pde" or args.task == "both" else (
                    [("u", 0, 1)] if args.task == "forward" else [("a", 0, 0)])
                for field, pc, tc in fields:
                    for j, row in enumerate(rows):
                        a = pred[j, pc][spatial].double()
                        b = truth[j, tc][spatial].double()
                        row[f"rel_l2_{field}_{region}"] = float((a-b).norm()/b.norm().clamp_min(1e-12))
        torch.save(dict(prediction=pred, indices=ids, sample_ids=pack["sample_ids"][lo:hi]), prediction_path)
        write_json(record_path, dict(start=lo, stop=hi, rows=rows, runtime=runtime,
            prediction_sha256=sha256(prediction_path), observation_mask_sha256=tensor_hash(mask),
            active_noisy_readings_sha256=tensor_hash(noisy), active_clean_readings_sha256=tensor_hash(clean)))
        print(f"PROGRESS {args.method}/{args.task}/{args.distribution}/{args.case} {hi}/{stop} "
              f"peak={runtime['peak_allocated_bytes']/2**30:.2f}GiB", flush=True)


if __name__ == "__main__":
    main()
