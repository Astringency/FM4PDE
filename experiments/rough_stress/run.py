"""Paired frozen-checkpoint distribution and observation-budget evaluation.

No fitting or tuning is performed. The baseline adapter defines physical
fields and sensor layouts; FM consumes exactly those same saved observations.
All errors are recomputed per sample in physical units with float64 norms.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import csv
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
TASKS = {"forward": "sparse_forward", "inverse": "sparse_inverse", "both": "sparse_solution"}
METHODS = ("recfno", "senseiver", "voronoicnn")
DISTRIBUTIONS = ("id", "smooth", "rough", "rough2", "rough3")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tensor_hash(value):
    x = value.detach().cpu().contiguous()
    return hashlib.sha256(str(x.dtype).encode() + str(tuple(x.shape)).encode() + x.numpy().tobytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def git_commit(root):
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def source_name(pde, distribution):
    n = 1000 if distribution in {"rough2", "rough3"} else 10000
    if pde == "nsnonbounded":
        return f"{pde}/{pde}_test_{n}-128-128-10_{distribution}.mat"
    if pde == "burger":
        return f"burgers/burger_test_{n}-128-128_{distribution}.mat"
    return f"{pde}/{pde}_test_{n}-128-128_{distribution}.mat"


def baseline_imports(root):
    sys.path.insert(0, str(Path(root).resolve()))
    from baselines.common.data_adapter import build_default_registry
    return build_default_registry()


def slice_tree(value, start, stop, count):
    import torch
    if torch.is_tensor(value) and value.ndim and value.shape[0] == count:
        return value[start:stop]
    if isinstance(value, dict):
        return {k: slice_tree(v, start, stop, count) for k, v in value.items()}
    if isinstance(value, list) and len(value) == count:
        return value[start:stop]
    return value


def prepare(args):
    import torch
    from baselines.common.sensors import build_observation_tensors
    registry = baseline_imports(args.baseline_root)
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    # Checkpoint selection is frozen before observing any stress-test score.
    matrix = [json.loads(line) for line in Path(args.matrix).read_text().splitlines() if line.strip()]
    checkpoints = {}
    for row in matrix:
        if row["pde"] != args.pde or row["baseline"] not in METHODS or row["task"] not in TASKS.values():
            continue
        if int(row["seed"]) != 1:
            continue
        source = Path(args.baseline_output_root) / "runs" / row["output_dir"].split("/runs/", 1)[1]
        summary_path = source / "summary.json"
        summary = json.loads(summary_path.read_text())
        if summary.get("status") != "success":
            raise ValueError(f"Incomplete source run: {source}")
        checkpoint = source / Path(summary["checkpoint_path"]).name
        task = next(k for k, v in TASKS.items() if v == row["task"])
        key = f"{row['baseline']}/{task}"
        if key in checkpoints:
            raise ValueError(f"Ambiguous source checkpoint: {key}")
        checkpoints[key] = dict(path=str(checkpoint), sha256=sha256(checkpoint),
            source_summary=str(summary_path), source_summary_sha256=sha256(summary_path),
            source_run_id=row["run_id"], source_task=row["task"], source_seed=1,
            source_num_sensors=500, source_sensor_seed=1)
    if len(checkpoints) != 9:
        raise ValueError(f"Expected 9 frozen task checkpoints, found {len(checkpoints)}")
    path = root / f"{args.pde}_checkpoints.json"
    if path.exists() and json.loads(path.read_text()) != checkpoints:
        raise ValueError("Refusing to change frozen checkpoint selection")
    write_json(path, checkpoints)
    for distribution in args.distributions:
        destination = root / "inputs" / f"{args.pde}_{distribution}.pt"
        receipt = destination.with_suffix(".json")
        if destination.exists() and receipt.exists():
            if sha256(destination) != json.loads(receipt.read_text())["pack_sha256"]:
                raise ValueError(f"Existing pack is corrupt: {destination}")
            continue
        source = Path(args.data_root) / source_name(args.pde, distribution)
        initial_stat = source.stat()
        raw = registry.load_raw(args.pde, args.data_root, split="test", max_samples=args.count,
            strict_size=True, data_files={"test": [source_name(args.pde, distribution)]}, load_full_trajectory=False)
        raw = registry.to_canonical(raw, args.pde)
        full = raw["full_tensor"]
        if full.shape != (args.count, 2, 128, 128) or not torch.isfinite(full).all():
            raise ValueError(f"Invalid paired physical fields: {full.shape}")
        ids = raw["global_sample_ids"]
        if len(ids) != args.count or len(set(ids)) != args.count:
            raise ValueError("Missing or duplicate physical sample IDs")
        masks = {}
        for n in (400, 500):
            obs = build_observation_tensors(full[:, :1], n, "random_per_sample", seed=1,
                sample_ids=ids, split="test", sensor_budget_mode="total", build_voronoi_grid=False)
            masks[n] = obs["mask"].to(torch.uint8)
        if not torch.all(masks[400] <= masks[500]):
            raise ValueError("400 observations must be a subset of the corresponding 500")
        source_digest = sha256(source)
        final_stat = source.stat()
        if (initial_stat.st_size, initial_stat.st_mtime_ns) != (final_stat.st_size, final_stat.st_mtime_ns):
            raise ValueError(f"Source changed while being read: {source}")
        pack = dict(pde=args.pde, distribution=distribution, count=args.count, raw=raw,
                    masks=masks, sample_ids=ids, source_indices=torch.arange(args.count))
        destination.parent.mkdir(parents=True, exist_ok=True)
        tmp = destination.with_suffix(".tmp")
        torch.save(pack, tmp)
        tmp.replace(destination)
        write_json(receipt, dict(source=str(source), source_sha256=source_digest,
            source_size=final_stat.st_size, source_mtime_ns=final_stat.st_mtime_ns,
            pack_sha256=sha256(destination), count=args.count, first_index=0, last_index=args.count-1,
            physical_tensor_sha256=tensor_hash(full), masks={str(k): tensor_hash(v) for k,v in masks.items()},
            baseline_commit=git_commit(args.baseline_root), code_commit=git_commit(ROOT)))
        print(f"PREPARED {args.pde}/{distribution}: {args.count} inputs", flush=True)


def load_pack(args):
    import torch
    path = Path(args.input_root or args.output_root) / "inputs" / f"{args.pde}_{args.distribution}.pt"
    receipt = json.loads(path.with_suffix(".json").read_text())
    if sha256(path) != receipt["pack_sha256"]:
        raise ValueError(f"Input pack hash mismatch: {path}")
    return torch.load(path, map_location="cpu", weights_only=False), receipt


def evaluate_errors(pred, truth, task, method):
    import torch
    rows = []
    channels = [("a", 0, 0), ("u", 1, 1)] if task == "both" or method == "fm4pde" else (
        [("u", 0, 1)] if task == "forward" else [("a", 0, 0)])
    for index in range(len(pred)):
        row = {}
        for name, pc, tc in channels:
            p, t = pred[index, pc].double(), truth[index, tc].double()
            row[f"rel_l2_{name}"] = float(torch.linalg.vector_norm(p-t) / torch.linalg.vector_norm(t).clamp_min(1e-12))
            row[f"rmse_{name}"] = float(torch.mean((p-t).square()).sqrt())
        rows.append(row)
    return rows


def run(args):
    import torch
    from experiments.aligned_sampling.run_inference import configure_runtime
    configure_runtime(args.device, False, args.threads)
    pack, input_receipt = load_pack(args)
    root = Path(args.output_root)
    count = pack["count"]
    start, stop = args.start, min(args.stop or count, count)
    if not 0 <= start < stop <= count:
        raise ValueError("Invalid sample range")
    cell = root / "results" / args.pde / args.method / args.task / args.distribution / f"obs{args.num_obs}"
    cell.mkdir(parents=True, exist_ok=True)
    identity = dict(pde=args.pde, method=args.method, task=args.task, distribution=args.distribution,
        num_obs=args.num_obs, count=count, code_commit=git_commit(ROOT), input_receipt=input_receipt,
        sensor_policy="baseline_v3_seed1_nested_400_in_500_shared_across_fields",
        missing_policy="native_variable_observations_zero_grid_missing_mask",
        batch_size=args.batch_size, host=socket.gethostname(), torch=torch.__version__,
        device=args.device, tf32=False)
    if args.method != "fm4pde":
        registry = baseline_imports(args.baseline_root)
        from baselines.run import load_baseline_checkpoint, _make_inference_batch, _to_device_batch_for_eval
        ckpts = json.loads((Path(args.input_root or args.output_root) / f"{args.pde}_checkpoints.json").read_text())
        ckpt = ckpts[f"{args.method}/{args.task}"]
        if sha256(ckpt["path"]) != ckpt["sha256"]:
            raise ValueError("Checkpoint hash mismatch")
        model = load_baseline_checkpoint(ckpt["path"], map_location=args.device,
            expected={"baseline":args.method,"pde":args.pde,"task":TASKS[args.task],
                      "seed":1,"num_sensors":500,"source_train_run_id":ckpt["source_run_id"]},
            require_provenance=True).eval()
        identity.update(checkpoint=ckpt, baseline_commit=git_commit(args.baseline_root), backend=model.backend_metadata())
        if model.backend_metadata().get("fallback_used"):
            raise ValueError("A fallback model is not allowed")
    else:
        from experiments.aligned_sampling.run_inference import effective_config, observation_batch, infer
        from sampling.model_io import load_fm4pde_checkpoint_bundle
        protocol = json.loads(Path(args.fm_protocol).read_text())
        dist = args.distribution if args.distribution in {"id","smooth","rough"} else "rough"
        setting = "sparse_joint" if args.task == "both" else TASKS[args.task]
        reference = next(c for c in protocol["cells"] if c["cohort"]=="supervised" and c["pde"]==args.pde and c["distribution"]==dist and c["setting"]==setting)
        reference = dict(reference, config=dict(reference["config"]))
        reference["config"].update(num_obs=args.num_obs, test_type=args.distribution,
            data_paths={}, data_path=input_receipt["source"], shared_mask=True)
        checkpoint = Path(args.fm_checkpoint)
        identity.update(checkpoint=dict(path=str(checkpoint),sha256=sha256(checkpoint)),
            fm_reference_cell=reference["cell_id"], fm_protocol_sha256=sha256(args.fm_protocol),
            fm_config=reference["config"])
        noise_bank = None
        if args.noise_root:
            from experiments.rough_stress.noise import NoiseBank
            noise_bank = NoiseBank(args.noise_root, args.noise_cache)
            identity["noise_replay"] = dict(manifest_sha256=noise_bank.manifest_sha256,
                protocol=noise_bank.manifest["protocol"])
        bundle = load_fm4pde_checkpoint_bundle(str(checkpoint), args.pde, args.device,
            model_profile=reference["config"].get("model_profile"))
        ma = pack["masks"][args.num_obs] if args.task in {"forward","both"} else torch.zeros_like(pack["masks"][args.num_obs])
        mu = pack["masks"][args.num_obs] if args.task in {"inverse","both"} else torch.zeros_like(pack["masks"][args.num_obs])
        data = dict(coef_ground_truth=pack["raw"]["full_tensor"][:,:1],
            sol_ground_truth=pack["raw"]["full_tensor"][:,1:2],sample_ids=pack["sample_ids"],
            source_indices=pack["source_indices"], pde_params=pack["raw"].get("pde_params",{}),
            masks=dict(coef=ma,sol=mu,metadata={"num_obs":args.num_obs}))
    identity_path = cell / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError(f"Run identity changed: {cell}")
    write_json(identity_path, identity)
    for lo in range(start, stop, args.batch_size):
        hi = min(lo+args.batch_size,stop)
        batch_path = cell / f"batch_{lo:04d}_{hi:04d}.pt"
        record_path = batch_path.with_suffix(".json")
        if record_path.exists() and batch_path.exists():
            if sha256(batch_path) != json.loads(record_path.read_text())["prediction_sha256"]:
                raise ValueError(f"Corrupt saved prediction: {batch_path}")
            continue
        if args.method != "fm4pde":
            batch = registry.make_task(slice_tree(pack["raw"],lo,hi,count),args.pde,TASKS[args.task],
                num_sensors=args.num_obs,sensor_mode="random_per_sample",sensor_budget_mode="total",
                seed=1,experiment_mode="debug",build_voronoi_grid=args.method in {"recfno","voronoicnn"})
            if batch.metadata.get("deferred_dynamic_sensors"):
                from baselines.common.data_adapter import PDEBatchDataset,pde_collate
                ds=PDEBatchDataset(batch)
                batch=pde_collate([ds[i] for i in range(len(ds))])
            expected_mask=pack["masks"][args.num_obs][lo:hi].float().expand_as(batch.mask)
            if not torch.equal(batch.mask,expected_mask):
                raise ValueError("Baseline did not consume the frozen observations")
            actual_mask_hash=tensor_hash(batch.mask[:,:1].to(torch.uint8))
            view=_to_device_batch_for_eval(_make_inference_batch(batch),args.device)
            if args.device.startswith("cuda"):
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            begin=time.perf_counter()
            with torch.inference_mode():
                pred=model.predict_physical(view)
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            runtime=dict(seconds=time.perf_counter()-begin,
                peak_allocated_bytes=torch.cuda.max_memory_allocated() if args.device.startswith("cuda") else 0,
                peak_reserved_bytes=torch.cuda.max_memory_reserved() if args.device.startswith("cuda") else 0)
            pred=pred.detach().cpu()
        else:
            indices=list(range(lo,hi))
            cfg=effective_config(reference,checkpoint,args.device,indices,count,cell)
            gt,masks,hashes=observation_batch(data,cfg,indices,args.device)
            with noise_bank.replay(indices) if noise_bank is not None else nullcontext():
                pred,runtime=infer(cfg,bundle,gt,masks,indices)
            if noise_bank is not None:
                runtime["noise_manifest_sha256"] = noise_bank.manifest_sha256
            actual_mask_hash=tensor_hash(pack["masks"][args.num_obs][lo:hi])
        if not torch.isfinite(pred).all():
            raise ValueError("Nonfinite prediction")
        truth=pack["raw"]["full_tensor"][lo:hi]
        rows=evaluate_errors(pred,truth,args.task,args.method)
        for i,row in enumerate(rows,lo):
            row.update(index=i,sample_id=pack["sample_ids"][i])
        tmp=batch_path.with_suffix(".tmp")
        torch.save(dict(prediction=pred,sample_ids=pack["sample_ids"][lo:hi],indices=list(range(lo,hi))),tmp)
        tmp.replace(batch_path)
        write_json(record_path,dict(start=lo,stop=hi,rows=rows,runtime=runtime,
            observation_mask_sha256=actual_mask_hash,prediction_sha256=sha256(batch_path)))
        print(f"PROGRESS {args.method}/{args.task}/{args.distribution}/obs{args.num_obs} {hi}/{stop} peak={runtime['peak_allocated_bytes']/2**30:.2f}GiB",flush=True)
    print(f"FINISHED RANGE {cell} [{start},{stop})",flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=["prepare","run"])
    parser.add_argument("--output-root",required=True)
    parser.add_argument("--input-root")
    parser.add_argument("--baseline-root")
    parser.add_argument("--baseline-output-root")
    parser.add_argument("--matrix")
    parser.add_argument("--data-root")
    parser.add_argument("--pde",default="poisson")
    parser.add_argument("--count",type=int,default=1000)
    parser.add_argument("--distributions",nargs="+",default=list(DISTRIBUTIONS),choices=DISTRIBUTIONS)
    parser.add_argument("--distribution",choices=DISTRIBUTIONS)
    parser.add_argument("--method",choices=[*METHODS,"fm4pde"])
    parser.add_argument("--task",choices=list(TASKS))
    parser.add_argument("--num-obs",type=int,choices=[400,500],default=500)
    parser.add_argument("--batch-size",type=int,default=8)
    parser.add_argument("--start",type=int,default=0)
    parser.add_argument("--stop",type=int)
    parser.add_argument("--device",default="cuda:0")
    parser.add_argument("--threads",type=int,default=2)
    parser.add_argument("--fm-protocol")
    parser.add_argument("--fm-checkpoint")
    parser.add_argument("--noise-root")
    parser.add_argument("--noise-cache")
    args=parser.parse_args()
    if args.baseline_root:
        sys.path.insert(0,str(Path(args.baseline_root).resolve()))
    sys.path.insert(0,str(ROOT))
    (prepare if args.mode=="prepare" else run)(args)


if __name__=="__main__":
    main()
