"""Paired Poisson experiment for pre-proposal versus post-proposal endpoint guidance.

Uses archived physical test fields, observation masks and the original checkpoint.
Only masked measurements enter inference. No ground-truth-dependent selection.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments.aligned_sampling.input_sources import load_cell
from experiments.aligned_sampling.run_inference import (
    configure_runtime, digest, effective_config, infer, observation_batch,
    relative_differences, runtime_environment, score, tensor_digest,
)
from sampling.model_io import load_fm4pde_checkpoint_bundle
from sampling.metrics import write_json

VARIANTS = {"original": "current_state_chain_rule", "proposal": "proposal_state_chain_rule"}


def setup(args):
    args.root.mkdir(parents=True, exist_ok=True)
    if (args.root / "protocol.json").exists():
        raise FileExistsError("An existing experiment protocol must not be overwritten")
    source = json.loads((args.reference / "protocol.json").read_text())
    cells = {task: next(c for c in source["cells"] if c["cell_id"] == f"supervised/poisson/id/{task}")
             for task in ["sparse_forward", "sparse_inverse", "sparse_joint"]}
    indices = sorted(np.random.default_rng(20260919).choice(1000, args.count, replace=False).tolist())
    assert digest(args.checkpoint) == cells["sparse_joint"]["checkpoint"]["sha256"]
    data = load_cell(args.reference, cells["sparse_joint"])
    # A portable, full source input copy is small enough for this 1000-case set.
    # Retain all rows so canonical noise indices and source hashes stay explicit.
    torch.save(data, args.root / "inputs.pt")
    write_json(args.root / "protocol.json", dict(
        reference=str(args.reference), reference_sha256=digest(args.reference / "protocol.json"),
        checkpoint=str(args.checkpoint), checkpoint_sha256=digest(args.checkpoint),
        inputs_sha256=digest(args.root / "inputs.pt"), cells=cells,
        indices=indices, seeds=[0, 1, 2], count=args.count, batch_size=args.batch_size,
        variants=VARIANTS, steps=100, tf32=False,
        selection="Uniform random sample without replacement, seed 20260919, before candidate evaluation",
        comparison="Identical weights, input cases, masks, initial noise, every bridge noise, guidance weights, schedule, clipping and time grid; only endpoint evaluation pair and differentiated state change",
        primary_metrics={"sparse_forward": ["rel_l2_u"], "sparse_inverse": ["rel_l2_a"],
                         "sparse_joint": ["rel_l2_a", "rel_l2_u"]},
        uncertainty="Paired bootstrap over input cases after averaging the three noise seeds; seeds are not treated as independent cases",
        caveat="ID exploratory comparison, 500 sensors per observed field; no tuning or population-wide claim",
    ))
    print("SETUP", indices, flush=True)


def prepared(args):
    protocol = json.loads((args.root / "protocol.json").read_text())
    assert digest(args.root / "inputs.pt") == protocol["inputs_sha256"]
    assert digest(protocol["checkpoint"]) == protocol["checkpoint_sha256"]
    data = torch.load(args.root / "inputs.pt", map_location="cpu", weights_only=False)
    configure_runtime("cuda:0", False, 2)
    bundle = load_fm4pde_checkpoint_bundle(protocol["checkpoint"], "poisson", "cuda:0", prefer_ema=False)
    return protocol, data, bundle


def config_batch(protocol, data, task, indices, seed, variant, root, scale=1.0, steps=100):
    cell = copy.deepcopy(protocol["cells"][task])
    cell["config"]["sample_seed"] = seed
    cfg = effective_config(cell, protocol["checkpoint"], "cuda:0", indices, 1000, root)
    cfg.gradient_target = VARIANTS[variant]
    cfg.num_steps = steps
    cfg.stochastic_guidance_coeff *= scale
    cfg.validate()
    # Archived joint masks are the same source used by all three sparse tasks.
    # Disable the unobserved field exactly as input_sources.load_cell does.
    task_data = dict(data, masks=dict(data["masks"]))
    if task == "sparse_forward":
        task_data["masks"]["sol"] = torch.zeros_like(data["masks"]["sol"])
    elif task == "sparse_inverse":
        task_data["masks"]["coef"] = torch.zeros_like(data["masks"]["coef"])
    gt, masks, hashes = observation_batch(task_data, cfg, indices, "cuda:0")
    return cfg, gt, masks, hashes


def tracked_infer(cfg, bundle, gt, masks, indices):
    import sampling.sampler_wrappers as sampler
    original = sampler._stochastic_bridge_noise_like
    hashes = []

    def tracked(*args, **kwargs):
        value = original(*args, **kwargs)
        hashes.append(tensor_digest(value))
        return value

    sampler._stochastic_bridge_noise_like = tracked
    try:
        pred, metadata = infer(cfg, bundle, gt, masks, indices)
    finally:
        sampler._stochastic_bridge_noise_like = original
    assert len(hashes) == cfg.num_steps
    metadata["bridge_noise_sha256"] = hashes
    return pred, metadata


def evaluate(pred, data, indices, cfg, gt, masks):
    from sampling.state import SplitState
    from sampling.losses import compute_guidance_losses
    from sampling.metrics import per_sample_metrics
    # Evaluate the same saved physical prediction; truth is used here only.
    x = pred.to(cfg.device)
    physical = SplitState(x[:, :1], x[:, 1:2])
    with torch.no_grad():
        losses = compute_guidance_losses(physical, gt, masks, cfg)
        extra = per_sample_metrics(physical, gt, masks, losses)
    assert losses.pde_residual_status != "error"
    rows = score(pred, data, indices, cfg)
    for row, aux in zip(rows, extra):
        for key in ["obs_rel_l2_a", "obs_rel_l2_u", "pde_residual_norm"]:
            row[key] = aux[key]
    return rows


def pilot(args):
    p, data, bundle = prepared(args)
    environment = runtime_environment("cuda:0", False)
    target = args.root / "pilots" / args.task
    target.mkdir(parents=True, exist_ok=True)
    records = []
    # Check actual full-trajectory peak memory at increasing batch sizes first.
    for size in sorted({1, p["batch_size"]}):
        indices = p["indices"][:size]
        pair = {}
        for variant in VARIANTS:
            free, _ = torch.cuda.mem_get_info()
            if size > 1:
                one = next(r for r in records if r["batch_size"] == 1 and r["variant"] == variant)
                assert one["peak_allocated_bytes"] * size * 1.4 < free - 2 * 2**30, "Insufficient GPU memory margin"
            cfg, gt, masks, hashes = config_batch(p, data, args.task, indices, 0, variant, args.root)
            pred, meta = tracked_infer(cfg, bundle, gt, masks, indices)
            rows = evaluate(pred, data, indices, cfg, gt, masks)
            record = dict(variant=variant, **meta, rows=rows)
            records.append(record)
            pair[variant] = meta
            print("PILOT", args.task, variant, size, meta["seconds"], meta["peak_allocated_bytes"], rows, flush=True)
            if size == p["batch_size"]:
                again, replay_meta = tracked_infer(cfg, bundle, gt, masks, indices)
                difference = relative_differences(again, pred)
                assert difference["max_relative"] < 1e-6, difference
                record["replay"] = difference
            del pred, gt, masks
            torch.cuda.empty_cache()
        assert pair["original"]["initial_noise_sha256"] == pair["proposal"]["initial_noise_sha256"]
        assert pair["original"]["bridge_noise_sha256"] == pair["proposal"]["bridge_noise_sha256"]
    write_json(target / "complete.json", dict(passed=True, records=records, environment=environment,
                                               protocol_sha256=digest(args.root / "protocol.json")))


def run(args):
    p, data, bundle = prepared(args)
    cert = json.loads((args.root / "pilots" / args.task / "complete.json").read_text())
    assert cert["passed"] and cert["protocol_sha256"] == digest(args.root / "protocol.json")
    target = args.root / "runs" / args.task
    target.mkdir(parents=True, exist_ok=True)
    environment = runtime_environment("cuda:0", False)
    assert environment["gpu_uuid"] == cert["environment"]["gpu_uuid"]
    write_json(target / "environment.json", environment)
    indices = p["indices"]
    batch_size = p["batch_size"]
    for seed in p["seeds"]:
        for start in range(0, len(indices), batch_size):
            group = indices[start:start + batch_size]
            records = {}
            # Alternate order to reduce systematic warmup/temperature timing bias.
            variants = list(VARIANTS) if (seed + start // batch_size) % 2 == 0 else list(reversed(VARIANTS))
            for variant in variants:
                path = target / f"{variant}_seed{seed}_batch{start:04d}.pt"
                if path.exists():
                    payload = torch.load(path, map_location="cpu", weights_only=False)
                    assert payload["indices"] == group and payload["variant"] == variant
                else:
                    cfg, gt, masks, hashes = config_batch(p, data, args.task, group, seed, variant, args.root)
                    pred, meta = tracked_infer(cfg, bundle, gt, masks, group)
                    rows = evaluate(pred, data, group, cfg, gt, masks)
                    payload = dict(indices=group, seed=seed, variant=variant, prediction=pred, rows=rows,
                                   runtime=meta, input_hashes=hashes, config=cfg.asdict())
                    torch.save(payload, path)
                    write_json(path.with_suffix(".json"), dict(sha256=digest(path), indices=group,
                        seed=seed, variant=variant, rows=rows, runtime=meta, input_hashes=hashes))
                    print("BATCH", args.task, seed, start, variant, meta["seconds"],
                          {key: float(np.mean([r[key] for r in rows])) for key in ["rel_l2_a", "rel_l2_u"]}, flush=True)
                    del pred, gt, masks
                records[variant] = payload
            b, c = records["original"], records["proposal"]
            assert b["input_hashes"] == c["input_hashes"]
            for key in ["initial_noise_sha256", "bridge_noise_sha256"]:
                assert b["runtime"][key] == c["runtime"][key]
            configs = [dict(b["config"]), dict(c["config"])]
            for cfg in configs:
                cfg.pop("gradient_target")
            assert configs[0] == configs[1]
    write_json(target / "complete.json", dict(paired=True, cases=len(indices), seeds=p["seeds"],
                                               finished_unix=time.time()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["setup", "pilot", "run"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--task", choices=["sparse_forward", "sparse_inverse", "sparse_joint"])
    args = parser.parse_args()
    globals()[args.mode](args)


if __name__ == "__main__":
    main()
