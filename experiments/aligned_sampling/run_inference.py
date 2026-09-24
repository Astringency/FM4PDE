"""Frozen FM inference; no training, selection, mask generation, or source writes.

Each input row selects the same row of a canonical Gaussian pool seeded with
the archived sample_seed. Every call restarts that stream. Runtime batch size
therefore does not change the random path. This deliberately does not replay
historical non-NS runs whose random stream restarted at smaller batch sizes.

Production requires a successful full-step pilot for this PDE/task family,
model, code, GPU model, precision, and maximum batch size. Outputs live below the explicit output root
and globally unique job ID; completed batch directories are never overwritten.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
TASK = "baseline_pairing_20260915_57k"
RANDOM_POLICY = "canonical_pool_archived_seed_fixed_input_row_v1"
GRADIENT_OPERATOR = "stock_separate_component_gradients_v1"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def json_digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def tensor_digest(value):
    """SHA256 of str(dtype), compact JSON shape, and contiguous CPU raw bytes."""
    x = value.detach().cpu().contiguous()
    h = hashlib.sha256(str(x.dtype).encode())
    h.update(json.dumps(list(x.shape), separators=(",", ":")).encode())
    h.update(x.numpy().tobytes())
    return h.hexdigest()


def write_json_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def safe_relative(value):
    p = Path(value)
    if p.is_absolute() or ".." in p.parts or not p.parts:
        raise ValueError(f"Unsafe relative identifier: {value!r}")
    return p


def resolved(base, name):
    p = Path(name)
    return p if p.is_absolute() else base / p


def code_identity():
    files = [
        Path(__file__),
        *sorted((ROOT / "sampling").glob("*.py")),
        *sorted((ROOT / "models").glob("*.py")),
        ROOT / "data" / "specs.py",
        ROOT / "data" / "transform.py",
    ]
    for name in ["input_sources.py"]:
        p = Path(__file__).parent / name
        if p.exists():
            files.append(p)
    return {str(p.relative_to(ROOT)): digest(p) for p in files}


def runtime_environment(device, tf32):
    import torch

    d = torch.device(device)
    result = dict(
        host=socket.gethostname(),
        pid=os.getpid(),
        python=sys.version,
        torch=torch.__version__,
        cuda=torch.version.cuda,
        device=str(d),
        tf32=tf32,
        deterministic_algorithms=True,
        visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        code_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
    )
    if d.type == "cuda":
        props = torch.cuda.get_device_properties(d)
        result.update(
            gpu=props.name,
            gpu_uuid=str(props.uuid),
            gpu_total_bytes=props.total_memory,
            capability=list(torch.cuda.get_device_capability(d)),
        )
    return result


def configure_runtime(device, tf32, threads=2):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    torch.set_num_threads(threads)
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    if torch.device(device).type == "cuda":
        torch.cuda.set_device(device)


def effective_config(cell, checkpoint, device, indices, count, output_dir=None):
    from sampling.config import AblationConfig
    from sampling.data import finalize_ground_truth_config

    values = copy.deepcopy(cell["config"])
    values.update(
        checkpoint_path=str(checkpoint),
        device=str(device),
        output_dir=str(
            Path(output_dir or ROOT / "outputs" / TASK / "unwritten_kernel").resolve()
        ),
        batch_size=len(indices),
        offset=int(indices[0]),
        initial_noise_source_batch_size=count,
        initial_noise_source_indices=list(indices),
        save_plots=False,
        save_intermediate=False,
        save_per_sample_curves=False,
        empty_cache_each_step=False,
        allow_synthetic_data=False,
    )
    cfg = finalize_ground_truth_config(AblationConfig(**values))
    cfg.validate()
    if cfg.num_steps != 100 or cfg.dtype != "float32":
        raise ValueError("This frozen main rerun requires 100 steps and float32")
    if any(
        float(v or 0) != 0
        for v in (cfg.noise_level, cfg.noise_level_coef, cfg.noise_level_sol)
    ):
        raise ValueError(
            "Noisy observations require a separately frozen observation payload"
        )
    if (
        cfg.obs_l2_reference_mse_zeta_a is not None
        or cfg.obs_l2_reference_mse_zeta_u is not None
    ):
        raise ValueError("This rerun does not recalibrate guidance coefficients")
    return cfg


def _slice_params(value, indices, count, device):
    import torch

    if torch.is_tensor(value):
        if value.ndim and value.shape[0] == count:
            value = value[indices]
        return value.to(device)
    if isinstance(value, dict):
        return {k: _slice_params(v, indices, count, device) for k, v in value.items()}
    return value


def observation_batch(data, cfg, indices, device):
    """Only masked physical measurements enter the inference ground truth."""
    import torch
    from sampling.data import PDEGroundTruth
    from sampling.masks import PairMasks
    from sampling.losses import GROUND_TRUTH_FIELD_PDE_PARAM_KEYS

    count = len(data["sample_ids"])
    coef = data["coef_ground_truth"][indices].to(device=device, dtype=torch.float32)
    sol = data["sol_ground_truth"][indices].to(device=device, dtype=torch.float32)
    ma = data["masks"]["coef"][indices].to(device=device, dtype=torch.float32)
    mu = data["masks"]["sol"][indices].to(device=device, dtype=torch.float32)
    if ma.shape != coef.shape or mu.shape != sol.shape:
        raise ValueError("Frozen masks must have exactly the corresponding field shape")
    if not bool(((ma == 0) | (ma == 1)).all() & ((mu == 0) | (mu == 1)).all()):
        raise ValueError("Frozen observation masks must be binary")
    aa, uu = coef * ma, sol * mu
    params = _slice_params(data.get("pde_params", {}), indices, count, device)
    bad = set(params) & GROUND_TRUTH_FIELD_PDE_PARAM_KEYS
    if bad:
        raise ValueError(
            f"Frozen main residual requires public scalar parameters only: {bad}"
        )
    metadata = dict(
        sample_ids=[str(data["sample_ids"][i]) for i in indices],
        sample_offsets=[int(data["source_indices"][i]) for i in indices],
        offset=int(indices[0]),
        batch_size=len(indices),
        synthetic=False,
        inference_truth_scope="masked_measurements_only",
    )
    pair = uu if cfg.pde == "burger" else torch.cat([aa, uu], 1)
    gt = PDEGroundTruth(cfg.pde, aa, uu, pair, params, ["coef"], ["sol"], metadata)
    masks = PairMasks(ma, mu, copy.deepcopy(data["masks"].get("metadata", {})))
    hashes = {
        name: tensor_digest(tensor)
        for name, tensor in dict(
            coef_truth=coef,
            sol_truth=sol,
            coef_mask=ma,
            sol_mask=mu,
            observed_coef=aa,
            observed_sol=uu,
        ).items()
    }
    return gt, masks, hashes


def infer(cfg, bundle, gt, masks, indices, *, steps=None, observations=None):
    """Reuse stock model, conditioning, sampler, losses, schedules, and updates."""
    import torch
    import sampling.runner as r

    device = torch.device(cfg.device)
    net, normalizer, payload = bundle
    r._set_seed(cfg.sample_seed)
    scalar, _ = r._scalar_conditioning_for_sampling(
        checkpoint_payload=payload, gt=gt, config=cfg, device=device
    )
    classes, _ = r._class_conditioning_for_sampling(
        checkpoint_payload=payload,
        pde=cfg.pde,
        batch_size=len(indices),
        device=device,
        cfg_scale=cfg.cfg_scale,
    )
    extras = {**classes, **(scalar or {})} or None
    r._check_sampling_channels(gt, normalizer, payload)
    model = getattr(net, "model", net)
    if isinstance(model, torch.nn.Module):
        model.eval()
        if any(
            isinstance(m, torch.nn.modules.batchnorm._BatchNorm)
            for m in model.modules()
        ):
            raise ValueError(
                "Batch normalization is not permitted in this batching protocol"
            )
    calls = [0]

    def count(*_):
        calls[0] += 1

    hook = (
        model.register_forward_hook(count)
        if hasattr(model, "register_forward_hook")
        else None
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    try:
        grid = r.make_time_grid(
            cfg.time_grid, cfg.num_steps, device=device, eta=cfg.time_grid_eta
        )
        x = r._sample_initial_noise(cfg, gt, device)
        initial_hash = tensor_digest(x)
        actual_steps = cfg.num_steps if steps is None else steps
        for step in range(actual_steps):
            guided = r._has_guidance(cfg)
            cur = x.detach().requires_grad_(guided and cfg.gradient_target != "proposal_state_chain_rule")
            t, tn = grid[step], grid[step + 1]
            phase = r.phase_for_step(
                cfg.sampler_phase, cfg.switch_ratio, step, cfg.num_steps
            )
            out = r.sampler_step(
                net,
                cur,
                t,
                tn,
                phase,
                cfg.step_method,
                cfg.loss_state,
                device=device,
                model_extra=extras,
                stochastic_noise_source_batch_size=cfg.initial_noise_source_batch_size,
                stochastic_noise_source_indices=list(indices),
                deterministic_endpoint_mode=cfg.deterministic_endpoint_mode,
                deterministic_endpoint_time_grid=grid[step:],
                deterministic_rollout_checkpoint=cfg.deterministic_rollout_checkpoint,
                gradient_target=cfg.gradient_target,
            )
            physical = r._physical_from_model_state(out.x_loss_state, cfg, normalizer)
            losses = r.compute_guidance_losses(physical, gt, masks, cfg, observations=observations)
            if losses.pde_residual_status == "error":
                raise RuntimeError("PDE residual failed")
            affine = r.affine_coefficients(
                r.scheduler_coefficients(t, scheduler="CondOT"), training="velocity"
            )
            schedule = r.make_zeta_schedule(cfg, t, tn, affine.b_t, step=step)
            if guided:
                target = r._gradient_target_tensor(cfg, cur, out)
                if losses.metadata.get("loss_batch_reduction") != "mean_of_per_sample":
                    raise ValueError("Unexpected stock loss batch normalization")
                # Preserve the stock order: differentiate each active component,
                # apply component clipping, weight and sum, then global clipping.
                # Fusing the backward passes changes floating-point evaluation
                # and failed the real Helmholtz 100-step GPU pilot.
                gradient = r.compute_guidance_gradient(losses, target, schedule, cfg)
                x = r.apply_guidance_update(
                    out.x_raw_next, gradient, out, schedule, cfg
                ).detach()
            else:
                x = out.x_raw_next.detach()
        with torch.no_grad():
            physical = r._physical_from_model_state(x, cfg, normalizer)
            pred = (
                physical.sol
                if cfg.pde == "burger"
                else torch.cat([physical.coef, physical.sol], 1)
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        seconds = time.perf_counter() - start
        if not torch.isfinite(pred).all():
            raise RuntimeError("Nonfinite prediction")
        expected_nfe = actual_steps * (1 if cfg.step_method == "euler" else 2)
        if cfg.gradient_target == "proposal_state_chain_rule":
            expected_nfe += actual_steps - int(actual_steps == cfg.num_steps)
        if (
            hook is not None
            and cfg.sampler_phase == "stochastic"
            and cfg.cfg_scale == 1
            and calls[0] != expected_nfe
        ):
            raise RuntimeError(f"Unexpected forward count {calls[0]} != {expected_nfe}")
        return pred.detach().cpu(), dict(
            seconds=seconds,
            nfe=calls[0] if hook else expected_nfe,
            steps=actual_steps,
            batch_size=len(indices),
            initial_noise_sha256=initial_hash,
            peak_allocated_bytes=(
                torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
            ),
            peak_reserved_bytes=(
                torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0
            ),
            loss_gradient_reduction="sum_of_per_sample",
            clip_scope="per_sample",
            fused_weighted_gradient=False,
            gradient_operator=GRADIENT_OPERATOR,
            random_policy=RANDOM_POLICY,
        )
    finally:
        if hook is not None:
            hook.remove()


def relative_differences(pred, reference):
    import torch

    a, b = pred.double().flatten(1), reference.double().flatten(1)
    rel = torch.linalg.vector_norm(a - b, dim=1) / torch.linalg.vector_norm(
        b, dim=1
    ).clamp_min(1e-12)
    return dict(max_relative=float(rel.max()), max_absolute=float((a - b).abs().max()))


def score(pred, data, indices, cfg):
    """Float64 per-input field errors; full targets are used only after inference."""
    import torch

    aa = data["coef_ground_truth"][indices].double()
    uu = data["sol_ground_truth"][indices].double()
    pa = pred[:, :1].double()
    pu = pred[:, :1].double() if cfg.pde == "burger" else pred[:, 1:2].double()

    def error(a, b):
        return torch.linalg.vector_norm(
            (a - b).flatten(1), dim=1
        ) / torch.linalg.vector_norm(b.flatten(1), dim=1).clamp_min(1e-12)

    ea, eu = error(pa, aa), error(pu, uu)
    return [
        dict(
            index=i,
            sample_id=str(data["sample_ids"][i]),
            rel_l2_a=None if cfg.pde == "burger" else float(ea[j]),
            rel_l2_u=float(eu[j]),
        )
        for j, i in enumerate(indices)
    ]


def certificate_identity(cell, bindings, environment, code):
    return dict(
        cell_id=cell["cell_id"],
        config_sha256=json_digest(cell["config"]),
        **bindings,
        code_sha256=json_digest(code),
        torch=environment["torch"],
        cuda=environment["cuda"],
        gpu=environment.get("gpu"),
        gpu_uuid=environment.get("gpu_uuid"),
        tf32=environment["tf32"],
        random_policy=RANDOM_POLICY,
    )


def pilot_family(cell, bindings, environment, code):
    """Reuse validated kernels across distributions, masks, and scalar weights.

    Exact scientific configuration and input hashes remain in every result.
    This family only controls numerical/memory preflight, not result identity.
    """
    keys = [
        "pde",
        "task",
        "guidance_components",
        "guidance_operator",
        "obs_guidance_reduction",
        "pde_guidance_reduction",
        "loss_state",
        "gradient_target",
        "sampler_phase",
        "step_method",
        "time_grid",
        "num_steps",
        "clip_mode",
        "residual_mode",
        "pde_residual_region",
        "boundary_condition_mode",
        "boundary_residual_normalization",
        "ns_operator_mode",
        "coef_positive_mode",
        "model_profile",
        "dtype",
        "img_channels",
        "img_resolution",
    ]
    from sampling.config import AblationConfig

    c = AblationConfig(**cell["config"])
    family = {key: getattr(c, key) for key in keys}
    family.update(
        checkpoint_sha256=bindings["checkpoint_sha256"],
        code_sha256=json_digest(code),
        torch=environment["torch"],
        cuda=environment["cuda"],
        gpu=environment.get("gpu"),
        capability=environment.get("capability"),
        tf32=environment["tf32"],
        random_policy=RANDOM_POLICY,
        gradient_operator=GRADIENT_OPERATOR,
    )
    if c.pde == "nsnonbounded":
        family["ns_observation_regime"] = (
            "full" if cell["setting"].startswith("full_") else "sparse"
        )
    return family


def pilot(args, cell, data, bundle, checkpoint, bindings, environment, code, folder):
    import torch

    count = len(data["sample_ids"])
    if not 1 <= args.batch_size <= count:
        raise ValueError("Pilot batch size outside input pool")
    identity = certificate_identity(cell, bindings, environment, code)
    completed = folder / "pilot_complete.json"
    if completed.exists():
        old = json.loads(completed.read_text())
        if old["identity"] != identity or old["max_batch_size"] != args.batch_size:
            raise ValueError("Existing pilot has different bindings")
        print("PILOT_EXISTS", completed, flush=True)
        return

    def one(ids):
        cfg = effective_config(cell, checkpoint, args.device, ids, count, folder)
        gt, masks, hashes = observation_batch(data, cfg, ids, args.device)
        pred, meta = infer(cfg, bundle, gt, masks, ids)
        return pred, meta, cfg, gt, masks

    print("PILOT_REFERENCE", cell["cell_id"], "stock_batch1", flush=True)
    ref, ref_meta, cfg, gt, masks = one([0])
    # Authoritative reference: unmodified stock sampler/loss/gradient/update
    # functions, bypassing the outer runner's automatic guidance-policy hook.
    strict = one([0])[0]
    repeat_difference = relative_differences(strict, ref)
    tolerance = args.relative_tolerance
    if repeat_difference["max_absolute"] != 0:
        raise RuntimeError(f"Stock single-input repeat mismatch: {repeat_difference}")
    # Full 100-step batch-one peak includes the active PDE-guidance tail.
    if torch.device(args.device).type == "cuda":
        baseline = torch.cuda.memory_allocated(args.device)
        incremental = max(0, ref_meta["peak_allocated_bytes"] - baseline)
        free, total = torch.cuda.mem_get_info(args.device)
        conservative = args.batch_size * incremental + (512 << 20)
        if conservative > 0.85 * free:
            raise RuntimeError(
                f"Batch {args.batch_size} conservative extra-memory estimate {conservative} exceeds 85% of free {free}; choose smaller batch"
            )
    ids = list(range(args.batch_size))
    print("PILOT_BATCH", cell["cell_id"], args.batch_size, flush=True)
    batch, meta, *_ = one(ids)
    comparison = relative_differences(batch[:1], strict)
    if comparison["max_relative"] > tolerance:
        raise RuntimeError(f"Batch-size mismatch: {comparison}")
    # A second row plus reversed order detects accidental positional RNG use.
    checked = [0] if len(ids) == 1 else [0, len(ids) - 1]
    reorder, reorder_meta, *_ = one(list(reversed(checked)))
    reference = batch[checked].flip(0)
    permutation_difference = relative_differences(reorder, reference)
    if permutation_difference["max_relative"] > tolerance:
        raise RuntimeError(f"Input-order mismatch: {permutation_difference}")
    poisoned = dataclasses.replace(
        gt, coef=gt.coef + 3 * (1 - masks.coef), sol=gt.sol - 7 * (1 - masks.sol)
    )
    poisoned.pair = (
        poisoned.sol
        if cfg.pde == "burger"
        else torch.cat([poisoned.coef, poisoned.sol], 1)
    )
    hidden = infer(cfg, bundle, poisoned, masks, [0])[0]
    hidden_difference = relative_differences(hidden, strict)
    if hidden_difference["max_absolute"] != 0:
        raise RuntimeError(f"Hidden target dependence: {hidden_difference}")
    certificate = dict(
        status="pass",
        identity=identity,
        family=pilot_family(cell, bindings, environment, code),
        max_batch_size=args.batch_size,
        relative_tolerance=tolerance,
        repeat_difference=repeat_difference,
        batch_difference=comparison,
        gradient_operator=GRADIENT_OPERATOR,
        permutation_difference=permutation_difference,
        hidden_target_difference=hidden_difference,
        single=ref_meta,
        batch=meta,
        permutation=reorder_meta,
        pilot_input_rows=ids,
        scope="Full 100-step numerical/memory pilot; these predictions are excluded from production aggregates",
    )
    write_json_new(completed, certificate)
    print("PILOT_PASS", completed, json.dumps(meta), flush=True)


def existing_batches(folder, identity):
    completed = set()
    for path in sorted(folder.glob("batch_*/receipt.json")):
        receipt = json.loads(path.read_text())
        if receipt["identity"] != identity:
            raise ValueError(f"Resume bindings changed: {path}")
        if digest(path.parent / "prediction.pt") != receipt["prediction_sha256"]:
            raise ValueError(f"Completed prediction checksum mismatch: {path}")
        ids = receipt["indices"]
        if completed.intersection(ids):
            raise ValueError(f"Duplicate completed input rows: {path}")
        completed.update(ids)
    return completed


def production(
    args, cell, data, bundle, checkpoint, bindings, environment, code, folder
):
    import torch

    identity = certificate_identity(cell, bindings, environment, code)
    certificates = []
    for path in args.pilot_certificate:
        p = Path(path)
        certificates.extend(p.rglob("pilot_complete.json") if p.is_dir() else [p])
    matching = [json.loads(p.read_text()) for p in certificates]
    family = pilot_family(cell, bindings, environment, code)
    matching = [
        p for p in matching if p.get("status") == "pass" and p.get("family") == family
    ]
    if not any(p["max_batch_size"] >= args.batch_size for p in matching):
        raise ValueError(
            "Production needs a matching successful PDE/task/model/operator/precision/GPU-model family pilot"
        )
    count = len(data["sample_ids"])
    stop = count if args.stop is None else args.stop
    if not 0 <= args.start < stop <= count:
        raise ValueError("Invalid --start/--stop input-row range")
    done = existing_batches(folder, identity)
    todo = [i for i in range(args.start, stop) if i not in done]
    if todo:
        # Each newly started process/card first runs one actual input alone.
        # This prediction is diagnostic and is not a production record.
        sanity_cfg = effective_config(
            cell, checkpoint, args.device, todo[:1], count, folder
        )
        sanity_gt, sanity_masks, _ = observation_batch(
            data, sanity_cfg, todo[:1], args.device
        )
        _, sanity_meta = infer(sanity_cfg, bundle, sanity_gt, sanity_masks, todo[:1])
        print("CARD_SANITY_PASS", cell["cell_id"], json.dumps(sanity_meta), flush=True)
    for start in range(0, len(todo), args.batch_size):
        ids = todo[start : start + args.batch_size]
        cfg = effective_config(cell, checkpoint, args.device, ids, count, folder)
        gt, masks, hashes = observation_batch(data, cfg, ids, args.device)
        pred, meta = infer(cfg, bundle, gt, masks, ids)
        rows = score(pred, data, ids, cfg)
        key = json_digest(ids)[:20]
        destination = folder / f"batch_{key}"
        temporary = folder / (".partial_" + uuid.uuid4().hex)
        temporary.mkdir()
        # Actual input tensor hashes, rather than just expected manifest hashes,
        # bind the observations used by the kernel to this completed prediction.
        payload = dict(
            cell_id=cell["cell_id"],
            indices=ids,
            sample_ids=[str(data["sample_ids"][i]) for i in ids],
            source_indices=[int(data["source_indices"][i]) for i in ids],
            prediction=pred,
            coef_final=pred[:, :1],
            sol_final=pred[:, :1] if cfg.pde == "burger" else pred[:, 1:2],
            runtime_input_hashes=hashes,
            effective_config=cfg.asdict(),
            observation_mask_source="frozen_external",
            observation_masks_metadata=masks.metadata,
            historical_mask_config_scope="sensor_mode/shared_mask/mask_seed retained as provenance; frozen external masks control this execution",
            **bindings,
            metrics=rows,
            inference=meta,
        )
        torch.save(payload, temporary / "prediction.pt")
        with (temporary / "prediction.pt").open("rb") as stream:
            os.fsync(stream.fileno())
        receipt = dict(
            identity=identity,
            cell_id=cell["cell_id"],
            indices=ids,
            sample_ids=payload["sample_ids"],
            source_indices=payload["source_indices"],
            runtime_input_hashes=hashes,
            effective_config_sha256=json_digest(cfg.asdict()),
            config_sha256=json_digest(cfg.asdict()),
            observation_mask_source="frozen_external",
            observation_masks_metadata=masks.metadata,
            prediction_sha256=digest(temporary / "prediction.pt"),
            metrics=rows,
            source_hashes=code,
            **bindings,
            **meta,
        )
        write_json_new(temporary / "receipt.json", receipt)
        if destination.exists():
            raise FileExistsError(destination)
        temporary.rename(destination)
        directory = os.open(folder, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        print(
            "BATCH_COMPLETE",
            cell["cell_id"],
            ids[0],
            ids[-1],
            json.dumps(meta),
            flush=True,
        )


def validate_protocol(protocol, mode):
    """Pilot subsets cannot authorize a partial or unfrozen production study."""
    allowed = {"frozen", "pilot_only"} if mode == "pilot" else {"frozen"}
    if protocol.get("status") not in allowed:
        raise ValueError(
            f"Protocol status {protocol.get('status')!r} is not allowed for mode={mode}"
        )
    content = dict(protocol)
    recorded = content.pop("content_sha256", None)
    if recorded != json_digest(content):
        raise ValueError("Frozen protocol content hash mismatch")
    cells = protocol.get("cells", [])
    cell_map = {cell["cell_id"]: cell for cell in cells}
    if not cells or len(cell_map) != len(cells):
        raise ValueError("Protocol needs nonempty unique cell IDs")
    if any(
        type(cell.get("count")) is not int or cell["count"] != 1000 for cell in cells
    ):
        raise ValueError("Each complete input cell must contain exactly 1000 rows")
    if mode == "run" and (
        len(cells) != 57 or sum(cell["count"] for cell in cells) != 57000
    ):
        raise ValueError(
            "Production requires the complete frozen 57-cell/57000-row protocol"
        )
    return cell_map


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--cell", action="append", required=True)
    parser.add_argument("--mode", choices=["pilot", "run"], required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int)
    parser.add_argument("--pilot-certificate", type=Path, action="append", default=[])
    parser.add_argument("--relative-tolerance", type=float, default=3e-4)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.job_id):
        parser.error("--job-id must be a globally unique safe identifier")
    if args.batch_size < 1 or not 0 < args.relative_tolerance <= 0.005:
        parser.error(
            "Batch size must be positive; numerical tolerance must be in (0, .005]"
        )
    if args.mode == "run" and not args.pilot_certificate:
        parser.error("--pilot-certificate is required for production")
    args.protocol = args.protocol.resolve()
    base = args.input_root or args.protocol.parent
    protocol = json.loads(args.protocol.read_text())
    cell_map = validate_protocol(protocol, args.mode)
    for cell_id in args.cell:
        safe_relative(cell_id)
        if cell_id not in cell_map:
            raise KeyError(cell_id)
    configure_runtime(args.device, args.tf32, args.threads)
    environment = runtime_environment(args.device, args.tf32)
    code = code_identity()
    job = args.output_root.resolve() / "jobs" / args.job_id
    job.mkdir(parents=True, exist_ok=True)
    with (job / "worker.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_json_new(
            job / ("invocation_" + uuid.uuid4().hex + ".json"),
            dict(
                task=TASK,
                job_id=args.job_id,
                argv=sys.argv,
                environment=environment,
                source_hashes=code,
                protocol_sha256=digest(args.protocol),
            ),
        )
        from experiments.aligned_sampling.input_sources import load_cell
        from sampling.model_io import load_fm4pde_checkpoint_bundle

        bundle = None
        loaded_checkpoint = None
        for cell_id in args.cell:
            cell = cell_map[cell_id]
            weight = cell.get("checkpoint", cell.get("weights"))
            checkpoint = resolved(base, weight["path"])
            if loaded_checkpoint != (str(checkpoint), weight["sha256"]):
                if digest(checkpoint) != weight["sha256"]:
                    raise ValueError(f"Checkpoint hash mismatch: {checkpoint}")
                if bundle is not None:
                    del bundle
                    import torch

                    if torch.device(args.device).type == "cuda":
                        torch.cuda.empty_cache()
                bundle = load_fm4pde_checkpoint_bundle(
                    str(checkpoint),
                    cell["pde"],
                    args.device,
                    model_profile=cell["config"]["model_profile"],
                )
                loaded_checkpoint = (str(checkpoint), weight["sha256"])
            data = load_cell(base, cell, verify_hashes=True)
            if len(data["sample_ids"]) != cell.get("count", 1000):
                raise ValueError(f"Input count mismatch: {cell_id}")
            if len(set(data["sample_ids"])) != len(data["sample_ids"]):
                raise ValueError(f"Duplicate source-qualified input IDs: {cell_id}")
            bindings = dict(
                protocol_sha256=digest(args.protocol),
                checkpoint_sha256=weight["sha256"],
                input_truth_sha256=cell["truth_sha256"],
                input_masks_sha256=cell["masks_sha256"],
            )
            folder = job / safe_relative(cell_id)
            folder.mkdir(parents=True, exist_ok=True)
            operation = pilot if args.mode == "pilot" else production
            operation(
                args,
                cell,
                data,
                bundle,
                checkpoint,
                bindings,
                environment,
                code,
                folder,
            )
    print("JOB_COMPLETE", args.job_id, flush=True)


if __name__ == "__main__":
    main()
