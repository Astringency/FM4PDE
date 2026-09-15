"""Replay published NS experiments with the 44M checkpoint and its normalizer.

The stock sampler, batch membership, random seeds, observations and all other
scientific settings come from saved historical predictions. A job is committed
only after its saved tensors have been checked against the frozen input packet.
"""
from __future__ import annotations

import argparse
import copy
import csv
import dataclasses
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager, redirect_stderr, redirect_stdout

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PARAMETER_COUNT = 44_121_218
MODEL_SHA256 = "481a0db0f95db6a44197a16a8c0d45bd5075bfe34922bf85ab681275a5b697db"
RUNTIME_KEYS = {"checkpoint_path", "model_profile", "output_dir", "device",
                "save_plots", "save_intermediate", "save_per_sample_curves"}
PILOT_JOB_IDS = {"main/archive_0466", "main/archive_0505",
                 "guidance_controls/guidance_obs_pde/seed0/batch425",
                 "current_temporal/hermite_bridge", "current_temporal/near_endpoint_temporal"}


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def tensor_hash(value, *, raw=False):
    t = value.detach().cpu().contiguous()
    h = hashlib.sha256()
    if not raw:
        h.update(str(t.dtype).encode())
        h.update(json.dumps(list(t.shape), separators=(",", ":")).encode())
    h.update(t.numpy().tobytes())
    return h.hexdigest()


def tree_hash(value):
    import torch
    if isinstance(value, torch.Tensor):
        return {"tensor_sha256": tensor_hash(value)}
    if isinstance(value, dict):
        return {str(k): tree_hash(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [tree_hash(v) for v in value]
    return value


def to_device(value, device):
    import torch
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {k: to_device(v, device) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(to_device(v, device) for v in value)
    return copy.deepcopy(value)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(".partial_" + path.name + "_" + uuid.uuid4().hex)
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def load_json(path):
    return json.loads(Path(path).read_text())


def core_identity():
    paths = sorted(p for folder in ("sampling", "models", "data", "sde")
                   for p in (ROOT / folder).rglob("*.py"))
    paths.append(Path(__file__))
    return {str(p.relative_to(ROOT)): file_hash(p) for p in paths}


def scientific_config(config):
    return {k: v for k, v in config.items() if k not in RUNTIME_KEYS}


def effective_config(job, checkpoint, output, device):
    from sampling.config import AblationConfig
    original = copy.deepcopy(job["config"])
    unknown = set(original) - {f.name for f in dataclasses.fields(AblationConfig)}
    if unknown:
        raise ValueError(f"Unrecognized historical settings: {sorted(unknown)}")
    cfg = AblationConfig(**original)
    before = scientific_config(cfg.asdict())
    cfg.checkpoint_path = str(Path(checkpoint).resolve())
    cfg.model_profile = "light"
    cfg.output_dir = str(Path(output).resolve())
    cfg.device = device
    cfg.save_plots = False
    cfg.save_intermediate = False
    cfg.save_per_sample_curves = True
    cfg.validate()
    assert scientific_config(cfg.asdict()) == before
    assert cfg.pde == "nsnonbounded" and cfg.batch_size == len(job["sample_ids"])
    return cfg


def load_input(protocol_path, job):
    import torch
    path = Path(protocol_path).parent / job["input_file"]
    assert file_hash(path) == job["input_sha256"], path
    packet = torch.load(path, map_location="cpu", weights_only=False)
    assert packet["sample_ids"] == job["sample_ids"]
    assert packet["config"] == job["config"]
    assert json_hash(tree_hash(packet["inputs"])) == job["input_tensor_sha256"]
    return packet


def ground_truth(packet, device):
    import torch
    from sampling.data import PDEGroundTruth
    data = to_device(packet["inputs"], device)
    a, u = data["coef_ground_truth"], data["sol_ground_truth"]
    meta = data["ground_truth_metadata"]
    assert not meta.get("synthetic", False)
    return PDEGroundTruth("nsnonbounded", a, u, torch.cat([a, u], dim=1),
                          data["pde_params"], meta.get("channel_names_coef", ["w0"]),
                          meta.get("channel_names_sol", ["wT"]), meta)


def errors(pred, truth):
    import torch
    p, t = pred.detach().cpu().double().flatten(1), truth.detach().cpu().double().flatten(1)
    return (torch.linalg.vector_norm(p-t, dim=1) /
            torch.linalg.vector_norm(t, dim=1).clamp_min(1e-12)).tolist()


@contextmanager
def record_actual_inputs(packet, job, directory):
    """Observe stock inputs; only replace the raw loader for saved temporal auxiliaries."""
    import torch
    import sampling.runner as runner
    original_noise = runner._sample_initial_noise
    original_observation_noise = runner.add_observation_noise
    original_attach = runner.attach_near_endpoint_observations
    observed = {"observation_calls": []}

    def noise(*args, **kwargs):
        value = original_noise(*args, **kwargs)
        sha = tensor_hash(value, raw=True)
        expected = job.get("expected_initial_noise_sha256")
        if expected and sha != expected:
            raise AssertionError(f"Historical initial noise mismatch: {sha} != {expected}")
        observed["initial_noise_sha256"] = sha
        observed["initial_rng_state_sha256"] = tensor_hash(torch.cuda.get_rng_state(), raw=True)
        torch.save(value.detach().cpu(), directory / "initial_noise.pt")
        return value

    def observation_noise(*args, **kwargs):
        result = original_observation_noise(*args, **kwargs)
        observed["observation_calls"].append({
            "clean": result.clean.detach().cpu(), "noisy": result.noisy.detach().cpu(),
            "noise": result.noise.detach().cpu(), "metadata": result.metadata})
        return result

    def attach(config, gt, masks):
        if config.residual_mode != "near_endpoint_temporal":
            return original_attach(config, gt, masks)
        # The original result already saves exactly the permitted sparse temporal
        # observations. Reusing them avoids reloading or resampling the raw data.
        if "near_endpoint_temporal" not in gt.pde_params:
            raise ValueError("Frozen near-endpoint observations are missing")
        expected = packet["inputs"]["masks"]
        assert tensor_hash(masks.coef) == tensor_hash(expected["coef"])
        assert tensor_hash(masks.sol) == tensor_hash(expected["sol"])
        assert json_hash(tree_hash(gt.pde_params)) == json_hash(tree_hash(packet["inputs"]["pde_params"]))
        observed["temporal_auxiliary_source"] = "historical_saved_sparse_observations"
        return gt

    runner._sample_initial_noise = noise
    runner.add_observation_noise = observation_noise
    runner.attach_near_endpoint_observations = attach
    try:
        yield observed
    finally:
        runner._sample_initial_noise = original_noise
        runner.add_observation_noise = original_observation_noise
        runner.attach_near_endpoint_observations = original_attach


def validate_payload(packet, payload):
    import torch
    for field in ("coef", "sol"):
        assert torch.equal(payload[field + "_ground_truth"], packet["inputs"][field + "_ground_truth"])
        assert torch.equal(payload["masks"][field], packet["inputs"]["masks"][field])
        assert torch.isfinite(payload[field + "_final"]).all()
    assert json_hash(tree_hash(payload["pde_params"])) == json_hash(tree_hash(packet["inputs"]["pde_params"]))


def run_job(job, args, protocol, bundle, identity):
    import torch
    from sampling.masks import PairMasks
    import sampling.runner as runner
    folder = args.output / "jobs" / job["job_id"]
    folder.mkdir(parents=True, exist_ok=True)
    receipt = folder / "receipt.json"
    packet = load_input(args.protocol, job)
    if receipt.exists():
        r = load_json(receipt)
        assert r["protocol_sha256"] == file_hash(args.protocol)
        assert r["code_identity"] == identity and r["input_sha256"] == job["input_sha256"]
        for item in r["artifacts"].values():
            assert file_hash(folder / item["path"]) == item["sha256"]
        return r
    attempt = folder / ("attempt_" + uuid.uuid4().hex)
    attempt.mkdir()
    cfg = effective_config(job, args.checkpoint, attempt, args.device)
    expected_config = cfg.asdict()
    gt = ground_truth(packet, args.device)
    masks = packet["inputs"]["masks"]
    masks = PairMasks(masks["coef"].to(args.device), masks["sol"].to(args.device),
                      copy.deepcopy(masks.get("metadata", {})))
    calls = {"n": 0}
    def hook(*_): calls["n"] += 1
    handle = bundle[0].model.register_forward_hook(hook)
    try:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        with record_actual_inputs(packet, job, attempt) as observed:
            with (attempt / "run.log").open("w") as log, redirect_stdout(log), redirect_stderr(log):
                result = runner.run_single_ablation(cfg, checkpoint_bundle=bundle, ground_truth=gt,
                                                   observation_masks=masks)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        assert result["status"] == "ok", result
        assert cfg.asdict() == expected_config, "The sampler silently changed the configuration"
        result_path = Path(result["run_dir"]) / "result.pt"
        payload = torch.load(result_path, map_location="cpu", weights_only=False)
        validate_payload(packet, payload)
        torch.save(observed["observation_calls"], attempt / "observations.pt")
        assert len(observed["observation_calls"]) == 2
        for field, ob in zip(("coef", "sol"), observed["observation_calls"]):
            assert torch.equal(ob["clean"], packet["inputs"][field+"_ground_truth"] * packet["inputs"]["masks"][field])
        per_field = {f: errors(payload[k+"_final"], payload[k+"_ground_truth"])
                     for f, k in (("a", "coef"), ("u", "sol"))}
        assert all(math.isfinite(x) for v in per_field.values() for x in v)
        artifacts = {}
        for name, path in (("result", result_path), ("masks", result_path.parent / "masks.pt"),
                           ("observations", attempt / "observations.pt"),
                           ("initial_noise", attempt / "initial_noise.pt"),
                           ("curves", result_path.parent / "curves.csv"),
                           ("per_sample_curves", result_path.parent / "metrics_step_per_sample.csv")):
            artifacts[name] = {"path": str(path.relative_to(folder)), "sha256": file_hash(path)}
        r = {"status": "complete", "job_id": job["job_id"], "family": job["family"],
             "sample_ids": job["sample_ids"], "config": cfg.asdict(),
             "historical_config": job["config"], "input_sha256": job["input_sha256"],
             "input_tensor_sha256": job["input_tensor_sha256"],
             "protocol_sha256": file_hash(args.protocol), "model_sha256": file_hash(args.checkpoint),
             "parameter_count": PARAMETER_COUNT, "code_identity": identity,
             "normalizer": tree_hash(bundle[1].state_dict()),
             "artifacts": artifacts, "errors": per_field,
             "initial_noise_sha256": observed["initial_noise_sha256"],
             "initial_rng_state_sha256": observed["initial_rng_state_sha256"],
             "observations_sha256": json_hash(tree_hash(observed["observation_calls"])),
             "seconds": seconds, "nfe": calls["n"], "peak_bytes": torch.cuda.max_memory_allocated(),
             "finished_unix": time.time()}
        atomic_json(receipt, r)
        print("DONE", job["job_id"], f"{seconds:.2f}s", per_field, flush=True)
        return r
    finally:
        handle.remove()


def validate_protocol(path):
    p = load_json(path)
    assert p["status"] == "frozen" and p["parameter_count"] == PARAMETER_COUNT
    ids = [j["job_id"] for j in p["jobs"]]
    assert len(ids) == len(set(ids)) == p["expected_jobs"]
    assert sum(len(j["sample_ids"]) for j in p["jobs"]) == p["expected_predictions"]
    assert p["model_sha256"] == MODEL_SHA256
    for jid in ids:
        assert not Path(jid).is_absolute() and ".." not in Path(jid).parts
    return p


def audit(args):
    import torch
    protocol = validate_protocol(args.protocol)
    records = []
    for job in protocol["jobs"]:
        folder = args.output / "jobs" / job["job_id"]
        r = load_json(folder / "receipt.json")
        assert r["status"] == "complete" and r["sample_ids"] == job["sample_ids"]
        assert r["protocol_sha256"] == file_hash(args.protocol)
        assert r["model_sha256"] == protocol["model_sha256"]
        assert r["parameter_count"] == PARAMETER_COUNT
        assert r["code_identity"] == core_identity()
        packet = load_input(args.protocol, job)
        for item in r["artifacts"].values():
            assert file_hash(folder / item["path"]) == item["sha256"]
        payload = torch.load(folder / r["artifacts"]["result"]["path"], map_location="cpu", weights_only=False)
        validate_payload(packet, payload)
        assert json_hash(tree_hash(payload["normalizer"])) == protocol["normalizer_sha256"]
        assert tree_hash(payload["normalizer"]) == r["normalizer"]
        cfg = effective_config(job, r["config"]["checkpoint_path"], r["config"]["output_dir"], r["config"]["device"])
        assert cfg.asdict() == payload["config"] == r["config"]
        initial = torch.load(folder / r["artifacts"]["initial_noise"]["path"], map_location="cpu", weights_only=False)
        assert tensor_hash(initial, raw=True) == r["initial_noise_sha256"]
        if job.get("expected_initial_noise_sha256"):
            assert r["initial_noise_sha256"] == job["expected_initial_noise_sha256"]
        observations = torch.load(folder / r["artifacts"]["observations"]["path"], map_location="cpu", weights_only=False)
        assert json_hash(tree_hash(observations)) == r["observations_sha256"]
        from sampling.noise import add_observation_noise
        for k, ob in zip(("coef", "sol"), observations):
            assert torch.equal(ob["clean"], packet["inputs"][k+"_ground_truth"] * packet["inputs"]["masks"][k])
            level = r["config"].get("noise_level_"+k)
            level = r["config"]["noise_level"] if level is None else level
            expected = add_observation_noise(packet["inputs"][k+"_ground_truth"], packet["inputs"]["masks"][k],
                                             level, seed=r["config"]["noise_seed"]+(k=="sol"))
            # CPU/GPU reductions of observed variance can differ slightly.
            torch.testing.assert_close(ob["noisy"], expected.noisy, rtol=2e-6, atol=2e-7)
        a = errors(payload["coef_final"], payload["coef_ground_truth"])
        u = errors(payload["sol_final"], payload["sol_ground_truth"])
        assert a == r["errors"]["a"] and u == r["errors"]["u"]
        for n, sid in enumerate(job["sample_ids"]):
            records.append(dict(job_id=job["job_id"], family=job["family"], sample_id=sid,
                                rel_l2_a=a[n], rel_l2_u=u[n], seconds=r["seconds"], nfe=r["nfe"]))
    assert len(records) == protocol["expected_predictions"]
    target = args.output / "audit"
    target.mkdir(exist_ok=True)
    with (target / "per_sample.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    atomic_json(target / "audit.json", {"status": "pass", "complete": True,
                "jobs": len(protocol["jobs"]), "predictions": len(records),
                "protocol_sha256": file_hash(args.protocol), "model_sha256": protocol["model_sha256"],
                "per_sample_sha256": file_hash(target / "per_sample.csv"), "finished_unix": time.time()})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["pilot", "run", "audit"])
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--worker", default="g0")
    p.add_argument("--pilot-certificate", type=Path)
    p.add_argument("--job-ids", nargs="+")
    args = p.parse_args()
    args.protocol, args.output = args.protocol.resolve(), args.output.resolve()
    if args.mode == "audit":
        audit(args); return
    import torch
    from sampling.model_io import load_fm4pde_checkpoint_bundle
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    protocol = validate_protocol(args.protocol)
    assert args.checkpoint and file_hash(args.checkpoint) == protocol["model_sha256"]
    identity = core_identity()
    if args.mode == "run":
        assert args.pilot_certificate is not None
        cert = load_json(args.pilot_certificate)
        assert cert["status"] == "pass" and cert["protocol_sha256"] == file_hash(args.protocol)
        assert cert["code_identity"] == identity and cert["model_sha256"] == protocol["model_sha256"]
        assert cert["gpu"] == torch.cuda.get_device_name() and cert["torch"] == torch.__version__
        assert set(cert["jobs"]) == PILOT_JOB_IDS
    args.output.mkdir(parents=True, exist_ok=True)
    invocation = dict(host=socket.gethostname(), pid=os.getpid(), args={k:str(v) for k,v in vars(args).items()},
                      torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
                      tf32_matmul=False, tf32_cudnn=False, code_identity=identity,
                      protocol_sha256=file_hash(args.protocol), started_unix=time.time())
    atomic_json(args.output / "invocations" / f"{args.worker}_{uuid.uuid4().hex}.json", invocation)
    bundle = load_fm4pde_checkpoint_bundle(str(args.checkpoint), "nsnonbounded", args.device, model_profile="light")
    assert sum(p.numel() for p in bundle[0].model.parameters()) == PARAMETER_COUNT
    assert file_hash(args.checkpoint) == MODEL_SHA256
    assert json_hash(tree_hash(bundle[1].state_dict())) == protocol["normalizer_sha256"]
    jobs = protocol["jobs"]
    if args.job_ids:
        jobs = [j for j in jobs if j["job_id"] in args.job_ids]
        assert len(jobs) == len(args.job_ids)
    if args.mode == "pilot":
        assert set(args.job_ids or []) == PILOT_JOB_IDS, "The five declared stock pilot cases are required"
    free, _ = torch.cuda.mem_get_info()
    assert free > 12 * 2**30, ("Insufficient free GPU memory for the measured small-batch pilot", free)
    completed = []
    b1_peak = None
    state = args.output / "workers" / (args.worker + ".json")
    try:
        for job in jobs:
            stop = args.output / "workers" / (args.worker + ".STOP_AFTER_JOB")
            if stop.exists():
                atomic_json(state, dict(status="stopped", completed=completed, pid=os.getpid())); return
            lock_path = args.output / "jobs" / job["job_id"] / "job.lock"
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            with lock_path.open("a+") as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                atomic_json(state, dict(status="running", active_job=job["job_id"], completed=completed,
                                        pid=os.getpid(), updated_unix=time.time()))
                if args.mode == "pilot" and len(job["sample_ids"]) > 1:
                    free, _ = torch.cuda.mem_get_info()
                    assert b1_peak is not None and len(job["sample_ids"])*b1_peak < free*.8
                row = run_job(job, args, protocol, bundle, identity)
                if args.mode == "pilot" and len(job["sample_ids"]) == 1:
                    b1_peak = max(b1_peak or 0, row["peak_bytes"])
                completed.append(job["job_id"])
        atomic_json(state, dict(status="complete", completed=completed, pid=os.getpid(), finished_unix=time.time()))
        if args.mode == "pilot":
            atomic_json(args.output / "pilot_complete.json", dict(status="pass", jobs=completed,
                        protocol_sha256=file_hash(args.protocol), model_sha256=MODEL_SHA256,
                        code_identity=identity, torch=torch.__version__, gpu=torch.cuda.get_device_name()))
    except BaseException as exc:
        atomic_json(state, dict(status="error", error=repr(exc), completed=completed, pid=os.getpid(),
                                failed_unix=time.time()))
        raise


if __name__ == "__main__":
    main()
