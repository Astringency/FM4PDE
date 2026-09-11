"""Replay published layouts and inject the regional Fixed control, in isolation."""
from __future__ import annotations

import argparse
import copy
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
from contextlib import redirect_stdout, redirect_stderr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from plot.run_paper_ablation_revision import digest, write, physical_errors


def regional_errors(pred, truth, mask):
    import torch
    pred, truth = pred.double(), truth.double()
    regions = {"full": torch.ones_like(truth), "observed": mask.double()}
    regions["left"] = torch.zeros_like(truth)
    regions["left"][..., :truth.shape[-1] // 2] = 1
    regions["right"] = 1 - regions["left"]
    out = {}
    for name, region in regions.items():
        num = torch.linalg.vector_norm(((pred - truth) * region).flatten(1), dim=1)
        den = torch.linalg.vector_norm((truth * region).flatten(1), dim=1).clamp_min(1e-12)
        out[name] = (num / den).tolist()
    return out


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pdes", nargs="+", required=True)
    parser.add_argument("--variants", nargs="+", default=["random_replay", "legacy_fixed_replay", "fixed_left_half"])
    args = parser.parse_args()
    assert args.output.is_absolute()
    import torch
    from sampling.config import AblationConfig
    from sampling.model_io import load_fm4pde_checkpoint_bundle
    from sampling.masks import make_pair_masks
    from sampling.runner import run_single_ablation
    from scripts.tuning.compare_pde_guidance_schedules import combine_truths
    from experiments.fixed_region import left_half_pair_masks

    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    plan = json.loads(args.plan.read_text())
    plan_sha = digest(args.plan)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    for pde in args.pdes:
        source = args.inputs / pde
        target = args.output / "fixed_region" / pde
        target.mkdir(parents=True, exist_ok=True)
        with (target / "run.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            protocol = json.loads((source / "protocol.json").read_text())
            assert digest(source / "weights.pth") == protocol["weights_sha256"]
            assert digest(source / "truths.pt") == protocol["truth_sha256"]
            write(target / "environment.json", dict(commit=commit, plan_sha256=plan_sha,
                input_protocol_sha256=digest(source / "protocol.json"),
                weights_sha256=protocol["weights_sha256"], truth_sha256=protocol["truth_sha256"],
                python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(), visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                host=socket.gethostname(), pid=os.getpid(), started_unix=time.time(), tf32=False))
            truths = torch.load(source / "truths.pt", map_location="cpu", weights_only=False)
            gt = combine_truths(truths, [0], "cuda:0")
            bundle = load_fm4pde_checkpoint_bundle(str(source / "weights.pth"), pde, "cuda:0", model_profile="recommended")
            calls = [0]
            def hook(*_):
                calls[0] += 1
            handle = bundle[0].model.register_forward_hook(hook)
            for variant in args.variants:
                folder = target / variant
                receipt = folder / "receipt.json"
                if receipt.exists():
                    prior = json.loads(receipt.read_text())
                    assert prior["plan_sha256"] == plan_sha
                    assert Path(prior["result_path"]).exists()
                    print("EXISTS", pde, variant, flush=True)
                    continue
                folder.mkdir(parents=True, exist_ok=True)
                original = plan["pdes"][pde]["random" if variant == "random_replay" else "fixed"]
                conf = copy.deepcopy(original["config"])
                conf.update(checkpoint_path=str(source / "weights.pth"), output_dir=str(folder),
                    device="cuda:0", batch_size=1, offset=0, save_plots=False, save_intermediate=False,
                    save_per_sample_curves=True, allow_synthetic_data=False,
                    ablation_name="fixed_region_exact_tilt_20260911")
                if variant == "fixed_left_half":
                    conf["shared_mask"] = True
                    masks = left_half_pair_masks(gt.coef.shape, gt.sol.shape, conf["num_obs"],
                                                 conf["mask_seed"], device="cuda:0")
                else:
                    assert variant in {"random_replay", "legacy_fixed_replay"}
                    masks = make_pair_masks(gt.coef.shape, gt.sol.shape, conf["num_obs"], conf["sensor_mode"],
                        conf["shared_mask"], conf["mask_seed"], device="cuda:0",
                        num_sensor_columns=conf["num_sensor_columns"])
                cfg = AblationConfig(**conf)
                cfg.validate()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                calls[0] = 0
                start = time.perf_counter()
                with (folder / "run.log").open("w") as log, redirect_stdout(log), redirect_stderr(log):
                    result = run_single_ablation(cfg, checkpoint_bundle=bundle, ground_truth=gt,
                                                 observation_masks=masks)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                result_path = Path(result["run_dir"]) / "result.pt"
                payload = torch.load(result_path, map_location="cpu", weights_only=False)
                saved_masks = torch.load(result_path.parent / "masks.pt", map_location="cpu", weights_only=False)
                errors = {}
                for field, pred, truth, maskkey in [("a", "coef_final", "coef_ground_truth", "coef"),
                                                   ("u", "sol_final", "sol_ground_truth", "sol")]:
                    assert torch.equal(payload[truth], getattr(truths[0], maskkey))
                    errors[field] = regional_errors(payload[pred], payload[truth], saved_masks[maskkey])
                assert all(math.isfinite(v) for fields in errors.values() for vals in fields.values() for v in vals)
                row = dict(pde=pde, variant=variant, sample_ids=[0], config=conf, source_id=original["id"],
                    plan_sha256=plan_sha, commit=commit, input_protocol_sha256=digest(source / "protocol.json"),
                    result_path=str(result_path), result_sha256=digest(result_path),
                    mask_file_sha256=digest(result_path.parent / "masks.pt"), mask_metadata=saved_masks.get("metadata"),
                    errors=errors, status=result["status"], seconds=elapsed, nfe=calls[0],
                    peak_bytes=torch.cuda.max_memory_allocated(),
                    archive_errors={f: original["rel_l2_" + f] for f in ["a", "u"]})
                write(receipt, row)
                print("DONE", pde, variant, f"{elapsed:.2f}s", errors, flush=True)
            handle.remove()
            del bundle, truths, gt, masks, payload
            torch.cuda.empty_cache()
            write(target / "complete.json", dict(plan_sha256=plan_sha, variants=args.variants,
                                                 completed_unix=time.time()))


if __name__ == "__main__":
    main()
