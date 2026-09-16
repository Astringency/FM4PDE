"""Recompute final physical PDE scores on CPU from saved trace predictions."""

import argparse
import csv
import dataclasses
import hashlib
import json
import math
from pathlib import Path
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--pde", required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    info = manifest["cells"][args.pde]
    sys.path.insert(0, str(root / info["fm_code"]))
    import numpy as np
    import torch
    from sampling.config import AblationConfig
    from sampling.data import PDEGroundTruth
    from sampling.losses import compute_guidance_losses
    from sampling.masks import PairMasks
    from sampling.state import SplitState

    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    inp = root / info["input_root"]
    protocol = json.loads((inp / "protocol.json").read_text())
    truth = np.load(inp / "truths.npz")
    masks = np.load(inp / "masks.npz")
    records = []
    missing = []
    for method in ["FM4PDE", "DiffusionPDE"]:
        for sid in info["evaluation_ids"]:
            stem = root / "traces" / args.pde / f"{method}_1000_{sid}"
            receipt_path = stem.with_suffix(".json")
            if not receipt_path.exists():
                missing.append(f"{method}/{sid}")
                continue
            receipt = json.loads(receipt_path.read_text())
            assert sha(stem.with_suffix(".pt")) == receipt["prediction_sha256"]
            assert sha(stem.with_suffix(".csv")) == receipt["trace_sha256"]
            prediction = torch.load(
                stem.with_suffix(".pt"), map_location="cpu", weights_only=False
            )
            index = protocol["evaluation_ids"].index(sid)
            a = torch.from_numpy(truth[args.pde + "_a"][index : index + 1])
            u = torch.from_numpy(truth[args.pde + "_u"][index : index + 1])
            ma = torch.from_numpy(masks[f"{args.pde}_{sid}_a"])[None, None]
            mu = torch.from_numpy(masks[f"{args.pde}_{sid}_u"])[None, None]
            gt = PDEGroundTruth(
                args.pde,
                a,
                u,
                a if args.pde == "burger" else torch.cat([a, u], 1),
                {},
                ["a"],
                ["u"],
                {"synthetic": False},
            )
            saved_config = receipt["effective_fm_config"]
            fields = {f.name: f for f in dataclasses.fields(AblationConfig)}
            assert set(saved_config) <= set(fields)
            cfg = AblationConfig(
                **{k: v for k, v in saved_config.items() if fields[k].init}
            )
            for key, value in saved_config.items():
                if not fields[key].init:
                    setattr(cfg, key, value)
            cfg.device = "cpu"
            cfg.guidance_components = "obs_pde"
            cfg.zeta_pde = 1.0
            cfg.pde_guidance_reduction = "mse"
            cfg.obs_guidance_reduction = "mse"
            cfg.pde_residual_region = "full"
            with torch.no_grad():
                output = compute_guidance_losses(
                    SplitState(prediction["coef"], prediction["sol"]),
                    gt,
                    PairMasks(ma, mu, {}),
                    cfg,
                )
                double_output = compute_guidance_losses(
                    SplitState(prediction["coef"].double(), prediction["sol"].double()),
                    gt,
                    PairMasks(ma, mu, {}),
                    cfg,
                )
            value = float(output.L_pde)
            double_value = float(double_output.L_pde)
            rows = list(csv.DictReader(stem.with_suffix(".csv").open()))
            saved = float(rows[-1]["L_pde"])
            assert int(rows[-1]["step"]) == 1000
            assert (
                math.isfinite(value)
                and math.isfinite(saved)
                and math.isfinite(double_value)
            )
            delta = abs(value - saved)
            # Same stored fields and operator; CPU/GPU stencil and FFT rounding
            # may differ. This threshold is only a numerical consistency check.
            # Float32 spatial/time derivatives subtract nearly equal quantities
            # before division by grid spacing. Keep every measured discrepancy
            # and a float64 reference instead of requiring CPU/GPU bit equality.
            consistent = math.isclose(value, saved, rel_tol=1e-3, abs_tol=1e-7)
            records.append(
                {
                    "method": method,
                    "sample_id": sid,
                    "saved_gpu_L_pde": saved,
                    "recomputed_cpu_L_pde": value,
                    "absolute_difference": delta,
                    "relative_difference": delta / max(abs(saved), 1e-30),
                    "recomputed_cpu_float64_L_pde": double_value,
                    "consistent": consistent,
                    "prediction_dtype": str(prediction["coef"].dtype),
                    "prediction_sha256": receipt["prediction_sha256"],
                }
            )
    if missing and not args.allow_partial:
        raise ValueError(f"Missing {len(missing)}/40 traces for {args.pde}")
    failures = [r for r in records if not r["consistent"]]
    report = {
        "status": "error" if failures else "partial" if missing else "pass",
        "pde": args.pde,
        "manifest_sha256": sha(root / "manifest.json"),
        "operator_commit": info["fm_commit"],
        "operator_file_sha256": sha(
            root / info["fm_code"] / "sampling/pde_residuals.py"
        ),
        "loss_file_sha256": sha(root / info["fm_code"] / "sampling/losses.py"),
        "scope": "Final physical PDE MSE independently recomputed on CPU in the saved prediction dtype; no model inference.",
        "comparison_relative_tolerance": 1e-3,
        "comparison_absolute_tolerance": 1e-7,
        "records": records,
        "missing": missing,
        "failed_calls": len(failures),
    }
    out = root / "audit"
    out.mkdir(exist_ok=True)
    (out / f"{args.pde}_final_residual_audit.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(
        args.pde,
        report["status"],
        len(records),
        "failures",
        len(failures),
        "max_relative_difference",
        max((r["relative_difference"] for r in records), default=None),
    )
    if failures:
        raise AssertionError(
            f"{len(failures)} independently recomputed PDE scores differ"
        )


if __name__ == "__main__":
    main()
