#!/usr/bin/env python3
"""Freeze selected historical test slices; never write source data or train models.

The native loader performs its original float32 conversion. Independent NumPy
reads verify that conversion and axis selection, rather than claiming that the
original float64 MAT arrays are preserved bit for bit in a float32 cache.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def git_commit(path):
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def clone_tree(value):
    import numpy as np
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().contiguous().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {k: clone_tree(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(clone_tree(v) for v in value)
    return copy.deepcopy(value)


def tensors(value, prefix=""):
    import numpy as np
    import torch
    if isinstance(value, (torch.Tensor, np.ndarray)):
        array = value.detach().cpu().contiguous().numpy() if isinstance(value, torch.Tensor) else np.ascontiguousarray(value)
        return {prefix: {"shape": list(array.shape), "dtype": str(value.dtype),
                         "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest()}}
    result = {}
    if isinstance(value, dict):
        for key, child in value.items():
            result.update(tensors(child, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(value, (tuple, list)):
        for key, child in enumerate(value):
            result.update(tensors(child, f"{prefix}.{key}"))
    return result


def identical(a, b):
    import numpy as np
    import torch
    if type(a) is not type(b):
        return False
    if isinstance(a, torch.Tensor):
        return a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return a.dtype == b.dtype and np.array_equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(identical(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(identical(x, y) for x, y in zip(a, b))
    return a == b


def independently_read_source(source, pde, full, n):
    """Separate source slice/axis implementation; do not call native helpers."""
    import h5py
    import numpy as np
    import scipy.io
    descriptions = {}
    initials = None
    if pde == "darcy":
        keys = ("thresh_a_data", "thresh_p_data")
        with h5py.File(source, "r") as handle:
            arrays = []
            for key in keys:
                ds = handle[key]
                assert ds.shape == (128, 128, 10000), (key, ds.shape)
                arrays.append(np.moveaxis(ds[:, :, :n], -1, 0).astype(np.float32))
                descriptions[key] = {"original_shape": list(ds.shape), "original_dtype": str(ds.dtype),
                                     "slice": ":,:,0:1000", "axes": "H,W,N -> N,H,W"}
        value = np.stack(arrays, axis=1)
    elif pde == "nsnonbounded":
        with h5py.File(source, "r") as handle:
            assert handle["w0"].shape == (10000, 128, 128)
            assert handle["w"].shape == (10000, 128, 128, 10)
            w0 = handle["w0"][:n].astype(np.float32)
            if full:
                wt = handle["w"][:n].astype(np.float32)
                value = np.concatenate((w0[:, None], wt.transpose(0, 3, 1, 2)), axis=1)[:, None]
            else:
                wt = handle["w"][:n, :, :, -1].astype(np.float32)
                value = np.stack((w0, wt), axis=1)
            for key in ("w0", "w"):
                ds = handle[key]
                descriptions[key] = {"original_shape": list(ds.shape), "original_dtype": str(ds.dtype),
                    "slice": "0:1000,:,:,:" if full and key == "w" else "0:1000,:,:,-1" if key == "w" else "0:1000,:,:",
                    "axes": "N,H,W,T -> N,1,T+1,H,W with initial frame" if full else "N,2,H,W (initial, final)"}
            if "t" in handle:
                descriptions["t"] = {"values": handle["t"][:].tolist(), "original_dtype": str(handle["t"].dtype)}
    else:
        keys = {"poisson": ("f_data", "phi_data"), "helmholtz": ("f_data", "psi_data"),
                "burger": ("input", "output")}[pde]
        raw = scipy.io.loadmat(source, variable_names=list(keys))
        for key in keys:
            array = raw[key]
            descriptions[key] = {"original_shape": list(array.shape), "original_dtype": str(array.dtype),
                                 "slice": "0:1000,...", "axes": "unchanged"}
        if pde == "burger":
            value = raw["output"][:n].astype(np.float32)[:, None]
            initials = raw["input"][:n].astype(np.float32)
        else:
            value = np.stack([raw[key][:n].astype(np.float32) for key in keys], axis=1)
    return value, initials, descriptions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--baseline-code", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    if args.output.exists():
        raise FileExistsError(f"Refusing to replace existing output: {args.output}")
    args.output.mkdir(parents=True)
    sys.path.insert(0, str(args.baseline_code.resolve()))
    from baselines.common.data_adapter import build_default_registry
    registry = build_default_registry()
    plan = json.loads(args.plan.read_text())
    assert plan["n_per_cache"] == 1000
    loader_commit = git_commit(args.baseline_code)
    loader_source = args.baseline_code / "baselines/common/data_adapter.py"
    report = {"schema_version": "baseline-frozen-input-manifest-v1", "status": "running",
              "plan_sha256": sha256_file(args.plan), "loader_commit": loader_commit,
              "loader_source_sha256": sha256_file(loader_source), "freeze_commit": git_commit(Path(__file__).parent),
              "freeze_script_sha256": sha256_file(__file__), "entries": [], "source_files": {}}
    started = time.time()
    for entry in plan["entries"]:
        source = Path(entry["data_root"]) / entry["loader_options"]["data_files"]["test"][0]
        before = (source.stat().st_size, source.stat().st_mtime_ns)
        if str(source) not in report["source_files"]:
            print(f"Hashing source {source}", flush=True)
            report["source_files"][str(source)] = {"sha256": sha256_file(source), "bytes": before[0], "mtime_ns": before[1]}
        source_info = report["source_files"][str(source)]
        assert before == (source_info["bytes"], source_info["mtime_ns"])
        print(f"Freezing {entry['id']}", flush=True)
        raw = registry.load_raw(entry["pde"], Path(entry["data_root"]), **entry["loader_options"])
        assert raw["full_tensor"].shape[0] == 1000
        assert torch.equal(raw["sample_indices"], torch.arange(1000))
        assert raw["global_sample_ids"] == [f"{source.name}:{i}" for i in range(1000)]
        assert raw["split"] == "test"
        expected, initials, descriptions = independently_read_source(source, entry["pde"], entry["load_full_trajectory"], 1000)
        assert np.array_equal(raw["full_tensor"].numpy(), expected), entry["id"]
        if initials is not None:
            assert np.array_equal(raw["metadata"]["initial_1d"].numpy(), initials)
        del expected, initials
        shared_flag_check = None
        if len(entry["supported_load_full_trajectory"]) == 2:
            options = dict(entry["loader_options"], load_full_trajectory=True)
            alternate = registry.load_raw(entry["pde"], Path(entry["data_root"]), **options)
            shared_flag_check = identical(raw, alternate)
            assert shared_flag_check, entry["id"]
            del alternate
        raw = clone_tree(raw)
        provenance = {"source_mat_path": str(source), "source_mat_sha256": source_info["sha256"],
                      "source_indices": list(range(1000)), "source_arrays": descriptions,
                      "loader_commit": loader_commit, "loader_source_sha256": report["loader_source_sha256"],
                      "original_loader_commits": entry["original_loader_commits"],
                      "supported_load_full_trajectory": entry["supported_load_full_trajectory"],
                      "native_conversion": "Original loader converts numerical field arrays to torch.float32.",
                      "loader_options": entry["loader_options"],
                      "selected_run_bindings": entry["selected_runs"]}
        payload = {"schema_version": "baseline-frozen-raw-v1", "pde": entry["pde"], "dist": entry["dist"],
                   "load_full_trajectory": entry["load_full_trajectory"], "raw": raw, "provenance": provenance}
        cache = args.output / f"{entry['id']}.pt"
        tensor_manifest = tensors(raw)
        torch.save(payload, cache)
        reloaded = torch.load(cache, map_location="cpu", weights_only=False)
        assert identical(payload, reloaded)
        assert tensor_manifest == tensors(reloaded["raw"])
        assert before == (source.stat().st_size, source.stat().st_mtime_ns), "source changed during freeze"
        row = {"id": entry["id"], "cache_path": cache.name, "cache_sha256": sha256_file(cache),
               "cache_bytes": cache.stat().st_size, "pde": entry["pde"], "dist": entry["dist"],
               "load_full_trajectory": entry["load_full_trajectory"],
               "supported_load_full_trajectory": entry["supported_load_full_trajectory"],
               "n": 1000, "tensors": tensor_manifest, "source_mat_path": str(source),
               "source_mat_sha256": source_info["sha256"], "independent_source_slice_equal": True,
               "save_reload_exact": True, "static_flags_raw_tree_identical": shared_flag_check,
               "selected_run_count": len(entry["selected_runs"])}
        report["entries"].append(row)
        (args.output / "manifest.in_progress.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"PASS {entry['id']}: {row['cache_bytes']} bytes", flush=True)
        del raw, payload, reloaded
        gc.collect()
    report.update(status="pass", complete=True, elapsed_seconds=time.time() - started,
                  total_cache_bytes=sum(row["cache_bytes"] for row in report["entries"]),
                  selected_run_count=sum(row["selected_run_count"] for row in report["entries"]))
    assert len(report["entries"]) == 18 and report["selected_run_count"] == 186
    (args.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "complete", "elapsed_seconds", "total_cache_bytes", "selected_run_count")}), flush=True)


if __name__ == "__main__":
    main()
