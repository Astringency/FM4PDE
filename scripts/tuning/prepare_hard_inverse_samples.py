#!/usr/bin/env python3
"""Select hard inverse samples and materialize compact tuning/holdout datasets.

For every (test type, PDE) group, the script takes the 20 largest historical
coefficient errors. Odd hard ranks form the tuning split and even ranks form an
equally difficult holdout split. The resulting files contain ten samples each,
so all candidate parameters can be compared with the same batch shape.
"""

from __future__ import annotations

import argparse
import csv
import gc
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


TEST_TYPES = ("id", "smooth", "rough")
PDES = ("poisson", "helmholtz", "darcy", "nsnonbounded")
SCIPY_FIELDS = {
    "poisson": ("f_data", "phi_data"),
    "helmholtz": ("f_data", "psi_data"),
}
H5_FIELDS = {
    "darcy": ("thresh_a_data", "thresh_p_data"),
    "nsnonbounded": ("w0", "w"),
}


def _source_path(data_root: Path, pde: str, test_type: str) -> Path:
    if pde == "nsnonbounded":
        name = f"nsnonbounded_test_10000-128-128-10_{test_type}.mat"
    else:
        name = f"{pde}_test_10000-128-128_{test_type}.mat"
    return data_root / pde / name


def _finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Expected a finite number, got {value!r}")
    return number


def _read_main_inverse_rows(metrics_path: Path, pde: str) -> list[dict[str, str]]:
    """Return the unique 1000-sample main group, ignoring incidental debug runs."""
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("task") == "inverse" and row.get("pde") == pde:
                groups[row.get("ablation_group_key", "")].append(row)

    eligible = []
    for key, rows in groups.items():
        sample_ids = {int(float(row["sample_id"])) for row in rows}
        if len(rows) == 1000 and len(sample_ids) == 1000:
            eligible.append((key, rows))
    if len(eligible) != 1:
        details = [(key, len(rows), len({row.get('sample_id') for row in rows})) for key, rows in groups.items()]
        raise RuntimeError(
            f"Expected exactly one 1000-sample main group for {pde} in {metrics_path}; "
            f"found {len(eligible)}. Groups: {details}"
        )
    return eligible[0][1]


def select_hard_samples(
    results_root: Path,
    data_root: Path,
    output_root: Path,
    samples_per_group: int,
) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    split_counts: dict[tuple[str, str, str], int] = defaultdict(int)
    for test_type in TEST_TYPES:
        metrics_path = results_root / f"MAIN1000_100_TEST_{test_type}" / "metrics_per_sample_all.csv"
        if not metrics_path.is_file():
            raise FileNotFoundError(metrics_path)
        for pde in PDES:
            rows = _read_main_inverse_rows(metrics_path, pde)
            ranked = sorted(
                rows,
                key=lambda row: (-_finite_float(row["rel_l2_a"]), int(float(row["sample_id"]))),
            )[:samples_per_group]
            if len(ranked) != samples_per_group:
                raise RuntimeError(f"Only found {len(ranked)} rows for {pde}/{test_type}")
            source_path = _source_path(data_root, pde, test_type).resolve()
            if not source_path.is_file():
                raise FileNotFoundError(source_path)
            for hard_rank, row in enumerate(ranked, start=1):
                split = "tune" if hard_rank % 2 else "holdout"
                split_key = (pde, test_type, split)
                subset_index = split_counts[split_key]
                split_counts[split_key] += 1
                subset_path = (output_root / "subsets" / split / f"{pde}_{test_type}.mat").resolve()
                manifest.append(
                    {
                        "test_type": test_type,
                        "pde": pde,
                        "hard_rank": hard_rank,
                        "split": split,
                        "source_sample_id": int(float(row["sample_id"])),
                        "historical_sample_index": int(float(row["sample_index"])),
                        "subset_index": subset_index,
                        "historical_rel_l2_a": _finite_float(row["rel_l2_a"]),
                        "historical_rel_l2_u": _finite_float(row["rel_l2_u"]),
                        "historical_pde_residual_norm": _finite_float(row["pde_residual_norm"]),
                        "historical_run_dir": row.get("run_dir", ""),
                        "historical_zeta_obs_u": _finite_float(row["zeta_obs_u"]),
                        "historical_zeta_pde": _finite_float(row["zeta_pde"]),
                        "source_data_path": str(source_path),
                        "subset_path": str(subset_path),
                    }
                )
    expected = len(TEST_TYPES) * len(PDES) * samples_per_group
    if len(manifest) != expected:
        raise RuntimeError(f"Expected {expected} manifest rows, got {len(manifest)}")
    return manifest


def _copy_scipy_subset(
    source_path: Path,
    pde: str,
    entries: list[dict[str, Any]],
    output_root: Path,
) -> None:
    import numpy as np
    from scipy.io import loadmat, savemat, whosmat

    coef_name, sol_name = SCIPY_FIELDS[pde]
    source_ids = [int(entry["source_sample_id"]) for entry in entries]
    print(f"READ {pde}/{entries[0]['test_type']} ground truth artifacts for {len(source_ids)} hard samples", flush=True)
    coef, sol = _load_historical_ground_truth(entries)
    selected = {coef_name: coef[:, 0], sol_name: sol[:, 0]}

    metadata_names = [name for name, _shape, _kind in whosmat(source_path) if name not in {coef_name, sol_name}]
    metadata = loadmat(source_path, variable_names=metadata_names)
    metadata = {key: value for key, value in metadata.items() if not key.startswith("__")}

    for split in ("tune", "holdout"):
        split_positions = [idx for idx, entry in enumerate(entries) if entry["split"] == split]
        split_entries = [entries[idx] for idx in split_positions]
        payload = dict(metadata)
        payload[coef_name] = selected[coef_name][split_positions]
        payload[sol_name] = selected[sol_name][split_positions]
        payload["source_sample_id"] = np.asarray(
            [entry["source_sample_id"] for entry in split_entries], dtype=np.int64
        ).reshape(-1, 1)
        output_path = output_root / "subsets" / split / f"{pde}_{entries[0]['test_type']}.mat"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        savemat(output_path, payload, do_compression=False)
        print(f"WRITE {output_path} ({len(split_entries)} samples)", flush=True)
    del selected
    gc.collect()


def _load_historical_ground_truth(entries: list[dict[str, Any]]) -> tuple[Any, Any]:
    """Recover exact source samples from compact artifacts of the main runs."""
    import numpy as np
    import torch

    by_run: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for position, entry in enumerate(entries):
        by_run[str(entry["historical_run_dir"])].append(
            (position, int(entry["historical_sample_index"]))
        )
    coef_samples: list[Any | None] = [None] * len(entries)
    sol_samples: list[Any | None] = [None] * len(entries)
    for run_dir, requested in by_run.items():
        result_path = Path(run_dir) / "result.pt"
        if not result_path.is_file():
            raise FileNotFoundError(result_path)
        result = torch.load(result_path, map_location="cpu", weights_only=False)
        coef_batch = result["coef_ground_truth"]
        sol_batch = result["sol_ground_truth"]
        for position, sample_index in requested:
            coef_samples[position] = coef_batch[sample_index].numpy().copy()
            sol_samples[position] = sol_batch[sample_index].numpy().copy()
        del result, coef_batch, sol_batch
    if any(value is None for value in coef_samples + sol_samples):
        raise RuntimeError("Failed to recover one or more historical ground-truth samples")
    return np.stack(coef_samples), np.stack(sol_samples)


def _copy_h5_subset(
    source_path: Path,
    pde: str,
    entries: list[dict[str, Any]],
    output_root: Path,
) -> None:
    import h5py
    import numpy as np

    coef_name, sol_name = H5_FIELDS[pde]
    print(f"READ {pde}/{entries[0]['test_type']} ground truth artifacts for {len(entries)} hard samples", flush=True)
    coef, sol = _load_historical_ground_truth(entries)
    with h5py.File(source_path, "r") as source:
        for split in ("tune", "holdout"):
            split_entries = [entry for entry in entries if entry["split"] == split]
            split_positions = [idx for idx, entry in enumerate(entries) if entry["split"] == split]
            source_ids = [int(entry["source_sample_id"]) for entry in split_entries]
            output_path = output_root / "subsets" / split / f"{pde}_{entries[0]['test_type']}.mat"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with h5py.File(output_path, "w") as target:
                for key, value in source.attrs.items():
                    target.attrs[key] = value
                target.attrs["n_samples"] = len(source_ids)
                target.attrs["hard_sample_source"] = str(source_path)

                if pde == "darcy":
                    target.create_dataset(
                        coef_name,
                        data=np.transpose(coef[split_positions, 0], (1, 2, 0)),
                    )
                    target.create_dataset(
                        sol_name,
                        data=np.transpose(sol[split_positions, 0], (1, 2, 0)),
                    )
                    for name in ("dataset_type", "generation_seed", "grf_alpha", "grf_tau", "shard_id"):
                        if name in source:
                            target.create_dataset(name, data=source[name][()])
                else:
                    target.create_dataset(coef_name, data=coef[split_positions, 0])
                    # The endpoint-secant inverse configuration only reads the final
                    # frame, so one saved endpoint frame is sufficient here.
                    target.create_dataset(sol_name, data=sol[split_positions, 0, :, :, None])
                    if "t" in source:
                        target.create_dataset("t", data=source["t"][()])
                    if "sample_seed" in source:
                        target.create_dataset(
                            "sample_seed",
                            data=np.asarray([source["sample_seed"][sample_id] for sample_id in source_ids]),
                        )
                target.create_dataset("source_sample_id", data=np.asarray(source_ids, dtype=np.int64))
            print(f"WRITE {output_path} ({len(source_ids)} samples)", flush=True)


def materialize_subsets(manifest: list[dict[str, Any]], output_root: Path) -> None:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for entry in manifest:
        grouped[(str(entry["pde"]), str(entry["test_type"]))].append(entry)
    for pde in PDES:
        for test_type in TEST_TYPES:
            entries = sorted(grouped[(pde, test_type)], key=lambda item: int(item["hard_rank"]))
            source_path = Path(str(entries[0]["source_data_path"]))
            if pde in SCIPY_FIELDS:
                _copy_scipy_subset(source_path, pde, entries, output_root)
            else:
                _copy_h5_subset(source_path, pde, entries, output_root)


def write_manifest(manifest: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=Path("outputs"))
    parser.add_argument("--data-root", type=Path, default=Path("/home/tat512/share/PDEdata"))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/inverse_hard_tuning"))
    parser.add_argument("--samples-per-group", type=int, default=20)
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args()
    if args.samples_per_group < 2 or args.samples_per_group % 2:
        parser.error("--samples-per-group must be an even integer >= 2")

    output_root = args.output_root.resolve()
    manifest = select_hard_samples(
        args.results_root.resolve(), args.data_root.resolve(), output_root, args.samples_per_group
    )
    manifest_path = output_root / "hard_samples.csv"
    write_manifest(manifest, manifest_path)
    print(f"WRITE {manifest_path} ({len(manifest)} rows)", flush=True)
    if not args.manifest_only:
        materialize_subsets(manifest, output_root)


if __name__ == "__main__":
    main()
