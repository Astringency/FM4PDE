#!/usr/bin/env python3
"""Prepare hard, good, and representative samples for balanced tuning.

The historical 1,000-sample main runs are used only to rank/select samples.
Every (PDE, task, test_type) group contributes:

* 20 hard samples (largest task-specific error),
* 20 good samples (smallest task-specific error), and
* 40 deterministic random samples from the remaining population.

Odd/even ranks form equally sized tune/holdout splits.  The first 20 records of
each split deliberately contain 10 hard + 5 good + 5 random samples, giving the
screening stage more tail power.  All 40 records contain 10 hard + 10 good +
20 random samples for the final paired holdout.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.tuning.prepare_hard_inverse_samples import (
    H5_FIELDS,
    SCIPY_FIELDS,
    _load_historical_ground_truth,
    _source_path,
)


TEST_TYPES = ("id", "smooth", "rough")
PDES = ("poisson", "helmholtz", "darcy", "nsnonbounded")
TASKS = ("both", "forward", "inverse")
STRATA = ("hard", "good", "random")


def _finite(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Expected a finite number, got {value!r}")
    return number


def _read_formal_groups(
    metrics_path: Path,
) -> dict[tuple[str, str], list[dict[str, str]]]:
    """Read one result CSV once and resolve every complete formal group."""
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            pde = row.get("pde", "")
            task = row.get("task", "")
            if pde in PDES and task in TASKS:
                groups[(pde, task, row.get("ablation_group_key", ""))].append(row)

    resolved: dict[tuple[str, str], list[dict[str, str]]] = {}
    for pde in PDES:
        for task in TASKS:
            candidates = [
                (key, rows)
                for (gpde, gtask, key), rows in groups.items()
                if (gpde, gtask) == (pde, task)
            ]
            eligible: list[tuple[str, list[dict[str, str]]]] = []
            for key, rows in candidates:
                sample_ids = {int(float(row["sample_id"])) for row in rows}
                if len(rows) != 1000 or len(sample_ids) != 1000:
                    continue
                if all(
                    math.isfinite(float(row[metric]))
                    for row in rows
                    for metric in ("rel_l2_a", "rel_l2_u", "pde_residual_norm")
                ):
                    eligible.append((key, rows))
            if len(eligible) != 1:
                details = [
                    (key, len(rows), len({row.get("sample_id") for row in rows}))
                    for key, rows in candidates
                ]
                raise RuntimeError(
                    f"Expected exactly one complete formal group for {pde}/{task} in "
                    f"{metrics_path}; found {len(eligible)}. Groups: {details}"
                )
            resolved[(pde, task)] = eligible[0][1]
    return resolved


def _selection_scores(rows: list[dict[str, str]], task: str) -> list[float]:
    if task == "forward":
        return [_finite(row["rel_l2_u"]) for row in rows]
    if task == "inverse":
        return [_finite(row["rel_l2_a"]) for row in rows]
    median_a = statistics.median(_finite(row["rel_l2_a"]) for row in rows)
    median_u = statistics.median(_finite(row["rel_l2_u"]) for row in rows)
    if median_a <= 0.0 or median_u <= 0.0:
        raise RuntimeError("Both-task selection requires positive median errors")
    return [
        max(_finite(row["rel_l2_a"]) / median_a, _finite(row["rel_l2_u"]) / median_u)
        for row in rows
    ]


def _stable_random_key(pde: str, task: str, test_type: str, sample_id: int) -> str:
    payload = f"balanced-sampling-v1|{pde}|{task}|{test_type}|{sample_id}".encode()
    return hashlib.sha256(payload).hexdigest()


def _select_strata(
    rows: list[dict[str, str]],
    *,
    pde: str,
    task: str,
    test_type: str,
    tail_count: int,
    random_count: int,
) -> list[tuple[str, int, dict[str, str], float]]:
    scores = _selection_scores(rows, task)
    ranked = sorted(
        zip(rows, scores),
        key=lambda item: (item[1], int(float(item[0]["sample_id"]))),
    )
    good = ranked[:tail_count]
    hard = list(reversed(ranked[-tail_count:]))
    excluded = {int(float(row["sample_id"])) for row, _score in good + hard}
    remaining = [
        (row, score)
        for row, score in ranked
        if int(float(row["sample_id"])) not in excluded
    ]
    random_rows = sorted(
        remaining,
        key=lambda item: _stable_random_key(
            pde, task, test_type, int(float(item[0]["sample_id"]))
        ),
    )[:random_count]
    selected: list[tuple[str, int, dict[str, str], float]] = []
    for stratum, values in (("hard", hard), ("good", good), ("random", random_rows)):
        selected.extend(
            (stratum, rank, row, score)
            for rank, (row, score) in enumerate(values, start=1)
        )
    expected = 2 * tail_count + random_count
    if len(selected) != expected or len({int(float(item[2]["sample_id"])) for item in selected}) != expected:
        raise RuntimeError(f"Overlapping or incomplete strata for {pde}/{task}/{test_type}")
    return selected


def select_samples(
    results_root: Path,
    data_root: Path,
    output_root: Path,
    *,
    tail_count: int,
    random_count: int,
) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    for test_type in TEST_TYPES:
        metrics_path = results_root / f"MAIN1000_100_TEST_{test_type}" / "metrics_per_sample_all.csv"
        if not metrics_path.is_file():
            raise FileNotFoundError(metrics_path)
        formal_groups = _read_formal_groups(metrics_path)
        for pde in PDES:
            source_path = _source_path(data_root, pde, test_type).resolve()
            if not source_path.is_file():
                raise FileNotFoundError(source_path)
            for task in TASKS:
                rows = formal_groups[(pde, task)]
                selected = _select_strata(
                    rows,
                    pde=pde,
                    task=task,
                    test_type=test_type,
                    tail_count=tail_count,
                    random_count=random_count,
                )
                group_entries: list[dict[str, Any]] = []
                for stratum, stratum_rank, row, score in selected:
                    split = "tune" if stratum_rank % 2 else "holdout"
                    group_entries.append(
                        {
                            "test_type": test_type,
                            "pde": pde,
                            "task": task,
                            "stratum": stratum,
                            "stratum_rank": stratum_rank,
                            "stratum_split_index": (stratum_rank - 1) // 2,
                            "split": split,
                            "source_sample_id": int(float(row["sample_id"])),
                            "historical_sample_index": int(float(row["sample_index"])),
                            "selection_score": score,
                            "historical_rel_l2_a": _finite(row["rel_l2_a"]),
                            "historical_rel_l2_u": _finite(row["rel_l2_u"]),
                            "historical_pde_residual_norm": _finite(row["pde_residual_norm"]),
                            "historical_run_dir": row.get("run_dir", ""),
                            "historical_zeta_obs_a": _finite(row["zeta_obs_a"]),
                            "historical_zeta_obs_u": _finite(row["zeta_obs_u"]),
                            "historical_zeta_pde": _finite(row["zeta_pde"]),
                            "historical_clip_threshold": _finite(row["clip_threshold"]),
                            "source_data_path": str(source_path),
                            "subset_path": str(
                                (
                                    output_root
                                    / "subsets"
                                    / split
                                    / f"{pde}_{task}_{test_type}.mat"
                                ).resolve()
                            ),
                        }
                    )

                for split in ("tune", "holdout"):
                    split_entries = [entry for entry in group_entries if entry["split"] == split]
                    by_stratum = {
                        stratum: sorted(
                            (entry for entry in split_entries if entry["stratum"] == stratum),
                            key=lambda entry: int(entry["stratum_split_index"]),
                        )
                        for stratum in STRATA
                    }
                    split_tail_count = tail_count // 2
                    protection_prefix = split_tail_count // 2
                    # Tail-focused 20-sample prefix, then the remaining
                    # protection/representative records.  With the defaults:
                    # 10 hard + 5 good + 5 random, followed by 5 good + 15 random.
                    ordered = [
                        *by_stratum["hard"],
                        *by_stratum["good"][:protection_prefix],
                        *by_stratum["random"][:protection_prefix],
                        *by_stratum["good"][protection_prefix:],
                        *by_stratum["random"][protection_prefix:],
                    ]
                    for subset_index, entry in enumerate(ordered):
                        entry["subset_index"] = subset_index
                manifest.extend(group_entries)
    return manifest


def _copy_scipy_group(entries: list[dict[str, Any]], output_root: Path) -> None:
    import numpy as np
    from scipy.io import loadmat, savemat, whosmat

    pde = str(entries[0]["pde"])
    task = str(entries[0]["task"])
    test_type = str(entries[0]["test_type"])
    source_path = Path(str(entries[0]["source_data_path"]))
    coef_name, sol_name = SCIPY_FIELDS[pde]
    coef, sol = _load_historical_ground_truth(entries)
    metadata_names = [
        name for name, _shape, _kind in whosmat(source_path) if name not in {coef_name, sol_name}
    ]
    metadata = {
        key: value
        for key, value in loadmat(source_path, variable_names=metadata_names).items()
        if not key.startswith("__")
    }
    for split in ("tune", "holdout"):
        indexed = sorted(
            ((idx, entry) for idx, entry in enumerate(entries) if entry["split"] == split),
            key=lambda item: int(item[1]["subset_index"]),
        )
        positions = [idx for idx, _entry in indexed]
        split_entries = [entry for _idx, entry in indexed]
        payload = dict(metadata)
        payload[coef_name] = coef[positions, 0]
        payload[sol_name] = sol[positions, 0]
        payload["source_sample_id"] = np.asarray(
            [entry["source_sample_id"] for entry in split_entries], dtype=np.int64
        ).reshape(-1, 1)
        output_path = output_root / "subsets" / split / f"{pde}_{task}_{test_type}.mat"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        savemat(output_path, payload, do_compression=False)
        print(f"WRITE {output_path} ({len(split_entries)} samples)", flush=True)
    del coef, sol
    gc.collect()


def _copy_h5_group(entries: list[dict[str, Any]], output_root: Path) -> None:
    import h5py
    import numpy as np

    pde = str(entries[0]["pde"])
    task = str(entries[0]["task"])
    test_type = str(entries[0]["test_type"])
    source_path = Path(str(entries[0]["source_data_path"]))
    coef_name, sol_name = H5_FIELDS[pde]
    coef, sol = _load_historical_ground_truth(entries)
    with h5py.File(source_path, "r") as source:
        for split in ("tune", "holdout"):
            indexed = sorted(
                ((idx, entry) for idx, entry in enumerate(entries) if entry["split"] == split),
                key=lambda item: int(item[1]["subset_index"]),
            )
            positions = [idx for idx, _entry in indexed]
            split_entries = [entry for _idx, entry in indexed]
            source_ids = [int(entry["source_sample_id"]) for entry in split_entries]
            output_path = output_root / "subsets" / split / f"{pde}_{task}_{test_type}.mat"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with h5py.File(output_path, "w") as target:
                for key, value in source.attrs.items():
                    target.attrs[key] = value
                target.attrs["n_samples"] = len(source_ids)
                target.attrs["balanced_sample_source"] = str(source_path)
                if pde == "darcy":
                    target.create_dataset(coef_name, data=np.transpose(coef[positions, 0], (1, 2, 0)))
                    target.create_dataset(sol_name, data=np.transpose(sol[positions, 0], (1, 2, 0)))
                    for name in ("dataset_type", "generation_seed", "grf_alpha", "grf_tau", "shard_id"):
                        if name in source:
                            target.create_dataset(name, data=source[name][()])
                else:
                    target.create_dataset(coef_name, data=coef[positions, 0])
                    target.create_dataset(sol_name, data=sol[positions, 0, :, :, None])
                    if "t" in source:
                        target.create_dataset("t", data=source["t"][()])
                    if "sample_seed" in source:
                        target.create_dataset(
                            "sample_seed",
                            data=np.asarray([source["sample_seed"][sample_id] for sample_id in source_ids]),
                        )
                target.create_dataset("source_sample_id", data=np.asarray(source_ids, dtype=np.int64))
            print(f"WRITE {output_path} ({len(split_entries)} samples)", flush=True)
    del coef, sol
    gc.collect()


def materialize_subsets(manifest: list[dict[str, Any]], output_root: Path) -> None:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for entry in manifest:
        grouped[(str(entry["pde"]), str(entry["task"]), str(entry["test_type"]))].append(entry)
    for pde in PDES:
        for task in TASKS:
            for test_type in TEST_TYPES:
                entries = grouped[(pde, task, test_type)]
                print(f"READ {pde}/{task}/{test_type} ({len(entries)} selected samples)", flush=True)
                if pde in SCIPY_FIELDS:
                    _copy_scipy_group(entries, output_root)
                else:
                    _copy_h5_group(entries, output_root)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_manifest(manifest: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in manifest:
        grouped[(row["pde"], row["task"], row["test_type"], row["stratum"])].append(row)
    summaries = []
    for key, rows in sorted(grouped.items()):
        summaries.append(
            {
                "pde": key[0],
                "task": key[1],
                "test_type": key[2],
                "stratum": key[3],
                "n": len(rows),
                "selection_score_mean": statistics.fmean(float(row["selection_score"]) for row in rows),
                "rel_l2_a_mean": statistics.fmean(float(row["historical_rel_l2_a"]) for row in rows),
                "rel_l2_u_mean": statistics.fmean(float(row["historical_rel_l2_u"]) for row in rows),
            }
        )
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=Path("outputs"))
    parser.add_argument("--data-root", type=Path, default=Path("/large_storage/zhangxf/PDEdata"))
    parser.add_argument(
        "--output-root", type=Path, default=Path("outputs/artifacts/balanced_sampling_tuning")
    )
    parser.add_argument("--tail-samples-per-group", type=int, default=20)
    parser.add_argument("--random-samples-per-group", type=int, default=40)
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args()
    if args.tail_samples_per_group < 4 or args.tail_samples_per_group % 4:
        parser.error("--tail-samples-per-group must be a multiple of 4 and >= 4")
    if args.random_samples_per_group != 2 * args.tail_samples_per_group:
        parser.error("--random-samples-per-group must equal 2 * --tail-samples-per-group")

    output_root = args.output_root.resolve()
    manifest = select_samples(
        args.results_root.resolve(),
        args.data_root.resolve(),
        output_root,
        tail_count=args.tail_samples_per_group,
        random_count=args.random_samples_per_group,
    )
    write_csv(output_root / "balanced_samples.csv", manifest)
    write_csv(output_root / "sample_strata_summary.csv", summarize_manifest(manifest))
    print(f"WRITE {output_root / 'balanced_samples.csv'} ({len(manifest)} rows)", flush=True)
    if not args.manifest_only:
        materialize_subsets(manifest, output_root)


if __name__ == "__main__":
    main()
