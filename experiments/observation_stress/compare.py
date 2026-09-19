"""Export complete four-method comparisons from independently audited errors."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

METHODS = ["recfno", "senseiver", "voronoicnn", "fm4pde"]
KEYS = ["study", "pde", "distribution", "case", "num_obs", "task", "field"]


def write_csv(path, rows):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-root", required=True)
    args = parser.parse_args()
    root = Path(args.audit_root)
    audit = json.loads((root/"audit.json").read_text())
    if not (audit["status"] == "complete" and audit["complete_cells"] == 564
            and audit["metric_rows"] == 75200 and audit["original_mat_files_verified"]
            and audit["poisson_parent_samples_and_gaussian_noise_verified"]):
        raise ValueError("A complete source-verified audit is required before final comparisons")
    source = root/"per_sample.csv"
    groups = {}
    count = 0
    with source.open() as handle:
        for row in csv.DictReader(handle):
            key = tuple(row[k] for k in KEYS)
            method = row["method"]
            assert method in METHODS
            values = groups.setdefault(key, {}).setdefault(method, {})
            sample = (int(row["index"]), row["sample_id"])
            assert sample not in values
            value = float(row["rel_l2"])*100
            assert np.isfinite(value) and value >= 0
            values[sample] = value
            count += 1
    assert count == 75200 and len(groups) == 188
    matrix, comparisons = [], []
    # The same resampling indices preserve pairing between all four methods.
    resamples = np.random.default_rng(20260919).integers(0, 100, size=(10000, 100))
    for key, group in sorted(groups.items()):
        assert set(group) == set(METHODS)
        samples = sorted(group["fm4pde"])
        assert len(samples) == 100 and [i for i, _ in samples] == list(range(100))
        assert all(set(group[m]) == set(samples) for m in METHODS)
        values = {m: np.array([group[m][sample] for sample in samples]) for m in METHODS}
        means = {m: float(values[m].mean()) for m in METHODS}
        best = min(METHODS[:-1], key=means.get)
        identity = dict(zip(KEYS, key))
        matrix.append(dict(**identity, count=100,
            **{m+"_mean_rel_l2_pct": means[m] for m in METHODS},
            best_baseline=best,
            fm_minus_best_baseline_pp=means["fm4pde"]-means[best],
            fm_relative_improvement_over_best_pct=(
                100*(means[best]-means["fm4pde"])/means[best] if means[best] else None)))
        for baseline in METHODS[:-1]:
            delta = values["fm4pde"]-values[baseline]
            interval = np.quantile(delta[resamples].mean(1), [.025, .975])
            comparisons.append(dict(**identity, baseline=baseline, count=100,
                fm_mean_rel_l2_pct=means["fm4pde"], baseline_mean_rel_l2_pct=means[baseline],
                fm_minus_baseline_pp=float(delta.mean()),
                paired_bootstrap_ci95_low_pp=float(interval[0]),
                paired_bootstrap_ci95_high_pp=float(interval[1]),
                fraction_samples_fm_lower=float((delta < 0).mean()),
                fm_median_rel_l2_pct=float(np.median(values["fm4pde"])),
                baseline_median_rel_l2_pct=float(np.median(values[baseline]))))
    assert len(matrix) == 188 and len(comparisons) == 564
    write_csv(root/"comparison_matrix.csv", matrix)
    write_csv(root/"paired_comparisons.csv", comparisons)
    protocol = dict(status="complete", groups=188, paired_comparisons=564,
        samples_per_method_and_group=100, source_per_sample_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        metric="Mean per-sample full-grid relative L2, expressed in percent; lower is better",
        delta="FM4PDE minus baseline, in percentage points; negative favors FM4PDE",
        best_baseline="Lowest cohort mean among the three baselines; never selected separately per sample",
        bootstrap=dict(seed=20260919, draws=10000, method="paired percentile", interval=.95,
            scope="Conditional on fixed checkpoints, layouts, sensor/noise draws and inference protocol; no multiplicity adjustment"))
    (root/"comparison_protocol.json").write_text(json.dumps(protocol, indent=2)+"\n")
    print("EXPORTED 188 four-method rows and 564 paired comparisons", flush=True)


if __name__ == "__main__":
    main()
