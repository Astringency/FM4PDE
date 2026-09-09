"""Read completed compact K-study results and review their scientific claims.

No sampling, raw-pool audit, PDE residual recomputation or figure rendering.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import re
import statistics

import numpy as np


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--paper", type=Path, required=True)
parser.add_argument("--output", type=Path)
args = parser.parse_args()
ROOT = args.paper.resolve()
SOURCE = ROOT / "source_data/conditional_scaling_0909"
OUTPUT = args.output or ROOT / "audit/revision_0909/conditional_scaling_final_review/independent_scientific_review.json"
KS = [1, 3, 10, 100, 1000]
TASK_FIELDS = [("forward", "u"), ("inverse", "a"), ("both", "a"), ("both", "u")]


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def csv_rows(name):
    with (SOURCE / ("conditional_scaling_" + name + ".csv")).open(newline="") as stream:
        return list(csv.DictReader(stream))


names = ["conditional_scaling_" + n for n in (
    "summary.csv", "paired_effects.csv", "timing.csv", "per_input.csv",
    "fields.npz", "table.tex", "final_manifest.json")]
before = {name: sha(SOURCE / name) for name in names}
summary_rows = csv_rows("summary")
effect_rows = csv_rows("paired_effects")
timing_rows = csv_rows("timing")
input_rows = csv_rows("per_input")
summary = {(r["task"], r["field"], int(r["K"])): r for r in summary_rows}
effects = {(r["task"], r["field"], int(r["K"])): r for r in effect_rows}
timings = {(r["task"], int(r["K"])): r for r in timing_rows}
inputs = {(r["task"], int(r["offset"]), int(r["K"])): r for r in input_rows}
assert len(summary) == len(summary_rows) == 20
assert len(effects) == len(effect_rows) == 16
assert len(timings) == len(timing_rows) == 15
expected = {(t, i, k) for t in ("forward", "inverse", "both")
            for i in range(1500, 1532) for k in KS}
assert len(inputs) == len(input_rows) == 480 and set(inputs) == expected
manifest = json.loads((SOURCE / "conditional_scaling_final_manifest.json").read_text())
assert manifest["complete"] and manifest["physical_inputs"] == 32
assert manifest["K"] == KS and manifest["simultaneous_interval_comparisons"] == 16

table = (SOURCE / "conditional_scaling_table.tex").read_text()
table_cells = {}
for line in table.splitlines():
    if re.match(r"^\d+ &", line):
        pieces = line.split(" & ")
        assert len(pieces) == 5
        table_cells[int(pieces[0])] = pieces[1:]
assert set(table_cells) == set(KS)

field_findings = []
interval_findings = []
table_checks = []
for column, (task, field) in enumerate(TASK_FIELDS):
    means = [float(summary[task, field, k]["mean_percent"]) for k in KS]
    values = np.array([[float(inputs[task, i, k]["rel_l2_" + field]) * 100
                        for k in KS] for i in range(1500, 1532)])
    assert np.isfinite(values).all()
    distinct_means = sorted(set(means))
    ranked = [KS[j] for j in sorted(range(5), key=lambda j: means[j])]
    for j, k in enumerate(KS):
        row = summary[task, field, k]
        assert int(row["n"]) == 32
        assert abs(statistics.mean(values[:, j]) - means[j]) < 1e-12
        assert abs(statistics.stdev(values[:, j]) - float(row["sd_percent"])) < 1e-12
        cell = table_cells[k][column]
        numbers = re.findall(r"\d+\.\d+", cell)
        assert numbers == [f"{means[j]:.2f}", f"{float(row['sd_percent']):.2f}"]
        is_best, is_second = means[j] == distinct_means[0], means[j] == distinct_means[1]
        assert ("\\mathbf{" in cell) == is_best
        assert ("\\dagger" in cell) == is_second
        table_checks.append(dict(task=task, field=field, K=k, value_and_sd_correct=True,
                                 best=is_best, second=is_second, unrounded_mean=means[j]))
        if k > 1:
            effect = effects[task, field, k]
            delta = values[:, j] - values[:, 0]
            assert abs(float(effect["mean_delta_pp"]) - statistics.mean(delta)) < 1e-12
            assert int(effect["improved_inputs"]) == np.count_nonzero(delta < 0)
            low, high = (float(effect[x]) for x in ("simultaneous_ci_low", "simultaneous_ci_high"))
            assert low < high < 0
            interval_findings.append(dict(task=task, field=field, K=k, reference_K=1,
                mean_delta_pp=float(effect["mean_delta_pp"]), adjusted_ci_pp=[low, high],
                conclusion="lower mean error than K=1; adjusted interval entirely below zero",
                improved_inputs=int(effect["improved_inputs"])))
    field_findings.append(dict(task=task, field=field, K=KS, means_percent=means,
        mean_trend="strictly decreasing at the five evaluated K values",
        ranking_K_ascending_error=ranked, best_K=ranked[0], second_K=ranked[1],
        K100_fraction_of_observed_K1_to_K1000_mean_reduction=(means[0]-means[3])/(means[0]-means[4]),
        K100_to_K1000_reduction_percentage_points=means[3]-means[4],
        K100_to_K1000_relative_mean_error_reduction_percent=100*(means[3]-means[4])/means[3],
        K1_to_K1000_relative_mean_error_reduction_percent=100*(means[0]-means[4])/means[0],
        individually_strictly_decreasing_inputs=int(np.all(np.diff(values, axis=1) < 0, axis=1).sum()),
        adjacent_improved_counts=(np.diff(values, axis=1) < 0).sum(axis=0).tolist(),
        adjacent_worsened_counts=(np.diff(values, axis=1) > 0).sum(axis=0).tolist(),
        adjacent_comparison_pairs=list(zip(KS[:-1], KS[1:]))))

latency_findings = []
for task in ("forward", "inverse", "both"):
    times = np.array([[float(inputs[task, i, k]["seconds"]) for k in KS]
                      for i in range(1500, 1532)])
    assert np.isfinite(times).all() and np.all(times > 0)
    medians = [float(timings[task, k]["median_seconds"]) for k in KS]
    assert np.allclose(np.median(times, axis=0), medians, rtol=0, atol=1e-12)
    latency_findings.append(dict(task=task, K=KS, median_seconds=medians,
        median_per_input_ratios_to_K1=[float(timings[task, k]["median_ratio_to_K1"]) for k in KS],
        K100_to_K1000_median_per_input_latency_ratio=float(np.median(times[:, 4]/times[:, 3])),
        K100_to_K1000_ratio_of_task_medians=medians[4]/medians[3],
        K1000_median_serial_reference_speedup=float(timings[task, 1000]["median_speedup_vs_serial"])))

# Read compact fields only to check their cohort/column mapping and representative
# field structure. The root review independently recomputes all 480 field metrics.
representative_arrays = []
with np.load(SOURCE / "conditional_scaling_fields.npz", allow_pickle=False) as fields:
    expected_keys = {f"{task}_{offset}_{suffix}" for task in ("forward", "inverse", "both")
                     for offset in range(1500, 1532)
                     for suffix in ("truth", "mask", *(f"K{k}" for k in KS))}
    assert set(fields.files) == expected_keys
    for task in ("forward", "inverse", "both"):
        for suffix in ("truth", "mask", *(f"K{k}" for k in KS)):
            key = f"{task}_1500_{suffix}"
            a = fields[key]
            assert a.shape == (2, 128, 128) and np.isfinite(a).all()
            if suffix == "mask":
                assert np.isin(a, (0, 1)).all() and np.array_equal(a.sum(axis=(1, 2)), [500, 500])
            representative_arrays.append(dict(key=key, shape=list(a.shape), dtype=str(a.dtype),
                array_sha256=hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()))

assert all(sha(SOURCE / name) == digest for name, digest in before.items())
result = dict(status="pass", confidence="share with the stated scope and caveats",
    scope="Independent scientific interpretation, unrounded table ranking and compact-source mapping; no sampling, raw-pool/PDE audit or figure QA repeated.",
    source_directory=str(SOURCE), source_sha256=before, reviewer_script_sha256=sha(Path(__file__)),
    physical_inputs=32, tasks=3, reported_task_fields=4, compact_rows=480,
    fields_array_keys=672, representative_field_arrays=representative_arrays,
    mean_sd_from_per_input_csv_agree=True, table_cells=table_checks,
    all_20_table_values_sd_and_rank_marks_correct=True, field_findings=field_findings,
    adjusted_interval_conclusions=interval_findings, all_16_adjusted_intervals_below_zero=True,
    interval_definition="Input-paired bootstrap vs K=1, 100000 resamples; Bonferroni 16-comparison adjustment; quantiles 0.05/32 and 1-0.05/32. Bootstrap recomputation is owned by the separate full numerical review.",
    latency_findings=latency_findings,
    suitable_manuscript_paragraphs=[
        r"On the 32 Poisson test inputs, mean reconstruction error decreases at every evaluated sample count for all four task--field combinations. At $K=1000$, the mean relative errors are $3.86\%$ for forward reconstruction, $17.82\%$ for inverse reconstruction, and $12.26\%$ and $2.83\%$ for the two joint-reconstruction fields, compared with $7.44\%$, $22.64\%$, $17.24\%$, and $5.12\%$ at $K=1$. All 16 Bonferroni-adjusted paired bootstrap intervals for differences from $K=1$ lie below zero.",
        r"The additional reduction becomes small at large sample counts. Averaging $K=100$ samples attains $97.0$--$98.8\%$ of the observed mean-error reduction from $K=1$ to $K=1000$. Increasing $K$ from $100$ to $1000$ lowers mean error by a further $0.060$--$0.093$ percentage points, while median averaged-estimate latency rises from approximately $79$--$86$ seconds to $788$--$859$ seconds. These results describe the tested Poisson model and observation protocol; individual-input errors need not decrease at every increment of $K$."
    ],
    additional_supported_observations=[
        "K=3 lowers mean errors by 12.5%--25.0% relative to K=1, with median within-input latency ratios 1.037--1.057.",
        "K=10 lowers mean errors by 18.2%--39.3% relative to K=1, with median within-input latency ratios 1.859--1.885.",
        "K=1000 is the lowest observed mean and K=100 the second-lowest in every field. These are descriptive ranks, not significant-difference claims for K=1000 versus K=100."
    ],
    cannot_claim=[
        "Universal improvement for every input or monotonic inputwise error: from K=100 to K=1000, errors worsen for 14/32 forward-u, 6/32 inverse-a, 7/32 joint-a and 13/32 joint-u inputs.",
        "A statistically significant gain from K=100 to K=1000: the reported adjusted intervals compare larger K only with K=1.",
        "Convergence to the exact physical solution, a zero limiting error, or optimality of K=1000 beyond the five tested counts.",
        "That the 1000 draws provide 1000 independent test inputs: intervals resample 32 physical inputs and are conditional on the fixed model and observations.",
        "Uncertainty calibration, posterior correctness or validity of the averaged fields solely from these error curves; PDE/observation behavior requires its own evidence.",
        "The same quantitative gains on all eleven PDEs, other distributions, models, observation layouts or step counts.",
        "A measured serial runtime or end-to-end speedup over DiffusionPDE: the serial curve is K times K=1, and this averaged-estimate timing scope differs from the controlled FM--Diffusion comparison.",
        "Hardware-independent timing: per-input K comparisons use the same device, but pooled summaries combine A100 and A800 executions.",
        "Exact equality between separate timed batches and canonical accuracy prefixes: upstream TF32 path checks use their recorded tolerance.",
        "A global precision/cost optimum without an explicit error tolerance or latency budget."
    ],
    required_fixes=[], nonblocking_notes=[
        "settings.tex contains an old source comment saying numerical results will be added after completion; it is a comment and does not affect compiled prose.",
        "Final full compact-metric/bootstrap verification is performed by root; rendered figure QA is performed by the settings reviewer. This JSON does not replace either gate."
    ])
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(dict(status=result["status"], output=str(OUTPUT), sha256=sha(OUTPUT),
                     table_cells=20, adjusted_intervals=16, required_fixes=[])))
