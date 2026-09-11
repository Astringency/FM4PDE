"""Validate all 32 input-level comparisons and write paired uncertainty."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from plot.run_paper_ablation_revision import digest, write

METHODS = {
    "plugin_stochastic100": "Plug-in, stochastic, 100 steps",
    "exact_flow100": "Exact tilted velocity, 100 Euler steps",
    "exact_flow200": "Exact tilted velocity, 200 Euler steps",
    "posterior_draw": "Direct Gaussian posterior draw",
    "posterior_mean": "Gaussian posterior mean (estimator)",
}


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cohort = args.results / "gaussian_tilt/cohort"
    cp = json.loads((cohort / "cohort_protocol.json").read_text())
    completed = json.loads((cohort / "complete.json").read_text())
    assert completed["input_ids"] == cp["input_ids"] == list(range(1500, 1532))
    arrays = {method: [] for method in METHODS}
    times = {method: [] for method in METHODS}
    checks, refinement, table_rows = [], [], []
    for i in cp["input_ids"]:
        source = cohort / f"input{i}"
        assert json.loads((cohort / f"input{i}_execution.json").read_text())["exit_code"] == 0
        ip = json.loads((source / "protocol.json").read_text())
        assert ip["input_ids"] == [i] and ip["seeds"] == [0, 1, 2]
        inputs = np.load(source / "inputs.npz")
        truth = inputs["physical_truth"]
        assert np.all(inputs["masks"].sum(axis=(-2, -1)) == 500)
        predictions = {}
        for method in METHODS:
            names = ["posterior_mean"] if method == "posterior_mean" else [f"{method}_seed{s}" for s in cp["seeds"]]
            values = []
            for name in names:
                receipt = json.loads((source / (name + ".json")).read_text())
                predfile = source / (name + ".npz")
                pred = np.load(predfile)["physical_prediction"]
                assert pred.shape == truth.shape == (1, 2, 128, 128)
                errors = np.linalg.norm((pred - truth).reshape(2, -1), axis=1) / np.linalg.norm(truth.reshape(2, -1), axis=1)
                assert np.isfinite(pred).all() and np.isfinite(errors).all()
                assert np.allclose(errors, [receipt["errors"][f][0] for f in ["a", "u"]], rtol=1e-12, atol=1e-13)
                assert receipt["numerical"]["max_relative_residual"] <= 1.01e-9
                values.append(errors)
                times[method].append(receipt["seconds"])
                predictions[name] = pred
                checks.append(dict(input_id=i, method=name, relative_errors=errors.tolist(),
                    prediction_sha256=digest(predfile), input_sha256=digest(source / "inputs.npz"),
                    protocol_sha256=digest(source / "protocol.json")))
            arrays[method].append(np.mean(values, axis=0))
        for seed in cp["seeds"]:
            diff = predictions[f"exact_flow200_seed{seed}"] - predictions[f"exact_flow100_seed{seed}"]
            refinement.append(np.linalg.norm(diff.reshape(2, -1), axis=1) / np.linalg.norm(truth.reshape(2, -1), axis=1))
    arrays = {k: np.asarray(v) for k, v in arrays.items()}
    statistics = {}
    for method, x in arrays.items():
        statistics[method] = dict(mean_percent=(100 * x.mean(axis=0)).tolist(),
            input_sd_percent=(100 * x.std(axis=0, ddof=1)).tolist(), median_seconds=float(np.median(times[method])),
            inputwise_seed_mean_errors=x.tolist())
        table_rows.append([METHODS[method], *[f"${100*x[:,j].mean():.2f}\\pm{100*x[:,j].std(ddof=1):.2f}$" for j in [0, 1]],
                           f"${np.median(times[method]):.2f}$"])
    rng = np.random.default_rng(20260911)
    resample = rng.integers(0, 32, size=(25000, 32))
    base = arrays["plugin_stochastic100"]
    new = arrays["exact_flow100"]
    delta = new - base
    draws = delta[resample].mean(axis=1)
    ci = np.quantile(draws, [.0125, .9875], axis=0)
    improvement = 100 * (base.mean(axis=0) - new.mean(axis=0)) / base.mean(axis=0)
    primary = {f: dict(delta_percentage_points=float(100 * delta[:, j].mean()),
        paired_97p5_interval_percentage_points=(100 * ci[:, j]).tolist(),
        relative_improvement_percent=float(improvement[j]),
        significantly_better=bool(ci[1, j] < 0), significantly_worse=bool(ci[0, j] > 0),
        practical_5percent_improvement=bool(improvement[j] >= 5)) for j, f in enumerate(["a", "u"])}
    tex = "\n".join([r"\begin{table}[!htbp]", r"\centering\small\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}lrrr@{}}\toprule", r"Control & $a$ error (\%) & $u$ error (\%) & Seconds \\\midrule",
        *[" & ".join(row) + r" \\" for row in table_rows], r"\bottomrule\end{tabular}",
        r"\caption{Poisson controls with the same full-resolution Gaussian prior, 500 noiseless observations per field, and fixed observation-energy weights. Errors are means and sample SDs of the inputwise three-seed mean over 32 ID inputs; the posterior mean is deterministic. Seconds give median CPU time per output in this implementation. Exact velocity refers to the Gaussian conditional moment; the 100- and 200-step flows retain Euler discretization error. The direct draw is Gaussian posterior sampling up to the linear-solve tolerance. The final row is a point estimator. These controls change the trained prior and are not a same-checkpoint FM4PDE improvement claim.}",
        r"\label{tab:exact-tilt-gaussian}", r"\end{table}", ""])
    (args.output / "gaussian_tilt_table.tex").write_text(tex)
    summary = dict(primary=primary, methods=statistics, n_inputs=32, seeds=cp["seeds"],
        refinement_100_to_200=dict(median_truth_normalized_percent=(100 * np.median(refinement, axis=0)).tolist(),
                                  max_truth_normalized_percent=(100 * np.max(refinement, axis=0)).tolist()),
        source_protocol_sha256=digest(cohort / "cohort_protocol.json"), validation_count=len(checks),
        interpretation="Primary comparison changes exact-vs-plug-in guidance and deterministic-vs-stochastic transition together while fixing the Gaussian prior. It does not isolate a sole drift replacement for the learned FM prior.",
        inference_scope="Bootstrap unit is physical input, conditional on the fixed masks and three seeds. No 1000-input or seed-population claim.")
    write(args.output / "gaussian_tilt_summary.json", summary)
    write(args.output / "gaussian_tilt_validation.json", dict(checks=checks, analysis_sha256=digest(__file__),
        assessment="Share with explicit Gaussian-prior, sampler, and finite-step caveats"))
    print(json.dumps({"primary": primary, "methods": {k: {kk: vv for kk, vv in v.items() if kk != "inputwise_seed_mean_errors"} for k, v in statistics.items()},
                      "refinement": summary["refinement_100_to_200"], "validated": len(checks)}, indent=2))


if __name__ == "__main__":
    main()
