"""Audit the pairing and summarize all predeclared cases, including adverse outcomes."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    p = json.loads((args.root / "protocol.json").read_text())
    indices, seeds = p["indices"], p["seeds"]
    full_rows, summary = [], {}
    keys = ["rel_l2_a", "rel_l2_u", "obs_rel_l2_a", "obs_rel_l2_u", "pde_residual_norm"]
    primary_rows = []
    for task in p["cells"]:
        target = args.root / "runs" / task
        assert json.loads((target / "complete.json").read_text())["paired"]
        paired = {}
        times = {v: [] for v in p["variants"]}
        memory = {v: [] for v in p["variants"]}
        for seed in seeds:
            for start in range(0, len(indices), p["batch_size"]):
                group = indices[start:start + p["batch_size"]]
                batch = {}
                for variant in p["variants"]:
                    receipt = target / f"{variant}_seed{seed}_batch{start:04d}.json"
                    row = json.loads(receipt.read_text())
                    assert sha(receipt.with_suffix(".pt")) == row["sha256"]
                    assert row["indices"] == group
                    assert [r["index"] for r in row["rows"]] == group
                    expected_nfe = 100 if variant == "original" else 199
                    assert row["runtime"]["nfe"] == expected_nfe
                    times[variant].append(row["runtime"]["seconds"])
                    memory[variant].append(row["runtime"]["peak_allocated_bytes"])
                    for item in row["rows"]:
                        assert all(np.isfinite(item[key]) for key in keys)
                        paired[(variant, seed, item["index"])] = item
                        full_rows.append(dict(task=task, variant=variant, seed=seed, **item))
                    batch[variant] = row
                assert batch["original"]["input_hashes"] == batch["proposal"]["input_hashes"]
                for key in ["initial_noise_sha256", "bridge_noise_sha256"]:
                    assert batch["original"]["runtime"][key] == batch["proposal"]["runtime"][key]
        metrics = {}
        for key in keys:
            b = np.array([[paired[("original", s, i)][key] for i in indices] for s in seeds])
            c = np.array([[paired[("proposal", s, i)][key] for i in indices] for s in seeds])
            base, candidate = b.mean(0), c.mean(0)
            delta = candidate - base
            rng = np.random.default_rng(20260919)
            draw = rng.integers(0, len(indices), size=(20000, len(indices)))
            bootstrap = delta[draw].mean(1)
            metrics[key] = dict(original=float(base.mean()), proposal=float(candidate.mean()),
                relative_improvement_pct=float(100 * (1-candidate.mean()/base.mean())) if base.mean() else None,
                difference=float(delta.mean()), paired_difference_ci95=np.quantile(bootstrap,[.025,.975]).tolist(),
                improved_cases=int((delta < 0).sum()), total_cases=len(indices),
                original_median=float(np.median(base)), proposal_median=float(np.median(candidate)),
                seed_original_means=b.mean(1).tolist(), seed_proposal_means=c.mean(1).tolist(),
                seed_improvement_pct=(100*(1-c.mean(1)/b.mean(1))).tolist() if base.mean() else None)
            if key in p["primary_metrics"][task]:
                primary_rows.append((task, key, metrics[key]))
        summary[task] = dict(metrics=metrics,
            seconds_per_sample={v: float(sum(t)/(len(indices)*len(seeds))) for v,t in times.items()},
            median_seconds_per_batch={v: float(np.median(t)) for v,t in times.items()},
            time_ratio=float(sum(times["proposal"])/sum(times["original"])),
            peak_allocated_gib={v: float(max(m)/2**30) for v,m in memory.items()},
            cases=len(indices), seeds=seeds, all_noise_and_inputs_paired=True)
    out = args.root / "report"
    out.mkdir(exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    with (out / "per_sample.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(full_rows[0]))
        writer.writeheader()
        writer.writerows(full_rows)
    lines = ["# Poisson: post-proposal endpoint guidance", "",
        f"{len(indices)} randomly selected ID cases × {len(seeds)} sampling seeds × 3 sparse tasks. "
        "128×128 grid; 500 sensors per observed field; original checkpoint and guidance settings; "
        "100 stochastic Euler steps. Only the evaluation pair and differentiated state change.", "",
        "Every initial and bridge noise tensor, physical input and observation mask was SHA256-paired. "
        "Primary metrics are mean relative L2 error (%), lower is better. "
        "Bootstrap intervals resample cases after averaging seeds (20,000 replicates), not seed–case pairs.", "",
        "| Task | Field | Original (%) | Proposal (%) | Relative improvement | Better cases | Difference CI95 (pp) |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for task, key, value in primary_rows:
        lo, hi = np.array(value["paired_difference_ci95"]) * 100
        lines.append(f"| {task} | {key} | {value['original']*100:.4f} | {value['proposal']*100:.4f} | "
            f"{value['relative_improvement_pct']:+.2f}% | {value['improved_cases']}/{len(indices)} | [{lo:+.4f}, {hi:+.4f}] |")
    lines += ["", "A negative difference means the proposal variant has lower error. These are "
        "exploratory unadjusted intervals, with several task/field comparisons.", "",
        "| Task | PDE RMS original | PDE RMS proposal | Original s/sample | Proposal s/sample | Time ratio |",
        "|---|---:|---:|---:|---:|---:|"]
    for task, row in summary.items():
        metric = row["metrics"]["pde_residual_norm"]
        lines.append(f"| {task} | {metric['original']:.6g} | {metric['proposal']:.6g} | "
            f"{row['seconds_per_sample']['original']:.3f} | {row['seconds_per_sample']['proposal']:.3f} | {row['time_ratio']:.3f}× |")
    lines += ["", "PDE RMS is the repository's final predicted-pair physical residual metric; "
        "it is distinct from reconstruction error and from the total weighted guidance objective. "
        "Timing includes inference and noise hashing, excludes model/input loading and saving; "
        "variants alternate order on the same RTX 4090 at batch size 4. Original uses 100 neural forward "
        "calls and proposal uses 199 (the terminal endpoint is identity). Extra diagnostic calls are excluded.", "",
        "## Interpretation of the theoretical defect", "",
        r"The original correction uses $g_k^-=\nabla_x\mathcal L(E_{t_k}(x_k))$. The new correction uses "
        r"$g_k^+=\nabla_z\mathcal L(E_{t_{k+1}}(z))|_{z=\widetilde x_{k+1}}$, with the proposal detached from "
        "its generating graph. The endpoint Jacobian and physical-unit decoding remain in the gradient. "
        "The raw proposal, guidance schedule, guidance multiplier and clipping are unchanged.", "",
        r"For $a_k=\gamma_k\rho(g_k^+)$, smoothness gives "
        r"$F_{k+1}(x_{k+1})\le F_{k+1}(\widetilde x_{k+1})-a_k(1-La_k/2)\|g_k^+\|^2$. "
        "The stale-gradient mismatch term disappears from this correction bound, but descent still "
        "requires a suitable step size. Objective transport, endpoint prediction error and incomplete "
        "observations remain. Eliminating the mismatch does not ensure lower field error or PDE residual.", "",
        "The code uses the original t_k-based PDE gate and guidance multiplier for both variants. "
        "Diagnostic comparisons hold these weights fixed while changing endpoint time/state. "
        "Observable raw endpoint-loss changes are not called p_k because local objective minima are unknown.", "",
        "## Reproduction", "", "See protocol.json, code.bundle, pilots/, runs/, diagnostics/ and logs/. "
        "The code exposes --override gradient_target=proposal_state_chain_rule for sample.py with "
        "endpoint loss, Euler integration and single-step endpoint prediction. The default remains unchanged."]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    if all((args.root / "diagnostics" / f"{task}.json").exists() for task in p["cells"]):
        import torch
        from experiments.aligned_sampling.run_inference import tensor_digest
        diagnostics = {}
        for task in p["cells"]:
            record = json.loads((args.root / "diagnostics" / f"{task}.json").read_text())
            baseline = torch.load(args.root / "runs" / task / "original_seed0_batch0000.pt",
                                  map_location="cpu", weights_only=False)
            assert record["final_prediction_sha256"] == tensor_digest(baseline["prediction"])
            assert record["initial_noise_sha256"] == baseline["runtime"]["initial_noise_sha256"]
            trace = record["trace"]
            diagnostics[task] = dict(original_path_reproduced_bitwise=True,
                fresh_lower_than_stale=sum(x["objective_after_fresh"]<x["objective_after_stale"] for x in trace),
                fresh_objective_ascent_steps=[x["step"] for x in trace if x["objective_after_fresh"]>x["objective_at_proposal"]],
                checked_steps=len(trace), checked_cases=len(record["indices"]), last_step=trace[-1])
        (out / "diagnostic_summary.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
        with (out / "report.md").open("a") as stream:
            stream.write("\n## Gradient transport diagnostic\n\n"
                "All diagnostic trajectories reproduce the original seed-0, first-batch predictions bitwise. "
                "Each probe compares corrections on the same original-path proposal; objective values "
                "are means over the four diagnostic cases.\n\n")
            for task, value in diagnostics.items():
                stream.write(f"- {task}: fresh correction yields lower endpoint objective than stale "
                    f"correction at {value['fresh_lower_than_stale']}/{value['checked_steps']} checked times; "
                    f"fresh correction increases the objective relative to its proposal at steps "
                    f"{value['fresh_objective_ascent_steps']}.\n")
            stream.write("\nThus a correctly located gradient does not remove the step-size restriction. "
                "The gradient diagnostic covers four cases, not the full error-estimation sample.\n")
    budget_root = args.root / "budget_control"
    if (budget_root / "complete.json").exists():
        plan = json.loads((budget_root / "protocol.json").read_text())
        records = {}
        total_seconds = 0.
        for seed in seeds:
            for start in range(0, len(indices), p["batch_size"]):
                file = budget_root / f"original_seed{seed}_batch{start:04d}.json"
                record = json.loads(file.read_text())
                assert sha(file.with_suffix(".pt")) == record["sha256"]
                proposal = json.loads((args.root / "runs" / "sparse_forward" /
                                      f"proposal_seed{seed}_batch{start:04d}.json").read_text())
                assert record["indices"] == proposal["indices"]
                assert record["input_hashes"] == proposal["input_hashes"]
                assert record["runtime"]["nfe"] == plan["steps"]
                assert record["runtime"]["initial_noise_sha256"] == proposal["runtime"]["initial_noise_sha256"]
                assert record["runtime"]["bridge_noise_sha256"][:100] == proposal["runtime"]["bridge_noise_sha256"]
                total_seconds += record["runtime"]["seconds"]
                for a, b in zip(record["rows"], proposal["rows"]):
                    assert a["index"] == b["index"]
                    records[seed, a["index"]] = (a["rel_l2_u"], b["rel_l2_u"])
        b = np.array([[records[s, i][0] for i in indices] for s in seeds])
        c = np.array([[records[s, i][1] for i in indices] for s in seeds])
        delta = (c-b).mean(0)
        draw = np.random.default_rng(20260919).integers(0, len(indices), size=(20000, len(indices)))
        budget = dict(original_steps=plan["steps"], proposal_steps=100,
            original_mean=float(b.mean()), proposal_mean=float(c.mean()),
            relative_improvement_pct=float(100*(1-c.mean()/b.mean())),
            paired_difference_ci95=np.quantile(delta[draw].mean(1), [.025,.975]).tolist(),
            improved_cases=int((delta<0).sum()), cases=len(indices),
            original_seconds_per_sample=total_seconds/(len(indices)*len(seeds)),
            proposal_seconds_per_sample=summary["sparse_forward"]["seconds_per_sample"]["proposal"],
            hardware="RTX 4090; budget control on GPU 1, main forward pair on GPU 0; approximate wall-time matching",
            selection=plan["selection"])
        (out / "budget_summary.json").write_text(json.dumps(budget, indent=2) + "\n")
        lo, hi = np.array(budget["paired_difference_ci95"])*100
        with (out / "report.md").open("a") as stream:
            stream.write(f"\n## Approximate runtime control: forward task\n\n"
                f"Original {plan['steps']} steps: {b.mean()*100:.4f}% solution error, "
                f"{budget['original_seconds_per_sample']:.3f} s/sample. "
                f"Proposal 100 steps: {c.mean()*100:.4f}%, "
                f"{budget['proposal_seconds_per_sample']:.3f} s/sample. "
                f"Relative improvement: {budget['relative_improvement_pct']:+.2f}%; "
                f"paired difference CI95: [{lo:+.4f}, {hi:+.4f}] percentage points.\n\n"
                "The original step count is selected from runtime only. Guidance coefficients remain "
                "unchanged. This control has the same cases, seeds and initial noise; its first 100 "
                "bridge noise draws agree, but their associated time grid differs. "
                "The control runs on the other RTX 4090, so runtime matching is approximate.\n")
        print("BUDGET", json.dumps(budget), flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
