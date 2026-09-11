"""Recompute every layout score from saved physical tensors and render the paper artifacts."""
from __future__ import annotations
import argparse
import copy
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "plot")]
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.ticker import MaxNLocator
from publication_style import use_times_new_roman
from plot_paper_ablation_fields import FIELD_CMAP, ERROR_CMAP
from export_paper_ablation_tables import table, ranked_numbers, RANK_NOTE
from export_paper_seed_ensemble import NAMES
from run_paper_ablation_revision import PDES, digest, write

VARIANTS = {"random": "random_replay", "per_sample_random": "per_input_replay",
            "fixed": "fixed_left_half", "grid": "grid_replay", "sensor_column": "columns_replay"}


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plan = json.loads((args.results / "sources/fixed_region_plan.json").read_text())
    receipts, fields, checks = {}, {}, []
    for pde in PDES:
        truth_ref = None
        for variant in [*VARIANTS.values(), "legacy_fixed_replay"]:
            receipt_path = args.results / "fixed_region" / pde / variant / "receipt.json"
            row = json.loads(receipt_path.read_text())
            relative = Path(row["result_path"]).relative_to(plan["primary_output"])
            path = args.results / relative
            assert digest(path) == row["result_sha256"]
            payload = torch.load(path, map_location="cpu", weights_only=False)
            masks = torch.load(path.parent / "masks.pt", map_location="cpu", weights_only=False)
            assert digest(path.parent / "masks.pt") == row["mask_file_sha256"]
            key = pde, variant
            receipts[key] = row
            pair = torch.cat([payload["coef_ground_truth"], payload["sol_ground_truth"]], dim=1)
            if truth_ref is None:
                truth_ref = pair
            assert torch.equal(pair, truth_ref), key
            for f, prefix in [("a", "coef"), ("u", "sol")]:
                gt = payload[prefix + "_ground_truth"].double().numpy()
                pred = payload[prefix + "_final"].double().numpy()
                mask = masks[prefix].double().numpy()
                error = np.linalg.norm(pred - gt) / np.linalg.norm(gt)
                obs_error = np.linalg.norm((pred - gt) * mask) / np.linalg.norm(gt * mask)
                assert np.isfinite(error) and np.isfinite(obs_error)
                assert np.isclose(error, row["errors"][f]["full"][0], rtol=1e-12, atol=1e-13)
                assert np.isclose(obs_error, row["errors"][f]["observed"][0], rtol=1e-12, atol=1e-13)
                expected = 640 if variant == "columns_replay" else 500
                assert np.all(mask.sum(axis=(-2, -1)) == expected), key
                if variant == "fixed_left_half":
                    assert np.count_nonzero(mask[..., 64:]) == 0
                    assert np.array_equal(masks["coef"][:, :1], masks["sol"][:, :1])
                fields[pde, variant, f] = gt[0, 0], pred[0, 0], mask[0, 0], error, obs_error
                checks.append(dict(pde=pde, variant=variant, field=f, relative_error=error,
                    observed_relative_error=obs_error, locations_per_channel=expected,
                    result_path=row["result_path"], result_sha256=row["result_sha256"],
                    mask_sha256=row["mask_file_sha256"], code_commit=row["commit"]))
    rows = []
    for pde in PDES:
        for f in (["u"] if pde == "burger" else ["a", "u"]):
            values = [100 * fields[pde, variant, f][3] for variant in VARIANTS.values()]
            rows.append([NAMES[pde], f"${f}$", *ranked_numbers(values)])
    caption = ("Relative field errors in percent on ID input 0 for each PDE, using 100 stochastic Euler steps. "
        "Fixed uses 500 locations sampled without replacement with seed 0 inside the left half of the grid; "
        "the right half has no observations, and both fields share these locations. "
        "Random, per-input random, and grid also use 500 locations per field; five complete columns use 640. "
        "All layouts were rerun with the same frozen checkpoints and settings in one numerical environment. "
        "These are single-input comparisons, not population estimates." + RANK_NOTE)
    (args.output / "ablation_layout_fields.tex").write_text(table(caption, "tab:ablation-sensor-layout",
        ["PDE", "Field", "Random", "Per-input random", "Fixed", "Grid", "Columns"], rows, font_size="footnotesize"))
    summary = []
    for pde in PDES:
        row = dict(pde=pde)
        for f in (["u"] if pde == "burger" else ["a", "u"]):
            for variant in ["random_replay", "legacy_fixed_replay", "fixed_left_half"]:
                row[variant + "_" + f + "_percent"] = 100 * fields[pde, variant, f][3]
            for region in ["left", "right", "observed"]:
                row["fixed_" + f + "_" + region + "_percent"] = 100 * receipts[pde, "fixed_left_half"]["errors"][f][region][0]
        summary.append(row)
    headers = ["pde"] + sorted(set().union(*(set(r) for r in summary)) - {"pde"})
    with (args.output / "layout_summary.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=headers)
        writer.writeheader()
        writer.writerows(summary)

    use_times_new_roman()
    plt.rcParams.update({"font.size": 9.1, "axes.titlesize": 9.1, "axes.labelsize": 9.1,
        "xtick.labelsize": 9.1, "ytick.labelsize": 9.1, "axes.linewidth": .5, "pdf.fonttype": 42})
    variants = ["random_replay", "fixed_left_half", "grid_replay", "columns_replay"]
    values = {f: [fields["helmholtz", v, f] for v in variants] for f in ["a", "u"]}
    fig, axes = plt.subplots(4, 5, figsize=(6.4, 6.7), layout="constrained")
    axes[0, 0].set_axis_off()
    axes[0, 0].text(.5, .55, "Observed\nlocations", ha="center", va="center")
    axes[0, 0].text(.5, .2, r"$\mathbf{a}$: gold" + "\n" + r"$\mathbf{u}$: blue", ha="center", va="center")
    for j, title in enumerate(["Random", "Fixed (left)", "Grid", "Columns"], 1):
        ma, mu = values["a"][j - 1][2] > 0, values["u"][j - 1][2] > 0
        rgb = np.ones(ma.shape + (3,))
        rgb[ma], rgb[mu], rgb[ma & mu] = [.75, .57, .18], [.15, .39, .55], [.16, .16, .16]
        axes[0, j].imshow(rgb, origin="lower", interpolation="nearest")
        axes[0, j].set(title=title, xticks=[], yticks=[], xlabel=f"$m_a=m_u={int(ma.sum())}$")
    def maprow(index, arrays, labels, field, cmap):
        lo, hi = min(a.min() for a in arrays), max(a.max() for a in arrays)
        norm = TwoSlopeNorm(vmin=lo, vcenter=0, vmax=hi) if lo < 0 < hi else Normalize(lo, hi)
        for ax, array, label in zip(axes[index], arrays, labels):
            im = ax.imshow(array, origin="lower", cmap=cmap, norm=norm, interpolation="nearest")
            ax.set(xticks=[], yticks=[], xlabel=label)
            for spine in ax.spines.values():
                spine.set_linewidth(.3)
        axes[index, 0].set_ylabel(field)
        cb = fig.colorbar(im, ax=list(axes[index]), orientation="horizontal", fraction=.05, pad=.025, aspect=40, shrink=.8)
        cb.locator = MaxNLocator(4)
        cb.update_ticks()
        cb.ax.tick_params(length=2, pad=1)
    for index, f in enumerate(["a", "u"], 1):
        vals = values[f]
        maprow(index, [vals[0][0]] + [v[1] for v in vals],
            ["Truth"] + [f"{100*v[3]:.2f}% / {100*v[4]:.2f}%" for v in vals], rf"$\mathbf{{{f}}}$", FIELD_CMAP)
    vals = values["u"]
    maprow(3, [np.zeros_like(vals[0][0])] + [np.abs(v[1] - v[0]) for v in vals], [""] * 5,
           r"$|\widehat{\mathbf{u}}-\mathbf{u}|$", ERROR_CMAP)
    fig.suptitle("Helmholtz · observation layouts and reconstructions", fontsize=10)
    for ext in ["pdf", "png"]:
        fig.savefig(args.output / ("ablation_compact_layout_helmholtz." + ext), dpi=250)
    plt.close(fig)
    write(args.output / "layout_validation.json", dict(assessment="Share with single-input caveat",
        protocol_sha256=digest(args.results / "sources/fixed_region_plan.json"),
        plotter_sha256=digest(__file__), pdes=len(PDES), layouts_per_pde=6, field_checks=checks,
        output_sha256={p.name: digest(p) for p in args.output.iterdir() if p.suffix in {".pdf", ".png", ".tex", ".csv"}},
        checks="Saved tensor hashes; same truth across all six conditions; independent NumPy relative errors; counts; right-half zero; common Fixed field masks.",
        limitation="One archived ID input and sampling seed per PDE. Hardware-sensitive stochastic reconstructions can differ from earlier archives; all displayed layout conditions were replayed together."))
    print("Validated 66 runs and wrote layout figure, 21-row table, summary, and provenance.")


if __name__ == "__main__":
    main()
