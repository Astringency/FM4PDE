# JMLR revision figures

Run with the existing `fm4pde` environment:

```bash
python plot/plot_jmlr_revision.py --paper ../../C04Papers/fm4pde_jmlr
```

Use `--snapshot-training` only when creating the initial portable training-log
extract. The selected directories are the original entries in
`outputs/pretrained/training_summary.md`, not the newest run. Existing frozen
logs may not be silently replaced. The paper's `source_data/training_logs`
contains copied JSONL logs and source hashes.

To refresh DiffusionPDE, first create a **new** snapshot:

```bash
python plot/collect_diffusion_archive.py --output ../../C04Papers/fm4pde_jmlr/source_data/diffusion_snapshot_NEWDATE.json.gz
```

The collector only reads remote files via SSH. It does not launch experiments,
alter configs, or aggregate across live/unevaluated cells. Update the explicit
snapshot path in the figure script after review. The stored JSON contains
per-example metrics, hashes, configuration text, code and a completion inventory.

## Figure contracts

All artifacts are vector PDF plus PNG previews in the paper's `figures/`.
Typography is 8–10 pt before scaling, embedded TrueType, white background.
This is an externally authored journal article, so no product branding is used.
Every plot is checked in its PDF context before final handoff.

| Artifact | Question and evidence | Form and scale | Palette / noncolor encoding |
|---|---|---|---|
| main_comparisons | How do field errors vary across task, PDE and distribution? 264 field means from author-selected workbook cells. | Six heatmaps in three rows, same logarithmic color scale, exact cell labels; corresponding SDs in paper tables. Sparse protocols explicitly differ. | One blue root, printed values. |
| physics_burgers_comparisons | How do physics-based and trajectory methods compare under archived protocols? | Horizontal dots; physics axis log, Burgers axis linear from zero. No cross-protocol structured-mask ranking. | Four categorical roots, distinct marker shapes. |
| sampler_phase | How do all eight phase settings behave on all eleven available instances? | 11×8 heatmap, logarithmic scale, no invented CI. Retain the 2416% outlier. | One blue root, exact labels. |
| sampling_steps | How does the coupled sampler respond to budget? Seven observed budgets for each sampler/PDE. | Small-multiple log-log curves; lines connect discrete budgets, not an inferred convergence law. | Four categorical roots and shapes. |
| loss_state | How does evaluating the loss at current/proposed/endpoint states change error? | Eleven PDE rows, three dots per row, two sampler panels, log axis. | Three roots and shapes. |
| training_curves | How did the logged training and validation objective evolve? 150–300 epochs per PDE. | Log-loss curves, missing epochs blank, panels keep their own scales. | Blue/gold, solid/dashed. |

Source audit rows used for main plots are checked against the supplied workbooks
before rendering. `audit/figure_manifest.json` records source and script hashes.
`source_data/diffusion_summary.csv` records counts, finite-value checks and
completeness per metric. SD uses ddof=1, never SD=0 for n=1. Smooth attribution
for the legacy DiffusionPDE runs follows the author's instruction. Existing
wall times remain historical measurements, not a common-hardware benchmark.
