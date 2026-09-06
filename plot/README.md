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

## Verified supplements

Run these using the `fm4pde` Python environment. The raw-archive audit commands
require the mounted outputs; rendering uses the frozen paper extracts.

```bash
python plot/extract_endpoint_inventory.py
python plot/extract_main_configs.py
python plot/audit_loss_study.py
python plot/plot_verified_supplements.py
```

The main-config audit selects only the 66 author-approved cells used in the
paper, excludes separate diagnostic subdirectories, checks all 2,892 batch
configs, verifies weighted means and offsets, and retains complete config hashes.
Older Smooth configs missing gate fields are resolved from 210 full step logs,
not current defaults. MSE/RMS holdout errors are independently recomputed in
float64 from saved prediction/target tensors; source diagnostics and matched
masks are checked separately.

## New sampling confirmation (frozen 2026-09-06)

Code was committed locally and transferred via Git bundles into an isolated
checkout on server193. The actual sampler executes commit `42bae4a`, PyTorch
2.5.1/CUDA12.1, on RTX4090 GPUs. Postprocessing may have a later commit and records
its own script hash. Existing remote repositories and results are untouched.

Local inputs and collected results:
`../audit/fm4pde_jmlr_sampling_20260906/{inputs,results}`.
Remote checkout: `/home/zhangxf/C01Python/FM4PDE_jmlr_20260906`.

The runner has `prepare`, `probe`, `smoke` and `run` modes. Preparation freezes
4 pilot + 32 evaluation examples, excludes prior screening/holdout IDs, and
copies inference weights and real ground truths with SHA-256 hashes. A batch-one
memory probe gates the batch-four pilot. All four pilots passed. The evaluation
has 13 variants × 32 examples × 3 inference seeds per PDE, with no retraining or
new parameter selection. Each GPU task runs in its own tmux session; its session
exits when the command finishes. Never kill the user's other sessions.

```bash
python plot/collect_revision_sampling.py --dest ../audit/fm4pde_jmlr_sampling_20260906/results
python plot/check_revision_sampling.py \
  --inputs ../audit/fm4pde_jmlr_sampling_20260906/inputs \
  --output ../audit/fm4pde_jmlr_sampling_20260906/results --stage evaluation
python plot/summarize_revision_sampling.py \
  --inputs ../audit/fm4pde_jmlr_sampling_20260906/inputs \
  --results ../audit/fm4pde_jmlr_sampling_20260906/results \
  --dest ../audit/fm4pde_jmlr_sampling_20260906/confirmation_report
```

Collection never uses `--delete`. Add `--with-tensors` for prediction/mask audit
copies. Validation refuses missing groups, duplicates, errors, nonfinite
metrics, mismatched initial noise, or changed masks. A complete marker alone
is insufficient. Genuine failed evaluations must be reviewed and disclosed,
never silently dropped or replaced by a favorable rerun.

The postprocessor was exercised against all 52 pilot batches in a separate
`pilot_report` directory, with visible PILOT labels. Those plots are not paper
results. Final figures report paired physical-example intervals (seeds averaged
within example), all declared guidance/phase settings, measured time versus
accuracy, error trajectories, separate endpoint/state residuals, and weighted
gradient norms. Exact clipping/gate/correction traces remain in exported CSVs.
Timing includes the full sampler call, diagnostics and artifact writes, with
loading excluded. Per-example time is amortized batch cost, not latency.
The normalized rule fixes only the nominal scalar sum `c_N*(N+1)/2`; clipping,
physical activation, gradient paths and reinjection counts can still differ.
