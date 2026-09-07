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
Typography is sized for inclusion at the journal's six-inch text width, with
embedded TrueType fonts and a white background.
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
python plot/audit_baseline_counts.py
python plot/plot_verified_supplements.py
```

The main-config audit selects only the 66 author-approved cells used in the
paper, excludes separate diagnostic subdirectories, checks all 2,892 batch
configs, verifies weighted means and offsets, and retains complete config hashes.
Older Smooth configs missing gate fields are resolved from 210 full step logs,
not current defaults. MSE/RMS holdout errors are independently recomputed in
float64 from saved prediction/target tensors; source diagnostics and matched
masks are checked separately.

The baseline count audit verifies that all 267 current workbook identities match
the established row-to-run map. It rereads the raw metric arrays for all 235
populated cells, verifies 1,000 unique IDs per cell, and reproduces all 328
applicable field means and SDs with denominator `n-1`. The other 32 workbook
rows remain unavailable. It writes a portable compressed extract; use
`python plot/audit_baseline_counts.py --from-frozen` to repeat the aggregation
without mounted raw files. This does not recompute baseline predictions or
certify historical training-data validity or identical observation masks.

## New sampling confirmation (frozen 2026-09-06)

Code was committed locally and transferred via Git bundles into an isolated
checkout on server193. The actual sampler executes commit `42bae4a`, PyTorch
2.5.1/CUDA12.1, on RTX4090 GPUs. Postprocessing may have a later commit and records
its own script hash. Existing remote repositories and results are untouched.

Local inputs and collected results:
`../audit/fm4pde_jmlr_sampling_20260906/{inputs,results}`.
Remote checkout: `/home/zhangxf/C01Python/FM4PDE_jmlr_20260906`.

Completed 2026-09-07: all four PDEs, 1,248 batches and 4,992 example-runs
passed the full tensor audit. `confirmation_report` is the authoritative
complete report; all 52 variant summaries and four figures are included in
the paper. The revision's sampling tmux sessions have exited.

The runner has `prepare`, `probe`, `smoke` and `run` modes. Preparation freezes
4 pilot + 32 evaluation examples, excludes prior screening/holdout IDs, and
copies inference weights and real ground truths with SHA-256 hashes. A batch-one
memory probe gates the batch-four pilot. All four pilots passed. The evaluation
has 13 variants × 32 examples × 3 inference seeds per PDE, with no retraining or
new parameter selection. Each GPU task runs in its own tmux session; its session
exits when the command finishes. Never kill the user's other sessions.

```bash
python plot/collect_revision_sampling.py --dest ../audit/fm4pde_jmlr_sampling_20260906/results --with-tensors
python plot/check_revision_sampling.py \
  --inputs ../audit/fm4pde_jmlr_sampling_20260906/inputs \
  --output ../audit/fm4pde_jmlr_sampling_20260906/results --stage evaluation
python plot/summarize_revision_sampling.py \
  --inputs ../audit/fm4pde_jmlr_sampling_20260906/inputs \
  --results ../audit/fm4pde_jmlr_sampling_20260906/results \
  --dest ../audit/fm4pde_jmlr_sampling_20260906/confirmation_report
python plot/audit_revision_predictions.py \
  --inputs ../audit/fm4pde_jmlr_sampling_20260906/inputs \
  --results ../audit/fm4pde_jmlr_sampling_20260906/results \
  --dest ../audit/fm4pde_jmlr_sampling_20260906/confirmation_report
python plot/export_revision_sampling.py \
  --report ../audit/fm4pde_jmlr_sampling_20260906/confirmation_report \
  --paper ../../C04Papers/fm4pde_jmlr
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
Updated-state errors and residuals use `t_next`; guidance-endpoint residuals
and component gradients use their evaluation time `t`. Both clocks are
exported explicitly. Comparison figures use a 2-by-2 layout for four PDEs;
the trajectory figure is sized for the journal's six-inch text width.
Timing includes the full sampler call, diagnostics and artifact writes, with
loading excluded. Per-example time is amortized batch cost, not latency.
The normalized rule fixes only the nominal scalar sum `c_N*(N+1)/2`; clipping,
physical activation, gradient paths and reinjection counts can still differ.

`audit_revision_predictions.py` checks every saved prediction against the frozen
per-ID truth tensors and actual mask hashes, and recomputes both full-field and
observation errors in CPU float64. It does not independently discretize the
PDE residual or certify historical training-data independence. The paper exporter
requires complete formal results and tensor audits for all four PDEs, reconciles
the per-example error exports, and writes all 52 variant summaries without rank
selection. A complete single-PDE report may be generated for inspection in its
own directory, but it is refused by the paper exporter.

The residual trace is mean per-example RMS. Gradient and correction norms are
Euclidean norms over the entire four-example batch, then averaged over batches
and seeds. The sum of weighted observation-component norms is distinct from the
norm of their summed gradient. The CSVs also retain observation errors/losses
and paired physical-residual intervals. Initial-noise and mask pairing is
checked from actual tensor hashes; differing phase schedules and budgets do
not imply reinjection draws aligned at the same flow times.
# Matched inverse prediction timing (2026-09-07)

`run_matched_timing.py` is a separate batch-one study on Poisson and Darcy ID
inverse problems. It reuses the 4 pilot / 32 evaluation IDs already frozen for
sampling confirmation, with 500 fixed solution sensors, and compares FM4PDE
(25/50/100/200 steps), RecFNO, Senseiver, VoronoiCNN, and PDE-Opt
(50/100/500 iteration caps, retaining its original early stopping).
The current inverse FM weights and clipping thresholds must match every
corresponding sparse-ID archived configuration. No parameters are tuned.

All methods run in one Python/PyTorch environment on one GPU, with float32,
TF32 disabled and two CPU threads. Synchronized wall time starts with physical
CPU observations and includes construction, method-specific preprocessing,
transfer, prediction and physical decoding. Loading, warmup, error evaluation
and file writes are excluded. Voronoi preprocessing uses the canonical CPU
implementation for the two grid reconstruction networks. PDE-Opt retains its
archived initialization from the masked solution and best-state restoration.
Three baseline timing repetitions do not increase the number of independent
error observations. FM seeds are averaged within each physical example.

The pilot must pass repeat and hidden-target invariance checks on all four
pilot IDs; learned methods must also respond to perturbed observations.
Prediction-only FM output must exactly match the original instrumented sampler.
Zero PDE-Opt predictions after best-state restoration are recorded, not tuned
away. Every formal prediction, physical truth, actual mask, elapsed time,
optimizer status, and artifact checksum is retained for independent audit.
This experiment is distinct from the batch-four instrumented sampling timings.

```bash
python plot/run_matched_timing.py prepare --inputs INPUTS \
  --sampling-inputs SAMPLING_INPUTS --paper PAPER --baseline-root BASELINE_REPO
python plot/run_matched_timing.py pilot --inputs INPUTS --output RESULTS \
  --baseline-root BASELINE_REPO
python plot/run_matched_timing.py run --inputs INPUTS --output RESULTS \
  --baseline-root BASELINE_REPO
python plot/summarize_matched_timing.py --inputs INPUTS --results RESULTS --dest REPORT
python plot/export_matched_timing.py --inputs INPUTS --report REPORT --paper PAPER
```

For the actual 2026-09-07 execution, the authoritative local paths are
`../audit/fm4pde_jmlr_matched_timing_20260907/inputs_v2` and `results_v3`.
The earlier inputs/results preserve superseded engineering pilots and a
short interrupted timing launch; they must not enter paper summaries.
Formal inference is pinned to `180cdcf` in a separate remote checkout.
The report requires all 1,920 calls and independently checks actual masks,
frozen physical truths, saved predictions, deterministic repetitions, and
the original effective PDE-Opt configuration hash. Paper export retains all
20 method/budget rows, including exactly zero solver predictions.

Completed 2026-09-07: all 1,920 calls passed the full tensor audit and were
exported from `../audit/fm4pde_jmlr_matched_timing_20260907/report`.
The dedicated timing tmux session has exited. Both completed studies can be
replotted from their local archives without another remote inference run.

# Revised main tables, reconstructions, and generative latency (2026-09-07)

The eight original main tables are included at their task-specific locations
in Experiments. Their exact mean/SD blocks remain unchanged. The earlier
main-comparison heatmaps are retained as historical artifacts and are no longer
included in the manuscript. `export_matched_timing_tables.py` presents the
completed 32-example inverse cost study as mean/SD tables; its original
bootstrap and IQR tables remain in the appendix.

`prepare_diffusion_revision.py` reads original DiffusionPDE predictions and
existing metrics, remaps pickled CUDA tensors to CPU, and computes missing
metrics in a new output directory. The 2026-09-07 snapshot has all 26
100/1000-step PDE/task groups, each with 1,000 unique offsets and finite
field errors (26,000 records). It includes Burgers. No accuracy inference is
repeated and no original source, configuration, prediction or metric is changed.
`export_diffusion_revision.py` builds four task-specific Smooth comparisons,
with FM4PDE 100 / DiffusionPDE 100 / DiffusionPDE 1000 method columns.
Its 51 entries contain field-specific mean and sample SD; FM values and counts
are checked against the supplied workbook. The structured Burgers comparator
uses the five-spatial-column FM row, not the random-point row or time slices.

`plot_sampling_reconstructions.py` generates eight figures from actual saved
sampling-study tensors. Selection is fixed to the first predeclared evaluation
ID (425) and seed 0, before visual inspection. Four figures compare observation
and combined guidance, two show all phase controls, and two show normalized
step controls. No outcome is used to select the example. All prediction fields
within a comparison share color limits; all absolute errors share a zero-based
range, with no quantile clipping. Full-grid and observed relative L2 errors are
recomputed in CPU float64 and checked against original receipts. The figures
and a compact tensor NPZ, metric CSV and source/output hash manifest are exported.
The 32-example/three-seed aggregate results remain the evidence for mean effects.

`run_diffusion_fm_timing.py` freezes twenty common Smooth inputs plus a separate
pilot, and runs both methods at 100/1000 steps, batch one, on the same physical
GPU for each PDE. Resident GPU input-to-output timing includes initialization,
network calls, residuals, gradients, guidance and physical decoding; loading,
preprocessing/transfers, per-step scoring and writes are excluded. The adapter
extracts the original Diffusion update loop and records the generated source.
Both use float32 (including the explicit conversion of Diffusion's native
float64 state/residual), TF32 off, deterministic software and two CPU threads.
Heun is retained: Diffusion NFE=2N-1; FM NFE=N. This is a controlled-precision
latency study, separate from the archived native-precision accuracy results.

Each PDE passes repeat, hidden-field invariance, observation-sensitivity and
prediction-path equivalence checks. Each method/budget receives a full warm-up.
The four settings are shuffled within each of twenty inputs. GPU UUID, other
compute PIDs, utilization, clocks, temperature, memory and power are monitored
at one-second intervals and call boundaries. Contended attempts are retained
but not used as one of the twenty valid calls. No time-based trimming is allowed.
`export_diffusion_fm_timing.py` requires all 400 calls and independently verifies
input tensors, masks, prediction hashes, errors, NFE and contention telemetry
before producing two grouped bar panels (100/1000 steps). Bars and numeric
labels give mean seconds; error bars are sample SD (ddof=1), not SEM.

Authoritative controlled-timing inputs are
`../audit/diffusion_fm_revision_20260907/timing_inputs_v2`, protocol SHA-256
`0237fd018240c2d3f57720b40ce7fcec15cacca33b12ac588703df5ffe752217`.
Execution is pinned to FM commit `3f94ddb` and Diffusion commit `151e721` in
isolated Git checkouts on server193. The first launch ended before sampling
because Torch's UUID omits NVIDIA's `GPU-` prefix; `timing_results_v2` contains
the corrected study. The failed-launch directory must not enter final summaries.

```bash
python plot/export_diffusion_revision.py --paper PAPER --source SNAPSHOT
python plot/plot_sampling_reconstructions.py --study SAMPLING_STUDY --paper PAPER
python plot/export_matched_timing_tables.py --paper PAPER
python plot/export_diffusion_fm_timing.py --inputs INPUTS --results RESULTS --paper PAPER
python PAPER/audit/verify_presentation_0907.py --require-timing
```
