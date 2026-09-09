# Revision study entry points

All commands below are entry-point recipes, not evidence that a new run was executed. Use the source version, environment, checkpoint, and frozen protocol recorded for the study. A full current checkout alone does not make an earlier experiment byte-identical: source snapshots and environment records in this directory provide the corresponding versions.

## Paths and result ownership

The archive destination is server197:

```
/research_data/users/zhangxifeng/C01Python/FM4PDE/outputs/
  main/revision_20260909/
    ns_main_revision_0909/
    burgers_revision_20260908/
    diffusion_fm_timing_20260907/
  ablations/revision_20260909/
    paper_revision_20260908/
    ns_loss_spectrum_20260907/
    ns_checkpoint_comparison_20260908/
    conditional_scaling_20260909/
```

Each study retains a source-origin layer such as `server197/output`, `server193/output`, or `local_exports`, so identically named files from different sources are not overwritten. Resolve the authoritative input/result paths from the archive inventory before running a command. In the recipes, `$ABL`, `$NS_STUDY`, `$BURGERS`, `$OLD_NS`, and `$CHECKPOINT_STUDY` mean the appropriate restored study layout, not the parent directory above the origin layer. `$PAPER` is a working copy of the paper and its frozen source data; `$REPLAY` is a fresh output directory. Run Python commands from the FM4PDE repository root with its recorded environment. These variable names do not set system options or relocate files.

Frozen receipts often contain historical absolute paths. Preserve their bytes and record old-to-archive path mappings separately. Moving a file must not silently rewrite the protocol or receipt that hashes it. Dataset and checkpoint references must resolve to the same content hashes; preserve generated effective samplers with their result receipts.

## Original main comparisons and configuration evidence

The ordinary sampler is exposed by `scripts/sample/run_sample.sh` and the task-specific `configs/main/{forward,inverse,both}/*.yaml`. Historical main rows must be reproduced with the **saved configuration and historical checkpoint/code**, not by assuming the current YAML still denotes that experiment. The main paper has a separate new NS checkpoint from its old NS ablations.

The exact numerical source boundary is:

- `paper/source_data/main_figure_values.csv`: 264 field/method/distribution rows for the five main tables before the new NS replacement.
- `paper/audit/table_value_provenance.csv`: includes those rows plus many other experiments; do not replace or archive every `NS` row indiscriminately.
- `paper/source_data/main_configs_archive.json.gz` and `main_hyperparameters_verified.csv`: historical FM configurations and checkpoint paths.
- `paper/source_data/baseline_effective_counts_verified.csv` and `baseline_counts_archive.json.gz`: baseline workbook-row identities, exact `results_raw.jsonl` paths and hashes. Match the table's `results.xlsx` row identity and field to these records; do not archive the unrelated multi-terabyte baseline evaluations tree.
- `plot/extract_main_configs.py`, `plot/audit_baseline_counts.py`, `scripts/analysis/summarize_main1000_tuned1.py`, and `scripts/analysis/export_main1000_best_to_excel.py` are already tracked analysis entries. Their expected archive layouts and selection rules remain authoritative.

`plot/run_revision_sampling.py` with its check/audit/summarize/export companions is the separate fixed-input sampling-confirmation study. Its small-sample results must not be substituted for a 1,000-example main table.

## 744 revised ablations and the unchanged original controls

The original study has 1,060 configurations. Eight PDEs contribute 744 revised runs; 316 original records remain unchanged. Most factor sweeps use ID input 0. The six temporal studies and Poisson endpoint-correction controls retain their original scope. The old NS checkpoint is used here.

Tracked execution sequence:

```bash
python plot/run_paper_ablation_revision.py prepare --archive "$ORIGINAL_ARCHIVE" --inputs "$ABL/inputs"
python plot/run_paper_ablation_revision.py diagnose --inputs "$ABL/inputs" --output "$REPLAY/results" --pdes "$PDE"
python plot/select_paper_ablation_revision.py --inputs "$ABL/inputs" --output "$REPLAY/results" --pdes "$PDE"
python plot/run_paper_ablation_revision.py rerun --inputs "$ABL/inputs" --output "$REPLAY/results" --pdes "$PDE"
```

The commands show the selection stages; exact formal replay uses the already frozen per-PDE output `selection.json` and input protocol rather than selecting from formal inputs again. For a replay that skips diagnosis, first restore each frozen `selection.json` beneath `$REPLAY/results/<pde>/`. Preparation must target a new directory, not overwrite a preserved input cache. `plot/run_paper_revision_queue.py` can coordinate declared per-PDE stages; it is not needed for postprocessing.

Merge the verified revised fields with the retained original records:

```bash
python plot/collect_paper_ablation_fields.py \
  --archive "$ABL/archived_ablation_summary.csv" --inputs "$ABL/inputs" \
  --results "$ABL/output" --original-root "$ABL/original_predictions" \
  --output "$REPLAY/ablation_publication_snapshot"
python plot/export_paper_ablation_tables.py --source "$REPLAY/ablation_publication_snapshot" --output "$REPLAY/ablation_tables"
python plot/plot_paper_ablation_fields.py --source "$REPLAY/ablation_publication_snapshot" --output "$REPLAY/ablation_figures" \
  --contract reproducibility/revision_20260909/legacy_scripts/contracts/ABLATION_CHART_CONTRACT.md
python plot/plot_paper_ablation_sweeps.py --source "$REPLAY/ablation_publication_snapshot" --output "$REPLAY/ablation_figures"
```

Required data: original summary CSV, all original records referenced by that CSV, each PDE's input/weight/protocol/selection files, revised `result.pt`, `curves.csv` and receipts. Verify the actual collector's `--archive` input is the CSV, not the root archive directory. `legacy_scripts/paper_audit/revision_0909/qa_ablation_delivery.py` preserves the independent field-error checks and the final 85-figure verification; its paper paths must be restored explicitly. None of these steps calls for rerunning deleted main-text NS calibration/residual-selection sections.

## Eleven-PDE, 32-input, three-draw study and spectra

This is the original 1,056-prediction study, with 21 reported fields and input offsets 1500–1531. Keep the old NS model and frozen observation layout. It is distinct from the new K-scaling study.

```bash
python plot/run_paper_ablation_revision.py ensemble --inputs "$ABL/inputs" --output "$REPLAY/results" --pdes "$PDE"
python plot/export_paper_seed_ensemble.py --inputs "$ABL/inputs" --results "$ABL/output" --output "$REPLAY/ensemble_complete" \
  --plan reproducibility/revision_20260909/legacy_scripts/contracts/ENSEMBLE_ANALYSIS_PLAN.md
python plot/plot_paper_seed_ensemble.py --source "$REPLAY/ensemble_complete" --output "$REPLAY/ensemble_figures" \
  --contract reproducibility/revision_20260909/legacy_scripts/contracts/ENSEMBLE_CHART_CONTRACT.md
```

The exporter retains single draws and physical-field averages, uses input-level sample SD and paired bootstrap, and verifies complete field/seed coverage. Its output NPZ/CSV/manifest files are inputs for the eleven spectra. Preserve `analysis_plan.md`/plan hash and all saved prediction tensors; do not count three draws as three independent physical inputs.

Matched Poisson/Darcy baseline frequency diagnostics use `plot/export_matched_spectra.py --study "$MATCHED_STUDY" --output "$REPLAY/spectra"`, then `plot/plot_paper_frequency_revision.py --source "$REPLAY/spectra" --output "$REPLAY/frequency_figures"`. Required artifacts include matched predictions, masks/truth and baseline source mappings. The plotting source contains 1,216 spectral records. `legacy_scripts/paper_audit/revision_0909/rank_frequency_table.py` preserves the final unrounded ranking rule for the four displayed metrics. `plot/spectral_diagnostics.py` implements the common FFT/DCT and field-filtering definitions.

## Burgers supplementation

The formal extension added 7,000 FM/Diffusion calls and combined them with fixed baseline and archived random-FM scores. There are two distinct observation geometries: random space-time observations and structured time layers. Do not replace a time-layer mask with a spatial-column mask.

Preparation/execution entries are `plot/prepare_burgers_revision.py`, `plot/run_burgers_revision.py` (`freeze`, `worker`), and `plot/run_burgers_parallel.py` (immutable redistribution and native workers). Their CLI takes the input protocol, FM/Diffusion weights and `--diffusion-root`. Preserve `sampling_protocol_v3.json`, `burgers_parallel_assignment_2242.json`, 4,448 prior and 2,552 redistributed calls, all generated workers/samplers, exact pilots and executor receipts.

```bash
python plot/export_burgers_revision.py export --inputs "$BURGERS/burgers_inputs" \
  --sampling-protocol "$BURGERS/burgers_inputs/sampling_protocol_v3.json" \
  --results "$BURGERS/burgers_results_193" "$BURGERS/burgers_results_197" "$BURGERS/burgers_results_216" \
  --fm-random "$BURGERS/burgers_fm_random" --output "$REPLAY/burgers_complete"
```

The exporter recomputes geometry, fields, errors and complete 1,000-input cells. The preserved `legacy_scripts/paper_revision_20260908/check_burgers_parallel.py --require-complete`, run in the restored study layout, separately verifies worker ownership and exact native-pilot evidence. `plot/watch_burgers_revision.py` already provides repository collection/monitoring; old generated per-shard Python files belong to the result archive, not a second production implementation.

## New NS main experiment and controlled timing

All producers are in `plot/run_ns_main_revision_0909.py`. The new main model has 44,121,218 parameters; its full and sparse inverse weights were chosen from two candidates on four Smooth development inputs, one trajectory per input, seed 0. This is separate from the older 32-input/three-seed checkpoint comparison. Preserve `inputs/protocol.json`, input NPZs, actual checkpoint/normalizer, `selection.json`, per-worker environments, complete prediction tensors and receipts. Use the already frozen selection for formal replay.

The recorded evaluation partition is A100 batches of 64 for offsets 2000–2699 and RTX4090 batches of 16 for offsets 2700–2999, with TF32 enabled in the formal main runs. Exact flags and code versions are in environment/receipt records. A one-machine replay may use another batch partition only after its equivalence checks; do not silently treat such a replay as the original hardware configuration.

CPU postprocessing:

```bash
python plot/export_ns_main_revision_0909.py main --results "$NS_STUDY/main_results" --output "$REPLAY/tables"
python plot/audit_ns_main_residual_0909.py --audit "$NS_STUDY" --results "$NS_STUDY/main_results" --output "$REPLAY/residual_audit.json"
```

Publication requires all 15 cells × 1,000 examples, exact IDs 2000–2999, 100 NFEs, finite predictions, matching result/protocol hashes, independent raw error/observation recomputation and the extra NumPy residual audit. The root study layout expected by the residual audit includes the hashed `weights.pth`; retain the archive's recorded link/file mapping. No partial output constitutes this gate. The eight-table tool's copied CLI is documented in `legacy_scripts/README.md`.

NS timing is an independent 80-call study: 20 common inputs × two methods × 100/1000 steps. It uses strict float32 with TF32 disabled on one RTX4090. Restore DiffusionPDE commit `151e721b9991404154ad4b430ab85cdaedbaa399` for `--diffusion-root`. The production entries are `run_ns_main_revision_0909.py timing_prepare/timing_run`, `run_diffusion_fm_timing.py`, and `diffusion_timing_adapter.py`. Keep prediction tensors, generated effective sampler, complete telemetry, uncontended checks, checkpoint/source hashes and all 80 receipts.

```bash
python plot/export_ns_main_revision_0909.py timing --results "$NS_STUDY/timing_results" --output "$REPLAY/timing_tables"
python plot/export_diffusion_fm_timing.py --inputs "$OLD_TIMING/inputs" --results "$OLD_TIMING/results" \
  --paper "$PAPER" --pde-overrides "$NS_TIMING_OVERRIDES_JSON"
```

The override JSON maps `nsnonbounded` to its own `inputs` and `results`; the four non-NS PDEs retain their original 320 calls. There are 400 total calls after replacing the old NS timing. Never merge the two NS protocols while claiming a single hash. The preserved `check_ns_timing_statistics.py` independently recomputes receipt-level mean/SD and the five-PDE speed-ratio range. The plotting exporter preserves final ordinary text ≥9 pt.

## Old NS checkpoint and residual studies

These are supplemental diagnostics on the old model, not alternate source rows for the new main NS table.

- Checkpoint comparison: `plot/run_ns_checkpoint_comparison.py`, `prepare_ns_checkpoint_weights.py`, `report_ns_checkpoint_comparison.py`; 32 fixed Smooth inputs, three seeds, and the protocol's full variant set. Preserve `results_v2/evidence`, `weights_map.json`, input cache and Diffusion reference predictions.
- Four-input step probe: `plot/run_ns_checkpoint_step_probe.py` and `report_ns_checkpoint_step_probe.py`. The copied overlap checker verifies all 36 repeated 100-step inverse predictions. Historical collectors and launch commands are retained under `legacy_scripts/ns_checkpoint_comparison_20260908/` only as source records.
- Common-input loss/calibration work: `run_ns_loss_study.py`, `audit_ns_loss_results.py`, `export_ns_loss_spectra.py`, `run_ns_guidance_calibration.py`, `audit_ns_guidance_calibration.py`, `export_ns_guidance_calibration.py`, and `compare_ns_sampling_strategies.py`. Use `--require-complete` for the corresponding audit and retain all 1,728 original loss-study calls; the copied complete-report checker independently validates input-level statistics and spectra.
- True-trajectory residual diagnostics: `plot/audit_ns_true_trajectory.py` and `export_ns_true_trajectory.py`. They require actual archived intermediate trajectories and must not be confused with endpoint-only inference. The viscous figure can be redrawn with the exported helper or preserved CPU-only redraw script; no inference is involved.
- Directional/guidance and Rough Diffusion supplements already have tracked `run_ns_guidance_cross.py`, `report_ns_guidance_cross.py`, `run_ns_rough_diffusion.py` and `report_ns_rough_diffusion.py`. Their stored results retain their own inputs and model scopes.

## New conditional sample scaling and training plots

The new K-scaling workflow is already tracked: `plot/run_conditional_sample_scaling.py`, `validate_conditional_sample_scaling.py`, `export_conditional_sample_scaling.py`, `collect_conditional_sample_scaling.py`, and `plot_conditional_sample_scaling.py`. Its inputs and selection are explicit CLI arguments. It uses K ∈ {1,3,10,100,1000}, 32 Poisson inputs and forward/inverse/joint tasks; do not mix its raw rows or checkpoint scope into the old three-draw study. Its complete protocol and final inventory govern exact output paths.

Training-curve rendering uses:

```bash
python plot/plot_training_curves_revision.py --paper "$PAPER" --new-ns-log "$NEW_NS_TRAINING_LOG"
```

Retain original eleven-model logs and their SHA256 manifest plus the separate new NS log and training metadata. The revised three figures have twelve model panels, explicitly distinguishing old ablation NS from new main NS. Do not run `plot_jmlr_revision.py` indiscriminately over an integrated paper: its older all-in-one path writes historical main/training source data as well as figures.
