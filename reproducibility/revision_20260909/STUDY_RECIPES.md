# Revision study entry points

Commands below describe study entry points. Actual CPU relocation checks are reported separately in `RELOCATION_TEST.md` (local) and `RELOCATION_SERVER197.md` (complete server archive), with their JSON receipts; the recipes do not imply new sampling was executed. Use the source version, environment, checkpoint, and frozen protocol recorded for the study. A full current checkout alone does not make an earlier experiment byte-identical: source snapshots and environment records in this directory provide the corresponding versions.

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

Each study retains a source-origin layer such as `server197/output`, `server193/output`, or `local_exports`, so identically named files from different sources are not overwritten. Resolve the authoritative input/result paths from the archive inventory before running a command. In the recipes, `$ABL`, `$NS_STUDY`, `$BURGERS`, `$OLD_NS`, and `$CHECKPOINT_STUDY` mean the appropriate restored study layout, not the parent directory above the origin layer. `$PAPER` is a separate working copy restored from `paper_assets_final_20260909` (the current tool default), containing the paper and frozen source data; `$REPLAY` is a fresh output directory. Run Python commands from the FM4PDE repository root with its recorded environment. These variable names do not set system options or relocate files.

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

## Exact historical baseline inputs and checkpoint inference

The selected historical matrix contains 156 main cells plus 30 Burgers cells. `baseline_freeze_plan.json` maps them to original evaluation summaries/configurations and eighteen native raw caches. The baseline addenda preserve 54 selected neural checkpoints (48 main and six Burgers), with fifteen main files referenced from already identical archived copies. They do not copy the unrelated multi-terabyte results parent or the full training MAT collections.

`BASELINE_FROZEN_REPLAY.md` gives the complete command for `replay_baseline_frozen.py`, including the original summary/configuration hashes, selected cache manifest, historical evaluation commit and saved-artifact references. Resolve those fields from the addendum's `archive_dependency_bindings.json` and the exact cache run bindings. The target cache directory is `outputs/main/revision_20260909/legacy_baseline_extracts/dependencies/frozen_inputs`; first require its independent copy/verify receipt. Preserve its manifest SHA `feb694a06731452bdae3b10f5bc6812107f7ff481c82b200d22c8e9c8f20f3ba`.

The default wrapper operation audits all 1,000 sample/mask identities while checking selected saved artifacts; `--predict --checkpoint` separately enables native checkpoint inference within original test-batch boundaries. Use a checkout of the original evaluation commit, rather than the cache loader's current default version. The original training `eval_only` CLI is unsuitable for cache-only inference because it first loads training/validation data. FNO/DeepONet need their recorded native dependencies, absent from the server197 FM environment. Six Burgers VIVID cells did not save their trained inverse operator and cannot rerun the original learned model from test caches. Six 4D-Var cells use per-instance optimization without a trained checkpoint. Neither case justifies substituting another method's weights.

The native consumer has now been exercised with real archived-source inputs and original models. All 186 test data/mask contracts passed (138,000 mask records and full-field empty contracts), and thirty saved artifacts from ten representative cells matched bitwise. Both Burgers truth/mask bundles matched for three indices each. Four limited checkpoint replays retained original batches 16/16/8: Darcy RecFNO sparse forward, Poisson Senseiver sparse inverse and Helmholtz VoronoiCNN sparse joint on server197 CPU/PyTorch 2.8.0+cu128, and Poisson iFNO full inverse on server193 CPU in the original PyTorch 2.12.1+cu130 environment. They computed 160 batch members and exported twelve selected predictions; no complete test cell or training run was repeated. Input/mask equality is exact, while the stored CPU-to-original-CUDA prediction differences remain explicitly recorded.

The self-contained evidence is `verification_runs/baseline_frozen_replay/final_evidence/`, including 196 receipts and twelve predictions whose copied bytes match their source hosts. `delivery_manifest.json` has SHA `9113d1531d09ae0854f0a68abce9a5cbc714e80c0521d4524b798a4fdef44af1`; the final gate has SHA `cdf3add29c92763c51014e3f31df5329b73a482acc276878a16ecdaa576f1f30`. Consult `interface_validation_summary.json` for exact commands, environment differences and per-model prediction comparisons. This completed interface check does not remove the missing VIVID checkpoint or historical training-data boundaries.

## Old Diffusion saved results and native input view

These historical 26 cells contain 26,000 original predictions (13 tasks at 100 and 1,000 steps), distinct from the newer Burgers and NS comparisons. After verification of the old-result and native-input archive receipts, the following reproduces their full saved-result statistics on CPU from the exact frozen NPZ truth and original masks. It reads neither the large original MAT files nor model weights:

```bash
FM_ARCHIVE_REPO=/research_data/users/zhangxifeng/C01Python/FM4PDE
OLD_DM="$FM_ARCHIVE_REPO/outputs/main/revision_20260909/diffusion_original_main"
DIFF_NATIVE_INPUTS="$OLD_DM/native_frozen_inputs/frozen_inputs_v3"
DIFF_OLD_RESULTS="$OLD_DM/server216/outputs"

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  python reproducibility/revision_20260909/audit_diffusion_native_inputs.py \
  --frozen-inputs "$DIFF_NATIVE_INPUTS" \
  --manifest-sha256 818bdbfc3c7e4888110f92eda0fd1a2d9bf5e5fbe98874c749ee80f3eff04f15 \
  --result-root "$DIFF_OLD_RESULTS" \
  --output "$REPLAY/diffusion_native_checks"
```

The complete source-side numerical check has already passed. It compared 19,000 available metric JSON files and 7,000 stored final-loss references (2,000 scalar Burgers losses and 5,000 paired-field loss dictionaries), with maximum absolute field-error difference 2.6646e-14. Every prediction and available metric file then matched the independently verified destination archive SHA. The immutable reference summaries live under `$OLD_DM/native_frozen_inputs/full_saved_prediction_audit`; the source-side full check and subsequent destination hash association are separate evidence, not a claim that the whole numerical audit already reran at the destination.

For **new native inference**, use `materialize_diffusion_inputs.py` and the full commands in `DIFFUSION_INPUT_REPLAY.md`. This constructs compact MAT v5/HDF5 files from the verified NPZs and emits 26 native `generate_pde.py` commands; it does not start sampling. The exact keys, dtypes and axes are preserved, including Darcy `H,W,N` and NS's singleton final frame. The generated files have their own hashes and are not the original MATs. Only the data, weight and new output paths change; original numerical settings remain fixed. Use the emitted CLI so the recovery direction, unobserved-field guidance and offsets are applied correctly. Check device-dependent observation masks against archived masks before treating a new trajectory as the same-observation experiment.

Four old native files equal the corresponding baseline Smooth source MATs by whole-file SHA. The old unsuffixed Burgers MAT instead equals the baseline ID source and must not replace the revised Burgers comparison. The original Diffusion run records do not retain their producer Git HEAD. The captured `151e721` implementation and earlier `d494d733`/`2eb3c94` trees supply source history, not a recovered original run identity. `REPRODUCTION_COVERAGE.md/json` separates this limitation from the complete input/output correspondence checks.

## 744 revised ablations and the unchanged original controls

The original study has 1,060 configurations. Eight PDEs contribute 744 revised runs; 316 original records remain unchanged. Most factor sweeps use ID input 0. The six temporal studies and Poisson endpoint-correction controls retain their original scope. The old NS checkpoint is used here.

For the completed server197 archive, the concrete paths are:

```bash
FM_ARCHIVE_REPO=/research_data/users/zhangxifeng/C01Python/FM4PDE
ABL="$FM_ARCHIVE_REPO/outputs/ablations/revision_20260909/paper_revision_20260908"
ABL_INPUTS="$ABL/frozen_inputs"
ABL_SUMMARY="$ABL/local_exports/archived_ablation_summary/archived_ablation_summary.csv"
ARCHIVED_RESULTS_VIEW="$ABL/canonical_results"
ORIGINAL_RESULTS_ROOT="$FM_ARCHIVE_REPO"
```

The archive checks for the canonical results, eleven frozen inputs, summary CSV and historical exports must be complete before use. The summary CSV SHA256 is `2c077c491d51d556069e1b46a2fd8c4764dea02b0447648670c3785fa7299e0f`. The server replay creates new outputs in a separate verification directory; it does not overwrite `local_exports` or the raw results. Set `FM4PDE_FONT_DIR` to the directory containing the four hash-verified Times New Roman files for exact rendering; `RELOCATION_SERVER197.md` records the actual directory used.

For **offline recomputation**, first provide an eleven-PDE results view. Each `$ARCHIVED_RESULTS_VIEW/<pde>` must contain that PDE's frozen `selection.json`, completion markers, `main/` and `ensemble/` raw results. The archive inventory designates `outputs/ablations/revision_20260909/paper_revision_20260908/canonical_results/` for this unified tree, with the original host directories also preserved. Confirm that transfer and its checksum receipt have completed before using it. An independently constructed view must select exactly one authoritative directory per PDE. Existing absolute `records.json` paths are not portable. Rebuild a new snapshot from raw before plotting:

```bash
python plot/collect_paper_ablation_fields.py \
  --archive "$ABL_SUMMARY" --inputs "$ABL_INPUTS" \
  --results "$ARCHIVED_RESULTS_VIEW" --original-root "$ORIGINAL_RESULTS_ROOT" \
  --output "$REPLAY/ablation_publication_snapshot"
python plot/export_paper_ablation_tables.py --source "$REPLAY/ablation_publication_snapshot" --output "$REPLAY/ablation_tables"
python plot/plot_paper_ablation_fields.py --source "$REPLAY/ablation_publication_snapshot" --output "$REPLAY/ablation_figures" \
  --contract reproducibility/revision_20260909/legacy_scripts/contracts/ABLATION_CHART_CONTRACT.md
python plot/plot_paper_ablation_sweeps.py --source "$REPLAY/ablation_publication_snapshot" --output "$REPLAY/ablation_figures"
```

`$ORIGINAL_RESULTS_ROOT` precedes the portion of `source_result` after `/FM4PDE/`, which already starts with `outputs/ablations/`. For the still-preserved server197 originals this root is `/research_data/users/zhangxifeng/C01Python/FM4PDE`; an archived replacement must preserve that suffix or supply a corresponding view. The local `original_predictions/` used in this regression contains only 45 of the 316 unchanged raw predictions. The other 271 retain their exact archived summary values, but cannot be independently recomputed from this local subset. All 744 revised raw predictions are present.

The reusable CPU regression creates relative symlinks without copying raw data, then recomputes the old and new paths under the same two-thread environment. It checks all 1,060 records, 20 tables, the complete original three-draw study and 80 applicable figures. The `--results` argument selects a unified eleven-PDE directory; alternatively, `--result-map` maps each PDE to its authoritative archived directory. Use explicit `--archive`, `--inputs`, `--original-root`, `--snapshot` and `--ensemble` for the origin-preserving archive. `--original-manifest` also requires all 316 original raw tensors and independently recomputes their field errors. Use a new output directory each time:

```bash
python reproducibility/revision_20260909/check_ablation_relocation.py \
  --study "$ABL" --archive "$ABL_SUMMARY" --inputs "$ABL_INPUTS" \
  --results "$ARCHIVED_RESULTS_VIEW" --original-root "$ORIGINAL_RESULTS_ROOT" \
  --original-manifest reproducibility/revision_20260909/in_place_reference_checks.json \
  --snapshot "$ABL/local_exports/ablation_publication_snapshot" \
  --ensemble "$ABL/local_exports/ensemble_complete" \
  --figure-reference-manifest reproducibility/revision_20260909/ablation_final_figure_reference.json \
  --verify-source-hashes --output "$REPLAY/ablation_relocation"
```

The old local subset limitation does not apply to the full server archive: all 316 original controls are available there. Their float64-recomputed norms may differ from the historical recorded precision. The optional diagnostic check below substitutes those independent norms only in a fresh table snapshot and verifies that every displayed value and rank is unchanged:

```bash
python reproducibility/revision_20260909/check_original_control_tables.py \
  --snapshot "$REPLAY/ablation_relocation/snapshot" \
  --control-audit "$REPLAY/ablation_relocation/original_control_recomputation.json" \
  --published-tables "$REPLAY/ablation_relocation/relocated_tables" \
  --output "$REPLAY/original_control_table_precision"
```

`local_exports/ablation_publication_figures` and `ensemble_complete/figures` retain earlier figure layouts. Final figure references are the 80 entries in `ablation_final_figure_reference.json` and the complete `paper_assets_final_20260909/figures` archive. Numerical CSV/NPZ references remain in the historical exports.

For **new sampling with the frozen formal protocol**, use the recorded producer checkout and environment. Restore each frozen `selection.json` beneath a fresh `$NEW_RESULTS/<pde>/`, and run:

```bash
python plot/run_paper_ablation_revision.py rerun --inputs "$ABL_INPUTS" --output "$NEW_RESULTS" --pdes "$PDE"
```

Repeat for the eight revised PDEs. Collect the new output using `--results "$NEW_RESULTS"`; do not accidentally point that comparison at `$ARCHIVED_RESULTS_VIEW`. Unchanged controls still come from the original summary and original-result root. Re-estimating selection is a separate experiment. The historical `prepare` → `diagnose` → `select_paper_ablation_revision.py` stages require the original dataset/checkpoint/configuration sources and recommendation JSON; they are not prerequisites for recomputing frozen archived outputs. If preparation is necessary, target a fresh input directory and use the recorded source version, not current default YAML. `run_paper_revision_queue.py` is a sampling coordinator, not an offline collector.

Required data: original summary CSV, each PDE's frozen inputs/protocol/selection, revised `result.pt`, `curves.csv` and receipts, and the original control predictions needed by the requested plots. Weights are needed for new sampling; these CPU ablation collectors do not load them. Verify the actual collector's `--archive` input is the CSV, not the root archive directory. `legacy_scripts/paper_audit/revision_0909/qa_ablation_delivery.py` preserves the independent field-error checks and the final 85-figure verification; its paper paths must be restored explicitly. None of these steps calls for rerunning deleted main-text NS calibration/residual-selection sections.

## Eleven-PDE, 32-input, three-draw study and spectra

This is the original 1,056-prediction study, with 21 reported fields and input offsets 1500–1531. Keep the old NS model and frozen observation layout. It is distinct from the new K-scaling study.

```bash
python plot/export_paper_seed_ensemble.py --inputs "$ABL_INPUTS" --results "$ARCHIVED_RESULTS_VIEW" --output "$REPLAY/ensemble_complete" \
  --plan reproducibility/revision_20260909/legacy_scripts/contracts/ENSEMBLE_ANALYSIS_PLAN.md
python plot/plot_paper_seed_ensemble.py --source "$REPLAY/ensemble_complete" --output "$REPLAY/ensemble_figures" \
  --contract reproducibility/revision_20260909/legacy_scripts/contracts/ENSEMBLE_CHART_CONTRACT.md
```

These commands reanalyze archived raw predictions. For new sampling, first restore each PDE's frozen selection beneath `$NEW_RESULTS/<pde>/`, run `python plot/run_paper_ablation_revision.py ensemble --inputs "$ABL_INPUTS" --output "$NEW_RESULTS" --pdes "$PDE"` for all eleven PDEs, and pass that same `$NEW_RESULTS` to the exporter only after completion. The exporter retains single draws and physical-field averages, uses input-level sample SD and paired bootstrap, and verifies complete field/seed coverage. Its output NPZ/CSV/manifest files are inputs for the eleven spectra. Preserve `analysis_plan.md`/plan hash and all saved prediction tensors; do not count three draws as three independent physical inputs. `collect_conditional_sample_scaling.py` belongs to the separate new K study and is not an offline collector for this original three-draw study.

Matched Poisson/Darcy baseline frequency diagnostics use `plot/export_matched_spectra.py --study "$MATCHED_STUDY" --output "$REPLAY/spectra"`, then `plot/plot_paper_frequency_revision.py --source "$REPLAY/spectra" --output "$REPLAY/frequency_figures"`. Required artifacts include matched predictions, masks/truth and baseline source mappings. The plotting source contains 1,216 spectral records. `legacy_scripts/paper_audit/revision_0909/rank_frequency_table.py` preserves the final unrounded ranking rule for the four displayed metrics. `plot/spectral_diagnostics.py` implements the common FFT/DCT and field-filtering definitions.

## Burgers supplementation

The formal extension added 7,000 FM/Diffusion calls and combined them with fixed baseline and archived random-FM scores. There are two distinct observation geometries: random space-time observations and structured time layers. Do not replace a time-layer mask with a spatial-column mask.

Preparation/execution entries are `plot/prepare_burgers_revision.py`, `plot/run_burgers_revision.py` (`freeze`, `worker`), and `plot/run_burgers_parallel.py` (immutable redistribution and native workers). Their CLI takes the input protocol, FM/Diffusion weights and `--diffusion-root`. Preserve `sampling_protocol_v3.json`, `burgers_parallel_assignment_2242.json`, 4,448 prior and 2,552 redistributed calls, all generated workers/samplers, exact pilots and executor receipts.

```bash
BURGERS_STUDY=/research_data/users/zhangxifeng/C01Python/FM4PDE/outputs/main/revision_20260909/burgers_revision_20260908
python plot/export_burgers_revision.py export \
  --inputs "$BURGERS_STUDY/local/burgers_inputs" \
  --sampling-protocol "$BURGERS_STUDY/local/burgers_inputs/sampling_protocol_v3.json" \
  --results "$BURGERS_STUDY/server193/burgers_output_v3" \
            "$BURGERS_STUDY/server197/burgers_output_v3" \
            "$BURGERS_STUDY/server216/burgers_output_v3" \
  --fm-random "$BURGERS_STUDY/server197/burgers_fm_random" \
  --output "$REPLAY/burgers_complete"
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

The complete new-NS study has now actually been reread from `outputs/main/revision_20260909/ns_main_revision_0909/local/complete_local_ns_study` on server197. [NS_ARCHIVED_RECOMPUTATION.md](NS_ARCHIVED_RECOMPUTATION.md) gives the precise archived-path command, isolated source commit, environment and report. All 15,000 examples and 450 batches reproduce the five stored metrics and 15-cell mean/SD values under the frozen reductions; independent residual/runtime checks also match. The separate float64 FFT residual diagnostic remains within the original CUDA-float32 comparison tolerance. All 1,146 source files retain their hashes. This 158.7-second, two-thread CPU check repeats only saved-result computations. Its exact report SHA is `5ca94e91a2c2315103dd40739e30d081660756b1665929b6d48f112ff8c4568f`; full evidence is included with the final paper-assets audit archive. No model sampling or training is implied.

The completed eight-table candidate export can be checked independently against its preserved pre-update sources and the complete NS summary:

```bash
python reproducibility/revision_20260909/check_ns_table_export.py \
  --paper "$PAPER" --study "$NS_STUDY" \
  --export-root "$PAPER/audit/revision_0909/ns_table_updates"
```

This verifies all 312 displayed main/comparison cells and rank marks, unchanged source rows, complete input/output hashes and eight PDF pages. It writes `final_export_qa.json` beneath the audit directory, including explicit exceptions to any claim that Rough has the largest error. The preserved `prepared_sources.json` is the pre-update baseline, not the already integrated canonical CSVs; do not overwrite it when validating the final candidate export.

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

The four shards and collector have now terminated completely: all 96,000 canonical trajectories, 106,944 total timed trajectories and 480 result receipts passed their completion gates. Independent verification passed 2,361 compact-field/statistical comparisons, twenty unrounded table cells, sixteen adjusted paired intervals, fifteen latency summaries and the three standalone figures. The final scientific review is `paper/audit/revision_0909/conditional_scaling_final_review/final_scientific_review.json`, SHA `8d3a539b5118a72981c4b62a340968d4880e1631693fdf4357a114aa5d4d0801`; its final data-manifest SHA is `9a6579ef292c7bf3e2c5d3d42cb27144a389f3a537de61f599c72374141bbe18`.

The approved `archive_inventory.conditional_scaling_final.json` has SHA `c473fa1979647f40144b515cbae49550b8b17edcdda6f628bbc29b98452f9ad9`. Its three entries total 14,385,940,996 bytes. Transfer started at 2026-09-09 18:37:55 +08:00 and is not yet declared complete. Require the independent copy/verify receipt for each entry before replaying from the destination. The target mapping is:

```bash
K_STUDY=/research_data/users/zhangxifeng/C01Python/FM4PDE/outputs/ablations/revision_20260909/conditional_scaling_20260909
K_AUDIT="$K_STUDY/local_audit"
# Original raw layouts:
# "$K_STUDY/server197/results/production"
# "$K_STUDY/server216/results/production"
# Compact host exports used by the checker:
# "$K_AUDIT/server197" and "$K_AUDIT/server216"
```

Restore the current `paper_assets_final_20260909` into a separate working copy and set `$PAPER` to it. The recommended `build_reproduction_paper.py --output <new-build-directory>` restores and verifies compilation into `<new-build-directory>/paper`; that `paper` subdirectory is the value of `$PAPER`. Its `source_data/conditional_scaling_0909` contains the final merged CSV/NPZ and summaries. Set `$REPLAY` to a fresh verification directory. The following command writes a new JSON report beneath it; `--output` is a new report filename, not an existing frozen result or a directory:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  python reproducibility/revision_20260909/verify_conditional_scaling_summary.py \
  --paper "$PAPER" --audit "$K_AUDIT" \
  --output "$REPLAY/conditional_scaling/numerical_review.json"
```

The checker recomputes both field errors and observation MSEs from all 480 physical means, then checks the twenty mean/SD/pointwise-interval summaries, sixteen paired intervals and fifteen latency summaries. It also compares every merged numerical row and all 672 arrays with the two complete host exports. Those manifests retain all 480 raw-result checksums for exactly 96 distinct task/input groups. Original absolute source-path strings are preserved as provenance; the command reads the explicit `--paper` and `--audit` roots and does not rewrite those records. Raw-pool identity, PDE residuals and stochastic-noise correspondence remain established by the upstream host audits, rather than being newly reconstructed by this compact-data check. The checker does not sample models or grant a new scientific/visual review. The prior one-pool path-regression receipt remains historical evidence alongside this complete verification.

Training-curve rendering uses:

```bash
python plot/plot_training_curves_revision.py --paper "$PAPER" --new-ns-log "$NEW_NS_TRAINING_LOG"
```

Retain original eleven-model logs and their SHA256 manifest plus the separate new NS log and training metadata. The revised three figures have twelve model panels, explicitly distinguishing old ablation NS from new main NS. Do not run `plot_jmlr_revision.py` indiscriminately over an integrated paper: its older all-in-one path writes historical main/training source data as well as figures.

## Exact figure environment and mapping checks

The final local figures used Python 3.12.11, Matplotlib 3.11.0, FreeType 2.14.3, Pillow 12.2.0 and NumPy 2.4.6. The full 59-distribution observation is `environments/local_final_plot_20260909.requirements.txt`, with interpreter, font hashes and scope in the adjacent JSON. Existing remote environment observations remain unchanged. Set the font path from the repository root:

```bash
export FM4PDE_FONT_DIR="$PWD/reproducibility/revision_20260909/dependencies/fonts/times_new_roman"
```

The four private font files are present at that relative path locally and on server197; `environments/final_plot_fonts_20260909.json` records their hashes. Reproduce the rendering versions in an isolated environment to compare exact PNG pixels. Scientific values and table ranks matched on server197's different rendering stack; its 80 PNGs are not byte- or pixel-identical to the final local images. Neither the shared server environment nor the final paper figures was changed.

The data/label mapping comparison consumes the two exports' three figure subdirectories and field annotation CSV. It verifies all 80 figures, captions and source mappings independently of rasterization:

```bash
python reproducibility/revision_20260909/check_ablation_figure_mapping.py \
  --reference "$REFERENCE_REPLAY" --replay "$SERVER_REPLAY_MANIFESTS" \
  --output "$REPLAY/figure_data_label_mapping.json"
```

`collect_plot_environment.py --output <new-observation.json> --font-dir "$FM4PDE_FONT_DIR"` records the installed plotting packages and fonts without importing PyTorch or initializing CUDA. It refuses to overwrite an existing observation.
