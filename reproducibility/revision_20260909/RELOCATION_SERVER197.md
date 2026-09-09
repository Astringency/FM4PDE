# Complete server197 ablation archive replay

**The numerical results agree.** On 2026-09-09, the completed server197 archive was reread in an isolated Git checkout using two CPU threads and no CUDA. All 744 revised ablations, 316 original controls and 1,056 original three-draw predictions were checked. The 80 figures were regenerated with the server rendering environment; their pixels are not claimed to match the final local figures.

## Actual paths and source versions

The raw study is `/research_data/users/zhangxifeng/C01Python/FM4PDE/outputs/ablations/revision_20260909/paper_revision_20260908`. The authoritative subdirectories are `frozen_inputs/`, `canonical_results/`, and `local_exports/{ablation_publication_snapshot,ensemble_complete}`. The original CSV is `local_exports/archived_ablation_summary/archived_ablation_summary.csv` (SHA256 `2c077c491d51d556069e1b46a2fd8c4764dea02b0447648670c3785fa7299e0f`). Their archive checks completed before the replay.

The 316 original-control tensors remain in their independently verified `outputs/ablations/...` locations. `--original-root /research_data/users/zhangxifeng/C01Python/FM4PDE` resolves those recorded suffixes. All 316 actual hashes matched `in_place_reference_checks.json`; no missing local subset was substituted.

The independent review directory is `/research_data/users/zhangxifeng/C01Python/FM4PDE/reproducibility/revision_20260909/verification_runs/ablation_server197_20260909`. Its `code/` checkout was restored from a locally created Git bundle at `b059d6a6b7e9d71eca6c89c6076bdaec734002c5`; `replay/` contains fresh outputs and a 24-link relative reading view. The optional precision check used a separate Git worktree at `a7d098a`. No source tensor, protocol, original snapshot, canonical paper file or live producer was edited. All 8,862 consumed result/protocol/receipt/mask/curve files retained their SHA256 values after the replay.

The exact complete invocation, every child command, source hash and log hash are preserved in `RELOCATION_SERVER197.json`. The reusable archive command is in `STUDY_RECIPES.md`; it specifies the CSV, eleven-PDE inputs/results, original root and original hash manifest, historical snapshot/ensemble, final figure-reference manifest and a new output directory. It also enables `--verify-source-hashes` and uses `--font-dir` for the private fonts.

## Numerical and table checks

| Check | Result |
|---|---|
| 744 revised ablations | All raw tensors reread; original result hashes, selected settings and terminal metrics verified |
| Complete 1,060-record inventory | All non-path record values and the unrounded CSV exactly match between direct archive paths and the new relative view |
| 316 original controls | 632 coefficient/solution norms independently recomputed in float64; all pass the recorded-precision tolerance |
| Twenty table fragments | All values, formatting and ranks identical to the historical-snapshot export |
| Original-control precision sensitivity | Substituting all 632 independently recomputed norms into a new diagnostic snapshot leaves all 20 table fragments byte-identical |
| Original three-draw study | All 1,056 raw predictions verified; 21 summary, 672 per-input and 24,192 frequency rows exactly match the historical CSVs |
| Spectral arrays | All 84 arrays exactly equal, both between reading paths and against historical NPZ data |
| Eighty figure data/label mappings | 38 field figures, 30 sweeps and 12 ensemble figures retain their source/configuration/PDE/field mappings and identical generated captions |

The 744-rerun comparison with historical floating-point reductions has 1,132 different stored error values, with maximum absolute difference `2.220446049250313e-15` and relative difference `1.7317480407323052e-15`. The independent original-control float64 norms differ from their older reported values by at most `3.020880052773123e-7` absolute and `1.444894316281089e-7` relative. These precision differences do not change any displayed table value or rank. No formula or old data was changed to force equality.

Figure checks cover all 110 raw result/curve sources used by the field panels, every sweep's PDE/group/condition/source-ID mapping, and the ensemble PDE/field/estimator mappings. All 168 field annotations were checked: 157 numerical strings differ only at reduction roundoff, at most `1.2789769243681803e-13` absolute (`4.424663844301175e-15` relative). Other annotation fields and all captions match. The separate figure-mapping checker can be rerun on the local reference manifests and the downloaded server manifests; the raw results remain in the server archive.

## Rendering and typography

| Library | Final local figures | Server197 replay |
|---|---|---|
| Python | 3.12.11 | 3.12.11 |
| Matplotlib | 3.11.0 | 3.10.7 |
| FreeType | 2.14.3 | 2.6.1 |
| Pillow | 12.2.0 | 11.3.0 |
| NumPy | 2.4.6 | 2.3.3 |

All four font files have matching hashes, but the rendering libraries differ. None of the 80 PNGs is pixel-identical, and only one retains the exact pixel dimensions. The numerical figure data and labels pass the separate comparisons above. Poisson spectra, Wave guidance and a three-PDE seed sweep were inspected visually and remain legible and correctly labelled; the revised spectrum ticks do not overlap. The sampled spectrum PDF embeds Times New Roman regular and italic fonts. No server-rendered figure replaced a canonical paper figure, and no shared environment was changed.

`environments/local_final_plot_20260909.json` records the actual final plotting interpreter, versions, 59 installed distributions and font hashes without importing PyTorch or initializing CUDA. Its `.requirements.txt` is the full installed-version list. `environments/ablation_render_comparison_20260909.json` records the observed differences. Exact PNG reproduction requires the final local rendering versions, including the FreeType build, as well as the fonts.

The four private font files (4,227,900 bytes total) are present in **both** local and server197 FM4PDE trees under `reproducibility/revision_20260909/dependencies/fonts/times_new_roman/`. Paths and hashes are in `environments/final_plot_fonts_20260909.json`. Restore their lookup from the repository root with:

```bash
export FM4PDE_FONT_DIR="$PWD/reproducibility/revision_20260909/dependencies/fonts/times_new_roman"
```

The original Windows lookup remains the default when that environment variable is absent. Its explicit-path regression reproduced all 12 local ensemble PNGs exactly before the remote replay.

## Resources and retained artifacts

The replay ran from 07:37:30 to 07:41:55 UTC. Sequential child time was 248.576 seconds, with peak child RSS 840.4 MiB. Preflight found 942,054 MiB available memory and load 1.51; CUDA was hidden from every processor. Both task-specific tmux sessions ended and were absent from the final session listing. Other sessions were untouched.

The server review directory retains `code.bundle`, `precision_code.bundle`, `code/`, `precision_code/`, `dependencies/` (launch, resource and archive-gate receipts), `replay/` (new snapshots, tables, three-draw statistics, all 80 PDF/PNG pairs, before/after source hashes, raw-control audit and logs), and `original_control_table_precision/`. The publication archive remains authoritative for the final figures; older `local_exports` figure folders intentionally preserve earlier layouts. Local small-report and representative-image copies are in `verification_runs/ablation_server197_evidence/`. The source/target artifact list is `RELOCATION_SERVER197_ARTIFACTS.json`.
