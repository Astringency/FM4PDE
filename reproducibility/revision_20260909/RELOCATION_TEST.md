# Ablation and original three-draw relocation regression

**Passed on 2026-09-09.** The subsequent complete server197 archive replay, including independent recomputation of all 316 original controls, is recorded in `RELOCATION_SERVER197.md` / `.json`. This report preserves the earlier local-subset test. The same frozen raw results were processed through their original local paths and a new eleven-PDE view inside FM4PDE. All numerical comparisons between those two paths were exact. No GPU, remote transfer, original-data write, protocol edit or production-code change was performed.

## Executed command

Working directory: `/home/tat512/C01Python/FM4PDE`. Python: `/home/tat512/.conda/envs/fm4pde/bin/python`.

```bash
/home/tat512/.conda/envs/fm4pde/bin/python reproducibility/revision_20260909/check_ablation_relocation.py --study /home/tat512/C01Python/audit/paper_revision_20260908 --output /home/tat512/C01Python/FM4PDE/reproducibility/revision_20260909/verification_runs/ablation_relocation --paper-figures /home/tat512/C04Papers/fm4pde_jmlr/figures
```

The driver hides CUDA and sets OpenMP, MKL, OpenBLAS and NumExpr thread counts to two for each child. All processors run sequentially. Total child runtime was 352.220 seconds; peak child RSS was 825.7 MiB. The report records the exact command, source SHA256, runtime and log hash for every child.

## Results

| Check | Result |
|---|---|
| Revised raw ablations | All 744 predictions reread; original result hashes and terminal metrics checked by the collector |
| Complete ablation inventory | 1,060 records, including 316 unchanged controls; all non-path values exactly equal between old/new path recomputations |
| Unrounded field CSV | Byte-identical between same-environment original/new path recomputations |
| Table export | All 20 TeX fragments byte-identical; also identical when exported from the historical snapshot |
| Original three-draw raw | All 1,056 predictions, across 11 PDEs and 21 fields, reread and verified |
| Three-draw statistics | 21 summary rows, 672 input/field rows and 24,192 frequency rows byte-identical; also byte-identical to historical CSVs |
| Spectral arrays | All 84 arrays elementwise equal; also equal to the historical NPZ arrays |
| Figures | 38 field panels, 30 sweep figures and 12 ensemble/spectrum figures successfully rendered as PDF and PNG |
| Existing paper figures | All 80 PNGs exactly pixel-identical to the integrated paper figures |

Wave guidance and Poisson spectra were also inspected visually. Field labels, metric annotations and legends remained legible, and the revised symlog ticks did not overlap. Since every PNG is pixel-identical, this replay introduces no figure-layout change. The five separate matched-frequency figures are outside this 744/three-draw task and were not regenerated.

## Historical floating-point difference

The historical ablation snapshot used a different numerical reduction environment. Against the new two-thread recomputation, 1,376 field-error values differ at roundoff scale: maximum absolute difference `3.9968028886505635e-15`, maximum relative difference `7.157387328832945e-15`. Other non-path record values are unchanged. The same-environment old/new path comparison is exact, all displayed table values and ranks are identical, and all figure pixels are identical. No formula or old raw data was changed to force agreement. Full differences are saved in `verification_runs/ablation_relocation/historical_ablation_roundoff.json`.

## Reading view and original controls

The new view has 24 relative symlinks: eleven input directories, eleven authoritative result directories, the original-control subtree and the archived summary CSV. It copies no raw tensors. `view_manifest.json` records each new path, relative link target and resolved original target. The collector creates a fresh `snapshot/records.json`; all 789 available prediction paths and all 744 receipt paths point through the new view. Historical `source_result` strings are preserved as provenance.

The local original-control subset contains 45 of the 316 unchanged prediction tensors. The other 271 are absent locally and continue to use their unchanged archived summary values. This replay does not claim to independently recompute those missing tensors. Separately, the archive agent verified that all 316 original `source_result` tensors exist on server197 and recorded their SHA256 values in `in_place_reference_checks.json`; its 45 comparable local copies matched. That existence/hash check is distinct from numerical recomputation.

For those original paths, `--original-root /research_data/users/zhangxifeng/C01Python/FM4PDE` is valid while the original tree remains preserved: the collector appends the suffix after `/FM4PDE/`, already beginning with `outputs/ablations/`. The unified relocated archive target is `outputs/ablations/revision_20260909/paper_revision_20260908/canonical_results/`; its transfer verification remains an independent archive task.

## Reuse after archive migration

Use `check_ablation_relocation.py --result-map <JSON>` to select the eleven authoritative PDE directories in the archive; the JSON keys are the PDE names and values are directory paths. Explicit `--inputs`, `--original-root`, `--snapshot` and `--ensemble` override default study locations. The driver refuses to reuse an existing output directory. Frozen snapshots and receipts are never edited.

For ordinary offline publication exports, use the rebuilt snapshot and explicit results view as shown in `STUDY_RECIPES.md`. For new sampling, restore the frozen per-PDE selections into a fresh result tree and pass that same tree to both producer and exporter. Re-running prepare/diagnose/selection is not required for archive recomputation and would be a different experiment.

## Deliverables

- `check_ablation_relocation.py`: reusable CPU view/recomputation/comparison driver.
- `RELOCATION_TEST.json`: complete checked results, exact executed child commands and hashes.
- `verification_runs/ablation_relocation/`: relative view, both recomputed snapshots, three table exports, both three-draw exports, all 80 PDF/PNG pairs and logs. This generated directory is ignored by Git.
- `STUDY_RECIPES.md`: separate offline-recomputation and new-sampling instructions.
