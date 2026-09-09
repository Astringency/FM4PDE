# Preserved revision scripts

This directory contains 17 selected scripts and three registered analysis/figure specifications that were outside FM4PDE. All 20 files are byte-for-byte copies; their originals remain in place. The files total about 83 kB. Source paths, SHA256 hashes, purposes, required inputs, outputs, imports, and replay restrictions are recorded in [external_script_inventory.json](../external_script_inventory.json).

The inventory examined 193 standalone external audit scripts and counted another 1,859 files in historical code/dependency snapshots. No executable Python, shell, or notebook file was found under the paper's `source_data/`; its manifests, CSVs, workbooks, compressed records, and field arrays are result inputs, not missing production code. Other `C01Python` repositories are listed as separate projects or external source dependencies. Their version snapshots are handled by `../source_repositories.json` and the root reproducibility documentation.

| Directory | Retained purpose |
|---|---|
| `paper_audit/revision_0909/` | Independent raw ablation/figure verification; unrounded spectral ranking; gated NS table statistics; actual NS configuration extraction; NS timing reduction; dataset metadata; analytic viscous figure QA. |
| `paper_revision_20260908/` | Burgers native-call ownership, preserved-call hashes, redistributed jobs, and exact pilot checks. |
| `ns_checkpoint_comparison_20260908/` | Historical checkpoint and step-probe overlap verification, final reporting, and original launch/collection records. |
| `ns_loss_spectrum_20260907/` | Independent complete old-NS loss-study statistics, bootstrap, conditional-mean identity and spectral decomposition checks. |
| `contracts/` | Registered 11-PDE/32-input/three-seed analysis plan and ablation/ensemble figure contracts used by existing exporters. |

Use [STUDY_RECIPES.md](../STUDY_RECIPES.md) for the supported producer, audit, and plotting entry points already in `plot/` and `scripts/`. The 744 ablation reruns, all 11 PDEs in the original three-draw study, Burgers supplementation, new NS main evaluation, and controlled timing require no new external production script.

## Verify this collection

From the FM4PDE repository root:

```bash
python reproducibility/revision_20260909/legacy_scripts/verify_inventory.py
```

This checks all copy hashes, parses Python files, and runs `bash -n` on the four historical shell scripts. It executes no experiment, imports no study module, and performs no network operation. It also reports any originals that have changed since this snapshot. Original source paths can be unavailable on another machine while the preserved copies remain verifiable.

## Replay scope

The copied files intentionally preserve their original bytes, including original path assumptions. Many legacy checks derive the study or paper directory from `__file__`; others contain explicit historical paths. **Do not run all files in this directory as a batch.** Use the current repository CLI whenever available. To replay an individual legacy check, provide the documented study/paper layout and adapt path-only configuration explicitly; retain the copy's hash and a record of that adaptation. The original numerical expressions and frozen input protocols should remain unchanged.

The eight-table tool is directly reusable with explicit paths:

```bash
python reproducibility/revision_20260909/legacy_scripts/paper_audit/revision_0909/ns_table_update.py prepare \
  --paper "$PAPER" --study "$NS_STUDY" --output "$PAPER/audit/revision_0909/ns_table_updates"
```

Its `export` mode requires complete collection, the independent raw audit, 15 complete cells of 1,000 examples, and agreement with all 15,000 per-sample records. It writes candidate fragments beneath `paper/audit` only. Its copied version is a snapshot of an active audit tool; refresh that copy and manifest if the original changes during final integration. Do not replace the frozen old source statistics before the prepared-source gate has been used.

The four `launch*.sh` files and two collectors preserve historical host, GPU, checkout and shard settings. They are provenance records, not current launch instructions. The old checkpoint finalizer includes fixed historical narrative values and must not be used to generate new-main-experiment conclusions. The complete old-NS loss check belongs to the supplement's original checkpoint study; it does not use the new 44M NS model.

## Exclusions and remaining data dependencies

Pure TeX/prose integration, reference checking, manuscript relocation and table typography scripts are outside numerical reproduction and remain in the paper audit. Historical partial/pilot checks superseded by complete audits are inventoried but not duplicated. Full historical FM4PDE checkouts and vendored third-party libraries belong in source snapshots. Automatically generated Diffusion samplers and Burgers workers remain with the exact receipts/raw results that hash them; their generators are already tracked in `plot/`.

Reproduction also needs checkpoint weights and normalizers, fixed input caches/masks, original data/row mappings, receipts, raw predictions, and the recorded external baseline version. In particular, controlled NS timing uses DiffusionPDE commit `151e721b9991404154ad4b430ab85cdaedbaa399`. These are not replaced by a CSV or by this code collection. Migration to server197 `outputs/{main,ablations}/revision_20260909/<study>/` is managed separately; this task neither moved results nor changed a remote checkout.
