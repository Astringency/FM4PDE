# Complete Navier–Stokes main recomputation from the archive

The server197 archive was independently reread on 2026-09-09 using two CPU
threads and no CUDA. All 15,000 examples, 15 cells and 450 original batches
passed. This repeats saved-result computations; it does not run model inference
or change the selected model, parameters, observations or original predictions.

The unchanged exporter reproduces all five numeric columns of the 15,000-row
CSV and all ten mean/SD columns of the 15-cell summary exactly. The only
nonnumeric change is the explicit relocation of the 15,000 `result` paths.
The independent NumPy residual/runtime audit is also identical to its frozen
report, including the largest residual differences. The 1,146 archived files
(7,149,064,114 bytes) pass full hashes before and after recomputation. Original
protocol, configuration, input, receipt and checkpoint bytes are unchanged.

A separate export replaces the stored CUDA float32 PDE MSEs with independently
computed NumPy float64 FFT residuals. The largest per-example differences are
1.9179106894307019e-7 absolute and 1.102904492267298e-5 relative, within the
original audit tolerance. Across the 15 cells, the largest PDE-MSE mean and
sample-SD differences are 7.201141233662001e-9 and 8.292218496921966e-9,
respectively. All non-PDE metrics remain exactly equal. These alternative
exports are diagnostics and do not overwrite the frozen scientific results.

## Evidence and storage

- [NS_ARCHIVED_RECOMPUTATION.json](NS_ARCHIVED_RECOMPUTATION.json) is an exact
  copy of the complete remote report; SHA256 is
  `5ca94e91a2c2315103dd40739e30d081660756b1665929b6d48f112ff8c4568f`.
- [NS_ARCHIVED_RECOMPUTATION_MANIFEST.json](NS_ARCHIVED_RECOMPUTATION_MANIFEST.json)
  is the exact delivery manifest; SHA256 is
  `786ab15fb01b1c5898efe41660306dba7958e8b7e8e3e231d0d85bfa898ce163`.
  Its paths are relative to `audit/revision_0909/ns_archive_recompute/` within
  the paper archive. It lists 20 files totaling 26,686,424 bytes, excluding
  the manifest itself. The complete evidence is included by the final
  `paper_assets.py` capture of the paper's `audit/` directory, rather than
  duplicated as a new experimental-data archive.
- The original local evidence is
  `/home/tat512/C04Papers/fm4pde_jmlr/audit/revision_0909/ns_archive_recompute/`.
  It contains the exact launch, environment and dependency receipts, both
  full per-example CSVs and summaries, independent residual/runtime audit,
  stage logs and a local-copy hash check. The whole directory is frozen.
- The independent remote output is
  `/research_data/users/zhangxifeng/C01Python/FM4PDE/reproducibility/revision_20260909/verification_runs/ns_main_cpu_20260909T074242Z/recomputed/`.
  All downloaded outputs match the remote report's hashes; the complete
  report itself was independently hashed on both hosts.

The verified source study is
`/research_data/users/zhangxifeng/C01Python/FM4PDE/outputs/main/revision_20260909/ns_main_revision_0909/local/complete_local_ns_study`.
Its `.revision_archive_receipt.json` SHA256 is
`290c44ef780b62c9aa85e933cbdd79e23257166faec147c6fe5f90489d467262`.
The verification checks that receipt, the entire archived file tree, pinned
protocol/selection/model/input/summary hashes and all 450 raw batches before
numerical work. No source path strings inside original records are rewritten.

## Repeating the verification

The actual source commit is
`e4351366436ec912109ec71aa719476d620ef4f7`, retained in the source archives and
Git history. Restore it in an isolated checkout. Set `NS_REPLAY_CODE` to that
checkout, `NS_STUDY` to the verified study above and `NS_REPLAY_OUTPUT` to a
new directory below `FM4PDE/reproducibility/revision_20260909/verification_runs/`.
The output must not be inside the immutable study. After checking resources,
run in a separate task-specific tmux session:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  NUMEXPR_NUM_THREADS=2 \
  /research_data/users/zhangxifeng/.conda/envs/fm4pde/bin/python \
  "$NS_REPLAY_CODE/reproducibility/revision_20260909/verify_ns_archived_main.py" \
  --study "$NS_STUDY" --output "$NS_REPLAY_OUTPUT" \
  --receipt-sha256 290c44ef780b62c9aa85e933cbdd79e23257166faec147c6fe5f90489d467262
```

The completed run used Python 3.12.11, PyTorch 2.8.0+cu128 and NumPy 2.3.3;
CUDA was explicitly hidden. Its complete package versions and code hashes
are in the evidence's `recomputed/environment.json`. It completed in 158.70
seconds, returned zero and left no task tmux session. This elapsed time is
verification cost, not a sampling-time measurement. A repeat must retain the
original comparison tolerances and report every discrepancy rather than
modify the archived inputs or outputs.

For the separate old-model ablation and original three-draw replay, see
[RELOCATION_SERVER197.md](RELOCATION_SERVER197.md). The new main NS model
remains distinct from the old NS model used there. Broader inference and
training boundaries are recorded in
[REPRODUCTION_COVERAGE.md](REPRODUCTION_COVERAGE.md).
