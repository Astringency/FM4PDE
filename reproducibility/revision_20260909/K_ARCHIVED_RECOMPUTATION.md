# Complete K replay from the server197 archive

The complete saved-result replay passed on 2026-09-09. All 96 canonical pools
(96,000 saved trajectories) were reread from the verified archive on server197.
The two host exports reproduce all 480 CSV rows, 7,680 numerical values and 672
arrays exactly, including shape and dtype. Both CSV and NPZ files are also
byte-identical to their original host exports. No model inference or training
was performed.

| Original producer | Pools | Rows / arrays | CPU elapsed | Source files / bytes |
|---|---:|---:|---:|---:|
| server197 | 24 | 120 / 168 | 50.96 s | 284 / 3,561,553,404 |
| server216 | 72 | 360 / 504 | 144.18 s | 1,079 / 10,730,368,835 |

Both computations ran on server197 with two CPU threads and CUDA hidden.
The actual interpreter was
`/research_data/users/zhangxifeng/.conda/envs/fm4pde/bin/python`.
The canonical exporter SHA was
`14e7c78936fcc10a372a957dcd4a44a5665a609b6eddd7b7f8f65c0f31846cbb`;
the canonical checkouts were `5425152` and `e69e1f0`, respectively. The second
verification wrapper came from the independently synchronized `b911b2c` checkout.
Complete commands, resource observations and input hashes are in the JSON report.

The first 72-pool attempt passed the numerical comparisons, then rejected the
order of three environment records: original `[2,3,1]`, archive directory
iteration `[3,1,2]`. Each complete record is identical when matched by its unique
shard identifier. The corrected wrapper records this order difference and compares
records by `(num_shards, shard_index)`; a complete second CPU replay passed.
The initial attempt is preserved. Other manifest differences are only result
paths, resolved input/selection paths, exporter identity and resource observations.
No source manifest, numerical field or protocol was rewritten.

The separate compact checker used a newly restored final paper working copy and
the actual archived `local_audit` directory. All 2,361 comparisons passed: 480
physical means, twenty summaries, sixteen adjusted paired intervals and fifteen
latency summaries. The largest absolute differences were `3.6082e-15` for field
errors, `8.6736e-19` for observation MSE and `4.4409e-16` for statistics, identical
to the earlier compact check and within its declared tolerance. All 1,490 restored
paper files retained their hashes. This run did not repeat LaTeX compilation.

Across both raw hosts and `local_audit`, all 1,381 source files
(14,385,940,996 bytes) match their independently verified archive receipts and
retain their hashes before and after these computations. All owned tmux sessions
exited. The complete 67-entry result archive is documented separately in
[ARCHIVE_COMPLETED_67.md](ARCHIVE_COMPLETED_67.md).

## Run from the archive

Use a new output directory and the final wrapper containing `b911b2c` or later.
The default server197 shell has no bare `python`.

```bash
FM=/research_data/users/zhangxifeng/C01Python/FM4PDE
FM_PYTHON=/research_data/users/zhangxifeng/.conda/envs/fm4pde/bin/python
K_STUDY="$FM/outputs/ablations/revision_20260909/conditional_scaling_20260909"
REPLAY="$FM/reproducibility/revision_20260909/verification_runs/k_new_archive_replay"
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2

"$FM_PYTHON" "$FM/reproducibility/revision_20260909/verify_conditional_scaling_archived_host.py" \
  --repo "$FM" --archive "$K_STUDY/server197/results" \
  --inputs "$K_STUDY/frozen_inputs/poisson" \
  --selection "$FM/plot/conditional_sample_scaling_selection.json" \
  --receipt-sha256 5905709acb8764ad8917603b4c7215aeaa0e7621eb36e06aa2d95458caee4409 \
  --expected-jobs 24 --output "$REPLAY/server197"

"$FM_PYTHON" "$FM/reproducibility/revision_20260909/verify_conditional_scaling_archived_host.py" \
  --repo "$FM" --archive "$K_STUDY/server216/results" \
  --inputs "$K_STUDY/frozen_inputs/poisson" \
  --selection "$FM/plot/conditional_sample_scaling_selection.json" \
  --receipt-sha256 452a6ae73858014b58d4d2a1b8a342d2a773159555ab0e104a53df68e265ca94 \
  --expected-jobs 72 --output "$REPLAY/server216"

"$FM_PYTHON" "$FM/reproducibility/revision_20260909/paper_assets.py" restore \
  --destination "$REPLAY/compact/paper"
"$FM_PYTHON" "$FM/reproducibility/revision_20260909/verify_conditional_scaling_summary.py" \
  --paper "$REPLAY/compact/paper" --audit "$K_STUDY/local_audit" \
  --output "$REPLAY/compact/numerical_review.json"
```

The raw exporter does not use `--allow-partial`. It checks receipt hashes,
original truths, frozen configurations, masks, prefix relations and variance
identities. The compact checker independently recomputes errors and statistics;
it relies on the raw-host receipts for saved-pool/PDE/stochastic-path correspondence.
Neither command repeats sampling-time measurements or grants a new scientific
or visual review.

## Evidence

[K_ARCHIVED_RECOMPUTATION.json](K_ARCHIVED_RECOMPUTATION.json) records all report
hashes and actual commands. The detailed evidence is under
`verification_runs/k_full_archive_replay_20260909_1912/` on both the local machine
and server197; the final host reports are in `server197/` and
`server216_verified/`, with the compact result in `compact/`. The small
self-contained `K_ARCHIVED_RECOMPUTATION_EVIDENCE.tar.gz` contains 51 files;
`K_ARCHIVED_RECOMPUTATION_EVIDENCE_MANIFEST.json` lists their sizes and hashes.
The repeated NPZ files remain at the remote output paths; their hashes are
recorded and equal the original exports already in the result archive.
