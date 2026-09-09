# Replaying the archived external baselines

`replay_baseline_frozen.py` provides a direct inference interface for the
selected historical baseline runs. It uses the original baseline repository's
data adapter, checkpoint loader, normalization, hidden-target view and
`predict_physical` method. It does not call `baselines.run.main`: that CLI builds
training and validation datasets before entering its `eval_only` branch.

This interface does not reproduce training. No new baseline accuracy results
are implied by the interface checks below.

## Data and model boundaries

The selected matrix contains 156 paired-field cells and 30 Burgers cells.
The 156 cells use 48 saved models; the 18 Burgers reconstruction-network cells
use six more. The six Burgers 4D-Var cells have no learned model and repeat
the original per-instance optimization. The six VIVID cells are different:
each trained an inverse-observation VCNN on 45,000 training trajectories with
5,000 validation trajectories for 20 epochs before variational refinement.
All six effective configurations set `train_inverse_operator=True`,
`normalize=False` and `save_checkpoint=False`. Their recorded code saves a
checkpoint only when the last flag is true. The inspected evaluation and
associated training directories contain no saved VCNN. Consequently these
six original VIVID predictions can be rescored, but cannot be regenerated
from the frozen test inputs alone. Repeating the original training procedure
requires its training data and yields a newly trained model; the other six
Burgers network checkpoints cannot substitute for VIVID.

The 18 caches produced by `freeze_baseline_inputs.py` contain exactly the
native `registry.load_raw` dictionaries for original test indices 0–999:

* Nine static caches cover Poisson, Helmholtz and Darcy in three distributions.
  The loader ignores `load_full_trajectory`; freezing verifies that both flag
  values produce identical complete dictionaries. Original run flags remain
  unchanged in the replay arguments.
* Six NS caches retain separate endpoint and complete-trajectory dictionaries
  for each distribution. In particular, nine original NS joint-reconstruction
  cells requested a trajectory; endpoint-only caches cannot replace them.
* Three Burgers caches retain the complete time–space field and the native
  `initial_1d` metadata. The six existing Burgers truth/mask bundles remain
  separate evidence for their respective observation layouts.

The original loader converts field arrays to float32. The freeze procedure
independently rereads and compares the selected source arrays after precisely
that conversion and axis transformation. It records the entire source MAT
file SHA256, source field keys, original shape/dtype, selected slice, loader
commit, and each cached array's shape/dtype/C-order SHA256. The cache-file
SHA256 is separate. A derived cache is never presented as having the original
MAT-file hash. Original filename-based global IDs, file metadata, channel
names, physical parameters and time coordinates remain in the raw dictionary.

## Source selection and integrity gates

Use the original **evaluation** commit from each run's complete summary.
The seven selected commits are retained in the baseline Git history. The
default dependency checkout is not a substitute for those seven versions.
`verification_runs/baseline_frozen_replay/loader_compatibility.json` verifies
AST equality of the listed raw loaders and direct helpers across the seven
commits and the cache loader version; this limited check does not assert
model or task-adapter equivalence.

The cache manifest must be complete (`status=pass`, 18 entries, 186 selected
run bindings). The wrapper checks the cache-file hash, all cached-array
hashes/shapes/dtypes, original source-MAT identity, all 1,000 sample indices
and filename-based IDs, and the exact original summary/config hashes in the
cache's run binding. The caller must first verify the archive receipt covering
that manifest and its dependencies. Source inputs, configurations, protocols,
weights and receipts remain read-only.

Training configuration arguments are supplemented by actual evaluation fields
from the complete summary, including task, sensor seed, sensor mode, count,
noise, condition mode and trajectory flag. The native task adapter and
`PDEBatchDataset` then construct observations. The complete 1,000-example
mask-contract digest must equal the original `split_mask_manifest.test`.
This digest is not an array hash: with `--reference-manifest` the selected
saved artifacts also undergo bitwise comparisons of input, target, full field,
coordinates, masks, observation values/coordinates and physical metadata.

Saved neural models load through `load_baseline_checkpoint`, including the
original model configuration, data specification, normalization statistics and
training provenance. Predictions use `predict_physical`; its iFNO inverse
handling is retained. The wrapper never estimates new normalization statistics.
The native `_make_inference_batch` removes hidden full/target values before
prediction. Trained methods require their original checkpoint. Only the
zero-training 4D-Var branch can run without one.

## Commands

The following is a command template. Resolve every uppercase path/hash from
the verified archive manifest and the selected original run; do not substitute
the hash of a reformatted summary for the original summary file hash.

```bash
git clone BASELINE_HISTORY_BUNDLE ORIGINAL_BASELINE_CHECKOUT
git -C ORIGINAL_BASELINE_CHECKOUT checkout --detach ORIGINAL_EVALUATION_COMMIT

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /research_data/users/zhangxifeng/.conda/envs/fm4pde/bin/python \
  FM4PDE/reproducibility/revision_20260909/replay_baseline_frozen.py \
  --baseline-code ORIGINAL_BASELINE_CHECKOUT \
  --cache VERIFIED_CACHE.pt --cache-manifest VERIFIED_CACHE_MANIFEST.json \
  --summary ORIGINAL_SUMMARY.json --summary-sha256 ORIGINAL_SUMMARY_SHA256 \
  --config ORIGINAL_EFFECTIVE_CONFIG.json --config-sha256 ORIGINAL_CONFIG_SHA256 \
  --reference-manifest ORIGINAL_SAMPLE_MANIFEST.jsonl \
  --reference-root VERIFIED_SAVED_SAMPLE_DIRECTORY \
  --indices 0,17,999 --output NEW_VERIFICATION_DIRECTORY
```

This default audits data/observations without model inference. For a saved
neural baseline add `--predict --checkpoint VERIFIED_ORIGINAL_CHECKPOINT.pt`.
Its SHA256 must match the original summary. For zero-training 4D-Var, add
`--predict` without a checkpoint. VIVID without a checkpoint is rejected.
The server197 FM environment supports the cache audit; it does not include
NeuralOperator or DeepXDE. FNO/iFNO/DeepONet inference requires the recorded
server193 baseline environment or a separately reconstructed equivalent
environment. Do not replace their official implementations with available
local approximations. The archived environment manifests identify the original
package versions.

The selected indices are evaluated within their original complete test-batch
boundaries, including the final short batch. Thus a three-index check can
predict additional members of those three batches; only the requested indices
are exported. `--indices all --predict` is the explicit full-cell inference
mode. No full-cell baseline replay has been run as part of preparing this
interface.

Place each remote check in its own tmux session and a new directory below
`FM4PDE/reproducibility/revision_20260909/verification_runs/`, after checking
available resources. Transfer local source changes through Git to an isolated
checkout. Never run against writable copies of original result directories.

## Validation status and interpretation

Three unit tests verify evaluation-vs-training seed precedence, static flag
sharing without argument mutation, and rejection of altered arrays, source
hashes or global IDs. Unit arrays are explicitly artificial test fixtures.
All 186 complete summaries were fetched read-only and matched their previously
recorded file hashes. The completed real-cache validation is retained in
`verification_runs/baseline_frozen_replay/final_evidence/`. Its
`validation_complete.json` has SHA256
`cdf3add29c92763c51014e3f31df5329b73a482acc276878a16ecdaa576f1f30`.
`delivery_manifest.json` binds every delivered evidence file; individual
receipts retain their original execution paths and environment facts.

The checks cover all 186 complete test mask contracts, including 138,000
sample-specific mask records and the original empty contracts for full-field
observations. Ten representative cells compare indices 0, 17 and 999 against
30 original saved artifacts. Fields, channel names, axes, coordinates,
observations, masks and the checked physical metadata are bitwise equal.
These cases include both NS loader modes and both Burgers observation
patterns. The six selected Burgers samples also match the previously archived
truth/mask bundles bitwise. All 18 cache-file hashes and all arrays used by the
native consumers pass their shape, dtype, finite-value and array-hash checks.

Four original saved models were loaded without training and run on CPU using
their original batches: 0--15, 16--31 and 992--999. This computes 160 predictions
across four models and exports only the 12 requested examples. The native
inference view has zero target and full-field tensors before model evaluation.
No full-cell baseline inference was performed.

| Model and task | CPU PyTorch | Largest relative difference from the three saved CUDA predictions |
| --- | --- | --- |
| RecFNO, Darcy sparse forward | 2.8.0+cu128 | 6.56892e-4 |
| Senseiver, Poisson sparse inverse | 2.8.0+cu128 | 3.76026e-6 |
| VoronoiCNN, Helmholtz joint reconstruction | 2.8.0+cu128 | 2.17279e-4 |
| iFNO, Poisson full inverse | 2.12.1+cu130 | 3.63977e-4 |

These finite differences are reported without treating cross-device predictions
as bitwise equal. Full-precision values, individual comparisons, checkpoint
hashes and normalization-tensor hashes are in
`interface_validation_summary.json` and the 196 original receipts. The first
three models used server197 with two CPU threads. Senseiver required the
original FairScale 0.4.13 package, copied byte-for-byte from server193 into a
private `PYTHONPATH`; its package manifest and archive are included. iFNO used
the original server193 baseline environment with two CPU threads because
server197 lacks its NeuralOperator dependencies. No model implementation or
global environment was replaced. The final controller commit is `0b77021`;
the individual receipts preserve earlier controller paths and identical wrapper
source hashes when a completed check was reused.

Replay receipts report the actual device, PyTorch version and TF32 flags.
They do not claim to recreate the historical runtime environment. Optional
prediction comparisons report maximum absolute and relative differences from
the saved predictions, rather than requiring cross-device bitwise equality.
The exported whole-target relative-L2 value is an interface diagnostic; it is
not a replacement for the original solution/coefficient split metrics or
their uncertainty estimates. Replay wall time is not a controlled timing
result.
