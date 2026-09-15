# Additive correction of eight Smooth cells

Preparation only: no inference, GPU dispatch, remote mutation, v1 edit, or Git
commit was performed by this worktree. The default prepared protocol has
`status=draft_correction`, so the unchanged runner refuses it in production.

The historical evidence is the immutable task sidecar
`provenance/historical_gate_mismatch_20260915.json`, SHA256
`80f6d347648e8acde8b547fd0bcd74e23c626a9dbf207af3f765d85ec7c64786`.
It binds the actual v1 protocol and eight executed v1 payload checks. The
historical traces establish a constant PDE weight of 0.1 at all 100 steps.
The v1 default starts that weight at flow time 0.8, leaving only 20 active steps.

## Exact change and selection

For each of `supervised` and `diffusion`, replace these four complete cells:

| PDE | Distribution | Setting | Rows |
|---|---|---|---:|
| Poisson | Smooth | sparse_forward | 1,000 |
| Poisson | Smooth | sparse_joint | 1,000 |
| Helmholtz | Smooth | sparse_forward | 1,000 |
| Darcy | Smooth | sparse_joint | 1,000 |

Set `pde_guidance_start_ratio=0.0` and `pde_guidance_ramp_ratio=0.0` explicitly.
This means that the multiplicative gate is **one** throughout inference; it
does not zero the PDE loss. Retain `zeta_pde=0.1`, all observation weights,
clipping, masks, truths, checkpoint bytes, 100 steps, precision, and canonical
input-row random streams. The recovered historical config remains unedited in
provenance. The single effective config change is start ratio 0.8 to 0.0.

The selected final study is 49 complete v1 cells plus 8 complete v2 cells.
There is no fallback to v1 within a replaced cell and no selection using errors.
All 57,000 v1 predictions remain archived, including the 8,000 superseded rows.

## Prepared bundle and validation

`prepare_gatefix_v2.py` creates a new directory exclusively. It writes:

- `protocol.json`: all 57 cells, preserving the current runner's full-protocol
  requirement; only the declared eight configs are corrected.
- `selection.json`: 57 complete-cell choices, bound to the original source
  protocol SHA or new protocol SHA, plus immutable source results directories.
- `correction_slices.json`: sixteen distinct 500-row slices for just the eight
  replacements. Resource and certificate fields remain unset for the scheduler.
- `validation.json`: all 57 actual dataclass resolutions; unchanged 49 configs;
  eight exact start-ratio differences; real CPU schedule checks proving 20 to
  100 active PDE steps and identical observation schedules.

The checked draft is in `revision_pairing_0915/gatefix_v2_draft/`. Its protocol
SHA256 is `493c619fe562e6f872af4154f71d6086ec1c704f5d1c19f87078a1b8d17aa29e`.
`preparation_checks.json` records ten passing preparation checks, including
draft rejection, frozen runner compatibility, eight-cell selection, complete
slice coverage, all 171 path identities, wrong-evidence rejection, and overwrite
refusal. No model or input tensor was read; physical artifact rehashing remains
mandatory in the unchanged runner and final auditor.

After review, generate a fresh frozen bundle (no sampling is launched):

```bash
python revision_pairing_0915/prepare_gatefix_v2.py \
  --v1-protocol /PATH/TO/V1_TASK/protocol.json \
  --evidence /PATH/TO/V1_TASK/provenance/historical_gate_mismatch_20260915.json \
  --output-dir /PATH/TO/NEW_FROZEN_BUNDLE --freeze
```

Deploy that bundle at `V1_TASK/corrections/v2/` on either server. Relative
`../../inputs/...` paths reuse each server's existing read-only frozen cache.
The `--freeze` output gets a different SHA from the draft; all downstream
bindings must use the frozen bytes. Do not edit a generated bundle in place.

## Smallest execution path

1. Keep the runner and its `code_identity()` source files unchanged. New
   preparation, orchestration, and audit scripts are outside that identity.
2. Perform corrected 100-step pilots for the four PDE/task families above,
   keeping the already validated batch size/precision/device type. In
   particular, keep Helmholtz at its validated batch size of one. The existing
   family identity does not bind the gate or scalar weights, so an old pilot
   certificate alone is insufficient evidence that early guidance is finite
   and fits memory. No tolerance or scientific parameter should be adjusted.
3. Run each of the sixteen slices with the unchanged `run_inference.py`, using
   an explicitly allocated device, verified certificate, `--cell`, `--start`,
   `--stop`, and unique `corr_v2_*` job ID from the plan. Use
   `--protocol V1_TASK/corrections/v2/protocol.json` and
   `--output-root V1_TASK/corrections/v2`. Omit `--input-root`, because all paths
   resolve against the new protocol directory. Wrap calls in `run_stage.py` to
   retain their exact commands, logs, and exit codes.
4. Sync completed correction batches and their receipts through the existing
   local relay mechanism, using new paths beneath `corrections/v2`. Do not
   place them in the top-level v1 `jobs/` tree. Preserve `--ignore-existing`
   and hidden-partial exclusions; audit must reject any same-name conflict.

The old `run_queue.py` requires all 114 slices and the task-root basename.
The old `wait_finalize.py` requires those slices and all ten original workers.
Neither should be pointed at an eight-cell queue or have its frozen v1 inputs
rewritten. A small new correction queue/wait wrapper is still required; it can
reuse the existing runner/stage/receipt-coverage functions. Its readiness gate
should use the sixteen declared slices and their identities, not an assumed
number of active GPUs.

## Required selected-result audit and completion

The existing auditor intentionally binds every prediction to a single protocol
SHA. Combining v1 and v2 files in one directory does not constitute a valid
audit. Do not relabel old payloads or receipts as v2.

A thin selected-result auditor is still needed. It should validate the frozen
selection manifest, keep the existing audit checks, and call
`audit_outputs.audit_batch(path, original_source_cell, data, original_protocol_sha)`
for each selected source cell. For every inherited cell, first prove that its
resolved scientific config, model, inputs, and random-stream definition equal
the target protocol. Recompute float64 errors from the selected predictions;
enforce exactly 1,000 unique rows per cell, 57,000 total, cohort totals 45,000 and
12,000, and one unchanged executed source snapshot. Explicitly record each row's
source revision/protocol as well as the target protocol and selection SHA.
Only the eight declared v1 cells may be excluded as superseded. Unknown cells,
duplicates in the selected revision, missing receipts, and identity/hash errors
must fail the strict audit.

Reuse the existing NS Smooth sensitivity helpers with the original mandatory
`provenance/ns_existing_selection_overlap.json`. All eight NS cells are
inherited unchanged; regenerate the 996-row supplementary summaries and their
32 physical overlap checks, rather than dropping this existing audit condition.

Create a new `reference_metrics_v2.json` by retaining the original reference
records/metrics and source hashes exactly, changing only its target protocol
binding and adding the source v1 reference SHA as lineage. The unchanged
`compare_results.py` refuses references bound to a different protocol. Its
strict success still requires 57 complete cells, 73 historical-comparison rows,
and 203 baseline-comparison rows.

Write selected audit, comparison, and completion markers only under the new
correction directory. Mark corrected completion only after strict audit
`status=pass`, `complete=true`, 57,000 verified unique rows, complete NS
sensitivity, and strict comparison `status=complete`. The v1 finalizer may
finish its own immutable archive independently; its v1 completion marker does
not certify the corrected study. Partial or failed correction jobs must retain
clear incomplete/error status and must not silently substitute old results.

## Implementation anchors in the existing worktree

- `run_inference.py:117`: runtime-only effective overrides and actual config
  resolution; `:221`: unchanged schedule/stock gradient path; `:484`: full
  protocol gate; `:502`: explicit cell/output selection.
- `sampling/guidance.py:68`: multiply the independent PDE gate;
  `:87`: start/ramp semantics.
- `run_queue.py:103`: fixed 114-slice validation; `:261`: completed slice receipt
  coverage; `:372`: fixed task-root check.
- `wait_finalize.py:77`: original ten-worker/114-slice completion gate.
- `audit_outputs.py:97`: independent per-batch source identity, executed tensor,
  configuration, source, and float64 error checks; `:181`: mandatory NS sidecar;
  `:345`: current single-protocol batch audit call.
- `compare_results.py:98`: audit and reference protocol binding.

The scientific correction restores the documented historical schedule; it does
not imply that its errors will improve. Early PDE gradients can change numerical
stability, memory, and runtime, which is why corrected pilots are necessary.
