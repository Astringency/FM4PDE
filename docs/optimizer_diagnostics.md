# Adam continuation diagnostics

`--resume` restores the checkpoint's optimizer parameter groups, including its
learning rate and betas. `--optimizer_betas` initializes a new optimizer and does
not override a saved group. To intentionally change betas while retaining the
saved first moment, second moment, and step counter, add:

```text
--resume_optimizer_betas 0.9 0.99
```

The native loader prints the effective restored betas and learning rates, and
saves `effective_optimizer_betas` in the arguments. Changing betas does not erase their prior history; the moments
adapt gradually under the new coefficients.
For a separate beta1 comparison, use `--resume_optimizer_betas 0.8 0.999`
with the same LR, checkpoint and data order. The screening study changes one
coefficient at a time; it does not yet recommend changing both together.

To intentionally rebuild the LR schedule, add `--resume_reset_lr_schedule` with
the desired `--lr`, `--min_lr`, `--lr_scheduler`, and `--warmup_epochs`. The new
schedule spans only the remaining epochs (`--epochs` minus the restored start
epoch). Adam moments and counters remain intact. Without this flag, exact resume
preserves the saved LR/schedule even when command-line LR settings differ.

This matters for the June Helmholtz, Darcy and Burgers checkpoints: their stored
linear scheduler has `end_factor=1e-4` and base LR `1e-4`, yielding `1e-8`, although
their saved argument `min_lr` says `1e-6`. Restoring their old scheduler also
restores that discrepancy; current scheduler construction alone does not fix it.

The separate `experiments.optimizer_diagnostics.study` entry point compares a
constant saved LR, LR `1e-5`, LR `3e-5`, beta1 `0.8`, and beta2 `0.99`. Beta
comparisons hold LR at `1e-5`; all trials restore the same model, Adam moments and
counter, and replay identical training inputs, noise, time and dropout streams.
The protocol records actual values after loading. Float32 parameters, forward
and backward calculations are used with TF32 enabled, and effective batch is 64.

The screen trains on 4,096 inputs from the first original training shard,
preserving membership in the original 50,000-example train/validation split.
There are 256 development inputs and 512 separate confirmation inputs. Four
time-stratified noise draws are averaged per input before computing paired
intervals. This is a local optimization screen; it does not establish a global
capacity limit or conditional sampling improvements. Early June checkpoints may
have seen validation-pool examples during earlier pretraining.

Saved diagnostics include every candidate's losses and effective hyperparameters,
layerwise gradient norms, missing/zero gradients, actual update-to-weight ratios,
unchanged weight fractions, gradient/update alignment, and Adam second-moment
scale. A frozen-weight probe estimates gradient signal/noise and momentum
alignment over eight independent minibatches. Random time, noise and dropout
can produce nonzero gradients even near an optimum, so a nonzero gradient norm
alone is not evidence that a larger LR will improve generalization.

The 2026-09-19 canonical experiment root is on server197:

```text
/research_data/users/zhangxifeng/C01Python/FM4PDE/outputs/pretrained/optimizer_diagnostics_20260919/
```

`inputs/` preserves source identities and split indices. `runs/<pde>/` holds
diagnostics, full optimizer checkpoints, process exit records and independent
audits. `report/` contains the consolidated results. A800 jobs use temporary
server216 caches; their output files must pass SHA256 verification before atomic
publication to the canonical root. Each training task runs in its own tmux.

## Paired difficult-case sampling

`hard_sampling setup --root ROOT --pde PDE` fixes the 16 worst solution-field
errors from the existing 1,000-case ID sparse-observation evaluation before any
new candidate predictions are read. `hard_sampling queue` compares the original
checkpoint, the LR-only checkpoint and the selected intervention at 128 updates.
All five PDEs use seeds 0 and 1, the original 100-step stochastic sampler,
identical masks and guidance, and matching initial and all 100 bridge noises.
The Helmholtz evaluation uses batch 1; the other PDEs use batch 4.

Training gains are measured against a freshly sampled original-model baseline at
the same batch size. The archived NS predictions showed strong batch dependence;
archived errors select cases but are not the denominator for new improvements.
A repeated original-model execution must agree within relative error 1e-6 before
candidate sampling proceeds. Saved batches retain identities and all noise hashes.

`extend_ns queue` continues both NS arms from update 128 to update 512 on the full
original 45,000-example training split, preserving validation membership and the
original normalizer. Both arms use the same training inputs and random streams.
`sample_continuations` evaluates their final snapshots on the same fixed cases.

`sampling_audit` independently recomputes physical-space relative L2 errors from
saved predictions using CPU float64 and verifies hashes, configurations, masks and
noise pairing. `sampling_figures` exports all 16 solution fields with common
truth/prediction and error color scales within each case. Quantitative comparisons
average the two seeds within each input before paired bootstrap over 16 inputs.
This selected difficult set cannot estimate population-wide performance.

`finalize_sampling` audits each PDE as it finishes, exports figures and writes
`report/SAMPLING.md`. Remote sampling results are SHA256-verified before publishing
under `hard_sampling/<pde>/` (NS uses `hard_sampling/` itself). The collector waits
for all five sampling comparisons, both NS continuations, audits and figures; it
must not clean working caches based on the optimizer screen alone.

Darcy's initial two seeds differed in the direction of the solution-error change.
`seed_replicates queue --root ROOT --pde darcy` therefore adds independent seeds 2
and 3 for exactly the same sixteen cases and all three checkpoints. The initial
selection and predictions are retained unchanged. `seed_replicates combine`
audits these new physical predictions and reports four-seed averages and each
seed separately in `hard_sampling/darcy/FOUR_SEEDS.md`. This is an exploratory
follow-up prompted by the initial results, not a new randomly selected test set.
