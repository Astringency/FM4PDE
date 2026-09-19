# Adam continuation diagnostics

`--resume` restores the checkpoint's optimizer parameter groups, including its
learning rate and betas. `--optimizer_betas` initializes a new optimizer and does
not override a saved group. To intentionally change betas while retaining the
saved first moment, second moment, and step counter, add:

```text
--resume_optimizer_betas 0.8 0.99
```

The native loader prints the effective restored betas and learning rates, and
saves `effective_optimizer_betas` in the arguments. Changing betas does not erase their prior history; the moments
adapt gradually under the new coefficients.

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
