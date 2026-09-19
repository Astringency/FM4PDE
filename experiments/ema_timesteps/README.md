# EMA and timestep continuation study

This experiment continues the five original PDE checkpoints without resetting
their raw weights, Adam moments, update counters, or normalizers. It uses all
45,000 inputs in each original training split and an effective batch of 64.
Each epoch contains 703 updates; eight shuffled inputs are omitted per epoch.

## Comparisons

Every arm maintains fixed-decay EMA=0.999, initialized from the original weights,
and evaluates raw and EMA separately. EMA warmup is disabled for this study.

| Arm | Training times | Adam betas |
| --- | --- | --- |
| uniform | Independent uniform | (0.9, 0.999) |
| stratified_uniform | Shuffled uniform strata in each effective batch | (0.9, 0.999) |
| logit_normal | sigmoid(N(0,1)), without importance weighting | (0.9, 0.999) |
| beta1_05 | Independent uniform | (0.5, 0.999) |

Poisson and NS use LR=3e-6; Helmholtz, Darcy, and Burgers use LR=1e-6.
All arms ramp LR over the first 128 updates, then hold it constant. Weight decay
is retained from the source checkpoint. Data order, Gaussian noise, and dropout
streams are matched; time sampling has a separate random stream.

Logit-normal changes the time weighting of the training objective. Validation
always uses equal-mass uniform time bins, irrespective of the training arm.
The absolute endpoint CFM error can include irreducible target variance, so a
large endpoint loss alone does not establish where more training is useful.

## Execution

1. `prepare` reconstructs the complete original train split, preserves its
   normalizer, and freezes 512 development and 512 confirmation inputs.
2. `study profile` measures microbatch memory and the original model's errors in
   ten time bins. A reviewed `time_sampling_plan.json` is required before training.
3. `study queue --epochs 2` screens all four arms. This is an intermediate screen,
   not the planned end of training. Review development and physical sampling,
   retain uniform as a control, and extend justified branches to ten epochs.
4. `sampling setup` retains the historical 16 difficult cases and draws 16
   ordinary ID cases before new checkpoint evaluation. Both cohorts are
   exploratory, rather than independent final tests.
5. `sampling queue` first recomputes the original-model baseline, then evaluates
   raw and EMA from immutable snapshots. Inputs, masks, batch, sampler settings,
   initial noise, and all 100 bridge-noise draws must match. Two sampling seeds
   are averaged per input before the paired bootstrap over inputs.
6. Freeze checkpoint choices and their hashes in a manifest before using
   `confirmation`. It evaluates the reserved 512 inputs under a fixed uniform
   time distribution. Historical validation reuse remains a limitation for the
   four older checkpoints; this procedure does not undo earlier data exposure.
7. `audit` checks Adam and EMA counters, snapshot hashes, raw/EMA state mappings,
   training coverage, and independently recomputes physical-space prediction
   metrics in float64.

Continuation resumes from the complete `last.pth` state. Epoch 2, 5, and 10
snapshots are immutable and become available to sampling only after their SHA256
ready marker is written. Do not run two training processes for the same arm.

The canonical output directory is on server197. Server216 uses a temporary A800
cache. Input/checkpoint relay scripts verify hashes before publishing ready
markers; results must be returned and verified before cleaning that cache.
Training and physical sampling have separate named tmux sessions.

`extend` waits for the four screening arms, checks their actual saved states,
and resumes a reviewed entry in `extension_plan.json`. A per-GPU lock serializes
longer runs; `sample_extension` separately waits for the initial nine sampling
variants and evaluates epoch 5/10 raw and EMA snapshots. The training and
sampling publication helpers verify all transferred files before publishing.

A reviewed `overlap_screening` plan entry may let the prespecified uniform
control continue on an idle GPU of the same profiled type after its own two-epoch
process has exited successfully and its complete saved state has been audited.
The remaining screening arms keep their original schedules. Every arm must still
pass its state audit before recipe selection; adaptive alternative extensions
retain the default requirement to wait for all four screening arms. Record any
controller handoff and preserve the immutable checkpoints and scientific worker
checkout. This scheduling option changes no training or evaluation parameter.

## Native training support

The native trainer also supports explicit raw-checkpoint-to-EMA resume:
`--resume <checkpoint> --use_ema --resume_add_ema --ema_decay 0.999 --no-ema_warmup`.
Once resuming a checkpoint that already contains EMA, omit `--resume_add_ema`.
Saved EMA history is restored. Native validation reports both raw and EMA, ten
time bins, and channels, with a fixed stream isolated from training RNG.
Native EMA advances only when the accumulated optimizer update succeeds; an AMP
overflow that skips an update also skips EMA. This is checked with real CPU AMP
and CUDA fused AdamW, including gradient accumulation and scaler growth/backoff.

Use `--timestep_sampling uniform|stratified_uniform|logit_normal|legacy_skewed`
for explicit training time selection. The old skewed flag remains supported.
`--resume_optimizer_betas 0.5 0.999` changes betas after restoring Adam state;
`--optimizer_betas` alone configures a new optimizer and does not override resume.
LR changes on native resume require `--resume_reset_lr_schedule` and explicit
schedule settings.

Use `export_resume --source <immutable snapshot> --output <new file> --pde <pde>`
for a native-resume copy. Study snapshots retain some inherited pretraining
arguments and scaler metadata. The exporter records that provenance, normalizes
the effective precision, EMA, time sampling, batch and optimizer arguments, and
clears the inapplicable scaler state. It verifies the real native loader and
both inference loaders before publishing the copy; model, optimizer, normalizer
and stored RNG tensors must remain exactly equal to the source. The original
experimental snapshot and its sampling identities are preserved.

The native CLI still needs explicit run arguments, including an epoch limit
greater than the exported checkpoint's next epoch. The receipt lists effective
settings and that next epoch. Native training uses its own data order, random
stream, validation and per-microbatch stratification; use `study run` to continue
the exact experiment trajectory. Inference explicitly chooses `prefer_ema=True`
or `False`; an export contains both weights and does not select a winner.

## Interpretation

Nonzero stochastic gradients and nonzero CFM loss do not prove undertraining.
A lower fixed FM loss does not establish better guided physical sampling.
At 703 updates, fixed EMA=0.999 still gives about 49.5% cumulative weight to its
initial state; the two-epoch screen should not be treated as a mature EMA trial.
The intended ten epochs provide 7,030 updates, with about 0.09% initial EMA weight.
No training duration alone certifies convergence or guarantees large gains.

All continuation arms use FP32/TF32. The original NS training used BF16, so
original-to-continuation comparisons also include a precision change.

Verification so far: ten targeted tests passed on local PyTorch 2.12 and remote
PyTorch 2.8; nine existing EMA/checkpoint/optimizer/train-loop tests also passed
locally. These results are not a claim that the entire repository test suite ran.
The additional AMP tests passed on both local PyTorch 2.12 and remote PyTorch 2.8.
Native-resume export checks also passed on the real two-epoch Helmholtz and NS
checkpoints under PyTorch 2.8, including exact Adam-state comparison and explicit
raw/EMA inference loading. Every final selected export runs those checks again.
