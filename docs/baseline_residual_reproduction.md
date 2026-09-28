# FM4PDE-OFM and CoCoGen comparisons

Run the shell examples from the FM4PDE repository root unless a subshell explicitly changes directories.

These commands evaluate Darcy and Navier–Stokes, forward and inverse tasks, on 100 fixed realizations from each of ID, Smooth and Rough: 1,200 predictions per method. Sampling uses 100 steps. Weights, training normalizers and guidance parameters are fixed. Darcy uses the predicted physical coefficient directly. The NS secant residual averages the dynamics at the two endpoints.

## FM4PDE-OFM

Set `PYTHON_BIN` to the environment with the official OFM dependencies, `DATA_ROOT` to the physical test files used by the FM4PDE main comparison, `GENERATIVE_BASELINE_ROOT` to the `FunDPS_DDIS_ECI_OFM` checkout (including `adapters/` and `official/OFM/`), and `OFM_DATA_ROOT` to the compact training-data manifests. Under `OFM_CHECKPOINT_ROOT`, the runner selects `poisson/epoch_220.pt`, `helmholtz/epoch_220.pt`, `darcy/epoch_100.pt`, `nsnonbounded/epoch_50.pt`, and `burger/epoch_190.pt`. These are the actual adopted checkpoints, shared across tasks and distributions. `OFM_CHECKPOINT_<PDE>` can specify the same weights at a different location.

```bash
bash scripts/sample/comparison/sparse_forward_inverse_ofm.sh \
  --pdes darcy nsnonbounded --device cuda:0 --output outputs/paper/ofm
```

This calls the existing architecture runner:

```bash
bash scripts/sample/ablations/velocity_architecture.sh \
  --methods FM4PDE-OFM --pdes darcy nsnonbounded \
  --device cuda:0 --output outputs/paper/ofm
```

The OFM predictions serve both the main comparison and architecture ablation. Run the latter's native branch separately with `--methods FM4PDE`: its main-comparison weights differ from its architecture-ablation weights.

`configs/experiments/ablations/velocity_architecture.yaml` fixes batches, singleton cases and seeds. Do not rebatch OFM: its GP random draws depend on batch membership. Distribute whole batches with `--num-shards N --shard-index I` and separate output directories. These two PDEs have 465 batches covering exactly 1,200 sample settings. Receipts save source, checkpoint, normalizer and data hashes, resolved settings and prediction checksums. Resume refuses changed identities. Pool all shards' `errors.csv` rows before computing means and sample standard deviations.

## CoCoGen

Set `COCOGEN_ROOT` to the CoCoGen checkout and `PYTHON_BIN` to its environment. Set `COCOGEN_CHECKPOINT_DARCY`, `COCOGEN_CONFIG_DARCY`, `COCOGEN_CHECKPOINT_NSNONBOUNDED`, and `COCOGEN_CONFIG_NSNONBOUNDED` to the archived control-trained checkpoints and their training YAML files. Normalizers must be the archived training statistics.

Prepare the same physical inputs and 500-point masks as the FM4PDE main comparison:

```bash
export FM4PDE_ROOT="$PWD"
(cd "$COCOGEN_ROOT" && "$PYTHON_BIN" -m cocogen_eval.prepare_comparison \
  --pdes darcy nsnonbounded --output inputs/common_comparison)
```

`DATA_ROOT` must point to the same fixed test files as the FM4PDE main run. Alternatively, use the archived `common_inputs` files directly. Set `COCOGEN_NORMALIZERS` below to the directory containing `darcy/normalizer.json` and `nsnonbounded/normalizer.json`.

```bash
bash scripts/sample/comparison/cocogen.sh \
  --pdes darcy nsnonbounded --tasks forward inverse \
  --count 100 --batch-size 32 \
  --inputs "$COCOGEN_ROOT/inputs/common_comparison" \
  --normalizers "$COCOGEN_NORMALIZERS" \
  --device cuda:0 --output "$COCOGEN_ROOT/outputs/common_comparison"
```

The original 100-step, four-RePaint and 50+10-physics-correction settings are in `CoCoGen/configs/sampling.yaml`. No guidance parameter was tuned. CoCoGen's maximum physical update and backtracking are retained. Both methods call the current FM4PDE residual implementation.

CoCoGen's shard options select whole PDE/task/distribution cells. Pool all 12 cells for reporting. Saved predictions include their truths, masks and IDs; receipts include residual source hashes and input identities.

The earlier CoCoGen NS table used a different test cohort. This unified rerun changes its inputs and masks as well as the residual. Differences against that table must not be attributed solely to the secant formula.
