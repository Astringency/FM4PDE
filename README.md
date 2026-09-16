# Guided Flow Matching for Forward and Inverse PDE Problems with Sparse Observations: Algorithm and Theory

Sparse observations of partial differential equations (PDEs) often leave
input fields, solution fields, or both only partially known. We propose
FM4PDE, a flow-matching method that learns a joint prior over these fields
for each equation and uses the same prior for forward, inverse, and joint
reconstruction. During inference, observation and PDE residuals guide
sampling through predicted endpoints expressed in physical units.
We develop deterministic, stochastic, and hybrid samplers with gradient
clipping and establish finite-step bounds under local regularity and
geometric assumptions, with explicit error floors. Experiments on static and time-dependent benchmark PDEs examine
reconstruction accuracy and computational cost across the three tasks.
The results show competitive reconstruction accuracy on several tasks
and faster sampling than DiffusionPDE in controlled comparisons.

## Training

Run the commands below from this repository's root in Bash (Linux or WSL).
Create the environment from [environment.yml](environment.yml); datasets and
trained weights must be supplied separately.

```bash
conda env create -f environment.yml
conda activate fm4pde
export DATA_ROOT=/path/to/PDEdata
export PYTHON_BIN=python
```

The training entry is `main(args)` in [train.py](train.py).
It loads data, fits the normalization, constructs the model, and manages
validation and checkpoints. The flow-matching objective and parameter updates
are implemented by `train_one_epoch()` in [training/train_loop.py](training/train_loop.py);
the same file provides `validate_one_epoch()`.
[configs/training_data.yaml](configs/training_data.yaml) lists the five training
shards for each equation.

For example, launch Poisson training directly on one GPU:

```bash
python train.py \
  --dataset poisson \
  --data_path "$DATA_ROOT/" \
  --train_data_config configs/training_data.yaml \
  --data_size 5 \
  --epochs 300 --batch_size 4 --accum_iter 16 \
  --lr 0.0001 --lr_scheduler warmup_cosine \
  --model_profile recommended --device cuda \
  --output_dir outputs/pretrained/formal/poisson
```

The Bash launcher [scripts/training/run_train.sh](scripts/training/run_train.sh)
sets the per-GPU batch size and gradient accumulation and supports distributed
training and checkpoint resume:

```bash
# Inspect the training command.
PDE=poisson DRY_RUN=true bash scripts/training/run_train.sh

# Train one equation, or select several equations and two GPUs.
PDE=poisson bash scripts/training/run_train.sh
PDE_LIST="poisson helmholtz nsnonbounded" NPROC_PER_NODE=2 \
  bash scripts/training/run_train.sh

# Resume one equation from a training checkpoint.
PDE=poisson RESUME=/path/to/checkpoint.pth \
  bash scripts/training/run_train.sh
```

Omit both `PDE` and `PDE_LIST` to train all eleven supported equations.
The launcher defaults to 300 epochs and an effective batch size of 64;
`EPOCHS`, `TARGET_EFFECTIVE_BATCH`, and `OUTPUT_DIR` override these settings.
Navier–Stokes (`nsnonbounded`) uses the light model with 44,121,218
parameters; select `--model_profile light` for a direct Python launch.

To export a compact inference checkpoint and select it for Poisson sampling:

```bash
python scripts/training/export_checkpoint.py \
  /path/to/training-checkpoint.pth /path/to/fm4poisson.pth
export CHECKPOINT_POISSON=/path/to/fm4poisson.pth
```

## Main Sampling

[sample.py](sample.py) delegates to `main()` in [sampling/runner.py](sampling/runner.py).
Its `run_single_ablation()` function implements a single sampling run for
both main experiments and ablations, using the samplers in
[sampling/sampler_wrappers.py](sampling/sampler_wrappers.py).
Main experiments load profiles from [configs/main](configs/main), organized
as `<task>/<pde>.yaml`.

The task names are `forward` (observe the input field and reconstruct the
solution), `inverse` (observe the solution and reconstruct the input), and
`both` (joint reconstruction from observations of both fields).
Burgers uses `both` for space–time trajectory reconstruction.

`DATA_ROOT` replaces the `datasets/` prefix in sampling profiles.
Use `CHECKPOINT_<PDE>` for a complete weight filename, or
`CHECKPOINT_ROOT` to replace `outputs/pretrained/` while keeping the
remaining subdirectories. An explicit `--override checkpoint_path=...`
takes precedence. Set the corresponding checkpoint for every selected PDE.

Examples of direct Python sampling:

```bash
# Sparse forward reconstruction on Poisson ID inputs.
python sample.py --config configs/main/forward/poisson.yaml \
  --override num_steps=100 --override num_obs=500 \
  --override output_dir=outputs/examples/poisson_forward

# Sparse inverse reconstruction on the Smooth distribution.
python sample.py --config configs/main/inverse/poisson.yaml \
  --override test_type=smooth --override num_steps=100 \
  --override output_dir=outputs/examples/poisson_inverse

# Joint reconstruction; the module entry is equivalent to sample.py.
python -m sampling.runner --config configs/main/both/poisson.yaml \
  --override batch_size=4 --override num_steps=100 \
  --override output_dir=outputs/examples/poisson_joint

# Full-field forward reconstruction on a 128 x 128 grid.
python sample.py --config configs/main/forward/poisson.yaml \
  --override num_obs=16384 \
  --override output_dir=outputs/examples/poisson_full_forward

# Burgers trajectory reconstruction from random space-time observations.
python sample.py --config configs/main/both/burger.yaml \
  --override sensor_mode=random --override num_obs=500 \
  --override output_dir=outputs/examples/burger_random
```

The scripts provide single runs and complete sweeps:

```bash
# One PDE/task.
PDE=poisson TASK=both TEST_TYPE=id \
  bash scripts/sampling/main/run_sample.sh

# Inspect all main jobs, then run them on two GPUs.
PLAN_ONLY=true bash scripts/sampling/main/run.sh
DEVICE_LIST="cuda:0 cuda:1" bash scripts/sampling/main/run.sh

# A smaller sweep with explicit PDE, task, and sample counts.
PDE_LIST="poisson helmholtz" TASK_LIST="forward inverse both" \
  TEST_TYPE=smooth NUM_SAMPLES=100 MAX_BATCH_SIZE=10 \
  OUTPUT_DIR=outputs/main_subset \
  bash scripts/sampling/main/run_sample_sweep.sh

# Burgers: both configured observation layouts.
bash scripts/sampling/main/run_sample_sweep_burger.sh
```

The complete `run.sh` workflow covers sparse forward, inverse, and joint
tasks and full-field forward/inverse tasks for Poisson, Helmholtz, Darcy, and
Navier–Stokes, plus the two Burgers layouts, on ID/Smooth/Rough. It defaults to
1,000 inputs and 100 steps per comparison. Use `OUTPUT_ROOT` to change
`outputs/main`, and `TEST_TYPE_LIST` to select distributions.
Matching completed jobs can be resumed.

For the saved aligned comparisons, use
[experiments/aligned_sampling/run_inference.py](experiments/aligned_sampling/run_inference.py)
through its launcher:

```bash
STUDY_ROOT=/path/to/saved_aligned_study \
CELL=supervised/poisson/id/sparse_joint \
OUTPUT_ROOT=outputs/reproductions/aligned \
  bash scripts/sampling/main/run_matched_cell.sh
```

This requires the study's `protocol.json`, saved inputs, observation masks,
and weights. The script runs a batch-consistency pilot before replaying the
selected cell. Use this entry to reproduce saved aligned inputs; the ordinary
main scripts construct observations from their current configurations.

## Ablations

Ordinary ablations use [sampling/sweep.py](sampling/sweep.py) to expand
[configs/ablations/paper.yaml](configs/ablations/paper.yaml), then call the same
[sampling/runner.py](sampling/runner.py) used for main sampling.
Their base profiles live in [configs/ablations/base](configs/ablations/base);
these preserve separate guidance weights and gradient limits. The datasets
default to ID.

```bash
# Inspect a sampler ablation directly in Python; omit --list to run it.
python -m sampling.sweep --grid configs/ablations/paper.yaml \
  --pde poisson --group sampler_phase --list

# Inspect all ordinary ablations.
PLAN_ONLY=true bash scripts/sampling/ablations/run.sh

# Run selected factors on two equations.
PDE_LIST="poisson nsnonbounded" \
  bash scripts/sampling/ablations/run.sh guidance_components sampler_phase

# Run step-budget and observation-density ablations concurrently.
PDE_LIST=poisson PARALLEL=true MAX_PARALLEL_TASKS=2 \
  DEVICE_LIST="cuda:0 cuda:1" OUTPUT_DIR=outputs/ablations/poisson \
  bash scripts/sampling/ablations/run.sh num_steps_by_sampler sensor_sparsity
```

The other available groups are `loss_state_by_sampler`, `sensor_mode`,
`noise_robustness`, `temporal_residual_mode`, and `statistics_stability`.
`BATCH_SIZE` controls the inputs in each ablation job (default: 1);
`OFFSET` selects their starting index.

Additional studies are exposed by
[scripts/sampling/ablations/run_study.sh](scripts/sampling/ablations/run_study.sh):

| Study argument | Experiment |
| --- | --- |
| `ensemble` | Repeated sampling: 32 inputs × 3 draws |
| `guidance` | Repeated guidance trajectories |
| `averaging` | Poisson conditional sample averaging |
| `layouts` | Random, fixed, grid, and column observations with shared/separate locations |
| `weights` | Observation/PDE guidance-weight sweeps |
| `prior` | Unconditional samples |

```bash
bash scripts/sampling/ablations/run_study.sh averaging --help
bash scripts/sampling/ablations/run_study.sh ensemble \
  --pdes nsnonbounded --inputs /path/to/prepared_inputs \
  --output /path/to/ensemble_results --checkpoint /path/to/fm4nsnonbounded.pth
bash scripts/sampling/ablations/run_study.sh weights \
  --pdes nsnonbounded --inputs /path/to/prepared_inputs \
  --anchors /path/to/anchors.json --output /path/to/weight_results \
  --checkpoint /path/to/fm4nsnonbounded.pth
```

These studies require their prepared inputs and selected settings, including
`selection.json` for `ensemble` and an anchors file for `weights`.
The ensemble, guidance, and weight studies read model defaults from
`configs/ablations/base/both/<pde>.yaml`, including the light NS model.

Paired FM4PDE/DiffusionPDE error–time trajectories are implemented in
[experiments/trajectories](experiments/trajectories):

```bash
bash scripts/sampling/ablations/run_traces.sh prepare --help
DIFFUSION_ROOT=/path/to/DiffusionPDE \
  bash scripts/sampling/ablations/run_traces.sh run --root /path/to/trace_study
bash scripts/sampling/ablations/run_traces.sh verify --root /path/to/trace_study
bash scripts/sampling/ablations/run_traces.sh plot \
  --root /path/to/trace_study --output /path/to/figures
```

The run, verification, and plotting commands require an already prepared study.
Further input preparation and figure tools are listed in [plot/README.md](plot/README.md).

## Baseline and other info

Companion experiment repositories:

- [RecFNO and other baselines](https://github.com/Astringency/FM4PDEbaseline.git):
  FNO, DeepONet, iFNO, RecFNO, Senseiver, VoronoiCNN, PINN-Sparse, PDE-Opt,
  PC-BNN, 4D-Var, and VIVID.
- [DiffusionPDE experiments](https://github.com/Astringency/DiffusionPDE.git).
- [CoCoGen experiments](https://github.com/Astringency/CoCoGen.git).

The `experiments/` directory is required for the retained reproduction
workflows. `scripts/sampling/main/run_matched.sh` imports
`experiments/aligned_sampling/run_inference.py`, and
`scripts/sampling/ablations/run_traces.sh` calls the preparation,
sampling, and verification code in `experiments/trajectories/`.
Keep this directory to preserve both workflows.

Navier–Stokes main sampling and ordinary ablations use endpoint-secant
residuals. Heat, Wave, Advection–Diffusion, Reaction–Diffusion, and Shallow Water
use Hermite bridges by default; Burgers uses full-trajectory finite differences.
Static equations use spatial residuals. Training optimizes flow matching;
its periodic generated-sample diagnostics retain their own residual settings.

Validate the published profiles and ablation combinations without sampling:

```bash
python -m sampling.validate_configs
# Also check that all configured datasets and checkpoints exist locally.
python -m sampling.validate_configs --check-assets
```
