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

## Environment and data

Run commands from the repository root in Bash on Linux or WSL.
[environment.yml](environment.yml) specifies the `fm4pde` environment
(Python 3.12 and PyTorch 2.8). Training and sampling examples use a CUDA GPU.

```bash
conda env create --name fm4pde --file environment.yml
conda activate fm4pde
export DATA_ROOT=/path/to/PDEdata
export PYTHON_BIN=python
```

[data/DataGen/gen_pde.sh](data/DataGen/gen_pde.sh) is the unified data-generation
entry for all eleven PDEs. It dispatches to these solvers:

| PDEs | Generation code | Runtime |
| --- | --- | --- |
| Poisson, Helmholtz, Darcy, Burgers | [data/DataGen/static/](data/DataGen/static/) (`generate_poisson.m`, `generate_inhom_helmholtz.m`, `generate_darcy.m`, `gen_burgers1.m`) | MATLAB |
| Navier–Stokes | [gen_nbns.py](data/DataGen/time_dependent/gen_nbns.py) | Python / PyTorch |
| Heat, Wave, Advection–Diffusion, Steady Heat Conduction | [generate_pair_h5s.py](data/DataGen/python/generate_pair_h5s.py) | Python |
| Reaction–Diffusion, Shallow Water | [gen_rd.py](data/DataGen/time_dependent/gen_rd.py), [gen_swe.py](data/DataGen/time_dependent/gen_swe.py) | Python; Shallow Water uses Clawpack/PyClaw |

```bash
# Preview generation for every PDE without running a solver.
PDE=all DRY_RUN=true OUT_ROOT="$DATA_ROOT" bash data/DataGen/gen_pde.sh

# Generate Poisson training and all five test distributions (requires MATLAB).
PDE=poisson TYPE=all OUT_ROOT="$DATA_ROOT" bash data/DataGen/gen_pde.sh

# Generate the two extra rough test sets for these five PDEs, 1,000 samples each.
for dataset_type in rough2 rough3; do
  PDE="poisson helmholtz darcy nsnonbounded burger" TYPE="$dataset_type" \
    OUT_ROOT="$DATA_ROOT" bash data/DataGen/gen_pde.sh
done

# Generate only the Heat and Wave training sets.
PDE="heat wave" TYPE=train OUT_ROOT="$DATA_ROOT" bash data/DataGen/gen_pde.sh
```

The defaults are five training shards of 10,000 samples each, 10,000 test samples
for each of ID/Smooth/Rough, and resolution 128. Poisson, Helmholtz, Darcy,
Navier–Stokes, and Burgers also support `rough2` and `rough3`, with **1,000 samples
per new test type**. Set `TEST_SAMPLES` to override the count for every selected
test type. `TYPE=all` includes all supported types for each PDE; requesting
`rough2` or `rough3` explicitly for other PDEs fails before generation starts.
Existing files are skipped unless `OVERWRITE=true`.
Set `OUT_ROOT="$DATA_ROOT"` explicitly for generation.
[configs/training_data.yaml](configs/training_data.yaml) lists the resulting
training files; [configs/main](configs/main) contains test-data paths.
Datasets and trained weights are stored separately from the code.

The five PDEs use the following extra rough GRF settings (`alpha` is called
`gamma` in Burgers):

| Test type | alpha | tau | Default samples via `gen_pde.sh` | Seed offset |
| --- | --- | --- | --- | --- |
| `rough` | 1.5 | 5 | 10,000 | 30,000,000 |
| `rough2` | 1.2 | 12 | 1,000 | 40,000,000 |
| `rough3` | 1.05 | 24 | 1,000 | 50,000,000 |

Lower alpha slows spectral decay; larger tau increases relative small-scale
power. Both changes create wider separation than a small alpha-only adjustment.
They apply to the source in Poisson/Helmholtz, the latent coefficient field in
Darcy (the thresholded coefficient still takes values 4 and 12), the initial
vorticity in Navier–Stokes, and the initial velocity in Burgers. Field variance
is not held fixed, so these are distribution shifts in both spectrum and
potential amplitude, rather than a fixed-variance smoothness sweep.

Files keep the existing schema and record the actual GRF parameters. New test
filenames end in `_rough2.mat` or `_rough3.mat`, for example
`poisson/poisson_test_1000-128-128_rough2.mat` and
`nsnonbounded/nsnonbounded_test_1000-128-128-10_rough3.mat`.
The MATLAB profiles live in
[get_generation_profile.m](data/DataGen/static/get_generation_profile.m);
the Python profiles live in
[generation_profiles.py](data/DataGen/generation_profiles.py).
The existing sampling configs still select ID/Smooth/Rough.

## Training

`main(args)` in [train.py](train.py) manages data loading, normalization, model
setup, and checkpoints. `train_one_epoch()` and `validate_one_epoch()` in
[training/train_loop.py](training/train_loop.py) implement training and validation.

```bash
# Train Poisson directly on one GPU.
python train.py --dataset poisson --data_path "$DATA_ROOT/" \
  --train_data_config configs/training_data.yaml --data_size 5 \
  --epochs 300 --batch_size 4 --accum_iter 16 \
  --lr 0.0001 --lr_scheduler warmup_cosine \
  --model_profile recommended --device cuda \
  --output_dir outputs/pretrained/formal/poisson

# Preview or run the Bash launcher.
PDE=poisson DRY_RUN=true bash scripts/training/run_train.sh
PDE=poisson bash scripts/training/run_train.sh
PDE_LIST="poisson nsnonbounded" NPROC_PER_NODE=2 \
  bash scripts/training/run_train.sh
PDE=poisson RESUME=/path/to/checkpoint.pth bash scripts/training/run_train.sh
```

[scripts/training/run_train.sh](scripts/training/run_train.sh) defaults to all
eleven PDEs, 300 epochs, and an effective batch size of 64, with gradient
accumulation and distributed training. Navier–Stokes uses the 44M light model;
select `--model_profile light` when launching it directly.

Export an inference checkpoint and select it for sampling:

```bash
python scripts/training/export_checkpoint.py \
  /path/to/training-checkpoint.pth /path/to/fm4poisson.pth
export CHECKPOINT_POISSON=/path/to/fm4poisson.pth
```

## Main Sampling

[sample.py](sample.py) calls `main()` in [sampling/runner.py](sampling/runner.py).
Its `run_single_ablation()` handles both main sampling and ablations.
Main profiles are stored as `configs/main/<task>/<pde>.yaml`.

`forward` observes the input field and reconstructs the solution; `inverse`
observes the solution and reconstructs the input; `both` reconstructs both
fields. Burgers uses `both` for trajectory reconstruction.

Sampling profiles use `DATA_ROOT` in place of `datasets/`. Set
`CHECKPOINT_<PDE>` to each weight filename, or use `CHECKPOINT_ROOT` to replace
`outputs/pretrained/` while preserving the remaining subdirectories.
`--override checkpoint_path=...` takes precedence.

```bash
# Sparse forward, inverse, and joint reconstruction.
python sample.py --config configs/main/forward/poisson.yaml \
  --override num_steps=100 --override num_obs=500 \
  --override output_dir=outputs/examples/forward
python sample.py --config configs/main/inverse/poisson.yaml \
  --override test_type=smooth --override output_dir=outputs/examples/inverse
python -m sampling.runner --config configs/main/both/poisson.yaml \
  --override batch_size=4 --override output_dir=outputs/examples/joint

# One run, a complete main sweep, or a selected subset.
PDE=poisson TASK=both TEST_TYPE=id bash scripts/sampling/main/run_sample.sh
PLAN_ONLY=true bash scripts/sampling/main/run.sh
DEVICE_LIST="cuda:0 cuda:1" bash scripts/sampling/main/run.sh
PDE_LIST="poisson helmholtz" TASK_LIST="forward inverse both" \
  TEST_TYPE=smooth NUM_SAMPLES=100 MAX_BATCH_SIZE=10 \
  OUTPUT_DIR=outputs/main_subset bash scripts/sampling/main/run_sample_sweep.sh
bash scripts/sampling/main/run_sample_sweep_burger.sh
```

For full-field observation on a 128 × 128 grid, set `--override num_obs=16384`.
For Burgers, use `configs/main/both/burger.yaml` with `sensor_mode=random`
or `sensor_column`.

The complete main sweep covers full-field forward/inverse and three sparse
tasks for Poisson, Helmholtz, Darcy, and Navier–Stokes, plus two Burgers layouts,
on ID/Smooth/Rough. Defaults are 1,000 inputs and 100 steps per comparison.
`OUTPUT_ROOT` changes `outputs/main`; `TEST_TYPE_LIST` selects distributions.
Matching completed jobs can be resumed.

Replay a saved aligned comparison with its original inputs and masks:

```bash
STUDY_ROOT=/path/to/saved_aligned_study \
CELL=supervised/poisson/id/sparse_joint \
OUTPUT_ROOT=outputs/reproductions/aligned \
  bash scripts/sampling/main/run_matched_cell.sh
```

This entry uses [experiments/aligned_sampling](experiments/aligned_sampling)
and runs a batch-consistency pilot before sampling. It requires the saved
`protocol.json`, input tensors, masks, and weights.

## Ablations

[sampling/sweep.py](sampling/sweep.py) expands
[configs/ablations/paper.yaml](configs/ablations/paper.yaml) and calls the shared
sampling runner. The separate [base profiles](configs/ablations/base) preserve
ablation guidance weights and gradient limits; datasets default to ID.

```bash
# Inspect the Python sweep; omit --list to run it.
python -m sampling.sweep --grid configs/ablations/paper.yaml \
  --pde poisson --group sampler_phase --list

PLAN_ONLY=true bash scripts/sampling/ablations/run.sh
PDE_LIST="poisson nsnonbounded" \
  bash scripts/sampling/ablations/run.sh guidance_components sampler_phase
PDE_LIST=poisson PARALLEL=true DEVICE_LIST="cuda:0 cuda:1" \
  bash scripts/sampling/ablations/run.sh num_steps_by_sampler sensor_sparsity
```

Other groups are `loss_state_by_sampler`, `sensor_mode`, `noise_robustness`,
`temporal_residual_mode`, and `statistics_stability`. `BATCH_SIZE` controls
inputs per job (default: 1); `OFFSET` selects the starting input.

[scripts/sampling/ablations/run_study.sh](scripts/sampling/ablations/run_study.sh)
exposes repeated draws (`ensemble`), guidance trajectories (`guidance`),
conditional averaging (`averaging`), observation layouts (`layouts`),
guidance-weight sweeps (`weights`), and unconditional samples (`prior`):

```bash
bash scripts/sampling/ablations/run_study.sh averaging --help
bash scripts/sampling/ablations/run_study.sh ensemble \
  --pdes nsnonbounded --inputs /path/to/prepared_inputs \
  --output /path/to/ensemble_results --checkpoint /path/to/fm4nsnonbounded.pth

# Error–time trajectories from an already prepared study.
DIFFUSION_ROOT=/path/to/DiffusionPDE \
  bash scripts/sampling/ablations/run_traces.sh run --root /path/to/trace_study
bash scripts/sampling/ablations/run_traces.sh plot \
  --root /path/to/trace_study --output /path/to/figures
```

Repeated studies require prepared inputs and selected settings (`selection.json`
for `ensemble`; `--anchors` for `weights`). Ensemble, guidance, and weight studies
read model defaults from `configs/ablations/base/both/<pde>.yaml`; use
`--checkpoint` or `--model-profile` to override them. See [plot/README.md](plot/README.md)
for preparation and figure tools, and `run_traces.sh prepare --help` for
trajectory input preparation.

## Baseline and other info

- [RecFNO and other baselines](https://github.com/Astringency/FM4PDEbaseline.git):
  FNO, DeepONet, iFNO, RecFNO, Senseiver, VoronoiCNN, PINN-Sparse, PDE-Opt,
  PC-BNN, 4D-Var, and VIVID.
- [DiffusionPDE experiments](https://github.com/Astringency/DiffusionPDE.git).
- [CoCoGen experiments](https://github.com/Astringency/CoCoGen.git).

Keep `experiments/`: the matched main-study launcher requires `aligned_sampling/`,
and the FM4PDE/DiffusionPDE error–time launcher requires `trajectories/`.

Validate all sampling profiles and ablation combinations without sampling:

```bash
python -m sampling.validate_configs
# Also require the configured datasets and weights to exist locally.
python -m sampling.validate_configs --check-assets
```
