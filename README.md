# Guided Flow Matching for Forward and Inverse PDE Problems with Sparse Observations: Algorithm and Theory

FM4PDE learns a joint prior over PDE inputs and solutions, then uses sparse
observations and physical residuals to guide forward, inverse, and joint
reconstruction. This repository contains the experiments in the revised manuscript.

The [reproduction guide](docs/reproduction.md) explains the required assets,
fixed random-draw protocols, execution checks and result files.

## Environment and data

Run from the repository root in Bash. Install [environment.yml](environment.yml)
and keep datasets and trained checkpoints outside Git.

```bash
conda env create -f environment.yml
conda activate fm4pde
export DATA_ROOT=/path/to/PDEdata
export CHECKPOINT_ROOT=/path/to/pretrained
export PYTHON_BIN=python
PDE=all DRY_RUN=true OUT_ROOT="$DATA_ROOT" bash data/DataGen/gen_pde.sh
PDE=poisson TYPE=all OUT_ROOT="$DATA_ROOT" bash data/DataGen/gen_pde.sh
```

[data/DataGen](data/DataGen) contains all eleven PDE generators. Static equations
and Burgers require MATLAB; Burgers also requires Chebfun, and shallow water uses
PyClaw. Defaults: five training shards of 10,000 samples; ID/Smooth/Rough test
sets of 10,000; Rough2/Rough3 sets of 1,000 for the five main PDEs.
[configs/training_data.yaml](configs/training_data.yaml) lists training inputs.
`CHECKPOINT_<PDE>` overrides an individual checkpoint, e.g. `CHECKPOINT_POISSON`.
Otherwise `CHECKPOINT_ROOT` replaces the `outputs/pretrained` prefix in the configs.

## Training

[configs/training.yaml](configs/training.yaml) records the appendix schedules,
scalar conditioning, and batch sizes: 45,000 training / 5,000 validation samples,
300 epochs, effective batch 64 on two GPUs.

```bash
bash scripts/train/poisson.sh --plan-only
bash scripts/train/poisson.sh
bash scripts/train/run_train.sh --pdes poisson helmholtz darcy nsnonbounded burger
```

One launcher per PDE is available in [scripts/train](scripts/train).
`--nproc` changes the GPU count while preserving effective batch 64.

## Main Sampling

Each script runs the FM4PDE portion of one manuscript paragraph. Configs in
[configs/main](configs/main) specify the 31 PDE/task profiles;
[configs/experiments/comparison](configs/experiments/comparison) specifies the
cases and protocols. Baseline execution is documented in the sibling repositories.
The main comparison manifest also records its selected guidance and sampler
overrides. These overrides are fixed across ID, Smooth, and Rough. Shared
profiles in `configs/main` remain the defaults for separate ablations, so
reproduce a manuscript experiment through its listed script and manifest.

```bash
bash scripts/sample/comparison/sparse_forward_inverse.sh --plan-only
bash scripts/sample/comparison/sparse_forward_inverse.sh --device cuda:0
bash scripts/sample/comparison/burgers_trajectory.sh --device cuda:0
```

| Paragraph | Script in `scripts/sample/comparison/` |
| --- | --- |
| Sparse forward/inverse reconstruction | `sparse_forward_inverse.sh` |
| Physics-based comparison, Smooth | `physics_based.sh` |
| Burgers random points / five time levels | `burgers_trajectory.sh` |
| Accuracy during sampling | `accuracy_during_sampling.sh` |
| Sampling time | `sampling_time.sh` |
| Reconstruction and physical consistency | `physical_consistency.sh` |

Comparisons use 100 realizations per setting; timing uses 20. Random observations
use 500 values per active field; Burgers structured observations use 640 values
at five complete time levels. Physical consistency includes reference re-solving;
set `MATLAB_BIN` and `CHEBFUN_ROOT` for Burgers. To include DiffusionPDE in
timing and error traces, set `DIFFUSION_ROOT` and `DIFFUSION_CHECKPOINT_ROOT`
(or `DIFFUSION_CHECKPOINT_<PDE>`). `--methods FM4PDE` runs the native method
without loading a DiffusionPDE model.

The controlled timing experiment uses the original fixed input bundle. Set
`TIMING_INPUT_ROOT` to its directory, containing `protocol.json`, `masks.npz`,
`source/timing_truths.npz`, and `weights/`. `TIMING_INPUT_ROOT_<PDE>` can select
a PDE-specific bundle. The runner verifies artifact checksums, reads the original
20 evaluation IDs, separate warmup ID, masks, checkpoint, and guidance settings,
and applies the current sampler and residual definition. The first 100-step
warmup must produce exactly the same fields as the standard sampling runner.
Generating new test data or choosing its first 20 rows does not reproduce this
fixed-input experiment. Error traces have a separate protocol recorded in
`accuracy_during_sampling.yaml`.

The current Helmholtz joint profile uses `clip_threshold: 150.0`. Joint
ablations, physical consistency, and timing use this value; the timing loader
discards the superseded `1e10` threshold in the archived input bundle. The
observation/PDE weights and `stochastic_guidance_coeff: 0.1` are unchanged.

## Ablations

Each file in [scripts/sample/ablations](scripts/sample/ablations) runs a complete
paragraph using its matching [experiment manifest](configs/experiments/ablations).

| Paragraph | Script name |
| --- | --- |
| Velocity architecture: U-Net / OFM | `velocity_architecture.sh` |
| Time grids | `time_grid.sh` |
| Observation and PDE guidance | `observation_pde_guidance.sh` |
| Guidance evaluation state | `guidance_evaluation_state.sh` |
| Deterministic, stochastic, and hybrid phases | `sampling_phases.sh` |
| Switching time: uniform and geometric D phases | `switching_time.sh` |
| Sampling steps | `sampling_steps.sh` |
| Observation density | `observation_density.sh` |
| Observation noise | `noise_level.sh` |
| Temporal residuals and true-endpoint diagnostics | `temporal_residuals.sh` |
| Observation layouts | `observation_layouts.sh` |
| Conditional averaging | `conditional_averaging.sh` |
| Additional PDE families | `additional_pde_families.sh` |

```bash
bash scripts/sample/ablations/time_grid.sh --device cuda:0
bash scripts/sample/ablations/temporal_residuals.sh --pdes heat wave --plan-only
bash scripts/sample/ablations/temporal_residuals.sh --truth-only --device cuda:0
```

Common options: `--pdes`, `--device`, `--output`, `--limit` (development subset),
`--override key=value`. Standard sampling and conditional averaging resume only
when saved identities match; other diagnostics rerun the selected group.

The [hybrid grid protocol](docs/hybrid_geometric_grid.md) specifies physical
switching times, stage budgets, and the command to regenerate the two switching
figures from saved run outputs.

The [shared inverse guidance settings](docs/unified_inverse_guidance.md) give
the physics-comparison, conditional-averaging, and Poisson-trajectory commands,
including their fixed inputs and observation protocols.

Including FM4PDE-OFM in the architecture comparison needs the sibling
`FunDPS_DDIS_ECI_OFM` checkout,
`OFM_DATA_ROOT` (compact data) and `OFM_CHECKPOINT_ROOT` (the selected
`<pde>/epoch_<n>.pt` files listed in the architecture manifest).
[configs/ofm_guidance.yaml](configs/ofm_guidance.yaml) contains its shared weights.
For the main FM4PDE-OFM and CoCoGen comparisons, including fixed test inputs,
observation masks, batch partitions and resume checks, see the
[baseline reproduction guide](docs/baseline_residual_reproduction.md).

For a partial rerun, select existing cases without changing their settings:

```bash
bash scripts/sample/ablations/sampling_phases.sh --pdes nsnonbounded reaction_diffusion
bash scripts/sample/ablations/sampling_steps.sh --phases deterministic --steps 100
bash scripts/sample/comparison/sparse_forward_inverse.sh --pdes darcy nsnonbounded
bash scripts/sample/comparison/sampling_time.sh --methods FM4PDE
bash scripts/sample/ablations/velocity_architecture.sh --methods FM4PDE --pdes darcy nsnonbounded
```

The secant residual is `(u-a)/T - (G_h(a)+G_h(u))/2`. Sampling guidance and
temporal diagnostics call the same implementation. `--truth-only` recomputes
all three losses on the 32 real endpoint pairs for each selected PDE without
sampling. Its receipt records the residual source hash and secant definition.
Physical consistency records the updated PDE residual alongside the independent
reference-solver defect; these are distinct metrics.

Saved physical predictions can be reevaluated without sampling:

```bash
bash scripts/sample/comparison/reevaluate_saved_residuals.sh \
  --manifest /path/to/saved_predictions.json --output outputs/residual_reevaluation
```

The JSON manifest has a `records` list. Each record specifies `pde` (Darcy or
NS), `method`, `distribution`, `count`, `truth_file`, `truth_sha256`, and a
`predictions` list of `{ "file": "...", "sha256": "..." }`. Truth files contain
a physical `truth` tensor or `raw.full_tensor`; prediction files contain either
`prediction` and `indices`, or native `coef_final`, `sol_final`, and `config`.
Every index from zero to `count-1` must occur once. The output includes field
errors, current PDE losses, and checksums. Comparing methods requires reevaluating
every saved prediction with the same residual definition before ranking them.

`--tasks` and `--test-types` filter tasks and test distributions; `--job-ids`
selects exact manifest cases for a retry. For independent
GPU workers, use `--num-shards N --shard-index i` with separate `--output`
directories for each worker. Sharding preserves the original case indices,
seeds, and sampling batch sizes. `--continue-on-error` records failed cases in
`failures.json`; those cases must be reported and are not valid completed runs.
The standard paper runner fixes CUDA kernel selection and disables TF32.
Conditional averaging retains its recorded TF32 and fused-gradient execution
path, as described in the shared inverse-guidance guide. Data files,
checkpoints, resolved configurations, source hashes, and saved predictions are
recorded for standard sampling runs. MAT files are cached in memory to avoid
repeated decompression; the physical input arrays are unchanged.

## Appendix prior samples

```bash
bash scripts/sample/appendix/unconditional_priors.sh --plan-only
bash scripts/sample/appendix/unconditional_priors.sh --device cuda:0
```

This entry generates all eleven appendix samples with 100 direct Euler steps,
then re-solves each generated input with the reference numerical operator.
It uses the same checkpoint variables as the sampling experiments. The fixed
seeds and known physical coefficients are recorded in
[the appendix manifest](configs/experiments/appendix/unconditional_priors.yaml).
The Poisson seed is 20260911; the other ten seeds are 20260910. Scalar-conditioned
models receive the recorded physical parameters; no observed fields enter these
draws. Each output contains physical fields, componentwise solver-relative errors,
checkpoint/source identities and the initial-noise checksum. Burgers requires
`MATLAB_BIN` and `CHEBFUN_ROOT`; shallow-water reference solving requires PyClaw.

## Baseline and other info

Baseline repositories: [FM4PDEbaseline](https://github.com/Astringency/FM4PDEbaseline),
[CoCoGen](https://github.com/Astringency/CoCoGen),
[DiffusionPDE](https://github.com/Astringency/DiffusionPDE), and
[FunDPS_DDIS_ECI_OFM](https://github.com/Astringency/FunDPS_DDIS_ECI_OFM)
(FunDPS, DDIS, ECI, OFM). Each README lists methods, upstream sources, and examples.

| Directory | Contents |
| --- | --- |
| `configs`, `data` | Paper settings, loaders, normalization, and generators |
| `flow_matching`, `models`, `torchdiffeq`, `training` | Flow objectives, networks, ODE solvers, and training |
| `sampling`, `experiments` | Guided samplers, paragraph runners, and paired diagnostics |
| `plot` | Figure renderers and support for saved manuscript results |
| `scripts` | Training and paragraph-level experiment launchers |

```bash
python -m sampling.validate_configs
```

## Citation

Please cite [our paper on arXiv](https://arxiv.org/abs/2605.25509):

```bibtex
@misc{zhang2026guidedflowmatching,
  title = {Guided Flow Matching for Forward and Inverse PDE Problems with Sparse Observations: Algorithm and Theory},
  author = {Xifeng Zhang and Jin Zhao},
  year = {2026},
  eprint = {2605.25509},
  archivePrefix = {arXiv},
  primaryClass = {stat.ML},
  url = {https://arxiv.org/abs/2605.25509}
}
```
