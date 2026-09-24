# Guided Flow Matching for Forward and Inverse PDE Problems with Sparse Observations: Algorithm and Theory

FM4PDE learns a joint prior over PDE inputs and solutions, then uses sparse
observations and physical residuals to guide forward, inverse, and joint
reconstruction. This repository contains the experiments in the revised manuscript.

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
set `MATLAB_BIN` and `CHEBFUN_ROOT` for Burgers. Timing and error traces require
`DIFFUSION_ROOT` and `DIFFUSION_CHECKPOINT_ROOT` (or `DIFFUSION_CHECKPOINT_<PDE>`).

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
| Switching time | `switching_time.sh` |
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
```

Common options: `--pdes`, `--device`, `--output`, `--limit` (development subset),
`--override key=value`. Standard sampling and conditional averaging resume only
when saved identities match; other diagnostics rerun the selected group.
Architecture comparison also needs the sibling `FunDPS_DDIS_ECI_OFM` checkout,
`OFM_DATA_ROOT` (compact data) and `OFM_CHECKPOINT_ROOT` (`<pde>/best.pt`).
[configs/ofm_guidance.yaml](configs/ofm_guidance.yaml) contains its shared weights.

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
