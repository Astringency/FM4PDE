# Guided Flow Matching for Forward and Inverse PDE Problems with Sparse Observations: Algorithm and Theory

FM4PDE learns a joint prior over PDE inputs and solutions, then uses sparse
observations and physical residuals to guide forward, inverse, and joint
reconstruction. This repository provides the implementation and experiment settings
for the paper.

## Environment and data

The PDE data repository is hosted on Hugging Face:
[XifengZhang/FM4PDE-pde-data](https://huggingface.co/datasets/XifengZhang/FM4PDE-pde-data).
Set `DATA_ROOT` to the downloaded data directory, preserving its subdirectories and filenames.

Run from the repository root in Bash. Install [environment.yml](environment.yml)
and keep datasets and trained checkpoints outside Git.

```bash
conda env create -f environment.yml
conda activate fm4pde
export DATA_ROOT=/path/to/PDEdata
export CHECKPOINT_ROOT=/path/to/pretrained
export PYTHON_BIN=python
```

To generate a new dataset, choose its output directory separately:

```bash
PDE=all DRY_RUN=true OUT_ROOT=/path/to/generated-data bash data/DataGen/gen_pde.sh
PDE=poisson TYPE=all OUT_ROOT=/path/to/generated-data bash data/DataGen/gen_pde.sh
```

[data/DataGen](data/DataGen) contains all eleven PDE generators. Static equations
and Burgers require MATLAB; Burgers also requires Chebfun, and shallow water uses
PyClaw. Defaults: five training shards of 10,000 samples; ID/Smooth/Rough test
sets of 10,000; Rough2/Rough3 sets of 1,000 for the five main PDEs.
Shallow-water training uses radius `Uniform(0.3, 0.7)` and interior depth `2.0`;
test generation uses `Uniform(0.4, 0.8)` and `Uniform(2, 3)`, respectively.
Generated files record interior depth in `inner_height`.
[configs/training_data.yaml](configs/training_data.yaml) lists training inputs.
`CHECKPOINT_<PDE>` overrides an individual checkpoint, e.g. `CHECKPOINT_POISSON`.
Otherwise `CHECKPOINT_ROOT` replaces the `outputs/pretrained` prefix in the configs.

## Training

Pretrained FM4PDE models for all eleven PDEs are available on Hugging Face:
[XifengZhang/FM4PDE-pretrained-models](https://huggingface.co/XifengZhang/FM4PDE-pretrained-models).
Set `CHECKPOINT_ROOT` to the downloaded repository root containing `formal/` to use these models for sampling.

[configs/training.yaml](configs/training.yaml) records the appendix schedules,
scalar conditioning, and batch sizes: 45,000 training / 5,000 validation samples,
300 epochs, effective batch 64 on two GPUs.

```bash
bash scripts/train/poisson.sh --plan-only
bash scripts/train/poisson.sh
bash scripts/train/run_train.sh --pdes poisson helmholtz darcy nsnonbounded burger
```

One launcher per PDE is available in [scripts/train](scripts/train).
`--nproc` changes the GPU count; per-GPU batch size times GPU count must divide 64.

## Main Sampling

[configs/main](configs/main) defines the PDE/task profiles, while
[comparison manifests](configs/experiments/comparison) specify the cases and
observation protocols. Run each experiment through its listed script.

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

Comparisons use 100 samples per setting; timing uses 20. Random observations
cover 500 locations per observed channel; Burgers structured observations use
640 values at five complete physical-time levels. Reference re-solving for
Burgers requires `MATLAB_BIN` and `CHEBFUN_ROOT`.

Timing uses the paper's joint-guidance parameters, stochastic updates, and the
included `timing_protocol.json` and `timing_masks.npz`. Inputs are Smooth for
Poisson, Helmholtz, Darcy and Navier–Stokes, and ID for Burgers. Timing and error
traces run both methods by default: set `DIFFUSION_ROOT` and
`DIFFUSION_CHECKPOINT_ROOT` (or `DIFFUSION_CHECKPOINT_<PDE>`), or select
`--methods FM4PDE`. Error traces use `accuracy_during_sampling.yaml`.

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
`--override key=value`. Sampling, architecture comparisons, conditional averaging,
and appendix priors support checked resume; timing and error traces rerun the selection.

Flow-time switching settings are in [switching_time.yaml](configs/experiments/ablations/switching_time.yaml);
[plot/switching_time.py](plot/switching_time.py) renders the comparison.

Physics comparisons, conditional averaging, and error traces use their respective
manifests in [configs/experiments](configs/experiments).

FM4PDE-OFM uses `GENERATIVE_BASELINE_ROOT`, `OFM_DATA_ROOT`, and `OFM_CHECKPOINT_ROOT`;
checkpoints are listed in [velocity_architecture.yaml](configs/experiments/ablations/velocity_architecture.yaml).
The comparison launchers are [sparse_forward_inverse_ofm.sh](scripts/sample/comparison/sparse_forward_inverse_ofm.sh)
and [cocogen.sh](scripts/sample/comparison/cocogen.sh); the latter uses `COCOGEN_ROOT`.
Guidance weights are in [configs/ofm_guidance.yaml](configs/ofm_guidance.yaml).

For a partial rerun, select existing cases without changing their settings:

```bash
bash scripts/sample/ablations/sampling_phases.sh --pdes nsnonbounded reaction_diffusion
bash scripts/sample/ablations/sampling_steps.sh --phases deterministic --steps 100
bash scripts/sample/comparison/sparse_forward_inverse.sh --pdes darcy nsnonbounded
bash scripts/sample/comparison/sampling_time.sh --methods FM4PDE
bash scripts/sample/ablations/velocity_architecture.sh --methods FM4PDE --pdes darcy nsnonbounded
```

The secant residual is `(u-a)/T - (G_h(a)+G_h(u))/2`. Sampling guidance and
temporal diagnostics share its implementation. `--truth-only` evaluates all three
temporal losses on real endpoint pairs without sampling. Physical consistency
reports PDE loss and reference-solver error separately.

Saved physical predictions can be reevaluated without sampling:

```bash
bash scripts/sample/comparison/reevaluate_saved_residuals.sh \
  --manifest /path/to/saved_predictions.json --output outputs/residual_reevaluation
```

This supports Poisson, Helmholtz, Darcy, and Navier–Stokes. The JSON manifest's
`records` specify `pde`, `method`, `distribution`, `count`, `truth_file`,
`truth_sha256`, and `predictions` entries with `file` and `sha256`.

`--tasks` and `--test-types` filter comparisons; `--job-ids` selects manifest
sampling cases. Independent GPU workers use `--num-shards N --shard-index i`
with separate output directories. `--continue-on-error` records sampling failures
in `failures.json`.

## Appendix prior samples

```bash
bash scripts/sample/appendix/unconditional_priors.sh --plan-only
bash scripts/sample/appendix/unconditional_priors.sh --device cuda:0
```

This generates eleven unguided samples with 100 Euler steps and reference
re-solving. Seeds and physical parameters are fixed in the
[appendix manifest](configs/experiments/appendix/unconditional_priors.yaml).
Burgers requires MATLAB and Chebfun; shallow-water reference solving requires PyClaw.

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
