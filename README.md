# Guided Flow Matching for Forward and Inverse PDE Problems with Sparse Observations: Algorithm and Theory

## Environment and data

Create the environment using [environment.yml](environment.yml) and download the
[PDE datasets](https://huggingface.co/datasets/XifengZhang/FM4PDE-pde-data) from Hugging Face.

```bash
conda env create -f environment.yml
conda activate fm4pde
export DATA_ROOT=/path/to/PDEdata
export CHECKPOINT_ROOT=/path/to/pretrained
export PYTHON_BIN=python
```

`DATA_ROOT` is the data directory, with the downloaded subdirectories preserved.

```bash
PDE=all DRY_RUN=true OUT_ROOT=/path/to/generated-data bash data/DataGen/gen_pde.sh
PDE=poisson TYPE=all OUT_ROOT=/path/to/generated-data bash data/DataGen/gen_pde.sh
```

## Training

Download pretrained models from [Hugging Face](https://huggingface.co/XifengZhang/FM4PDE-pretrained-models)
for the sampling examples. The training commands below use [configs/training.yaml](configs/training.yaml).

`CHECKPOINT_ROOT` points to the downloaded model repository containing `formal/`.

```bash
bash scripts/train/poisson.sh --plan-only
bash scripts/train/poisson.sh
bash scripts/train/run_train.sh --pdes poisson helmholtz darcy nsnonbounded burger
```

## Main Sampling

```bash
bash scripts/sample/comparison/sparse_forward_inverse.sh --plan-only
bash scripts/sample/comparison/sparse_forward_inverse.sh --device cuda:0
bash scripts/sample/comparison/burgers_trajectory.sh --device cuda:0
```

| Paragraph | Script in `scripts/sample/comparison/` |
| --- | --- |
| Sparse forward/inverse reconstruction | `sparse_forward_inverse.sh` |
| Physics-based comparison, Smooth | `physics_based.sh` |
| Burgers trajectory reconstruction | `burgers_trajectory.sh` |
| Accuracy during sampling | `accuracy_during_sampling.sh` |
| Sampling time | `sampling_time.sh` |
| Reconstruction and physical consistency | `physical_consistency.sh` |

`MATLAB_BIN` selects the MATLAB executable; `CHEBFUN_ROOT` points to Chebfun.

## Ablations

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
| Temporal residuals | `temporal_residuals.sh` |
| Observation layouts | `observation_layouts.sh` |
| Conditional averaging | `conditional_averaging.sh` |
| Additional PDE families | `additional_pde_families.sh` |

```bash
bash scripts/sample/ablations/time_grid.sh --device cuda:0
bash scripts/sample/ablations/temporal_residuals.sh --pdes heat wave --plan-only
```

```bash
bash scripts/sample/ablations/sampling_phases.sh --pdes nsnonbounded reaction_diffusion
bash scripts/sample/ablations/sampling_steps.sh --phases deterministic --steps 100
bash scripts/sample/comparison/sparse_forward_inverse.sh --pdes darcy nsnonbounded
bash scripts/sample/comparison/sampling_time.sh --methods FM4PDE
bash scripts/sample/ablations/velocity_architecture.sh --methods FM4PDE --pdes darcy nsnonbounded
```

```bash
bash scripts/sample/comparison/reevaluate_saved_residuals.sh \
  --manifest /path/to/saved_predictions.json --output outputs/residual_reevaluation
```

## Appendix prior samples

```bash
bash scripts/sample/appendix/unconditional_priors.sh --plan-only
bash scripts/sample/appendix/unconditional_priors.sh --device cuda:0
```

The appendix commands use
[unconditional_priors.yaml](configs/experiments/appendix/unconditional_priors.yaml).

## Baseline and other info

| Directory | Contents |
| --- | --- |
| `configs`, `data` | Paper settings, loaders, normalization, and generators |
| `flow_matching`, `models`, `torchdiffeq`, `training` | Flow objectives, networks, ODE solvers, and training |
| `sampling`, `experiments` | Guided samplers, paragraph runners, and paired diagnostics |
| `plot` | Figure renderers and support for saved manuscript results |
| `scripts` | Training and paragraph-level experiment launchers |

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
