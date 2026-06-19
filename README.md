# FM4PDE

FM4PDE is a Flow Matching codebase for generating, completing, and inverting PDE solution fields with observation and PDE residual guidance.

## Entrypoints

```bash
python train.py
python -m sampling.runner --config configs/ablations/base/heat.yaml
python -m sampling.sweep --grid configs/ablations/all_internal_ablation_grid.yaml --list
python -m sampling.aggregate outputs/ablations --output-dir outputs/ablations
```

`sample.py` delegates directly to `sampling.runner`.

## Data Interface

PDE channel definitions, labels, sample-level scalar parameters, aliases, default `loadby`, and residual family are centralized in `data/specs.py`.

The endpoint pair HDF5 format is addressed as `loadby: pair_h5`. Its on-disk datasets remain:

```text
input_data
output_data
```

Those dataset names are part of the file format. Scalar PDE parameters such as `alpha`, `c`, `b_x`, `b_y`, `kappa`, `u_D`, `T`, and `dt` are sample-level metadata. They are stored in `PDEloader.pde_params`, training `data_metadata.json`, checkpoint `data_metadata`, and sampling `PDEGroundTruth.pde_params`; they are not Flow Matching input channels.

`pair_h5` is a file-format/loadby name, not a directory name. Formal data paths use one PDE-named directory directly under the data root, for example `PDEdata/heat/heat_test_1000-128-128.h5`.

Non-bounded Navier-Stokes keeps its generated HDF5 layout exposed as `loadby: h5py`, usually with `w0` as the initial vorticity field and `w` as the vorticity trajectory. Sampling reads optional sample-level `nu`, `viscosity`, `T`, `total_time`, and `dt` from root attributes or datasets using `data/specs.py`. When `residual_mode: full_trajectory_fd` is requested, the `w` trajectory is normalized to `[B,T,C,H,W]` and used as explicit trajectory state; missing or ambiguous trajectory data raises `ValueError`.

## Residual Families

| PDE | residual_family | auto / mode | status |
| --- | --- | --- | --- |
| darcy | static | static | reliable |
| poisson | static | static | reliable |
| helmholtz | static | static | reliable |
| steady_heat_conduction | static | static_nonlinear_boundary | reliable |
| burger | full_time_space | full_time_space | reliable |
| heat | temporal_endpoint | hermite_bridge | approximate |
| wave | temporal_endpoint | hermite_bridge | approximate |
| advection_diffusion | temporal_endpoint | hermite_bridge | approximate |
| reaction_diffusion | temporal_endpoint | hermite_bridge | approximate |
| shallow_water | temporal_endpoint | hermite_bridge | approximate |
| nsnonbounded | temporal_endpoint | hermite_bridge | approximate; PDE guidance enabled |

Temporal endpoint PDEs support `hermite_bridge`, `endpoint_secant`, `near_endpoint_temporal`, and `full_trajectory_fd`. `endpoint_secant` is a coarse ablation mode. `near_endpoint_temporal` requires explicit near-endpoint observations and masks. `full_trajectory_fd` requires an explicit full trajectory tensor.

Burgers is not an endpoint pair residual: its single BCHW field contains a time-space grid and uses full finite differences. Steady Heat Conduction is static, not a time-dependent approximation.

See `docs/time_dependent_residuals.md` for details.

## Model Architecture Profiles

Model configs are centralized in `models/model_configs.py` and selected by
`model_profile`. The default `recommended` profile uses PDE-family architectures:

| PDEs | architecture family |
| --- | --- |
| poisson, heat | `light_smooth` |
| darcy, helmholtz, steady_heat_conduction | `elliptic_static` |
| advection_diffusion, reaction_diffusion | `temporal_endpoint_base` |
| wave, shallow_water, nsnonbounded | `temporal_endpoint_heavy` |
| burger | `full_time_space` |

The `light`, `base`, and `heavy` profiles are explicit architecture ablations.
`legacy_base` is only for reproducing old checkpoints and is not the formal
default. Recommended configs move attention away from downsample factor `2` to
coarser factors such as `8` and `16`, reducing the quadratic attention memory
cost on 128x128 data while preserving coarse global structure.

Burgers records `axis_semantics=BCHW_as_time_space_H_time_W_space` because `H`
is time and `W` is space. Sample-level scalar PDE parameters are loaded and used
by residuals; network FiLM scalar conditioning is currently metadata/TODO only
(`scalar_conditioning=false`) and scalar parameters are not added as constant
input channels.

Training checkpoints save `model_profile`, `model_config`,
`model_config_metadata`, `data_metadata`, `num_channels`, and
`checkpoint_schema_version`. Sampling reconstructs the model from checkpoint
`model_config` first; old checkpoints without this field require an explicit
profile override such as `model_profile=legacy_base`.

See `docs/model_architectures.md` for details.

## Sampling

Configs use flat `AblationConfig` YAML. Old `data/generate/model` YAML structures are rejected with a clear conversion error.

Single run:

```bash
python -m sampling.runner \
  --config configs/ablations/base/heat.yaml \
  --override residual_mode=hermite_bridge
```

Sweep:

```bash
python -m sampling.sweep \
  --grid configs/ablations/all_internal_ablation_grid.yaml \
  --group time_dependent_residual_mode
```

`configs/ablations/all_internal_ablation_grid.yaml` includes `nsnonbounded` in the top-level `base_configs`, so NS participates in the same guidance, sensor, noise, zeta, time-grid, clipping, residual-region, and statistics ablations as the other PDEs. The time-dependent residual-mode group also includes NS with the other temporal endpoint PDEs.

Formal base configs set `allow_synthetic_data: false`; dry-run smoke configs may use synthetic data.

## Training

Training uses `data/load.py::PDEloader` and `data/transform.py::PDEStandardizer`.

```bash
python train.py \
  --dataset heat \
  --data_path /large_storage/zhangxf/PDEdata/ \
  --output_dir outputs/pretrained/ \
  --epochs 500 \
  --batch_size 32 \
  --model_profile recommended
```

Checkpoints include model weights, normalizer, data shape, channel names, scalar PDE parameter keys, data specs, and residual families. Sampling requires a checkpoint normalizer; dry-run is the only path that creates an explicit identity normalizer.

## Data Generation

Endpoint pair HDF5 generation:

```bash
bash data/DataGen/run_generate_pair_h5s_50k_10k_fulltraj.sh
```

Only generate endpoint-pair test data:

```bash
SPLIT=test N_TEST=10000 bash data/DataGen/run_generate_pair_h5s_50k_10k_fulltraj.sh
```

Quick check:

```bash
python data/DataGen/python/generate_pair_h5s.py \
  --pde all \
  --out-root /tmp/fm4pde_pair_h5s \
  --quick-test \
  --overwrite
```

More details are in `data/DataGen/pde_data_generation_summary.md`, `docs/sampling.md`, and `docs/normalization.md`.

## Tests

The focused refactor tests can be run in the `fm4pde` environment with:

```bash
conda run -n fm4pde python -m pytest \
  tests/test_model_configs.py \
  tests/test_sampling_model_io.py \
  tests/test_training_metadata_model_config.py \
  tests/test_ablation_pair_h5_metadata.py \
  tests/test_loader_shapes.py \
  tests/test_pde_residuals_endpoint_and_static.py \
  tests/test_sweep_all_pdes.py \
  tests/test_ablation_config.py \
  tests/test_sampling_no_synthetic_formal.py -q
```

`sampling.config` works with or without `PyYAML`; the tests avoid requiring `PyYAML` directly.
