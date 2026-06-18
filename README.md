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
| nsnonbounded | temporal_endpoint | hermite_bridge | approximate |

Temporal endpoint PDEs support `hermite_bridge`, `endpoint_secant`, `near_endpoint_temporal`, and `full_trajectory_fd`. `endpoint_secant` is a coarse ablation mode. `near_endpoint_temporal` requires explicit near-endpoint observations and masks. `full_trajectory_fd` requires an explicit full trajectory tensor.

Burgers is not an endpoint pair residual: its single BCHW field contains a time-space grid and uses full finite differences. Steady Heat Conduction is static, not a time-dependent approximation.

See `docs/time_dependent_residuals.md` for details.

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

Formal base configs set `allow_synthetic_data: false`; dry-run smoke configs may use synthetic data.

## Training

Training uses `data/load.py::PDEloader` and `data/transform.py::PDEStandardizer`.

```bash
python train.py \
  --dataset heat \
  --data_path /large_storage/zhangxf/PDEdata/ \
  --output_dir outputs/pretrained/ \
  --epochs 500 \
  --batch_size 32
```

Checkpoints include model weights, normalizer, data shape, channel names, scalar PDE parameter keys, data specs, and residual families. Sampling requires a checkpoint normalizer; dry-run is the only path that creates an explicit identity normalizer.

## Data Generation

Endpoint pair HDF5 generation:

```bash
bash data/DataGen/run_generate_pair_h5s_50k_10k_fulltraj.sh
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
