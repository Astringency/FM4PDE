# FM4PDE Sampling

`python -m sampling.runner` is the only single-run sampling entrypoint. `python -m sampling.sweep` is the only grouped sweep entrypoint. The root `sample.py` delegates directly to `sampling.runner`.

## Config Schema

Sampling configs use the flat `AblationConfig` schema. YAML files with top-level `data`, `generate`, or `model` sections are rejected with a conversion error instead of being converted automatically.

Required data fields for formal runs:

- `pde`
- `task`
- `data_path`
- `loadby`
- `coef_name`
- `solution_name`
- `checkpoint_path`
- `model_profile`
- `img_channels`
- `img_resolution`
- `allow_synthetic_data: false`

Smoke configs may set `allow_synthetic_data: true`; formal configs should fail if the dataset path is missing.

`model_profile: recommended` is the formal default. `light`, `base`, and
`heavy` are architecture ablation profiles; `legacy_base` is only for explicit
old-checkpoint reproduction. Sampling reconstructs the model from checkpoint
`model_config` first, then records both the requested runtime profile and the
checkpoint architecture metadata in `run_metadata.json`.

## Pair HDF5 Metadata

Heat, Wave, Advection-Diffusion, and Steady Heat Conduction use the endpoint pair HDF5 format exposed as `loadby: pair_h5`. The HDF5 dataset names remain `input_data` and `output_data`; those names are part of the disk format and do not imply any special code path.

`pair_h5` is not a storage subdirectory. Formal data files are expected directly under `DATA_ROOT/<pde>/`, for example `PDEdata/heat/heat_test_1000-128-128.h5`.

Scalar or sample-level PDE parameters are loaded into `PDEGroundTruth.pde_params` and are not Flow Matching input channels:

- Heat: `alpha`, `T`, `total_time`, `dt`
- Wave: `c`, `T`, `total_time`, `dt`
- Advection-Diffusion: `b_x`, `b_y`, `kappa`, `T`, `total_time`, `dt`
- Steady Heat Conduction: `u_D` plus available solver/source diagnostics

Channel names, scalar parameter names, aliases, labels, default `loadby`, and `residual_family` come from `data/specs.py`.

## Residual Metadata

Every PDE residual writes:

- `residual_family`
- `temporal_derivative_mode`
- `endpoint_only`
- `uses_generated_trajectory`
- `uses_extra_temporal_observations`
- `requested_residual_mode`
- `resolved_residual_mode`
- `residual_status`

Status is not the same as family. Static PDEs and Burgers are `reliable`; endpoint temporal residuals are `approximate` because the default endpoint/sparse temporal guidance approximates time dynamics.

## Commands

Single run:

```bash
python -m sampling.runner \
  --config configs/ablations/base/heat.yaml \
  --override residual_mode=hermite_bridge \
  --override output_dir=outputs/ablations
```

List a sweep:

```bash
python -m sampling.sweep \
  --grid configs/ablations/all_internal_ablation_grid.yaml \
  --list
```

Run one sweep group:

```bash
python -m sampling.sweep \
  --grid configs/ablations/all_internal_ablation_grid.yaml \
  --group time_dependent_residual_mode
```

Aggregate:

```bash
python -m sampling.aggregate outputs/ablations --output-dir outputs/ablations
```

Each run writes `resolved_config.yaml`, `run_metadata.json`, `metrics_step.jsonl`, `metrics_final.json`, `curves.csv`, `summary.csv`, `result.pt`, and `masks.pt`.
