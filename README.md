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

Those dataset names are part of the file format. Scalar PDE parameters such as `alpha`, `c`, `b_x`, `b_y`, `kappa`, `u_D`, `T`, and `dt` are sample-level metadata. They are stored in `PDEloader.pde_params`, training `data_metadata.json`, checkpoint `data_metadata`, and sampling `PDEGroundTruth.pde_params`; they are not expanded into spatial Flow Matching input channels. Training can opt in to selected scalar metadata with `--scalar_conditioning_params`, which standardizes those values from training-set statistics and injects them through a UNet time-embedding MLP.

`pair_h5` is a file-format/loadby name, not a directory name. Formal data paths use one PDE-named directory directly under the data root, for example `PDEdata/heat/heat_test_1000-128-128.h5`.

Non-bounded Navier-Stokes keeps its generated HDF5 layout exposed as `loadby: h5py`, usually with `w0` as the initial vorticity field and `w` as the vorticity trajectory. Sampling reads optional sample-level `nu`, `viscosity`, `T`, `total_time`, and `dt` from root attributes or datasets using `data/specs.py`. The model currently predicts only `w0/wT`, so its endpoint fields in PDE loss always come from the model. A saved true `w` trajectory is never substituted for those outputs. The sole field-data exception is an explicitly selected `near_endpoint_temporal` residual, which may read only sparse masked observations at `dt` and `T-dt`.

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

Temporal endpoint models normally use `hermite_bridge` or `endpoint_secant`, both computed from predicted `a/q0` and `u/qT`. For Heat, Wave, Advection-Diffusion, Reaction-Diffusion, Shallow Water, and Non-bounded Navier-Stokes, formal sampling may use the sole `near_endpoint_temporal` exception: `q(dt)` reuses the q0 sensor mask and `q(T-dt)` reuses the qT sensor mask. Unobserved auxiliary values are removed before the residual is evaluated; unconditional periodic training evaluation rejects this mode. Burgers instead outputs the complete `[B,1,T,X]` prediction, and PDE loss finite-differences that predicted field at every interior time point.

Burgers is not an endpoint pair residual: its single BCHW field contains a time-space grid and uses full finite differences. Steady Heat Conduction is static, not a time-dependent approximation.

See `docs/time_dependent_residuals.md` for details.

## Model Architecture Profiles

Model configs are centralized in `models/model_configs.py` and selected by
`model_profile`. The default `recommended` profile uses PDE-family architectures:

| PDEs | architecture family |
| --- | --- |
| poisson | `light_smooth` |
| darcy, helmholtz, steady_heat_conduction | `elliptic_static` |
| heat, advection_diffusion, reaction_diffusion | `temporal_endpoint_base` |
| wave, shallow_water, nsnonbounded | `temporal_endpoint_heavy` |
| burger | `full_time_space` |

Current `recommended` profile details and raw UNet parameter counts are:

| PDE | family | data C in/out | effective input C | model channels | res blocks | channel mult | attention | value Fourier | coordinate Fourier | parameters |
| --- | --- | --- | ---: | ---: | ---: | --- | --- | --- | --- | ---: |
| `darcy` | `elliptic_static` | 2 / 2 | 20 | 128 | 4 | `(1, 2, 4)` | `(8, 16)` | false | true | 96,455,042 (96.46M) |
| `poisson` | `light_smooth` | 2 / 2 | 2 | 96 | 3 | `(1, 2, 4)` | `(16,)` | false | false | 44,121,218 (44.12M) |
| `helmholtz` | `elliptic_static` | 2 / 2 | 28 | 128 | 4 | `(1, 2, 4)` | `(8, 16)` | true | true | 96,464,258 (96.46M) |
| `nsnonbounded` | `temporal_endpoint_heavy` | 2 / 2 | 10 | 192 | 4 | `(1, 2, 4, 4)` | `(8, 16)` | true | false | 387,487,682 (387.49M) |
| `burger` | `full_time_space` | 1 / 1 | 23 | 128 | 4 | `(1, 2, 4)` | `(16,)` | true | true | 96,457,345 (96.46M) |
| `reaction_diffusion` | `temporal_endpoint_base` | 4 / 4 | 4 | 128 | 4 | `(1, 2, 4)` | `(16,)` | false | false | 96,438,916 (96.44M) |
| `shallow_water` | `temporal_endpoint_heavy` | 6 / 6 | 6 | 192 | 4 | `(1, 2, 4, 4)` | `(8, 16)` | false | false | 387,487,686 (387.49M) |
| `heat` | `temporal_endpoint_base` | 2 / 2 | 2 | 128 | 4 | `(1, 2, 4)` | `(16,)` | false | false | 96,434,306 (96.43M) |
| `wave` | `temporal_endpoint_heavy` | 4 / 4 | 38 | 192 | 4 | `(1, 2, 4, 4)` | `(8, 16)` | true | true | 387,539,524 (387.54M) |
| `advection_diffusion` | `temporal_endpoint_base` | 2 / 2 | 20 | 128 | 4 | `(1, 2, 4)` | `(16,)` | false | true | 96,455,042 (96.46M) |
| `steady_heat_conduction` | `elliptic_static` | 2 / 2 | 20 | 128 | 4 | `(1, 2, 4)` | `(8, 16)` | false | true | 96,455,042 (96.46M) |

`effective input C` includes internal value/coordinate Fourier feature channels
added inside `UNetModel`; `data C in/out` remains the standardized PDE data
channel count. Parameter counts are for the raw UNet with `use_ema=false`, so
they do not include optimizer state, EMA shadow weights, or checkpoint metadata.

The `light`, `base`, and `heavy` profiles are explicit architecture ablations.
`legacy_base` is only for reproducing old checkpoints and is not the formal
default. Recommended configs move attention away from downsample factor `2` to
coarser factors such as `8` and `16`, reducing the quadratic attention memory
cost on 128x128 data while preserving coarse global structure.

Burgers records `axis_semantics=BCHW_as_time_space_H_time_W_space` because `H`
is time and `W` is space. Sample-level scalar PDE parameters are loaded and used
by residuals. Network scalar conditioning is disabled by default
(`scalar_conditioning=false`); when explicitly enabled with
`--scalar_conditioning_params`, the selected parameters are added to the UNet
time embedding and are still not added as constant input channels.

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

Scalar conditioning is opt-in:

```bash
python train.py \
  --dataset heat \
  --data_path /large_storage/zhangxf/PDEdata/ \
  --eval_frequency -1 \
  --scalar_conditioning_params alpha T
```

The selected scalar names and train-set mean/std are written to `data_metadata.json` and checkpoint `data_metadata`. Periodic eval and sampling derive runtime scalar values from the current validation/test ground-truth `pde_params`, standardize them with the checkpoint/train statistics, and pass them through `extra["scalar_conditioning"]`.

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

## Citation

If you use this codebase, please cite:

```bibtex
@misc{zhang2026guidedflowmatchingforward,
      title={Guided Flow Matching for Forward and Inverse PDE Problems with Sparse Observations: Algorithm and Theory},
      author={Xifeng Zhang and Jin Zhao},
      year={2026},
      eprint={2605.25509},
      archivePrefix={arXiv},
      primaryClass={stat.ML},
      url={https://arxiv.org/abs/2605.25509},
}
```
