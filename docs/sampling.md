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
- `interior_residual_enabled`
- `bc_residual_enabled`
- `ic_residual_enabled`
- `endpoint_residual_enabled`
- `boundary_condition_type`
- `boundary_condition_source`
- `initial_condition_type`
- `initial_condition_source`
- `legacy_ignore_boundary`
- `residual_channels`

Status is not the same as family. Static PDEs and Burgers are `reliable`; endpoint temporal residuals are `approximate` because the default endpoint/sparse temporal guidance approximates time dynamics.

The sampling observation loss is now a masked MSE over actual observation entries:

```math
L_{obs} = \frac{\sum_i M_i (x_i - y_i)^2}{\sum_i M_i}
```

When a single-channel mask is broadcast over multiple state channels, the denominator counts the expanded observed entries. Empty masks are clamped to avoid NaNs.

The sampling PDE loss is assembled from differentiable residual components. Each component is reduced with its own MSE denominator, then the component losses are weighted and summed:

```math
L_{PDE} = L_{int} + \lambda_{bc} L_{bc} + \lambda_{ic} L_{ic} + \lambda_{ep} L_{ep}
```

```math
L_{int} = \operatorname{mean}_{j \in C_{int}} |R_{int,j}|^2
```
```math
L_{bc} = \operatorname{mean}_{j \in C_{bc}} |R_{bc,j}|^2
```
```math
L_{ic} = \operatorname{mean}_{j \in C_{ic}} |R_{ic,j}|^2
```
```math
L_{ep} = \operatorname{mean}_{j \in C_{ep}} |R_{ep,j}|^2
```

Different components are not concatenated and globally averaged for loss computation. They are logged as a concatenated residual field for diagnostics, but separate MSE denominators avoid changing lambda semantics when components have different point counts or channel counts.

The old boundary handling zeroed boundary entries in the PDE residual. That excluded boundary points from the interior equation but did not enforce BCs. Formal configs now use explicit BC residual channels where applicable. Periodic endpoint-false grids are enforced by periodic discrete operators instead of first/last value residual channels. The legacy behavior is available only with `legacy_ignore_boundary: true` or `boundary_condition_mode: legacy_ignore`.

New config fields:

- `enforce_boundary_conditions: true`
- `enforce_initial_conditions: true`
- `boundary_condition_mode: auto`
- `initial_condition_mode: auto`
- `bc_weight: 1.0`
- `ic_weight: 1.0`
- `endpoint_bc_weight: 1.0`
- `boundary_residual_normalization: sqrt_grid_over_mask`
- `allow_unknown_boundary_conditions: false`
- `legacy_ignore_boundary: false`

Supported `boundary_condition_mode` values are `auto`, `dirichlet_zero`, `neumann_zero`, `periodic`, `mixed`, `none`, `wall`, `open`, and `legacy_ignore`. Supported `initial_condition_mode` values are `auto`, `endpoint_initial`, `observed_initial`, `trajectory_initial`, `none`, and `legacy_ignore`.

Current formal config BC/IC sources:

- Darcy, Poisson, Helmholtz: `dirichlet_zero` from config/static dataset convention; IC disabled.
- Heat: `periodic` from pair-HDF5 generator/config; IC is added only when an observed or true initial field is passed.
- Wave: `periodic` from generator/config; velocity boundary follows the state channels.
- Advection-Diffusion: `periodic` from generator/config.
- Reaction-Diffusion: `neumann_zero` from generator metadata/config.
- Shallow Water: `open` from Clawpack extrapolation boundary/config.
- Burgers: spatial periodic boundary on the BCHW time-space field; IC residual requires a known initial slice.
- Steady Heat Conduction: `mixed` from generator metadata/config; bottom Dirichlet `u_D`, other sides zero Neumann.
- Non-bounded Navier-Stokes: scalar-vorticity PDE residual is enabled for sampling guidance on the periodic torus `Omega=[0,1)^2`. Velocity is reconstructed from vorticity with the periodic FFT stream-function solve `-Delta psi=omega`, `v=(partial_y psi, -partial_x psi)`. The default fixed forcing is `f_NS(x,y)=0.1(sin(2*pi*(x+y))+cos(2*pi*(x+y)))` on endpoint-false grid points `x_i=i/W`, `y_j=j/H`; explicit `forcing` overrides it.

For Non-bounded Navier-Stokes, guidance uses:

```math
r = \partial_\tau \omega + v \cdot \nabla \omega - \nu \Delta \omega - f_{NS}.
```

Endpoint-only modes such as `hermite_bridge` and `endpoint_secant` are approximate. `full_trajectory_fd` uses finite differences when an explicit full trajectory is available.

Endpoint-only temporal PDEs do not fabricate an IC residual from `q0-q0`. IC residuals are added from masked `observed_initial` only when coefficient/initial observations are active, or from explicitly supplied `true_initial` extra conditions. Inverse, `pde_only`, and unconditional runs do not inject initial observations through the PDE residual.

## PDE Residual Region

`pde_residual_region` now masks only the `interior` PDE residual component. Boundary, initial, and endpoint components are appended after the interior mask and are never intersected with observation masks or boundary-exclusion masks. This prevents `boundary_excluded`, `observed`, and `union_obs` from accidentally deleting explicit BC/IC/endpoint residuals.

For `near_endpoint_temporal`, the interior residual is already sparse-temporal masked by `mask_0` and `mask_T`; `pde_residual_region` is skipped for that mode and metadata records:

```yaml
pde_residual_region_applied_to: interior_only
pde_residual_region_skipped: true
reason: near_endpoint_temporal interior is already sparse-temporal masked
```

## Artifacts

`result.pt` stores PDE params after recursive CPU sanitization. For `near_endpoint_temporal`, full hidden near-endpoint frames are not saved. The artifact keeps:

- `q_dt_obs = q_dt * mask_0`
- `q_T_minus_dt_obs = q_T_minus_dt * mask_T`
- `mask_0`, `mask_T`, `dt`
- metadata with `full_near_endpoint_frames_saved: false`

Full trajectory fields are omitted unless `residual_mode: full_trajectory_fd` and `save_intermediate: true`; metadata records the omission so large hidden trajectories are not silently persisted.

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
