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

Supported `boundary_condition_mode` values are `auto`, `dirichlet_zero`, `neumann_zero`, `periodic`, `mixed`, `none`, `wall`, `open`, and `legacy_ignore`. Sampling PDE loss accepts `initial_condition_mode: auto`, `none`, or `legacy_ignore`; `auto` resolves to no reference-field IC component. Modes that compare against an observed or true initial field are rejected because observations belong in observation loss, not PDE loss.

Current formal config BC/IC sources:

- Darcy, Poisson, Helmholtz: `dirichlet_zero` from config/static dataset convention; IC disabled.
- Heat: `periodic` from pair-HDF5 generator/config; IC is added only when an observed or true initial field is passed.
- Wave: `periodic` from generator/config; velocity boundary follows the state channels.
- Advection-Diffusion: `periodic` from generator/config.
- Reaction-Diffusion: `neumann_zero` from generator metadata/config.
- Shallow Water: `open` from Clawpack extrapolation boundary/config.
- Burgers: spatial periodic boundary on the BCHW time-space field; IC residual requires a known initial slice.
- Steady Heat Conduction: `mixed` from generator metadata/config; bottom Dirichlet `u_D`, other sides zero Neumann.
- Non-bounded Navier-Stokes: scalar-vorticity PDE residual is enabled for sampling guidance on the periodic torus `Omega=[0,1)^2`. Velocity is reconstructed from vorticity with the periodic FFT stream-function solve `-Delta psi=omega`, `v=(partial_y psi, -partial_x psi)`. `ns_operator_mode=generator_dealiased` (default) applies the generator's 2/3 Fourier projection to the nonlinear product and forcing; `continuous_spectral` is available as a diagnostic. NaN/Inf fails immediately instead of being clamped.

For Non-bounded Navier-Stokes, guidance uses:

```math
r = \partial_\tau \omega + v \cdot \nabla \omega - \nu \Delta \omega - f_{NS}.
```

Endpoint-only modes such as `hermite_bridge` and `endpoint_secant` are approximate and use only the model-predicted `a/q0` and `u/qT`. Burgers is the only current model whose output itself is a full time-space field, so it alone uses full-trajectory finite differences during guidance and evaluation.

Forward, inverse, and both tasks always pass model-predicted `coef/a/q0` and `sol/u/qT` as the direct PDE residual fields; none substitutes `ground_truth.coef`, `ground_truth.sol`, an observed initial field, or a stored trajectory. The sole field-data exception is `near_endpoint_temporal`, which may additionally read sparse true observations at `dt` and `T-dt` for the six supported temporal-endpoint PDEs. Scalar coefficients, time intervals, forcing definitions, and boundary-condition parameters may also come from dataset metadata because they define the PDE rather than supply a target solution field.

## PDE Residual Region

`pde_residual_region` masks only the `interior` component. `coef_obs`, `sol_obs`, and `active_obs_union` explicitly select the task-active observation side; invalid task/side combinations fail. Deprecated `observed` and `union_obs` warn and resolve to `active_obs_union`. Boundary, initial, and endpoint components are never intersected with these masks.

`near_endpoint_temporal` is a valid formal sampling guidance/evaluation mode only for Heat, Wave, Advection-Diffusion, Reaction-Diffusion, Shallow Water, and Non-bounded Navier-Stokes. The predicted `q0/qT` remain the differentiated fields. `q(dt)` reuses the coef/q0 mask and `q(T-dt)` reuses the sol/qT mask, so there is no independent near-endpoint budget or seed. Training's unconditional periodic evaluator rejects this mode.

## Artifacts

`near_endpoint_temporal` artifacts sanitize hidden frames. They keep:

- `q_dt_obs = q_dt * mask_0`
- `q_T_minus_dt_obs = q_T_minus_dt * mask_T`
- `mask_0`, `mask_T`, `dt`
- metadata identifying `near_endpoint_temporal_sparse_observations` as the ground-truth field exception and recording that no full near-endpoint frame was retained

Stored ground-truth trajectory fields are excluded from sampling PDE loss and generated-sample metrics. Burgers needs no auxiliary trajectory parameter: its model output tensor is already the full predicted time-space field.

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

For batched sampling, relative L2 errors are computed independently for each sample and then averaged. The per-sample values are retained in the matching `*_per_sample` fields. PDE residual norms use the mean of per-sample RMS values.

PDE residual operators also keep sample-level physical parameters independent. In particular, batched Burgers evaluation broadcasts each sample's own `nu`, `T`/`trajectory_dt`, and `domain_length`; it does not reuse the first batch element's values. Step metrics record `pde_coef_input_source=model_output` and `pde_sol_input_source=model_output`. Normally `pde_uses_ground_truth_fields=false`; `near_endpoint_temporal` records it as true together with `pde_uses_ground_truth_endpoint_fields=false`, `pde_ground_truth_field_exception=near_endpoint_temporal_sparse_observations`, and the sparse auxiliary input sources. Training-time periodic generated-sample evaluation records equivalent `eval_pde_*` provenance fields.

Evaluation losses are independent of guidance flags and zeta weights: `eval_L_pde` is always computed from the generated model outputs when the selected residual mode is evaluable, while `guidance_L_pde` is the separately gated optimization objective. Enabled but disconnected guidance gradients raise an error. Batch clipping, observation normalization, noise scaling, and reported relative errors are all per sample. Every run writes one compact row per sample to `metrics_per_sample.csv`; `metrics_final.json` contains means/counts and batch-level status only. Set `save_per_sample_curves=true` only when the larger `metrics_step_per_sample.csv` is needed.

For joint-PDE checkpoints, labels are checkpoint-local contiguous IDs and the null ID is `num_classes`. Sampling uses standard classifier-free guidance `v_uncond + cfg_scale * (v_cond - v_uncond)`; the unconditional branch drops only the PDE label and retains scalar conditioning. Old joint checkpoints without the mapping/category layer are rejected, while single-PDE checkpoints remain unconditional and unchanged.
