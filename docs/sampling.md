# FM4PDE Sampling

`python -m sampling.runner` is the only single-run sampling entrypoint. `python -m sampling.sweep` is the only grouped sweep entrypoint. The root `sample.py` delegates directly to `sampling.runner`.

## Config Schema

Sampling configs use the flat `AblationConfig` schema. Unknown YAML fields are rejected.

Required data fields for formal runs:

- `pde`
- `task`
- `data_paths` with `id`, `smooth`, and `rough` entries
- `test_type` selecting one entry; `data_path` stores the resolved path
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
`heavy` are architecture ablation profiles. Sampling requires a complete schema
3 checkpoint, reconstructs the model from `model_config`, then records the
runtime profile and checkpoint architecture metadata in `run_metadata.json`.

## Pair HDF5 Metadata

Heat, Wave, Advection-Diffusion, and Steady Heat Conduction use the endpoint pair HDF5 format exposed as `loadby: pair_h5`. The HDF5 dataset names remain `input_data` and `output_data`; those names are part of the disk format and do not imply any special code path.

`pair_h5` is not a storage subdirectory. Formal data files are expected directly under `DATA_ROOT/<pde>/`, for example `PDEdata/heat/heat_test_10000-128-128_id.h5`. Use `--override test_type=id|smooth|rough`, or set `TEST_TYPE` when calling `scripts/sample/run_sample.sh`, to select the test distribution.

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
- `endpoint_residual_enabled`
- `boundary_condition_type`
- `boundary_condition_source`
- `residual_channels`

Status is not the same as family. Static PDEs and Burgers are `reliable`; endpoint temporal residuals are `approximate` because the default endpoint/sparse temporal guidance approximates time dynamics.

The sampling observation loss is now a masked MSE over actual observation entries:

```math
L_{obs} = \frac{\sum_i M_i (x_i - y_i)^2}{\sum_i M_i}
```

When a single-channel mask is broadcast over multiple state channels, the denominator counts the expanded observed entries. Empty masks are clamped to avoid NaNs.

The sampling PDE loss is assembled from differentiable residual components. Each component is reduced with its own MSE denominator, then the component losses are weighted and summed:

```math
L_{PDE} = L_{int} + \lambda_{bc} L_{bc} + \lambda_{ep} L_{ep}
```

```math
L_{int} = \operatorname{mean}_{j \in C_{int}} |R_{int,j}|^2
```
```math
L_{bc} = \operatorname{mean}_{j \in C_{bc}} |R_{bc,j}|^2
```
```math
L_{ep} = \operatorname{mean}_{j \in C_{ep}} |R_{ep,j}|^2
```

Different components are not concatenated and globally averaged for loss computation. They are logged as a concatenated residual field for diagnostics, but separate MSE denominators avoid changing lambda semantics when components have different point counts or channel counts.

Formal configs use explicit BC residual channels where applicable. Periodic
endpoint-false grids are enforced by periodic discrete operators instead of
first/last value residual channels.

New config fields:

- `enforce_boundary_conditions: true`
- `boundary_condition_mode: auto`
- `bc_weight: 1.0`
- `endpoint_bc_weight: 1.0`
- `boundary_residual_normalization: sqrt_grid_over_mask`
- `allow_unknown_boundary_conditions: false`

Supported `boundary_condition_mode` values are `auto`, `dirichlet_zero`,
`neumann_zero`, `periodic`, `mixed`, `none`, `wall`, and `open`. The predicted
initial field is an input to the PDE operator; there is no separate
ground-truth initial-condition loss.

Current formal boundary-condition sources:

- Darcy, Poisson, Helmholtz: `dirichlet_zero` from config/static dataset convention.
- Heat: `periodic` from pair-HDF5 generator/config.
- Wave: `periodic` from generator/config; velocity boundary follows the state channels.
- Advection-Diffusion: `periodic` from generator/config.
- Reaction-Diffusion: `neumann_zero` from generator metadata/config.
- Shallow Water: `open` from Clawpack extrapolation boundary/config.
- Burgers: spatial periodic boundary on the BCHW time-space field.
- Steady Heat Conduction: `mixed` from generator metadata/config; bottom Dirichlet `u_D`, other sides zero Neumann.
- Non-bounded Navier-Stokes: scalar-vorticity PDE residual is enabled for sampling guidance on the periodic torus `Omega=[0,1)^2`. Velocity is reconstructed from vorticity with the periodic FFT stream-function solve `-Delta psi=omega`, `v=(partial_y psi, -partial_x psi)`. `ns_operator_mode=generator_dealiased` (default) applies the generator's 2/3 Fourier projection to the nonlinear product and forcing; `continuous_spectral` is available as a diagnostic. NaN/Inf fails immediately instead of being clamped.

For Non-bounded Navier-Stokes, guidance uses:

```math
r = \partial_\tau \omega + v \cdot \nabla \omega - \nu \Delta \omega - f_{NS}.
```

Endpoint-only modes such as `hermite_bridge` and `endpoint_secant` are approximate and use only the model-predicted `a/q0` and `u/qT`. Burgers is the only current model whose output itself is a full time-space field, so it alone uses full-trajectory finite differences during guidance and evaluation.

Forward, inverse, and both tasks always pass model-predicted `coef/a/q0` and `sol/u/qT` as the direct PDE residual fields; none substitutes target endpoint fields or a stored trajectory. The sole field-data exception is `near_endpoint_temporal`, which may additionally read sparse true observations at `dt` and `T-dt` for the six supported temporal-endpoint PDEs. Scalar coefficients, time intervals, forcing definitions, and boundary-condition parameters may also come from dataset metadata because they define the PDE rather than supply a target solution field.

## PDE Residual Region

`pde_residual_region` masks only the `interior` component. `coef_obs`, `sol_obs`, and `active_obs_union` explicitly select the task-active observation side; invalid task/side combinations fail. Boundary and endpoint components are never intersected with these masks.

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
  --config configs/main/both/heat.yaml \
  --override residual_mode=hermite_bridge \
  --override output_dir=outputs/ablations
```

List a sweep:

```bash
python -m sampling.sweep \
  --grid configs/ablations/all_internal_ablation_grid.yaml \
  --pde poisson \
  --group guidance_components \
  --list
```

Run selected PDEs and one sweep group with shared runtime overrides:

```bash
python -m sampling.sweep \
  --grid configs/ablations/all_internal_ablation_grid.yaml \
  --pde heat \
  --pde wave \
  --group temporal_residual_mode \
  --override output_dir=outputs/ablations \
  --override device=cuda

PDE_LIST="poisson heat" \
PLAN_ONLY=true \
  bash scripts/run_ablations.sh guidance_components time_grid_by_sampler
```

`--pde`, `--group`, and `--override key=value` may each be repeated. Unknown
PDEs/groups and filters that select no jobs fail before sampling. The formal
grid has 1,041 jobs across 11 PDEs; the Poisson-focused grid has 111 jobs.

Formal experiment groups and per-PDE variant counts are:

| group | variants per applicable PDE |
| --- | ---: |
| `guidance_components` | 12 |
| `loss_state_by_sampler` | 6 |
| `sampler_phase` | 8 |
| `time_grid_by_sampler` | 12 |
| `num_steps_by_sampler` | 28 |
| `step_method_by_sampler` | 8 |
| `sensor_sparsity` | 5 |
| `sensor_mode` | 5 |
| `noise_robustness` | 4 |
| `temporal_residual_mode` | 3, on six temporal endpoint PDEs |
| `statistics_stability` | 5 |

The three time-discretization groups vary one factor at a time. Their hybrid
samplers use `switch_ratio=0.5`; the dedicated `sampler_phase` group tests both
hybrid directions at `0.2`, `0.5`, and `0.8`. Temporal residual comparison uses
`endpoint_secant`, `hermite_bridge`, and `near_endpoint_temporal` with a shared
500-observation budget. `auto` is omitted because it resolves to Hermite, while
full-trajectory modes are not valid endpoint-model approximations.

Formal sweep grids use `configs/ablations/formal_suite.yaml`. The suite selects
`configs/main/<task>/<pde>.yaml` after each task matrix value is expanded, so
task-specific zeta and other tuned defaults always come from the matching main
config. Burgers explicitly falls back to its sole `both` main config. The suite
then applies only formal-run overrides such as held-out test paths and
`allow_synthetic_data=false`; group/matrix, conditional, and command-line
overrides are applied afterward in that order.

Aggregate:

```bash
python -m sampling.aggregate outputs/ablations --output-dir outputs/ablations
```

Aggregation writes `ablation_report_metrics.csv` as the stable ablation-report
view.  It contains one selected row per `(pde, task, ablation_name)` and always
reports `rel_l2_a`, `rel_l2_u`, `L_obs_a`, `L_obs_u`, and `L_pde` as separate
columns.  `summary_latest_run_seed_grouped.csv` reports the mean, standard
deviation, standard error, 95% interval half-width, median, p90, minimum, and
maximum for the same five metrics.  A task-specific or balanced score may be
used for ranking, but it does not replace these component metrics in reports.

Each run writes `resolved_config.yaml`, `run_metadata.json`, `metrics_step.jsonl`, `metrics_final.json`, `curves.csv`, `summary.csv`, `result.pt`, and `masks.pt`.

For batched sampling, relative L2 errors are computed independently for each sample and then averaged. The per-sample values are retained in the matching `*_per_sample` fields. PDE residual norms use the mean of per-sample RMS values.

PDE residual operators also keep sample-level physical parameters independent. In particular, batched Burgers evaluation broadcasts each sample's own `nu`, `T`/`trajectory_dt`, and `domain_length`; it does not reuse the first batch element's values. Step metrics record `pde_coef_input_source=model_output` and `pde_sol_input_source=model_output`. Normally `pde_uses_ground_truth_fields=false`; `near_endpoint_temporal` records it as true together with `pde_uses_ground_truth_endpoint_fields=false`, `pde_ground_truth_field_exception=near_endpoint_temporal_sparse_observations`, and the sparse auxiliary input sources. Training-time periodic generated-sample evaluation records equivalent `eval_pde_*` provenance fields.

Evaluation losses are independent of guidance flags and zeta weights: `eval_L_pde` is always computed from the generated model outputs when the selected residual mode is evaluable, while `guidance_L_pde` is the separately gated optimization objective. Enabled but disconnected guidance gradients raise an error. Batch clipping, observation normalization, noise scaling, and reported relative errors are all per sample. Every run writes one compact row per sample to `metrics_per_sample.csv`; `metrics_final.json` contains means/counts and batch-level status only. Set `save_per_sample_curves=true` only when the larger `metrics_step_per_sample.csv` is needed.

For joint-PDE checkpoints, labels are checkpoint-local contiguous IDs and the null ID is `num_classes`. Sampling uses standard classifier-free guidance `v_uncond + cfg_scale * (v_cond - v_uncond)`; the unconditional branch drops only the PDE label and retains scalar conditioning. Old joint checkpoints without the mapping/category layer are rejected, while single-PDE checkpoints remain unconditional and unchanged.
