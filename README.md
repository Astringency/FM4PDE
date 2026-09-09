# FM4PDE

FM4PDE is a Flow Matching codebase for generating, completing, and inverting PDE solution fields with observation and PDE residual guidance.

## JMLR revision experiments

The [September 2026 reproduction guide](reproducibility/revision_20260909/README.md)
indexes the main and ablation experiments, frozen source versions, environments,
and consolidated results under `outputs/main/` and `outputs/ablations/`.
Use the recorded study configuration and model for reproducing a published row.

## Entrypoints

```bash
python train.py
python -m sampling.runner --config configs/main/both/heat.yaml
PLAN_ONLY=true OUTPUT_DIR=outputs/MAIN1000 bash scripts/sample/run_sample_sweep.sh
PLAN_ONLY=true OUTPUT_DIR=outputs/MAIN1000 bash scripts/sample/run_sample_sweep_burger.sh
python -m sampling.sweep --grid configs/ablations/all_internal_ablation_grid.yaml --list
PDE_LIST="poisson heat" PLAN_ONLY=true bash scripts/run_ablations.sh guidance_components
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

`pair_h5` is a file-format/loadby name, not a directory name. Formal data paths use one PDE-named directory directly under the data root, for example `PDEdata/heat/heat_test_10000-128-128_id.h5`. Each main PDE config provides `data_paths.id`, `data_paths.smooth`, and `data_paths.rough`; select one with `test_type` or `TEST_TYPE` in `scripts/sample/run_sample.sh`.

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
Recommended configs move attention away from downsample factor `2` to
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
`checkpoint_schema_version`. Sampling requires schema 3 checkpoints with
`model_config`, `model_config_metadata`, a plain model state, and a saved
normalizer.

See `docs/model_architectures.md` for details.

## Sampling

Configs use flat `AblationConfig` YAML. Unknown fields are rejected.

### Single run

Main sampling configs are task-specific: `configs/main/{both,forward,inverse}/`
contains one YAML per supported PDE/task combination. The ten non-Burgers PDEs
support all three tasks; Burger is intentionally available only as
`configs/main/both/burger.yaml`. Each config selects its active training or test
`data_path`; switch the active path in that YAML before starting a sweep if a
different split is required.

Run one PDE/task/batch through the shell entrypoint:

```bash
PDE=poisson \
TASK=forward \
SAMPLER_PHASE=stochastic \
BATCH_SIZE=10 \
OFFSET=0 \
DEVICE=cuda:0 \
OUTPUT_DIR=outputs/MAIN1000 \
  bash scripts/sample/run_sample.sh
```

Or call the Python runner directly:

```bash
python -m sampling.runner \
  --config configs/main/both/heat.yaml \
  --override residual_mode=hermite_bridge
```

`sample.py` delegates to the same runner. Set `VIS=true` for the shell
entrypoint, or pass `--vis` to the Python runner, to save a figure for the
completed batch.

### Resumable 1000-sample sweep

Use two sweeps when the four elliptic/NS PDEs need all three tasks but Burgers
needs only `both`. The main script defaults to `poisson`, `helmholtz`, `darcy`,
and `nsnonbounded` with `forward`, `inverse`, and `both`:

```bash
OUTPUT_DIR=outputs/MAIN1000 \
NUM_SAMPLES=1000 \
PDE_LIST="poisson helmholtz darcy nsnonbounded" \
TASK_LIST="forward inverse both" \
SAMPLER_LIST="stochastic" \
PARALLEL=true \
MAX_PARALLEL_TASKS=2 \
DEVICE_LIST="cuda:0 cuda:1" \
RESUME=true \
AGGREGATE=false \
  bash scripts/sample/run_sample_sweep.sh
```

Run the dedicated Burgers wrapper afterward. It fixes the selection to
`burger / both` and defaults to both `random` and `sensor_column` observation
modes:

```bash
OUTPUT_DIR=outputs/MAIN1000 \
NUM_SAMPLES=1000 \
SAMPLER_LIST="stochastic" \
BURGER_SENSOR_MODE_LIST="random sensor_column" \
PARALLEL=true \
MAX_PARALLEL_TASKS=2 \
DEVICE_LIST="cuda:0 cuda:1" \
RESUME=true \
AGGREGATE=true \
  bash scripts/sample/run_sample_sweep_burger.sh
```

For full observations on the four non-Burgers `MAIN1000_100_TEST` equations
across `id`, `rough`, and `smooth`, use the dedicated wrapper.  At the current
`128 x 128` resolution it sets `NUM_OBS=16384`, runs all three tasks, excludes
Burgers, and writes `outputs/main/MAIN1000_100_TEST_FULL_<test_type>`:

```bash
bash scripts/sample/run_sample_sweep_full.sh
```

Preview all three plans without sampling with
`PLAN_ONLY=true bash scripts/sample/run_sample_sweep_full.sh`.

Together, these stochastic-only commands create 12 non-Burgers PDE/task
experiments plus two Burgers sensor-mode experiments, or 14,000 sample results.
The current default `SAMPLER_LIST` is `stochastic`, so these are also the
default experiment counts.

Important sweep controls are:

| Variable | Default | Meaning |
| --- | --- | --- |
| `NUM_SAMPLES` | `1000` | Samples per PDE × task × sampler × sensor-mode experiment |
| `MAX_BATCH_SIZE` | `50` | Maximum samples handled by one runner process; reduce this if GPU memory is insufficient |
| `PDE_LIST` | four non-Burgers PDEs | PDEs handled by the main sweep script |
| `TASK_LIST` | `forward inverse both` | Tasks applied to every PDE in `PDE_LIST` |
| `SENSOR_MODE_LIST` | `random` | Space-separated sensor modes for the main sweep |
| `BURGER_SENSOR_MODE_LIST` | `random sensor_column` | Modes forced by the Burgers wrapper, independent of an inherited `SENSOR_MODE_LIST` |
| `PARALLEL` | `true` | Enable concurrent PDE/task/sensor-mode groups |
| `MAX_PARALLEL_TASKS` | `2` | Maximum concurrent PDE/task/sensor-mode runner processes |
| `DEVICE_LIST` | value of `DEVICE` | Space-separated devices assigned round-robin |
| `OUTPUT_DIR` | `outputs/MAIN1000_100` | Artifact and recovery-state root |
| `RESUME` | `true` | Reuse matching successful chunks and run only missing offsets |
| `PROGRESS_INTERVAL` | `1` | Live progress refresh interval in seconds |
| `VIS` | `true` | Save one comparison figure for each completed batch |
| `PLAN_ONLY` | `false` | Print completed samples and pending chunks without sampling |
| `AGGREGATE` | `true` | Aggregate successful metrics after the sweep |

For two GPUs, the example above runs at most one task per GPU. Setting
`MAX_PARALLEL_TASKS=8` with `DEVICE_LIST="cuda:0 cuda:1"` creates eight device
slots in round-robin order, so a fully occupied sweep runs at most four
independent processes on each GPU. Their model and batch memory are additive;
80 GB per A100 does not by itself guarantee that four processes will fit. Start
with two workers, then increase concurrency or `MAX_BATCH_SIZE` while monitoring
GPU memory.

The terminal displays one live row per PDE/task/sensor-mode group. For example, the
`poisson / forward` row reports its current sampler, sensor mode,
completed/total samples, `offset`, batch size, sampling step, state, and current
coefficient/solution relative L2 values. An `OVERALL` row combines progress
across all groups. Child runner output is retained in per-chunk log files
instead of overwriting the live table. With `PARALLEL=true`, two Burgers sensor
modes, `MAX_PARALLEL_TASKS=2`, and two devices, `random` runs on `cuda:0` while
`sensor_column` runs concurrently on `cuda:1`.

Preview the exact work plan and any samples recognized for recovery without
starting a model:

```bash
OUTPUT_DIR=outputs/MAIN1000 \
NUM_SAMPLES=1000 \
PLAN_ONLY=true \
  bash scripts/sample/run_sample_sweep.sh

OUTPUT_DIR=outputs/MAIN1000 \
NUM_SAMPLES=1000 \
PLAN_ONLY=true \
  bash scripts/sample/run_sample_sweep_burger.sh
```

### Burgers parameter tuning and inverse debug

The staged Burgers tuner evaluates zeta and clipping on disjoint offset blocks,
supports the `random` and `sensor_column` layouts, and resumes completed jobs:

```bash
STAGE=screen DEVICE_LIST="cuda:0 cuda:1" \
  bash scripts/tuning/run_burger_tuning.sh
STAGE=validate ZETA_OBS_U_LIST=409600 ZETA_PDE_LIST=10 \
  CLIP_THRESHOLD_LIST=50 DEVICE_LIST="cuda:0 cuda:1" \
  bash scripts/tuning/run_burger_tuning.sh
STAGE=samplers ZETA_OBS_U_LIST=409600 ZETA_PDE_LIST=10 \
  CLIP_THRESHOLD_LIST=50 DEVICE_LIST="cuda:0 cuda:1" \
  bash scripts/tuning/run_burger_tuning.sh
```

Set `CACHE_DIR` when data and checkpoints are on a slow remote mount. The
script copies complete inputs through a `.partial` file before starting jobs.
Use `PLAN_ONLY=true` to inspect the full experiment cross-product.

The ten non-Burgers main equations have a separate inverse/random debug pass:

```bash
DEVICE_LIST="cuda:0 cuda:1" \
  bash scripts/tuning/run_inverse_debug.sh
```

After the one-configuration debug pass, run the result-driven inverse grids in
two stages. Both stages keep stochastic sampling at 100 steps and evaluate
each parameter set on two disjoint offset blocks:

```bash
STAGE=stabilize DEVICE_LIST="cuda:0 cuda:1" \
  bash scripts/tuning/run_inverse_tuning.sh
STAGE=refine DEVICE_LIST="cuda:0 cuda:1" \
  bash scripts/tuning/run_inverse_tuning.sh
```

`stabilize` runs 108 jobs for the six equations that produced NaN/Inf or large
finite divergence. `refine` runs 96 jobs for Darcy, Poisson, Helmholtz, and
non-bounded Navier-Stokes. Each stage writes `selected_params.csv`, requiring
all expected offsets to complete before ranking a configuration by coefficient
relative-L2 mean, P90, and maximum. Use `PLAN_ONLY=true` to review the grids.

The scripts accept `PDE_DATA_ROOT`, `OUTPUT_DIR`, `NUM_STEPS`, `BATCH_SIZE`,
`RESUME`, and `AGGREGATE` overrides. See `docs/burger_tuning_results.md` for the
local Burger tuning evidence and `docs/inverse_tuning_results.md` for the first
inverse baseline and second-round grid rationale.

`RESUME=true` is enabled by default. Each tuning job identifier includes the
observation layout, sampler, zeta values, clipping, steps, offset, batch, and
seeds. Its `.complete` marker is written only after a successful
`metrics_final.json` and the matching `result.pt` are found. If a run is
interrupted, rerun the same command: completed jobs are retained and incomplete
jobs are scheduled again.

Artifacts preserve the existing MAIN1000-style layout:

```text
outputs/MAIN1000/<pde>/<task>/<ablation_name>/<timestamp>/
├── resolved_config.yaml
├── metrics_final.json
├── metrics_per_sample.csv
├── result.pt
└── figures/                         # present when VIS=true

outputs/MAIN1000/.sample_sweeps/<configuration-fingerprint>/
├── manifest.json
├── completed/                       # validated chunk completion markers
├── progress/                        # live runner progress files
├── logs/                            # per-chunk stdout/stderr
└── sweep.log
```

After a successful sweep, aggregation writes files such as
`summary_all_raw.csv`, `summary_all_grouped.csv`,
`metrics_per_sample_all.csv`, and `curves_grouped.csv` under `OUTPUT_DIR`.
For ablation reporting, `ablation_report_metrics.csv` is the compact
one-row-per-selected-run view and keeps `rel_l2_a`, `rel_l2_u`, `L_obs_a`,
`L_obs_u`, and `L_pde` in separate columns.

Create the concise Excel summaries under `outputs/summary` with:

```bash
python scripts/summary.py                  # main and ablations
python scripts/summary.py --exp main       # main only
python scripts/summary.py --exp ablations  # ablations only
```

### Ablation sweep

The formal ablation grid contains 1,060 jobs across all 11 PDEs. Select PDEs
and experiment groups either through the Python entrypoint or the shell wrapper:

```bash
python -m sampling.sweep \
  --grid configs/ablations/all_internal_ablation_grid.yaml \
  --pde heat \
  --group temporal_residual_mode \
  --list

PDE_LIST="poisson heat" \
OUTPUT_DIR=outputs/ablations \
DEVICE=cuda \
PLAN_ONLY=true \
  bash scripts/run_ablations.sh guidance_components time_grid_by_sampler
```

The formal groups are `guidance_components`, `loss_state_by_sampler`,
`sampler_phase`, `time_grid_by_sampler`, `num_steps_by_sampler`,
`step_method_by_sampler`, `sensor_sparsity`, `sensor_mode`,
`noise_robustness`, `deterministic_endpoint_bt`, `temporal_residual_mode`, and
`statistics_stability`. The `deterministic_endpoint_bt` group is initially
scoped to Poisson so its conclusions can be validated before transfer. The
remaining non-temporal groups expand over all 11 PDEs. Temporal residual
comparison is scoped to Heat, Wave, Advection-Diffusion, Reaction-Diffusion,
Shallow Water, and Non-bounded Navier-Stokes.

Time grid, step count, and integration method are deliberately independent
ablations rather than one large Cartesian product. The Poisson-focused grid in
`configs/ablations/all_ablation_grid.yaml` contains 138 jobs, including the 18
temporal-PDE residual jobs. `endpoint_secant`, `hermite_bridge`, and
`near_endpoint_temporal` all use the same 500-point endpoint observation budget.
The near-endpoint mode additionally requires saved near-endpoint frames or a
full trajectory in the formal dataset.

The formal ablation suite sets `allow_synthetic_data: false`; dry-run smoke configs may use synthetic data.

### Safeguarded deterministic sampling

CondOT uses `b_t = (1 - t) / t`, so the historical deterministic update
`delta_t * b_t * grad(L)` is singular at `t=0`.  The sampler now supports two
endpoint predictors and four independent stability controls:

- `deterministic_endpoint_mode=single_step` uses the paper's
  `x_t + (1 - t) v_t(x_t)` endpoint predictor.
- `deterministic_endpoint_mode=rollout` differentiably integrates the
  unguided ODE from the current time to 1.  It is substantially more expensive;
  keep `deterministic_rollout_checkpoint=true` for memory control.
- `deterministic_bt_mode=clipped_zero_at_t0` removes the singular first update
  and caps the scalar `delta_t * b_t` multiplier.
- `deterministic_guidance_start_ratio` and
  `deterministic_guidance_ramp_ratio` delay and smoothly enable all guidance.
- `deterministic_correction_max_rms` clips the actual per-sample state
  correction after weighting, making the trust region independent of image
  resolution and the PDE-specific zeta scale.  A value of 0 disables it.
- `deterministic_numerical_guard=true` rejects a sample update containing NaN
  or infinity instead of contaminating the remaining trajectory.

Run the compact stochastic/single-step/rollout comparison on a server with:

```bash
bash scripts/run_deterministic_sampling.sh

PDE_LIST="poisson darcy heat" BATCH_SIZE=4 \
  bash scripts/run_deterministic_sampling.sh

# List jobs or omit the expensive rollout branch.
PLAN_ONLY=true bash scripts/run_deterministic_sampling.sh
INCLUDE_ROLLOUT=false bash scripts/run_deterministic_sampling.sh
```

The Poisson pilot uses `bt_max_scale=0.0125`, an initial 0.02 no-guidance
window, a 0.04 ramp, `correction_max_rms=0.02`, and `zeta_pde=30` for all three
Poisson comparison branches.  Other PDEs inherit their own observation/PDE
weights from `configs/main/both`; the same trust-region settings are
intentionally retained to test transfer rather than silently retuning each
equation.

### Result visualization

Sampling visualization now uses four columns:

```text
Ground Truth | Sparse Observations | Prediction | Difference
```

Unobserved locations in the sparse-observation column have a white background;
observed locations use the same color limits as ground truth and prediction.
Coefficient and solution are shown as separate rows, except for single-field
PDEs such as Burgers. For a batched `result.pt`, the plotting function accepts
`sample_index` (default `0`):

```python
from plot.plot import plot_from_result_pt

plot_from_result_pt(
    "outputs/MAIN1000/poisson/forward/example/result.pt",
    "outputs/MAIN1000/poisson/forward/example/figures/sample_7.png",
    sample_index=7,
)
```

See `docs/config_reference.md` for the full sampling variable and output-file
reference.

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

Training files can optionally be selected explicitly with a YAML manifest. Relative
file names are resolved against `--data_path`; absolute file names are used directly.
One manifest can contain entries for every PDE:

```yaml
train_files:
  heat:
    - heat/heat_10000-128-128_0.h5
    - heat/heat_10000-128-128_1.h5
  poisson:
    - poisson/poisson_10000-128-128_1.mat
```

```bash
python train.py \
  --dataset heat \
  --data_path /large_storage/zhangxf/PDEdata/ \
  --train_data_config configs/training_data.yaml
```

For a single-PDE run, `train_files` may be a list directly. The manifest file order
is preserved, `--max_train_samples` caps the combined files, and `--data_size` is
ignored because the file set is explicit. If `--train_data_config` is omitted,
training keeps the existing automatic discovery and `--data_size` behavior.

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
