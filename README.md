# FM4PDE

FM4PDE is a Flow Matching codebase for generating, completing, and inverting PDE solution fields with observation and PDE residual guidance.

## Entrypoints

```bash
python train.py
python -m sampling.runner --config configs/ablations/base/heat.yaml
PLAN_ONLY=true OUTPUT_DIR=outputs/MAIN1000 bash scripts/sample/run_sample_sweep.sh
PLAN_ONLY=true OUTPUT_DIR=outputs/MAIN1000 bash scripts/sample/run_sample_sweep_burger.sh
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

### Single run

The five main sampling configs are `poisson`, `helmholtz`, `darcy`,
`nsnonbounded`, and `burger` under `configs/main/`. Each config selects its
active training or test `data_path`; switch the active path in that YAML before
starting a sweep if a different split is required.

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
  --config configs/ablations/base/heat.yaml \
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
SENSOR_MODE_LIST="random sensor_column" \
DEVICE_LIST="cuda:0 cuda:1" \
RESUME=true \
AGGREGATE=true \
  bash scripts/sample/run_sample_sweep_burger.sh
```

Together, these stochastic-only commands create 12 non-Burgers PDE/task
experiments plus two Burgers sensor-mode experiments, or 14,000 sample results.
The default `SAMPLER_LIST` is
`"stochastic deterministic hybrid_s2d"`; omitting the stochastic-only setting
would produce 36,000 results in the first sweep and 6,000 in the Burgers sweep.

Important sweep controls are:

| Variable | Default | Meaning |
| --- | --- | --- |
| `NUM_SAMPLES` | `1000` | Samples per PDE × task × sampler × sensor-mode experiment |
| `MAX_BATCH_SIZE` | `50` | Maximum samples handled by one runner process; reduce this if GPU memory is insufficient |
| `PDE_LIST` | four non-Burgers PDEs | PDEs handled by the main sweep script |
| `TASK_LIST` | `forward inverse both` | Tasks applied to every PDE in `PDE_LIST` |
| `SENSOR_MODE_LIST` | config value | Space-separated sensor modes; an empty value preserves each PDE YAML setting |
| `PARALLEL` | `false` | Enable concurrent PDE/task groups |
| `MAX_PARALLEL_TASKS` | `2` | Maximum number of concurrent PDE/task runner processes |
| `DEVICE_LIST` | value of `DEVICE` | Space-separated devices assigned round-robin |
| `RESUME` | `true` | Reuse matching successful chunks and run only missing offsets |
| `PROGRESS_INTERVAL` | `1` | Live progress refresh interval in seconds |
| `VIS` | `false` | Save one comparison figure for each completed batch |
| `PLAN_ONLY` | `false` | Print completed samples and pending chunks without sampling |
| `AGGREGATE` | `true` | Aggregate successful metrics after the sweep |

For two GPUs, the example above runs at most one task per GPU. Setting
`MAX_PARALLEL_TASKS=8` with `DEVICE_LIST="cuda:0 cuda:1"` creates eight device
slots in round-robin order, so a fully occupied sweep runs at most four
independent processes on each GPU. Their model and batch memory are additive;
80 GB per A100 does not by itself guarantee that four processes will fit. Start
with two workers, then increase concurrency or `MAX_BATCH_SIZE` while monitoring
GPU memory.

The terminal displays one live row per PDE/task group. For example, the
`poisson / forward` row reports its current sampler, sensor mode,
completed/total samples, `offset`, batch size, sampling step, state, and current
coefficient/solution relative L2 values. An `OVERALL` row combines progress
across all groups. Child runner output is retained in per-chunk log files
instead of overwriting the live table. Sensor modes remain ordered within their
PDE/task group, so the Burgers-only sweep uses one worker even when two devices
are listed.

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

`RESUME=true` is enabled by default. A chunk is skipped only when its full
configuration matches and `resolved_config.yaml`, a successful
`metrics_final.json`, and `result.pt` are all present. If a run is interrupted,
rerun the same command: completed chunks are retained and the interrupted or
missing offsets are scheduled again. Configuration changes, including
`sensor_mode`, produce a different fingerprint and do not silently reuse
incompatible results.

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

### Ablation sweep

The configuration-grid ablation entrypoint remains available separately:

```bash
python -m sampling.sweep \
  --grid configs/ablations/all_internal_ablation_grid.yaml \
  --group time_dependent_residual_mode
```

`configs/ablations/all_internal_ablation_grid.yaml` includes `nsnonbounded` in the top-level `base_configs`, so NS participates in the same guidance, sensor, noise, zeta, time-grid, clipping, residual-region, and statistics ablations as the other PDEs. The time-dependent residual-mode group also includes NS with the other temporal endpoint PDEs.

Formal base configs set `allow_synthetic_data: false`; dry-run smoke configs may use synthetic data.

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
