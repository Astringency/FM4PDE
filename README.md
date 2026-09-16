# FM4PDE

Flow matching for joint physical-field generation and conditional reconstruction.
The same trained network supports forward, inverse and joint tasks for each PDE.

## Setup and assets

Use the supplied `environment.yml` (environment name `fm4pde`). Training and
sampling require the datasets and trained weights; these are external to the code.

```bash
export DATA_ROOT=/path/to/PDEdata
export CHECKPOINT_ROOT=/path/to/pretrained
python -m sampling.validate_configs
```

`DATA_ROOT` replaces `datasets/` in sampling configurations. `CHECKPOINT_ROOT`
replaces `outputs/pretrained/` while retaining the remaining subdirectories.
Alternatively, set a complete checkpoint filename per equation, for example
`CHECKPOINT_NSNONBOUNDED=/path/to/fm4nsnonbounded.pth`.
Explicit `--override checkpoint_path=...` takes precedence. Add `--check-assets`
to the validator to check that all referenced data and weight files are installed.

## Training

```bash
PDE=poisson bash scripts/training/run_train.sh
PDE=nsnonbounded NPROC_PER_NODE=2 bash scripts/training/run_train.sh
PDE=poisson RESUME=/path/to/checkpoint.pth bash scripts/training/run_train.sh
```

Omit `PDE` to train all eleven equations, or provide `PDE_LIST`.
`configs/training_data.yaml` lists five training shards per equation. The launcher
supports gradient accumulation, distributed training and checkpoint resume.
`DRY_RUN=true` prints commands. `PYTHON_BIN` selects the Python executable.
Navier–Stokes uses the **44,121,218-parameter light model**, including its ablations.
Export an inference checkpoint with `python scripts/training/export_checkpoint.py SOURCE OUTPUT`.

## Main sampling

```bash
PLAN_ONLY=true bash scripts/sampling/main/run.sh
DEVICE_LIST="cuda:0 cuda:1" bash scripts/sampling/main/run.sh
PDE=poisson TASK=both TEST_TYPE=id bash scripts/sampling/main/run_sample.sh
```

The main launcher covers full-field forward/inverse and three sparse tasks for
Poisson, Helmholtz, Darcy and Navier–Stokes, plus the two Burgers layouts, on
ID/Smooth/Rough. It uses 1,000 inputs and 100 steps per comparison by default.
The default output is `outputs/main`; `OUTPUT_ROOT` selects a new root. Existing
successful sampler results can be resumed. `run_sample_sweep.sh` also accepts
`PDE_LIST`, `TASK_LIST`, `NUM_SAMPLES`, `MAX_BATCH_SIZE`, and `PLAN_ONLY=true`.

The paper's aligned main results use saved input and observation tensors. Replay
those exact inputs with the retained inference implementation:

```bash
STUDY_ROOT=/path/to/baseline_pairing_20260915_57k \
CELL=supervised/poisson/id/sparse_joint \
OUTPUT_ROOT=outputs/reproductions/aligned bash scripts/sampling/main/run_matched_cell.sh
```

This command runs the batch-consistency check and then the selected cell. It
requires the saved protocol, inputs, masks and weights. It preserves completed
results. `run_matched.sh --help` exposes individual pilot/production controls.
Fresh `run.sh` sampling generates observations from the supplied configurations;
use the matched entry point when reproducing the saved aligned comparisons.

## Ablations and repeated sampling

```bash
PLAN_ONLY=true bash scripts/sampling/ablations/run.sh
PDE_LIST="poisson nsnonbounded" bash scripts/sampling/ablations/run.sh sampler_phase
bash scripts/sampling/ablations/run_study.sh averaging --help
bash scripts/sampling/ablations/run_traces.sh --help
```

`configs/ablations/paper.yaml` contains the retained factor sweeps. The separate
`configs/ablations/base/<task>/<pde>.yaml` profiles preserve the paper's ablation
weights and gradient limits; main and ablation settings need not coincide.
All ablation datasets default to ID. Synthetic configurations live only in test fixtures.

| Study | Entry |
| --- | --- |
| Guidance, loss state, D/S and switches, steps, density, noise, temporal residual | `scripts/sampling/ablations/run.sh` |
| 32 inputs × 3 draws | `run_study.sh ensemble` |
| Repeated guidance trajectories | `run_study.sh guidance` |
| Poisson conditional averaging | `run_study.sh averaging` |
| Four observation layouts, each with shared/separate locations | `run_study.sh layouts` |
| Physical-guidance weights, including NS | `run_study.sh weights` |
| Unconditional examples | `run_study.sh prior` |
| FM4PDE/DiffusionPDE error–time trajectories | `run_traces.sh` |

The Python runners, input preparation tools and figure scripts are listed by
task in [Experiment and plotting tools](plot/README.md).

NS uses the same entries as every other PDE. Select it with
`PDE_LIST=nsnonbounded` for the ordinary ablation sweep, or
`--pdes nsnonbounded` for repeated studies. No model-specific experiment directory
or NS-only launcher is needed.

`ensemble`, `guidance`, and `weights` read checkpoint paths and model profiles
from `configs/ablations/base/both/<pde>.yaml`. These profiles already select the
44M NS model. `CHECKPOINT_<PDE>` and `CHECKPOINT_ROOT` work as for main sampling;
`--checkpoint PATH` overrides the weights for one selected PDE, and
`--model-profile light` or `auto` overrides the architecture choice.
`--config-dir` selects a different set of model defaults. The prepared inputs,
guidance parameters, seeds and weight-selection rule retain their existing meaning.
Choose a new output directory when changing the model: saved predictions from a
different checkpoint are never reused.

```bash
bash scripts/sampling/ablations/run_study.sh ensemble \
  --pdes nsnonbounded --inputs /path/to/prepared_inputs \
  --output /path/to/ensemble_results --checkpoint /path/to/fm4nsnonbounded.pth
bash scripts/sampling/ablations/run_study.sh weights \
  --pdes nsnonbounded --inputs /path/to/prepared_inputs \
  --anchors /path/to/anchors.json --output /path/to/weight_results \
  --checkpoint /path/to/fm4nsnonbounded.pth
```

Repeated studies still require their prepared physical inputs and selected
settings (`selection.json` for `ensemble`, `--anchors` for `weights`); changing
the checkpoint does not change those experimental controls. Use `--plan-only`
to inspect model choices without reading the inputs or running inference.

Sampling-trajectory code lives in `experiments/trajectories/` and figure exporters
in `plot/`. The trajectory entry accepts `prepare`, `run` (the default), `verify`,
and `plot`; each subcommand supports `--help`. Existing prepared trajectory inputs
retain their model profiles and sampler definitions.

```bash
bash scripts/sampling/ablations/run_traces.sh run --root /path/to/trace_study
bash scripts/sampling/ablations/run_traces.sh plot \
  --root /path/to/trace_study --output /path/to/figures
```

The complete layout comparison uses `run_study.sh layouts --inputs
/path/to/prepared_inputs --output /absolute/path/to/layout_results`, where the
input root contains `helmholtz/`. It runs Random, Fixed (left half), Grid and
Columns with both shared and separate locations.

Controlled FM4PDE/DiffusionPDE timing uses `plot/run_diffusion_fm_timing.py`
(`prepare` and `run` modes). It reads the model profile from the prepared
protocol, including the light NS checkpoint.

The Poisson/Darcy frequency comparison uses predictions from
`plot/run_matched_timing.py`. Its study directory contains `inputs_v2/` and
`results_v3/`. Compute the spectra and input-level statistics with:

```bash
python plot/export_matched_spectra.py \
  --study /path/to/frequency_study --output /path/to/frequency_report
python plot/summarize_spectral_evidence.py --report /path/to/frequency_report
```

These commands compute frequency-band errors, predicted/reference energy
ratios and coefficient alignment from the saved physical predictions.
`plot/export_frequency_tables.py` exports the resulting paper tables.

## Physical residuals

Navier–Stokes main sampling and ordinary ablations use **endpoint secants**.
Its temporal-residual comparison explicitly includes Hermite and near-endpoint
variants. Heat, Wave, Advection–Diffusion, Reaction–Diffusion and Shallow Water use
Hermite bridges by default; Burgers uses full-trajectory finite differences.
Static equations use their spatial residuals. The training objective is flow matching;
periodic generated-sample diagnostics retain `eval_residual_mode=auto`, which
resolves to Hermite for NS, as in its saved training log. See [residual definitions](docs/time_dependent_residuals.md),
[normalization](docs/normalization.md) and [sampling options](docs/sampling.md).

## Verification and local backups

`python -m pytest tests` runs CPU tests. GPU experiments require their external
assets and are not launched by configuration validation. Earlier parameter
searches and development scripts are preserved under ignored `bak/`, with file
hashes in `bak/release_20260916/manifest.json`. Existing experiment outputs and
local research notes are preserved.
