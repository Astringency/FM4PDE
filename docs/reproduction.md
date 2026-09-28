# Reproducing the manuscript experiments

Use the paragraph scripts listed in the [README](../README.md), with the
original datasets and released inference checkpoints. The experiment manifests
fix the task, input indices, observation protocol, seeds, batch partitions and
guidance settings. A newly generated dataset or a newly trained checkpoint is a
new realization of the experiment; it does not reproduce the saved predictions.

## Environment and assets

Install [environment.yml](../environment.yml). Set `PYTHON_BIN`, `DATA_ROOT`
and `CHECKPOINT_ROOT`, or the individual `CHECKPOINT_<PDE>` variables, as in the
README. The configuration loader relocates the paths; changing paths does not
change the physical arrays or the embedded training normalizer. Preserve the
original filenames and row order inside the data tree.

The archived A800/A100 inference runs used PyTorch 2.8.0 with CUDA 12.8. The
RTX 4090 timing runs used PyTorch 2.5.1 with CUDA 12.1. Equal seeds do not always
produce equal CUDA random streams across these environments. For individual
prediction replay, use the recorded software environment, batch partition and
checkpoint as well as the seed. Runtime measurements additionally depend on the
GPU and competing workloads.

Additional assets are experiment-specific:

| Experiment | Required assets |
| --- | --- |
| Native sampling and ablations | Original physical test files and FM4PDE checkpoints |
| Conditional averaging | Original Poisson ID file including rows 1500–1531; the recorded execution partitions in the [averaging guide](unified_inverse_guidance.md) |
| Controlled timing | `TIMING_INPUT_ROOT` bundle containing `protocol.json`, `masks.npz`, `source/timing_truths.npz` and `weights/` |
| FM4PDE-OFM | Official OFM code, compact training normalizers and the selected epoch checkpoints in the [baseline guide](baseline_residual_reproduction.md) |
| Burgers reference re-solving | MATLAB and Chebfun (`MATLAB_BIN`, `CHEBFUN_ROOT`) |
| Shallow-water prior reference re-solving | PyClaw |

Data and checkpoints are kept outside Git. A source checkout alone does not
contain these assets. The runners record checksums for the assets they actually
load; retain these receipts with the resulting fields.

## Run the recorded experiments

Start with configuration validation and a plan, then execute into a new output
directory:

```bash
python -m sampling.validate_configs
python -m sampling.validate_manuscript_sampling
python -m sampling.validate_secant_residual
bash scripts/sample/comparison/sparse_forward_inverse.sh --plan-only
bash scripts/sample/comparison/sparse_forward_inverse.sh \
  --device cuda:0 --output outputs/paper/main
```

Run the corresponding script for each additional paragraph. The README maps
all main comparisons, ablations and the eleven appendix priors to their scripts.
Main-table overrides are part of the experiment manifest: in particular, the
NS forward/inverse comparison uses S→D at flow time 0.5. The generic
`scripts/sample/run_sample.sh` executes one base profile; it does not substitute
for the paragraph manifest. Its `--plan-only` mode requires no data or weights;
`DRY_RUN=true` is different and executes with a placeholder velocity model.

Use `--pdes`, `--tasks`, `--test-types`, or `--job-ids` to select existing cases.
For example:

```bash
bash scripts/sample/comparison/physics_based.sh --pdes poisson helmholtz
bash scripts/sample/ablations/switching_time.sh --pdes nsnonbounded poisson
bash scripts/sample/appendix/unconditional_priors.sh --pdes heat wave
```

`--limit` deliberately runs a development subset. `--override` changes the
experiment and should be omitted when reproducing a published setting. Special
engines reject overrides of fixed cohort controls or settings they do not
implement. Invalid or empty selections fail instead of reporting a successful
empty run.

Sharding uses `--num-shards N --shard-index I` and a different output directory
for each worker. It distributes existing cases while preserving their seeds and
batch membership. Do not rebatch a stochastic experiment. Conditional averaging
uses its separately documented canonical 1,000-draw noise pool and execution
batch settings.

## Read and retain the results

Standard sampling writes per-case `receipt.json`, saved physical fields,
`errors.csv`, `summary.json`, and a `completion.json` for the selected workload.
The completion record must have zero failures and the expected number of cases.
If `--continue-on-error` is used, a `failures.json` records unsuccessful cases;
those cases cannot be dropped when reporting the experiment.

Receipts contain the resolved configuration and source, data, checkpoint and
prediction identities. Resuming a standard sampling or averaging run checks
these identities. After a source or configuration change, use a new output
directory rather than mixing results from different versions.

Pool per-sample rows from every shard before calculating manuscript means and
sample standard deviations; do not average shard means with unequal counts.
Relative errors are ratios in the result files. Multiply by 100 only for figure
or table displays that explicitly use percentages. PDE residual loss and the
independent reference-solver consistency defect are different quantities.

Training uses the eleven scripts in `scripts/train` and the schedules in
`configs/training.yaml`. `--test_run` is a short execution check, not the
300-epoch training protocol used to obtain the paper checkpoints.

## Checks performed on the public entrypoints

The September 2026 code audit checked every shell entrypoint from outside the
repository directory and ran representative real-data cases through each native
sampling engine: all eleven PDEs, S, D, both hybrid orders, both time-grid types,
and the three temporal residuals. It also exercised the five OFM priors, timing,
sampling trajectories, physical re-solving and the appendix prior pipeline.

All eleven training launchers completed a real short optimization, checkpoint
export, reload and finite velocity evaluation. Two-GPU training was additionally
checked for both FP32 and bfloat16 recipes. These checks establish executable
paths; they do not replace full training or running every case in a paragraph.
Representative predictions were replayed against the adopted manuscript
artifacts, including a complete 1,000-prediction inverse-averaging pool and all
eleven appendix prior fields.
