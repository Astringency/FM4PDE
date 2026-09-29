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

Full-file SHA-256 lists are provided for the 84 data files (55 training shards
and 29 test files) and all eleven released checkpoints. After downloading,
verify them from the corresponding repository root:

```bash
(cd "$DATA_ROOT" && sha256sum -c /path/to/FM4PDE/configs/release_data.sha256)
(cd "$CHECKPOINT_ROOT" && sha256sum -c /path/to/FM4PDE/configs/release_models.sha256)
```

The full data check reads about 697 GB; a missing file is reported explicitly.
The manuscript experiment manifests use all 29 test files: ID, Smooth and Rough
for the five main PDEs, ID for the six additional PDEs, and eight
`rough2`/`rough3` files for Poisson, Helmholtz, Darcy and Navier–Stokes in the
joint-reconstruction and physical-consistency comparison.
`python -m sampling.validate_configs --check-assets` checks all 55 training
shards, the test files selected by the published experiments, and the FM4PDE
checkpoints. It does not require unused optional test distributions.

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
| Controlled timing | Released data and FM4PDE weights; the IDs and manuscript-aligned masks are included in `configs/experiments/comparison/timing_protocol.json` and `timing_masks.npz` |
| FM4PDE-OFM | Official OFM code, compact training normalizers and the selected epoch checkpoints in the [baseline guide](baseline_residual_reproduction.md) |
| Burgers reference re-solving | MATLAB and Chebfun (`MATLAB_BIN`, `CHEBFUN_ROOT`) |
| Shallow-water prior reference re-solving | PyClaw |

Data and checkpoints are kept outside Git. A source checkout alone does not
contain these assets. The runners record checksums for the assets they actually
load; retain these receipts with the resulting fields.

The timing cohort is drawn with `np.random.default_rng(20260907).choice(1000,
21, replace=False)`: the first 20 IDs are evaluated and the last is used only
for warmup. For each ID `i`, `np.random.default_rng(20260907+i)` draws 500
positions without replacement for each paired field (successive draws from
the same generator). Burgers instead draws five physical-time levels, each
observed at all 128 spatial locations. Both methods use identical inputs and masks.
The timing loader regenerates this random selection and verifies it against
the checked-in records; no prediction or reconstruction error enters selection.
FM4PDE uses the joint guidance weights and clipping thresholds from
`configs/main/both/`, with stochastic updates as specified in the timing appendix.
Protocol version 2 corrects the earlier archived weights and Burgers mask
orientation. When `TIMING_INPUT_ROOT` selects an old bundle, its assets are
verified first, its Burgers masks are transposed to time slices, and current
manuscript guidance controls are applied. These corrections do not alter saved
measurements or imply that the published timing values have been rerun.

## Run the recorded experiments

Start with configuration validation and a plan, then execute into a new output
directory:

```bash
python -m sampling.validate_configs
python -m sampling.validate_manuscript_sampling
python -m sampling.validate_secant_residual
python -m sampling.validate_reproduction
# Check the current manuscript tables when its LaTeX source is available:
python -m sampling.validate_manuscript --manuscript /path/to/fm4pde_jmlr_revision.tex
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

Input indices are zero-based. The main comparisons use rows 0–99 of each
released test file; the temporal-residual and additional-PDE comparisons use
rows 0–31. The observation/PDE-guidance comparison uses rows 0–19 per PDE,
and the PDE-weight ablation uses rows 1500–1519. Both guidance tables report
means over 20 ID inputs, evaluated in one batch of 20 with the fixed seeds
and observation rules recorded in the manifest. Conditional averaging uses
Poisson rows 1500–1531. These are recorded cohorts, not samples selected
by reconstruction error. The timing cohort follows the separate draw described
above.

The main Poisson, Helmholtz, Darcy and Navier–Stokes comparisons use
`np.random.RandomState(0)`: for each input, draw 500 distinct flattened grid
indices for the first field, then another 500 for the second field. Both draws
are consumed even if only one field is observed. The physics-baseline masks
instead use the first eight bytes of
`SHA256("mask|1|test|<filename>:<input_index>|0")`, interpreted as a big-endian
integer modulo `2**63 - 1`, to seed a CPU Torch generator; the first 500 entries
of `torch.randperm(128 * 128)` are observed. The Smooth physics comparison uses
the original filename without the `_smooth` suffix. These rules are implemented
in `experiments/paper/observations.py`; the manifests select the appropriate
rule for each experiment.

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

Paper runners also record Python, PyTorch, CUDA/cuDNN and numerical-library
versions, device model, thread count, deterministic settings and TF32 flags.
The resumable engines compare these execution settings as part of the identity;
conditional averaging additionally records its execution batch size. Receipts
from older versions without this information require a new output directory.
Native sampling and appendix-prior source identities include MATLAB sources.

The Poisson and Helmholtz forward main comparisons retain the original batches
of eight (with the final four inputs run individually) and the common comparison
observation masks. Changing the batch size or generating new masks changes the
random experiment, even if the guidance weights and nominal seeds are unchanged.

The Burgers structured comparison uses batches of 16, with a final batch of
four. Its five observed physical-time levels per input follow the same
filename-and-index seeded rule as the physics baselines. The experiment runner
recreates these masks from the public test files; no baseline prediction files
are needed. Burgers has a single trajectory field, so the comparison protocols
leave the unused coefficient observation mask empty.

The Burgers random-point comparison uses the same filename-and-index rule to
draw 500 space-time positions without replacement. Its ID and Smooth batches
contain eight inputs, and its Rough batches contain sixteen (the final batch
contains four). Initial and per-step Gaussian noise are selected by input index
from a 1,000-row pool with seed zero. The Burgers physical-consistency cases
retain this same protocol. Poisson physical-consistency cases use batches of
sixteen, the filename-and-index observation rule, and a 100-row Gaussian pool
with seed zero. These settings are explicit in the experiment manifests.

Pool per-sample rows from every shard before calculating manuscript means and
sample standard deviations; do not average shard means with unequal counts.
Relative errors are ratios in the result files. Multiply by 100 only for figure
or table displays that explicitly use percentages. PDE residual loss and the
independent reference-solver consistency defect are different quantities.

`result.pt` now retains the supplied observation values and final per-sample
metrics. Observed-location relative errors use those measurements, including
noise; complete-field errors still use clean references. The result preview
plots saved measurements and reports the selected sample's errors. Older noisy
artifacts without stored measurements explicitly label the sensor panel as a
clean reference. `L_pde` in the
per-sample sampling metrics is the weighted sum of component MSEs. The RMS of
concatenated residual channels is a separate diagnostic and must not be squared
and substituted for this loss.

The paragraph runner's `errors.csv` and `summary.json` include `pde_loss`, the
arithmetic mean of the two fieldwise relative errors, and solver inconsistency
when re-solving was requested. Offline `pde_loss` is recomputed in float64 from
saved physical fields with the saved residual and boundary settings, so small
roundoff differences from the sampling-time float32 loss are possible. The old
`residual_mse` field in consistency receipts is an interior-region diagnostic,
not the manuscript's complete PDE loss. Use `pde_loss` for the Q comparison.

To re-evaluate an existing indexed run without resampling:

```bash
python -m experiments.paper.summarize /path/to/experiment-output
```

The observation/PDE-guidance manifest labels its two sample cohorts separately:
`guidance_components` (IDs 0–19) and `pde_weight` (IDs 1500–1519).
Summaries pool batches within each cohort; identical controls in the two
experiments remain separate. To repair summaries from older receipts that lack
these labels, supply the manifest without resampling or changing predictions:

```bash
python -m experiments.paper.summarize /path/to/experiment-output \
  --manifest configs/experiments/ablations/observation_pde_guidance.yaml
```

The Navier–Stokes reference solver now retains a zero Laplacian coefficient at
zero frequency, so viscosity does not damp mean vorticity. Only the
streamfunction division uses a guarded denominator. Existing re-solve metrics
are not updated automatically; any future comparison must evaluate all methods
with the same solver revision.

For the paired-field Q table, provide all four methods' per-sample CSVs with
the same 100 input IDs per PDE/distribution. Each file must contain full PDE
losses and solver errors; older files containing only `residual_mse` are not
accepted. Re-evaluate baseline predictions with
`scripts/sample/comparison/reevaluate_saved_residuals.sh` to obtain full losses.
Join solver errors using PDE, distribution, method and input ID if they are
stored separately. Multiple `--input` files for the same method combine shards;
duplicate or missing input IDs fail validation.

```bash
python -m experiments.paper.score \
  --input fm4pde=/path/to/fm4pde/errors.csv \
  --input recfno=/path/to/recfno/errors.csv \
  --input senseiver=/path/to/senseiver/errors.csv \
  --input voronoicnn=/path/to/voronoicnn/errors.csv \
  --output outputs/paper/joint_scores.json
```

The score averages per-sample metrics first, normalizes each metric by the
minimum across the four methods within the same setting, then takes the
geometric mean. It does not average per-sample Q values.

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

The September 29 follow-up checked the current manuscript's 32 FM4PDE and nine
OFM guidance rows, the main comparison's prescribed input IDs, and all eleven
training schedules. It added regression checks for noisy observed errors,
complete PDE loss, aggregation over samples, and Q computation. All 37 shell
entrypoints passed syntax and plan/help checks. These are separate from full
numerical reproduction. Re-evaluation of 8,000 adopted joint-comparison
predictions found that substituting complete PDE loss for the old interior
diagnostic changes 15 displayed Helmholtz baseline Q entries; method rankings
in all 20 settings remain unchanged. The manuscript table must use the same
loss definition as the evaluator when it is regenerated.
