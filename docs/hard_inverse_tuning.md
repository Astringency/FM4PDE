# Hard-sample inverse tuning for ID, smooth, and rough data

## Scope and selection rule

This workflow targets inverse coefficient recovery for Poisson, Helmholtz,
Darcy, and non-bounded Navier-Stokes. These are the four equations with a
complete, finite 1,000-sample inverse group in each of
`MAIN1000_100_TEST_{id,smooth,rough}`.

For every `(test_type, PDE)` group, `prepare_hard_inverse_samples.py` ranks the
1,000 records by coefficient relative L2 error (`rel_l2_a`) and keeps the worst
20. Odd ranks become a 10-sample tuning split; even ranks become an equally
difficult 10-sample holdout split. This produces 240 selected records in total:
120 for tuning and 120 for paired holdout validation. The exact source IDs and
historical metrics are recorded in
`artifacts/inverse_hard_tuning/hard_samples.csv`.

The split by alternating hard rank is deliberate: tuning and validation have
nearly identical difficulty while sharing no sample. Parameter selection uses

```text
0.5 * mean(rel_l2_a) + 0.3 * P90(rel_l2_a) + 0.2 * max(rel_l2_a)
```

across ID, smooth, and rough samples. Lower is better. The selected setting is
then compared with the baseline on the untouched holdout samples using the
same initial stochastic noise for every candidate. A candidate is eligible
only when mean PDE residual is no more than 1.25x baseline and mean solution
error is no more than 1.05x baseline in **each** of ID, smooth, and rough. The
per-distribution check prevents a good aggregate mean from hiding an OOD
regression. The observation-only candidate tests the limiting case explicitly
instead of assuming that ever-larger observation weights remain physically
acceptable.

## Prepare the hard subsets

Run this where the three 1,000-sample result sets and original PDE data are
available:

```bash
python scripts/tuning/prepare_hard_inverse_samples.py \
  --results-root outputs \
  --data-root /absolute/path/to/PDEdata \
  --output-root artifacts/inverse_hard_tuning \
  --samples-per-group 20
```

The compact tune/holdout files are approximately 1.3 MB each. Copy
`hard_samples.csv`, `subsets/`, and the repository/checkpoints to both A100
servers if they do not share storage.

The prepared inputs can also be transferred as one archive. From the repository
root on each server, extract it with:

```bash
tar -xzf artifacts/hard_inverse_inputs.tar.gz
```

Because `subsets/` contains generated binary data, it is intentionally ignored
by git and will not appear after a plain clone or pull.

## Run on two A100 80 GB servers

On server 0:

```bash
SERVER_RANK=0 \
ARTIFACT_ROOT=artifacts/inverse_hard_tuning \
bash scripts/tuning/run_hard_inverse_a100.sh
```

On server 1:

```bash
SERVER_RANK=1 \
ARTIFACT_ROOT=artifacts/inverse_hard_tuning \
bash scripts/tuning/run_hard_inverse_a100.sh
```

Server 0 runs Poisson and Helmholtz; server 1 runs Darcy and non-bounded
Navier-Stokes. The default is 10 tuning and 10 holdout samples per test type,
100 sampling steps, persistent checkpoint loading, and resumable jobs. Static
PDEs use a microbatch of 10. Non-bounded Navier-Stokes defaults to 4 because its
model and guidance graph are much larger; reduce it with
`NSNONBOUNDED_MICROBATCH=2` or `1` after an OOM.

Inspect commands without starting GPU work:

```bash
SERVER_RANK=0 PLAN_ONLY=true bash scripts/tuning/run_hard_inverse_a100.sh
SERVER_RANK=1 PLAN_ONLY=true bash scripts/tuning/run_hard_inverse_a100.sh
```

To retry only one phase or PDE subset:

```bash
PDE_LIST=nsnonbounded PHASE=tune NSNONBOUNDED_MICROBATCH=2 \
  bash scripts/tuning/run_hard_inverse_a100.sh

PDE_LIST=nsnonbounded PHASE=holdout NSNONBOUNDED_MICROBATCH=2 \
  bash scripts/tuning/run_hard_inverse_a100.sh
```

Keep the same `SAMPLES_PER_TEST_TYPE`, `NUM_STEPS`, and artifact root between
tuning and holdout. Job signatures include the sample shape, step count,
parameters, checkpoint, and subset fingerprints; incompatible old jobs are
marked stale and rerun instead of being silently reused.

## Outputs

Each server writes PDE-scoped summaries so concurrent runs do not overwrite
one another:

- `tune_summary_<pdes>.csv`: all complete candidates ranked by the robust score
- `selected_params_<pdes>.csv`: selected parameter per PDE
- `holdout_comparison_<pdes>.csv`: paired baseline-versus-winner validation
- `tune_per_sample_<pdes>.csv` and `holdout_per_sample_<pdes>.csv`: audit rows
- `runs/{tune,holdout}/.../job.json`: resumable job status and exact signature

If both servers write to the same shared artifact root, run a final combined
analysis after all jobs finish:

```bash
PDE_LIST=poisson,helmholtz,darcy,nsnonbounded PHASE=analyze \
  bash scripts/tuning/run_hard_inverse_a100.sh
```

If storage is independent, keep the two scoped result bundles or copy the
server-1 `runs/` subdirectories into server 0 before the combined analysis.

## Refined second-round sweep

After the initial 10+10 hard-sample run, use the refined candidate set to probe
the observed frontiers without rerunning completed candidate names. Poisson
tests 10x/12x/14x observation guidance, Helmholtz tests 20x/24x/28x, Darcy
crosses 32x/64x guidance with clip thresholds 75/100, and non-bounded NS tests
clip thresholds 125/150/200 plus two observation/clip interactions.

If results should live directly under `outputs/artifacts`, run:

```bash
SERVER_RANK=0 \
ARTIFACT_ROOT=outputs/artifacts/inverse_hard_tuning \
CANDIDATE_SET=refined \
ANALYSIS_LABEL=round2 \
bash scripts/tuning/run_hard_inverse_a100.sh

SERVER_RANK=1 \
ARTIFACT_ROOT=outputs/artifacts/inverse_hard_tuning \
CANDIDATE_SET=refined \
ANALYSIS_LABEL=round2 \
bash scripts/tuning/run_hard_inverse_a100.sh
```

The existing `runs/` tree and first-round summaries are preserved. New
candidate jobs are added under the same `runs/{tune,holdout}/...` hierarchy;
round-two summaries have names such as
`selected_params_poisson_helmholtz_round2.csv` and
`holdout_comparison_darcy_nsnonbounded_round2.csv`.

## Local Poisson pilot

The expanded 12-sample tuning pilot (four hard samples per test type) selected
16x observation guidance (`zeta_obs_u=5.76e9`) over the `3.6e8` baseline. On 12
disjoint paired holdout samples, mean coefficient relative L2 fell from 0.848
to 0.581 (31.4%), and all 12 samples improved. This setting nevertheless
failed the predeclared cross-distribution guardrail: its rough holdout PDE
residual was 1.291x baseline, above the 1.25x limit.

An 8x exploratory probe (`2.88e9`) reduced the same holdout mean to 0.633
(25.3% versus baseline), again with 12/12 wins, and passed the PDE and solution
guardrails in all three distributions. It is a useful fallback hypothesis, not
a final selection, because it was inspected after the 16x holdout result. More
aggressive values exposed the tradeoff clearly: 32x raised rough PDE residual
to 1.487x baseline on tuning and 1.607x on holdout; 64x was worse. The formal
A100 run keeps the full geometric grid plus observation-only limit and uses all
ten tune plus ten holdout samples per test type before recommending a final
value.
