# Burgers 100-step tuning results

## Selected main parameters

The shared 100-step configuration selected from the local RTX 3070 Ti runs is:

```yaml
sampler_phase: stochastic
num_steps: 100
zeta_obs_a: 0
zeta_obs_u: 409600
zeta_pde: 10
clip_mode: global_norm
clip_threshold: 50
num_obs: 500
num_sensor_columns: 16
```

All reported errors are full-field per-sample relative L2 values. Tuning and
validation used the held-out `burger_test_10000-128-128.mat`, `sample_seed=0`,
`mask_seed=0`, batch size 2, and disjoint offset blocks.

## Held-out results

| Observation mode | Validation offsets | n | Mean | Median | P90 | Max |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| random, 500 points | 7000-7001, 8000-8001, 9000-9001 | 6 | 2.27% | 2.15% | 3.24% | 3.91% |
| sensor_column, 16 columns | 2500-2501, 3500-3501, 4500-4501 | 6 | 3.04% | 1.99% | 5.93% | 8.50% |

The results meet a mean error below 5% on these held-out offsets. The sample
count is deliberately small enough to run on the local 8 GB GPU; the A100
scripts default to larger batches and should be used before treating these as
final benchmark estimates.

## Why clipping is required

With the old `clip_threshold=1e10`, observation zeta values of 4800 and above
produced gradient spikes during steps 2-5 and could diverge by several orders
of magnitude. Per-sample global clipping at 50 made large zeta values stable,
allowing stronger late-stage observation fitting.

## Five-column limitation

Keeping `num_sensor_columns=5` did not provide robust cross-offset performance.
For the selected zeta and clipping, six held-out offsets had mean 14.19%, median
9.83%, P90 30.17%, and maximum 43.29%. Increasing zeta as high as 3,276,800
reduced the difficult sample only to about 34.5%. On that same difficult pair,
10 columns gave 7.90% mean and 16 columns gave 2.58% mean. Consequently, the
main configuration uses 16 columns. If the formal observation budget must stay
at five columns, the below-10% target is not supported by these experiments.

## Sampler comparison

The sampler comparison used offsets 5500-5501 and 6500-6501 with the selected
zeta/clipping and 16 sensor columns.

| Mode | Sampler | n | Mean | Median | P90 | Max |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| random | stochastic | 4 | 2.32% | 1.76% | 3.74% | 4.55% |
| random | hybrid_s2d | 4 | 1.76% | 1.55% | 2.71% | 3.20% |
| sensor_column | stochastic | 4 | 2.70% | 2.32% | 4.38% | 5.13% |
| sensor_column | hybrid_s2d | 4 | 2.72% | 2.43% | 4.55% | 5.38% |

Deterministic sampling diverged under its current early-time `bt * step_size`
guidance scaling, and hybrid_d2s was much worse than the two rows above.
Stochastic was marginally best for `sensor_column`, while hybrid_s2d was best
for `random` and across both modes combined. Therefore the claim that
stochastic is universally best was not verified. The main config remains
stochastic because it is stable in both modes and already meets the target.
