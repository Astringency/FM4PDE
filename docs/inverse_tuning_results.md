# Inverse tuning status

## First 100-step A100 baseline

The initial pass used stochastic sampling, random observations, 500 observed
points, batch size 1, and offset 0. It is a numerical diagnostic rather than a
cross-offset benchmark.

| PDE | Outcome | coefficient relative L2 | solution relative L2 |
| --- | --- | ---: | ---: |
| Darcy | finite | 19.65% | 4.43% |
| Poisson | finite | 26.16% | 10.02% |
| Helmholtz | finite | 27.51% | 9.77% |
| Non-bounded Navier-Stokes | finite | 28.98% | 15.67% |
| Heat | finite divergence | 1,635,696% | 1,033,704% |
| Wave | finite divergence | 1,383,843% | 774,549% |
| Advection-Diffusion | finite divergence | 1,640,233% | 1,087,299% |
| Reaction-Diffusion | NaN/Inf near step 2 | - | - |
| Shallow Water | NaN/Inf near step 18 | - | - |
| Steady Heat Conduction | NaN/Inf near step 2 | - | - |

The three finite but divergent temporal equations and Steady Heat Conduction
still used placeholder `zeta_obs_u=1`, `zeta_pde=1`. Most failing configurations
also used `clip_threshold=1e10`, which did not constrain the early update.

## Second-round protocol

Run the stability screen first:

```bash
PDE_DATA_ROOT=/absolute/path/to/PDEdata \
STAGE=stabilize DEVICE_LIST="cuda:0 cuda:1" MAX_PARALLEL_TASKS=2 \
  bash scripts/tuning/run_inverse_tuning.sh
```

This evaluates nine observation/PDE zeta ratios per unstable equation with
global-norm clipping at 50. The tailored ranges are based on the measured
unweighted observation and PDE gradient norms at step 0.

Then run the accuracy screen:

```bash
PDE_DATA_ROOT=/absolute/path/to/PDEdata \
STAGE=refine DEVICE_LIST="cuda:0 cuda:1" MAX_PARALLEL_TASKS=2 \
  bash scripts/tuning/run_inverse_tuning.sh
```

This evaluates four observation-zeta levels and three PDE-zeta levels for each
of the four finite equations. Both stages use offsets 0 and 1000 with batch size
2, so a configuration must provide four finite samples to enter the automatic
ranking. The rank score is `0.5 * mean + 0.3 * P90 + 0.2 * max` of coefficient
relative L2; incomplete configurations are excluded.

Final validation should use new offset blocks such as 3000 and 5000, larger
batches, and at least two mask seeds before treating any candidate as a final
benchmark configuration.

## Provisional main configurations

The best finite settings from the four unique samples at offsets 0-1 and
1000-1001 have been copied to `configs/main/inverse`. The separate inverse
configs use random observations, stochastic 100-step sampling, and global-norm
clipping at 50. The `both` and `forward` directories retain their independent
baselines. These values reproduce the screening setup but remain provisional
until evaluation on disjoint offsets with more samples.

Some stability-screen winners resolved to `zeta_pde=0.0` because the original
integer-valued YAML fields truncated fractional CLI overrides. The main zeta
fields are now written explicitly as floats to prevent the same coercion during
subsequent tuning. A zero value records the configuration that actually ran; it
does not constitute a comparison against the intended 0.01 and 0.1 values.
