# Time-Dependent PDE Residual Modes

FM4PDE endpoint models still generate endpoint pairs only. The model output is not changed into a full trajectory:

- heat: `(u0, uT)`
- wave: `([u0, v0], [uT, vT])`
- advection_diffusion: `(u0, uT)`
- reaction_diffusion: `([u0, v0], [uT, vT])`
- shallow_water: `([h0, hu0, hv0], [hT, huT, hvT])`

All PDE residuals used by guidance are computed on the physical endpoint estimate passed into the guidance loss.

## `hermite_bridge`

`hermite_bridge` is the default mode for endpoint-only time-dependent PDEs when `residual_mode: auto`.

The residual constructs an endpoint-induced cubic Hermite bridge using endpoint PDE derivatives:

```text
q_t = F(q; params)
f0 = F(q0)
fT = F(qT)
```

The bridge is evaluated at a few interior collocation times, defaulting to `[0.25, 0.5, 0.75]`, and penalizes:

```text
dH/dt(s) - F(H(s))
```

This bridge is not a generated physical trajectory and is not used as a training target. It is a differentiable endpoint-only residual used for PDE-aware guidance. By default an endpoint integral consistency term is also appended:

```text
qT - q0 - 0.5 * T * (F(q0) + F(qT))
```

Relevant config fields:

```yaml
residual_mode: hermite_bridge
hermite_collocation_times: [0.25, 0.5, 0.75]
hermite_num_collocation: 0
hermite_include_integral_residual: true
hermite_integral_weight: 1.0
```

## `near_endpoint_temporal`

`near_endpoint_temporal` is an extra observation setting. It requires sparse temporal observations near both endpoints:

```text
q_dt ~= q(delta t)
q_T_minus_dt ~= q(T - delta t)
mask_0, mask_T
```

The residual uses local temporal differences:

```text
(q_dt - q0) / dt - F(q0)
(qT - q_T_minus_dt) / dt - F(qT)
```

The residual is masked and normalized by the observed mask count so sparse masks do not artificially shrink the mean loss. If the near-endpoint observations or masks are missing, this mode raises `ValueError`; it never falls back to hidden dense trajectories.

Relevant config fields:

```yaml
residual_mode: near_endpoint_temporal
num_near_endpoint_obs: 64
near_endpoint_sensor_mode: random
near_endpoint_mask_seed: 0
near_endpoint_shared_mask: true
```

The loader supports explicit trajectory data for `future_h5` via datasets such as `full_trajectory`, `trajectory`, `states`, or `solution_trajectory`. For `swe` it uses frames `1` and `-2`; for `rd` it uses frames `51` and `-2` because the endpoint pair starts at frame `50`.

## `endpoint_secant`

`endpoint_secant` keeps the old coarse two-time-level residual for ablations:

```text
(qT - q0) / T - F(endpoint midpoint or endpoint average)
```

Metadata includes:

```yaml
mode: endpoint_secant
warning: coarse two-time-level endpoint residual; not a full spatiotemporal PDE residual
two_time_level_approx: true
```

`legacy_endpoint_secant` is also accepted for compatibility. For shallow-water, `endpoint_secant` makes the time scale explicit; `legacy_endpoint_secant` records the legacy implicit unit-time behavior.

## Exceptions

Static PDE residuals keep their original behavior.

Burger uses its existing full time-space tensor residual because the data tensor itself contains an explicit time-space grid.

`nsnonbounded` PDE guidance remains disabled. It does not compute a pseudo residual unless a reliable vorticity transport residual is implemented.

## Example Commands

```bash
python -m sampling.runner --config configs/ablations/base/heat.yaml --override residual_mode=hermite_bridge
python -m sampling.runner --config configs/ablations/base/heat.yaml --override residual_mode=endpoint_secant
python -m sampling.runner --config configs/ablations/base/heat.yaml --override residual_mode=near_endpoint_temporal --override num_near_endpoint_obs=64
```
