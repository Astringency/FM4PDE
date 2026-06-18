# Time-Dependent PDE Residuals

FM4PDE separates residual family from residual status.

- `static`: Darcy, Poisson, Helmholtz, Steady Heat Conduction
- `full_time_space`: Burgers
- `temporal_endpoint`: Heat, Wave, Advection-Diffusion, Reaction-Diffusion, Shallow Water, Non-bounded Navier-Stokes

Burgers is the only current full time-space tensor residual: the BCHW tensor itself stores a one-dimensional PDE time-space field, with `H` as time and `W` as space. It uses `u_t + u u_x - nu u_xx` with finite differences and records `status=reliable`.

The other time-dependent PDEs use the unified temporal endpoint mode system below. Endpoint-only is the default in the current endpoint-pair FM setup, not the only valid temporal residual form.

## `hermite_bridge`

Default for `residual_mode: auto` on temporal endpoint PDEs.

```text
H(s) = h00(s) q0 + h10(s) T F(q0) + h01(s) qT + h11(s) T F(qT)
residual = dH/dt - F(H)
```

An optional endpoint integral consistency residual is appended:

```text
qT - q0 - 0.5 * T * (F(q0) + F(qT))
```

Metadata marks this as:

```yaml
residual_family: temporal_endpoint
temporal_derivative_mode: hermite_bridge
endpoint_only: true
uses_generated_trajectory: false
uses_extra_temporal_observations: false
status: approximate
```

## `near_endpoint_temporal`

Requires extra sparse temporal observations:

```text
q_dt ~= q(delta t)
q_T_minus_dt ~= q(T - delta t)
mask_0, mask_T, dt
```

It computes:

```text
r0 = (q_dt - q0) / dt - F(q0)
rT = (qT - q_T_minus_dt) / dt - F(qT)
```

The residual is masked and normalized by observed count so sparse masks do not shrink the loss. Missing auxiliary observations raise `ValueError`; there is no fallback to Hermite.

Metadata marks this as:

```yaml
residual_family: temporal_endpoint
temporal_derivative_mode: near_endpoint_sparse_fd
endpoint_only: false
uses_generated_trajectory: false
uses_extra_temporal_observations: true
requires_extra_temporal_observations: true
status: approximate
```

## `endpoint_secant`

Coarse two-endpoint residual for ablations only:

```text
(qT - q0) / T - F(q_mid)
```

Metadata includes:

```yaml
residual_family: temporal_endpoint
temporal_derivative_mode: endpoint_secant
endpoint_only: true
uses_generated_trajectory: false
uses_extra_temporal_observations: false
two_time_level_approx: true
status: approximate
warning: coarse two-time-level endpoint residual; not a full spatiotemporal PDE residual
```

## `full_trajectory_fd`

Reserved for full-trajectory FM or explicit trajectory auxiliary state. It requires `pde_params["full_trajectory"]` or `pde_params["trajectory"]` with layout `[B,T,C,H,W]` or `[B,C,T,H,W]`.

It computes finite-difference time derivatives across the trajectory and compares them with the same RHS `F(q; params)` used by Hermite and endpoint modes. If only endpoint state is present, this mode raises `ValueError`.

Metadata marks this as:

```yaml
residual_family: full_trajectory
temporal_derivative_mode: full_fd
endpoint_only: false
uses_generated_trajectory: true
uses_extra_temporal_observations: false
status: reliable
```

## RHS Definitions

- Heat: `F(u) = alpha * Laplacian(u)`
- Wave: `F([u,v]) = [v, c^2 Laplacian(u)]`
- Advection-Diffusion: `F(u) = -b_x u_x - b_y u_y + kappa Laplacian(u)`
- Reaction-Diffusion: FitzHugh-Nagumo RHS with Neumann Laplacian
- Shallow Water: standard 2D conservative flux RHS with clamped safe depth
- Non-bounded Navier-Stokes: 2D vorticity equation with periodic FFT stream-function velocity reconstruction

Any experiment using extra temporal observations or full trajectory state must preserve the metadata fields above so endpoint-only results are not mixed with sparse-temporal or full-trajectory results.
