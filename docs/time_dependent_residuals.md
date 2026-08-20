# Time-Dependent PDE Residuals

FM4PDE separates residual family from residual status.

- `static`: Darcy, Poisson, Helmholtz, Steady Heat Conduction
- `full_time_space`: Burgers
- `temporal_endpoint`: Heat, Wave, Advection-Diffusion, Reaction-Diffusion, Shallow Water, Non-bounded Navier-Stokes

Burgers is the only current full time-space tensor residual: the BCHW tensor itself stores a one-dimensional PDE time-space field, with `H` as time and `W` as space. It uses `u_t + u u_x - nu u_xx` with finite differences and records `status=reliable`.

The other time-dependent PDEs use predicted endpoint pairs. Their `q0/qT` inputs to PDE loss always come from the model. The sole field-data exception is `near_endpoint_temporal`: when explicitly selected for one of the six supported temporal-endpoint PDEs, it may additionally consume sparse masked ground-truth observations at `dt` and `T-dt`. It never substitutes a true endpoint or full trajectory for a model output.

Sampling guidance now treats observation loss as masked MSE over observed entries:

```math
L_{obs} = \frac{\sum_i M_i (x_i - y_i)^2}{\sum_i M_i}
```

The PDE loss computes MSE separately for each residual component, then applies the configured component weights:

```math
L_{PDE} = L_{int} + \lambda_{bc} L_{bc} + \lambda_{ic} L_{ic} + \lambda_{ep} L_{ep}
```

```math
L_{int} = \operatorname{mean}_{j \in C_{int}} |R_{int,j}|^2
```
```math
L_{bc} = \operatorname{mean}_{j \in C_{bc}} |R_{bc,j}|^2
```
```math
L_{ic} = \operatorname{mean}_{j \in C_{ic}} |R_{ic,j}|^2
```
```math
L_{ep} = \operatorname{mean}_{j \in C_{ep}} |R_{ep,j}|^2
```

Interior, boundary, initial, and endpoint components are not concatenated and globally averaged for loss computation. Separate MSE denominators prevent different component point counts or channel counts from changing the intended weights. Concatenated residual fields remain available only for logging and residual norm diagnostics.

Older code often zeroed the outer grid cells of the PDE residual. That was a boundary-excluded interior residual; it did not enforce boundary conditions. The current implementation keeps the interior residual on interior points and appends explicit, differentiable boundary and optional initial-condition residual channels where those residuals are physically meaningful.

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

Boundary residuals are evaluated for `q0`, every collocation state `H(s)`, and `qT`. The endpoint integral consistency residual is recorded as the endpoint component and is weighted by `endpoint_bc_weight`. If a future bridge construction exactly satisfies an endpoint condition, metadata should mark that fact and should not append a duplicate zero residual.

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

This mode is available in formal sampling for Heat, Wave, Advection-Diffusion, Reaction-Diffusion, Shallow Water, and Non-bounded Navier-Stokes. It is the sole guidance/evaluation exception that may pass ground-truth solution-field values into the PDE residual. Poisson, Burgers, and all other model families reject it, as does unconditional periodic training evaluation.

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

Here `q0` and `qT` are always model predictions. `q_dt` reuses the coef/q0 endpoint mask and `q_T_minus_dt` reuses the sol/qT endpoint mask. There is no separate near-endpoint observation budget, sensor mode, seed, or sharing flag. The loader zeroes unobserved entries immediately, and the loss boundary reapplies the masks before calling the residual. The auxiliary observations are detached; gradients flow through predicted endpoints only.

The residual is masked and normalized by observed count so sparse masks do not shrink the loss. Every batch sample must have at least one observation at each near-endpoint frame. Missing observations or empty masks raise `ValueError`; there is no fallback to Hermite. True initial fields, true endpoints, and true complete trajectories remain forbidden as PDE-loss inputs.

Metadata marks this as:

```yaml
residual_family: temporal_endpoint
temporal_derivative_mode: near_endpoint_sparse_fd
endpoint_only: false
uses_generated_trajectory: false
uses_extra_temporal_observations: true
requires_extra_temporal_observations: true
uses_ground_truth_fields: true
uses_ground_truth_endpoint_fields: false
ground_truth_field_exception: near_endpoint_temporal_sparse_observations
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

In sampling, this mode is reserved for Burgers because its model output is already `[B,1,T,X]`. The lower-level residual API can still consume an explicit trajectory for offline diagnostics, but stored ground-truth trajectories are rejected as guidance/evaluation inputs for endpoint-output models.

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
- Reaction-Diffusion: FitzHugh-Nagumo RHS with Neumann Laplacian in explicit `[B,C,Y,X]` layout; `dx` differentiates the last axis and `dy` the penultimate axis
- Shallow Water: standard 2D conservative flux RHS with clamped safe depth
- Non-bounded Navier-Stokes: scalar-vorticity 2D Navier-Stokes residual on the periodic torus. It reconstructs velocity from vorticity with an FFT stream-function solve and uses the fixed forcing `f_NS(x,y)=0.1(sin(2*pi*(x+y))+cos(2*pi*(x+y)))` when no explicit forcing is supplied.

For Non-bounded Navier-Stokes, the state is vorticity `omega` on `Omega=[0,1)^2` with endpoint-false periodic grid points `x_i=i/W`, `y_j=j/H`. The RHS is:

```text
F(omega) = -v_x omega_x - v_y omega_y + nu Delta omega + f_NS
```

with `-Delta psi = omega` and `v=(partial_y psi, -partial_x psi)`. Guidance uses:

```text
r = partial_tau omega - F(omega)
  = partial_tau omega + v dot grad omega - nu Delta omega - f_NS
```

The default viscosity is `nu=1e-3`; explicit `nu`, `viscosity`, or `forcing` in PDE params override the corresponding defaults. The default `generator_dealiased` operator applies the same 2/3 Fourier projection as the data generator to the nonlinear product and forcing. `continuous_spectral` is an explicit diagnostic alternative. Non-finite values fail immediately rather than being clamped. Endpoint-only Hermite and endpoint-secant modes remain approximate; they are not upgraded using a stored ground-truth trajectory.

For Heat, Wave, and Advection-Diffusion generated by the pair-HDF5 spectral solvers, the stored spatial grid is endpoint-false periodic: `np.linspace(0, 1, resolution, endpoint=False)`. The first and last samples are adjacent grid cells, not duplicate samples of the same physical point. Periodic BCs are therefore enforced by the discrete operator: `torch.roll` stencils with spacing `1/N`, not by a first-minus-last value residual. Metadata records:

```yaml
boundary_condition_type: periodic
boundary_enforced: true
boundary_enforced_by_operator: true
boundary_value_residual_applicable: false
grid_convention: endpoint_false_periodic
```

Only datasets that explicitly store a duplicate periodic endpoint should set `periodic_duplicate_endpoint: true`; only then is first/last value comparison applicable. Dirichlet, Neumann, open, wall, and mixed BCs still use explicit boundary residual fields.

## Boundary and Initial Conditions

Current resolved boundary conditions:

- Heat: periodic for pair-HDF5 generated data by default; heat generation can also emit Neumann data, in which case metadata/config must set `boundary_condition_mode: neumann_zero`.
- Wave: periodic.
- Advection-Diffusion: periodic.
- Reaction-Diffusion: homogeneous Neumann from the finite-volume generator.
- Shallow Water: open/extrapolation boundary from the Clawpack radial dam-break generator.
- Burgers: periodic along the spatial axis of the BCHW time-space field.
- Darcy, Poisson, Helmholtz: zero Dirichlet for the static datasets/configs.
- Steady Heat Conduction: mixed; bottom Dirichlet `u=u_D`, top/left/right zero Neumann.
- Non-bounded Navier-Stokes: periodic torus. The PDE residual is enabled in scalar-vorticity form. Periodicity is represented by FFT derivatives and endpoint-false periodic operators, so no first/last value matching residual is added unless a dataset explicitly marks duplicate periodic endpoints. IC residuals are added only from observed or explicitly supplied initial fields.

The residual metadata records whether interior, BC, IC, and endpoint components are enabled; the boundary and initial condition types; whether each condition came from config, metadata, data, or a confirmed default; unresolved conditions; and whether `legacy_ignore_boundary` was used.

`legacy_ignore_boundary: true` or `boundary_condition_mode: legacy_ignore` disables explicit BC and IC residuals for old-result ablations. It does not disable endpoint consistency residuals such as the Hermite integral component.

Any experiment using extra temporal observations or full trajectory state must preserve the metadata fields above so endpoint-only results are not mixed with sparse-temporal or full-trajectory results.
