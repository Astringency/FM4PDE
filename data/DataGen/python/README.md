# FM4PDE Python Future PDE Data Generators

This directory contains self-contained NumPy/SciPy/HDF5 generators for four FM4PDE-compatible datasets:

- `heat`: time-dependent heat equation, `u_t = alpha Delta u`
- `wave`: time-dependent wave equation, `u_tt = c^2 Delta u`
- `advection_diffusion`: `u_t + b_x u_x + b_y u_y = kappa Delta u`
- `steady_heat_conduction`: RecFNO-style nonlinear steady-state heat conduction

`heat` and `steady_heat_conduction` are different PDEs. The former is time-dependent and solved spectrally by time evolution. The latter is steady-state, nonlinear, and solved by Picard iteration over sparse finite-difference systems.

## Storage vs Model Tensors

The HDF5 storage layer only stores real physical fields as `[N,C,H,W]`. Spatially constant per-sample parameters are stored as scalar datasets `[N]`; globally fixed constants are stored as root attributes. By default the generators do not write constant parameter fields.

`data/load.py` materializes scalar parameters into constant `[N,1,H,W]` channels when constructing FM4PDE tensors. This keeps files smaller while preserving the even-channel input/output split expected by FM4PDE.

Use `--materialize-constant-fields` only for debugging or old-code inspection. It writes `materialized_data`; the loader still prefers reconstructing from `input_data`, `output_data`, scalar datasets, and attrs.

## PDE Schemas

### Heat

Equation:

```text
u_t = alpha Delta u,  (x,y) in [0,1]^2
```

Default boundary condition is periodic. Fourier wavenumbers use physical angular frequencies: `2*pi*np.fft.fftfreq(s, d=1/s)`. `--bc neumann` uses a DCT spectral solve; the current Neumann mode still samples the initial field with the periodic GRF sampler, so it should be treated as an approximate initial-condition choice.

Default `--alpha-mode random`:

- HDF5: `input_data=[u0]`, `output_data=[uT]`, `alpha=[N]`, optional `full_trajectory=[N,1,T,H,W]`
- Loader tensor: `[u0, alpha_field, uT, alpha_field]`
- Config: `configs/heat.yaml`, `img_channels=4`

Fixed-alpha mode:

```bash
python data/DataGen/python/generate_future_pdes.py --pde heat --alpha-mode fixed --alpha 0.001 ...
```

- HDF5 attrs: `fixed_alpha`, `alpha_random=False`
- Loader tensor: `[u0, uT]`
- Config: `configs/heat_fixed.yaml`, `img_channels=2`

### Wave

Equation:

```text
u_tt = c^2 Delta u
u(x,y,0)=u0,  u_t(x,y,0)=v0
```

Default `c` is fixed and stored as `fixed_c` in HDF5 attrs. The exact periodic spectral formula records both terminal displacement and velocity:

```text
u_hat(t,k) = u0_hat cos(c|k|t) + v0_hat sin(c|k|t)/(c|k|)
v_hat(t,k) = -c|k| u0_hat sin(c|k|t) + v0_hat cos(c|k|t)
```

The zero mode uses `u_hat_0(t)=u0_hat_0+t*v0_hat_0`, `v_hat_0(t)=v0_hat_0`.

- HDF5 default: `input_data=[u0,v0]`, `output_data=[uT,vT]`, fixed `c` attrs, optional `full_trajectory` containing only `u`
- Loader tensor fixed `c`: `[u0, v0, uT, vT]`
- Optional scalar random `c`: `--c-mode random` writes `c=[N]`; loader tensor is `[u0, v0, c_field, uT, vT, c_field]`
- `--variable-c` currently raises `NotImplementedError`; it does not use the Fourier exact formula

### Advection-Diffusion

Equation:

```text
u_t + b_x u_x + b_y u_y = kappa Delta u
```

Periodic spectral evolution:

```text
u_hat(t) = exp(-(kappa*(kx^2+ky^2) + i*(b_x*kx+b_y*ky))*t) * u0_hat
```

- HDF5: `input_data=[u0]`, `output_data=[uT]`, `b_x=[N]`, `b_y=[N]`, `kappa=[N]`, optional `full_trajectory`
- Loader tensor: `[u0, b_x_field, b_y_field, kappa_field, uT, b_x_field, b_y_field, kappa_field]`
- Config: `configs/advection_diffusion.yaml`, `img_channels=8`

### RecFNO-Style Steady Heat Conduction

Equation:

```text
-div(lambda(u) grad u) = f(x,y)
lambda(u) = 1 + 0.05*(u - 298)
```

Boundary conditions:

- Bottom: Dirichlet `u = u_D`
- Top/left/right: zero Neumann

The source `f` is a sum of Gaussian heat sources. `u_D` is sampled per sample. The solver uses Picard iteration:

```text
lambda^m = 1 + 0.05*(u^m - 298)
solve -div(lambda^m grad u^{m+1}) = f
```

Conductivity is clamped below by `0.1` to keep the elliptic system positive.

- HDF5: `input_data=[f]`, `output_data=[u]`, `u_D=[N]`, `picard_iters=[N]`, `converged=[N]`, `residual_norm=[N]`, padded source parameter arrays
- Loader tensor: `[f, u_D_field, u, u_D_field]`
- Config: `configs/steady_heat_conduction.yaml`, `img_channels=4`

This is a RecFNO-style dataset shape and PDE family. It is not a bitwise reproduction of any external RecFNO data file.

## File Layout

Default full generation writes:

```text
<DATA_ROOT>/heat/heat_10000-128-128_1.h5
...
<DATA_ROOT>/heat/heat_test_1000-128-128.h5

<DATA_ROOT>/wave/wave_10000-128-128_1.h5
...
<DATA_ROOT>/wave/wave_test_1000-128-128.h5

<DATA_ROOT>/advection_diffusion/advection_diffusion_10000-128-128_1.h5
...
<DATA_ROOT>/advection_diffusion/advection_diffusion_test_1000-128-128.h5

<DATA_ROOT>/steady_heat_conduction/steady_heat_conduction_10000-128-128_1.h5
...
<DATA_ROOT>/steady_heat_conduction/steady_heat_conduction_test_1000-128-128.h5
```

RecFNO-style small split:

```bash
python data/DataGen/python/generate_future_pdes.py \
  --pde steady_heat_conduction \
  --out-root <DATA_ROOT> \
  --recfno-split \
  --n-train 4000 \
  --n-val 1000 \
  --n-test 1000
```

## Commands

Full default FM4PDE generation:

```bash
python data/DataGen/python/generate_future_pdes.py \
  --pde all \
  --out-root <DATA_ROOT> \
  --resolution 128 \
  --n-train 50000 \
  --n-test 1000 \
  --train-shards 5 \
  --n-time 11 \
  --overwrite
```

CPU quick test:

```bash
python data/DataGen/python/generate_future_pdes.py \
  --pde all \
  --out-root /tmp/fm4pde_future_pdes \
  --resolution 16 \
  --n-train 8 \
  --n-test 4 \
  --train-shards 2 \
  --n-time 5 \
  --quick-test \
  --overwrite
python data/DataGen/python/test_future_pdes.py
python -m compileall -q data
```

Use `--no-trajectory` to reduce file size. Full trajectories at `50000 x 128 x 128 x 11` are much larger than endpoint-only data.

## Datacheck Reaction-Diffusion

`datacheck_generate.py` also writes a small FM4PDE-compatible 2D
reaction-diffusion dataset using:

```text
data/DataGen/time_dependent/pdebench/data_gen/src/sim_diff_react.py
```

The equation is:

```text
u_t = u - u^3 - k - v + Du * Delta u
v_t = u - v + Dv * Delta v
```

Reaction-diffusion uses `--n-save-steps` rather than `--n-time`. The default is
10 saved intervals on `[0, 1]`, so `tdim = n_save_steps + 1 = 11` saved nodes,
including the initial state. The default initial field is `--reaction-diffusion-init-mode grf`;
set it to `iid` to reproduce the older per-grid-point standard normal initial
condition. `solve_ivp` uses adaptive RK45, so the saved interval `0.1` is not a
fixed internal solver step.

## No-Leakage Design

- Train base seed defaults to `0`; validation base seed defaults to `5000000`; test base seed defaults to `10000000`.
- Sample seed is always `base_seed + global_sample_id`.
- Train, validation, and test are generated independently, never by slicing one pre-generated array.
- Shard identity does not affect sampling because `global_sample_id = shard_start + local_id`.
- Each PDE directory gets `no_leakage_check.json`; all-PDE runs also write a root-level `no_leakage_check.json`.

## FM4PDE Loading

Training data can be loaded with:

```python
from data.load import PDEloader

data, label = PDEloader("heat").load_data("<DATA_ROOT>/")
```

The loader reads train shards from `<DATA_ROOT>/<pde>/`. For sampling configs, use `loadby: future_h5`, `coef: input_data`, and `solution: output_data`; the data interface materializes scalar channels before splitting into coefficient and solution tensors.

## External References

PDEBench, iFNO, and RecFNO were used only as references for HDF5/trajectory organization and operator-learning data shape. These generators do not depend on their code, solvers, or data files.
