# FM4PDE Python Future PDE Data Generators

This directory contains self-contained NumPy/HDF5 generators for three FM4PDE-compatible operator-learning datasets:

- `heat`: `u_t = alpha Delta u`
- `wave`: `u_tt = c^2 Delta u`
- `advection_diffusion`: `u_t + b_x u_x + b_y u_y = kappa Delta u`

All default problems use the periodic domain `[0, 1]^2` on a `128 x 128` grid. Heat also exposes `--bc neumann` through a DCT spectral solver. Initial conditions are smooth periodic Gaussian random fields. Train and test splits are generated independently with non-overlapping seed ranges.

## File Layout

Default full generation writes:

```text
<DATA_ROOT>/heat/heat_10000-128-128_1.h5
...
<DATA_ROOT>/heat/heat_10000-128-128_5.h5
<DATA_ROOT>/heat/heat_test_1000-128-128.h5

<DATA_ROOT>/wave/wave_10000-128-128_1.h5
...
<DATA_ROOT>/wave/wave_10000-128-128_5.h5
<DATA_ROOT>/wave/wave_test_1000-128-128.h5

<DATA_ROOT>/advection_diffusion/advection_diffusion_10000-128-128_1.h5
...
<DATA_ROOT>/advection_diffusion/advection_diffusion_10000-128-128_5.h5
<DATA_ROOT>/advection_diffusion/advection_diffusion_test_1000-128-128.h5
```

## HDF5 Keys

Each file contains:

- `input_data`: `[N, C_in, H, W]`
- `output_data`: `[N, C_out, H, W]`
- `data`: `[N, C_in + C_out, H, W]`, directly loadable by `data/load.py`
- `full_trajectory`: optional `[N, 1, T, H, W]`
- `x`, `y`, `t`
- `sample_id`, `sample_seed`
- PDE parameter datasets such as `alpha`, `c`, `b_x`, `b_y`, `kappa`
- root attributes with equation, boundary condition, channel names, seed range, git commit if obtainable, and no-leakage notes

Channel definitions:

- Heat: input `[u0]`, output `[uT]`, `data=[u0,uT]`
- Wave default: input `[u0,v0]`, output `[uT,vT]`, `data=[u0,v0,uT,vT]`
- Advection-diffusion: input `[u0,b_x,b_y,kappa]`, output `[uT,b_x,b_y,kappa]`

The even channel counts keep FM4PDE's current pair split protocol intact.

## Commands

Full default generation:

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
python -m compileall -q data
```

Use `--no-trajectory` to reduce file size. Full trajectories at `50000 x 128 x 128 x 11` are much larger than endpoint-only data.

## No-Leakage Design

- Train base seed defaults to `0`; test base seed defaults to `10000000`.
- Sample seed is always `base_seed + global_sample_id`.
- Train and test are generated as separate splits, never by slicing one pre-generated array.
- Shard identity does not affect sampling because `global_sample_id = shard_start + local_id`.
- Each PDE directory gets `no_leakage_check.json`, and all-PDE runs also write a root-level `no_leakage_check.json`.

## FM4PDE Loading

Training data can be loaded with:

```python
from data.load import PDEloader

data, label = PDEloader("heat").load_data("<DATA_ROOT>/")
```

The loader reads train shards from `<DATA_ROOT>/<pde>/`. For sampling configs, use `loadby: future_h5`, `coef: input_data`, and `solution: output_data` with the test HDF5 file path.

## External References

PDEBench, iFNO, and RecFNO were used only as references for HDF5/trajectory organization and operator-learning data shape. These generators do not depend on their code, solvers, or data files.

