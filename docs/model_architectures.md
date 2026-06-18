# Model Architecture Profiles

FM4PDE uses a PDE-family architecture registry in `models/model_configs.py`.
The default profile is `recommended`; `light`, `base`, and `heavy` are explicit
architecture ablations; `legacy_base` is only for reproducing old checkpoints.

| PDEs | architecture family |
| --- | --- |
| poisson, heat | `light_smooth` |
| darcy, helmholtz, steady_heat_conduction | `elliptic_static` |
| advection_diffusion, reaction_diffusion | `temporal_endpoint_base` |
| wave, shallow_water, nsnonbounded | `temporal_endpoint_heavy` |
| burger | `full_time_space` |

Recommended configs avoid high-resolution attention at downsample factor `2`.
Attention is placed at coarser factors such as `8` and `16` because attention
cost is quadratic in spatial tokens, while PDE global structure can still be
captured at coarse scales. This reduces memory pressure on 128x128 data.

Burgers is special: its BCHW tensor is a time-space field where `H` is time and
`W` is space. The residual is a full time-space finite-difference residual, so
the architecture metadata records:

```text
architecture_family = full_time_space
axis_semantics = BCHW_as_time_space_H_time_W_space
```

Sample-level scalar PDE parameters are loaded, saved in metadata, and used by
residuals. Network FiLM scalar conditioning is a planned extension in this
patch: architecture metadata records `scalar_conditioning=false` and
`scalar_conditioning_params`, but scalar parameters are not materialized as
constant input fields.

Training checkpoints save:

```text
model_profile
model_config
model_config_metadata
data_metadata
num_channels
checkpoint_schema_version
```

Sampling first uses the checkpoint `model_config` to reconstruct the model. If a
checkpoint lacks `model_config`, pass an explicit `model_profile` such as
`legacy_base`; otherwise the loader raises a clear architecture mismatch error.
