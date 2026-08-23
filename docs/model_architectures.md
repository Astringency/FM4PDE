# Model Architecture Profiles

FM4PDE uses a PDE-family architecture registry in `models/model_configs.py`.
Training uses `model_profile=auto` by default: new training resolves it to
`recommended`, while resume training first inspects checkpoint architecture
metadata and uses the saved profile/config when available. `recommended` is the
PDE-family registry; `light`, `base`, and `heavy` are explicit architecture
ablations.

| PDEs | architecture family |
| --- | --- |
| poisson | `light_smooth` |
| darcy, helmholtz, steady_heat_conduction | `elliptic_static` |
| heat, advection_diffusion, reaction_diffusion | `temporal_endpoint_base` |
| wave, shallow_water, nsnonbounded | `temporal_endpoint_heavy` |
| burger | `full_time_space` |

Recommended configs avoid high-resolution attention at downsample factor `2`.
Attention is placed at coarser factors such as `8` and `16` because attention
cost is quadratic in spatial tokens, while PDE global structure can still be
captured at coarse scales. This reduces memory pressure on 128x128 data.

## Fourier Features

`with_value_fourier_features` enables value Fourier features. Value Fourier
applies `sin` and `cos` to input field values:

```text
sin(omega * input_value), cos(omega * input_value)
```

`with_coordinate_fourier_features` enables coordinate Fourier features.
Coordinate Fourier applies `sin` and `cos` to deterministic grid coordinates,
not to normalized input values:

```text
sin(omega * y_coord), cos(omega * y_coord)
sin(omega * x_coord), cos(omega * x_coord)
```

The two features can be enabled together, but they mean different things:
value Fourier lifts input field values; coordinate Fourier lifts grid
coordinates. Coordinate channels are generated inside `UNetModel.forward` from
tensor shape/device/dtype and are not part of `PDEStandardizer`.

The value Fourier and `concat_conditioning` hooks come from or are adapted from
the official Flow Matching image example UNet. They are engineering hooks in
the velocity network, not part of the mathematical definition of the Flow
Matching objective. `concat_conditioning` currently requires an explicit channel
contract and raises a clear error if used without one.

| PDE | value Fourier | coordinate Fourier | reason |
| --- | --- | --- | --- |
| poisson | false | false | Smooth elliptic baseline does not need value or location lift by default. |
| heat | false | false | Endpoint heat fields stay in native channels by default. |
| darcy | false | true | Coefficient/solution fields benefit from explicit spatial coordinates. |
| helmholtz | true | true | Oscillatory values and spatial phase both benefit from Fourier lifts. |
| steady_heat_conduction | false | true | Inhomogeneous material/location effects benefit from grid coordinates. |
| advection_diffusion | false | true | Spatially resolved transport can benefit from location features. |
| reaction_diffusion | false | false | Multi-channel endpoint states remain scale-sensitive by default. |
| wave | true | true | Oscillatory phase/frequency structure benefits from both lifts. |
| shallow_water | false | false | Conservative variables stay in native channels initially. |
| nsnonbounded | true | false | Periodic vorticity benefits from value lift, while coordinate Fourier is off to preserve periodic translation-equivariance. |
| burger | true | true | Full time-space residual benefits from value lift and explicit time/space coordinates. |

Burgers is special: its BCHW tensor is a time-space field where `H` is time and
`W` is space. The coordinate H axis is the time coordinate and the coordinate W
axis is the space coordinate. The architecture metadata records:

```text
architecture_family = full_time_space
axis_semantics = BCHW_as_time_space_H_time_W_space
```

For `nsnonbounded`, coordinate Fourier is off by default to avoid
introducing absolute position channels that can break the intended periodic
translation-equivariance of the vorticity data. Value Fourier remains enabled as
a vorticity value lift. Coordinate Fourier can be tested later as an ablation
profile.

## Checkpoint Metadata

Training checkpoints save:

```text
model_profile
requested_model_profile
resolved_model_profile
resume_architecture_metadata
model_config
model_config_metadata
data_metadata
num_channels
checkpoint_schema_version
```

Sampling requires a schema 3 checkpoint and uses its `model_config` to
reconstruct the model. Missing architecture metadata, normalizer data, or plain
model weights is an error.
Training resume follows the same preflight idea: `auto` uses checkpoint
architecture metadata, explicit profile mismatches fail unless
`--allow_model_profile_override` is passed, and checkpoint `model_config`
channel counts must match the current training data.
