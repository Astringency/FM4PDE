# FM4PDE Ablation Framework

This framework replaces the monolithic `sample.py` sampling loop with a configurable runner under `fm4pde_ablation/`. The goal is to make FM4PDE ablations reproducible, batched, and auditable while keeping the existing checkpoint format, PDE data format, and legacy config files usable.

## Why These Ablations

FM4PDE sampling couples several design choices: observation guidance, PDE residual guidance, loss evaluation state, deterministic/stochastic sampling, time grids, sensor masks, observation noise, and gradient clipping. The old script encoded many of these through implicit `if` branches and hard-coded constants. The new config separates task semantics from guidance components so each variable can be measured independently.

## Default Main Method

The default main method is:

- `guidance_components: obs_pde`
- `loss_state: endpoint`
- `sampler_phase: stochastic`
- `guidance_schedule: constant`
- `clip_mode: global_norm`
- `clip_threshold: 1e10`
- `pde_residual_region: full`
- `time_grid: uniform`
- `step_method: euler`

`loss_state: endpoint` is the recommended default because deterministic and stochastic samplers now both expose an explicit endpoint prediction.

## Ablation Variables

- `task`: `forward`, `inverse`, `both`, or `unconditional`. This defines which side is semantically conditioned on.
- `guidance_components`: `noguide`, `obs_only`, `pde_only`, `obs_pde`, `coef_obs_only`, `sol_obs_only`, or `both_obs`.
- `loss_state`: `xt`, `x_next`, `endpoint`, or `denoised_endpoint`.
- `sampler_phase`: `deterministic`, `stochastic`, `hybrid_d2s`, or `hybrid_s2d`.
- `switch_ratio`: fraction of steps spent in the first hybrid phase.
- `guidance_schedule`: `constant`, `delta`, `bt`, `cosine`, `polynomial`, or `obs_decay`.
- `clip_mode`: `none`, `global_norm`, or `per_component_norm`.
- `pde_residual_region`: `full`, `observed`, `boundary_excluded`, or `union_obs`.
- `sensor_mode`: `random`, `fixed`, `grid`, `sensor_column`, or `time_varying`.
- `noise_level`: sparse observation noise scale, applied as `obs + noise_level * std(obs) * eps`.
- `time_grid`: `uniform`, `geometric`, or `cosine`.
- `step_method`: `euler` or `midpoint`.

## Smoke Test

```bash
python -m fm4pde_ablation.runner --config configs/ablations/smoke.yaml --dry-run
scripts/ablations/smoke.sh
```

In a full FM4PDE environment with `torch`, `numpy`, `scipy`, `h5py`, and `pyyaml` installed, remove `--dry-run` to load the checkpoint and run the Poisson 5-step smoke test. If the configured data path is unavailable, `allow_synthetic_data: true` creates deterministic synthetic tensors for code-path validation only.

## Running Ablation Groups

```bash
scripts/ablations/run_guidance_components.sh --dry-run
scripts/ablations/run_loss_state.sh --dry-run
scripts/ablations/run_sampler_phase.sh --dry-run
scripts/ablations/run_guidance_schedule.sh --dry-run
scripts/ablations/run_clipping.sh --dry-run
scripts/ablations/run_pde_residual_region.sh --dry-run
scripts/ablations/run_sensor_noise.sh --dry-run
scripts/ablations/run_steps_timegrid.sh --dry-run
```

For real experiments, omit `--dry-run` after confirming data paths and GPU device. To inspect expanded jobs without running:

```bash
python -m fm4pde_ablation.sweep --grid configs/ablations/all_ablation_grid.yaml --list
```

## Output Structure

Each run writes:

```text
outputs/ablations/{pde}/{task}/{ablation_name}/{timestamp}/
  resolved_config.yaml
  run_metadata.json
  metrics_step.jsonl
  metrics_final.json
  result.pt
  masks.pt
  curves.csv
  summary.csv
  legacy_results.pkl
  figures/
```

Every step records `loss_state`, phase, actual zeta values, `bt`, `clip_scale`, observation/PDE gradient norms, relative errors, sparse observation errors, PDE residual norm, and wall-clock time.

## Collecting Tables

```bash
scripts/ablations/collect_results.sh outputs/ablations outputs/ablations/summary_all.csv
```

Use `summary_all.csv` for paper tables. `curves.csv` and `metrics_step.jsonl` contain error-time curves and per-step diagnostics.

## PDE Residual Status

- Reliable: `poisson`, `darcy`, `helmholtz`
- Approximate: `burger`, `reaction_diffusion`, `shallow_water`
- Placeholder: `nsnonbounded`

Reaction-Diffusion and Shallow Water use two-time-level approximations. Shallow Water uses conservative variables `[h, hu, hv]`. The current NS residual remains the legacy approximation and should not be interpreted as a full Navier-Stokes residual.

## Adding New PDEs

To add Heat, Wave, or Advection-Diffusion:

1. Add channel counts and residual status in `fm4pde_ablation/registry.py`.
2. Add channel names and data extraction logic in `fm4pde_ablation/data.py`.
3. Add split rules if the pair is not an even channel split in `fm4pde_ablation/state.py`.
4. Add a residual field function in `fm4pde_ablation/pde_residuals.py`.
5. Add a legacy or ablation config under `configs/`.
6. Add a smoke config and one unit test for shape/residual behavior.
