# FM4PDE Ablation Framework

`fm4pde_ablation.runner` is the official FM4PDE sampling and internal ablation entrypoint. `sample.py` is only a legacy compatibility wrapper: it maps old CLI arguments into `AblationConfig`, stores ignored legacy knobs such as `dt_sampler`, `lr_decay`, `freq_decay`, `perturb`, and `perturb_rate` under `config.extra`, prints a deprecation warning, and delegates to the runner.

The sweep entrypoint is `python -m fm4pde_ablation.sweep`. It expands grouped internal ablation grids without mixing in any external method.

## Main Defaults

- `guidance_components: obs_pde`
- `loss_state: endpoint`
- `sampler_phase: stochastic`
- `guidance_schedule: constant`
- `clip_mode: global_norm`
- `clip_threshold: 1e10`
- `pde_residual_region: full`
- `time_grid: uniform`
- `step_method: euler`

Supported `loss_state` values are `xt`, `x_next`, and `endpoint`. `denoised_endpoint` is accepted only for compatibility and maps to `endpoint` with a warning because there is no separate denoising process.

Supported sampler phases are `deterministic`, `stochastic`, `hybrid_d2s`, and `hybrid_s2d`.

Supported sensor modes are `random`, `fixed`, `grid`, `sensor_column`, and `per_sample_random`. `random` samples one spatial mask and shares it across the whole batch; `per_sample_random` samples an independent spatial mask for each batch item. The old `time_varying` name is a legacy alias for `per_sample_random` and is not a formal mode in grids.

`obs_only` means observation guidance on the side visible for the current task: coefficient observations for `forward`, solution observations for `inverse`, and both sides for `both`. `both_obs` is an explicit two-sided observation setting and is only valid for `task: both`; it is omitted from the main guidance grid to avoid duplicating `obs_only` under `task: both`.

## Future PDE Metadata

For Heat, Wave, Advection-Diffusion, and Steady Heat Conduction, spatially constant PDE parameters are not Flow Matching channels. The FM tensors contain only physical spatial fields:

- Heat: `[u0, uT]`
- Wave: `[u0, v0, uT, vT]`
- Advection-Diffusion: `[u0, uT]`
- Steady Heat Conduction: `[f, u]`

Scalar or sample-level parameters are loaded into `PDEGroundTruth.pde_params` and passed to PDE residuals:

- Heat: `alpha`
- Wave: `c`
- Advection-Diffusion: `b_x`, `b_y`, `kappa`
- Steady Heat Conduction: `u_D` plus available sample metadata such as source parameters and solver diagnostics

For Heat, Wave, and Advection-Diffusion residuals, the two-time-level derivative uses `(uT - u0) / T`. The residual code reads `T`, then `total_time`, then `dt` from `pde_params`; if none is present it uses `T=1.0` and records that default in residual metadata.

Steady Heat Conduction residual fields include the interior PDE residual plus boundary residuals in the same tensor: bottom Dirichlet `u[..., 0, :] - u_D`, and zero-Neumann residuals on top, left, and right boundaries.

If a checkpoint still expects old scalar-parameter channels, sampling fails with a channel mismatch and the checkpoint must be retrained under the current channel definition.

## Residual Status

- `reliable`: Darcy, Poisson, Helmholtz
- `approximate`: Burgers, Reaction-Diffusion, Shallow Water, Heat, Wave, Advection-Diffusion, Steady Heat Conduction
- `placeholder`: reserved for PDEs with documented but inactive residual plans
- `disabled`: no PDE guidance should be claimed or used

`nsnonbounded` PDE guidance is currently disabled. If a config requests `pde_only`, the runner warns and maps it to `noguide`; if it requests `obs_pde`, the runner warns and maps it to `obs_only`. The original request and effective guidance are recorded in metadata so NS runs are not interpreted as PDE-guided results.

## Commands

Smoke:

```bash
python -m fm4pde_ablation.runner --config configs/ablations/smoke.yaml --dry-run
scripts/ablations/smoke.sh --dry-run
```

List the formal grouped internal grid:

```bash
python -m fm4pde_ablation.sweep --grid configs/ablations/all_internal_ablation_grid.yaml --list
```

Run one selected group:

```bash
python -m fm4pde_ablation.sweep --grid configs/ablations/all_internal_ablation_grid.yaml --group zeta_sensitivity
```

Aggregate results:

```bash
scripts/ablations/aggregate_results.sh outputs/ablations outputs/ablations
```

This writes `summary_all_raw.csv`, `summary_all_grouped.csv`, and `curves_grouped.csv`.

Grouped summaries use stable `ablation_family` and `ablation_group_key` fields instead of `ablation_name`. Seeds, sample offsets, and batch size remain in `summary_all_raw.csv` only, so seed/offset repeats such as `statistics_seed_offset` aggregate into one grouped row.

## Outputs

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

`run_metadata.json` records git commit, checkpoint path, data path, data config path, PDE/task, channel names, loaded PDE parameter keys, seeds, offset/batch size, guidance settings, sampler settings, residual status, device, dtype, and torch/cuda versions.

`result.pt` stores final coefficient/solution fields, ground truth fields, masks, `pde_params`, resolved config, metrics, and optional intermediate sampler states.

Every step records `t`, `t_next`, `step_size`, phase, loss state, actual zeta values, guidance schedule factor, `bt`, gradient norms, `clip_scale`, observation losses, PDE residual norm, relative errors, sparse observation errors, and wall-clock time.
