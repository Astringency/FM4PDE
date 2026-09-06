# PDE guidance schedule and loss comparison

The sampling loss supports `pde_guidance_reduction: rms` in addition to the default `mse`. RMS changes only the optimization objective; `L_pde`, `eval_L_pde`, physical residual metrics, and reconstruction metrics retain their existing definitions.

For sample b and component c, RMS is `||r[b,c]||₂ / sqrt(N[b,c])`, where N counts active entries (including broadcast channels). Each component is averaged across samples, then interior, boundary, and endpoint losses are summed using the existing component weights. This is an unsquared L2 objective, distinct from MAE and the historical `legacy_l2_mean` objective `||r||₂ / N`. The norm implementation has a finite zero subgradient at exact zero residual.

Use the current guidance operator with this setting:

```yaml
pde_guidance_reduction: rms
clip_mode: global_norm
clip_threshold: 50.0
```

Clipping applies to each sample's **weighted total** observation-plus-PDE gradient. For batch size 2, the recorded whole-batch gradient norm can reach `50 * sqrt(2)` while each sample remains bounded by 50. `grad_norm_pde` records the component gradient before total clipping and weighting; it need not be below 50.

The current main profiles evaluate guidance on the endpoint prediction. The curve's `guidance_L_pde` records the optimization objective there; `L_pde` / `eval_L_pde` evaluates the generated sampling state. A large initial evaluation loss is not by itself evidence of a large endpoint guidance gradient.

## Paired experiment

Run the MSE schedule comparison, then the RMS comparison on all five PDEs' both tasks:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python scripts/tuning/compare_pde_guidance_schedules.py --root /tmp/pde_mse --pde-loss mse --clip-threshold 50
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python scripts/tuning/compare_pde_guidance_schedules.py --root /tmp/pde_rms --pde-loss rms --clip-threshold 50 --tasks both
```

The runner reads main configurations without changing them. It requires the existing real ID test artifacts under `outputs/main/MAIN1000_100_TEST_id` and canonical pretrained checkpoints. Saved ground-truth fields are reused for efficient data access; historical predictions are not used in the comparison. `--source-root` can locate a corresponding source artifact directory. Both commands must use the same seed and sample-count options.

Each candidate uses 100 uniform stochastic steps, 500 random noiseless observations, batch size 2, and per-sample gradient clipping at 50. Three schedules are compared:

- `late_hard`: start 0.8, ramp 0; 20 active steps.
- `late_ramp`: start 0.8, ramp 0.1; 19 positive-weight steps, reaching full weight at flow time 0.9.
- `always_on`: start 0, ramp 0; 100 active steps.

The current main PDE weight and 0.1, 1, 10 are deduplicated and tested separately per schedule. Four tuning IDs and eight disjoint holdout IDs are drawn once using seed 20260906. Each schedule selects its weight on tuning error alone. The comparison between loss forms additionally selects the schedule on tuning error alone; holdout error is not used to select the reported cross-loss winner. All paired candidates share initial-noise seeds, observation masks, targets, models, observation weights, and clipping.

Primary error is relative L2 of u for forward, a for inverse, and the arithmetic mean of a and u errors for both. Burger uses its full predicted solution field. MSE covers all available main tasks (13 combinations); the added RMS comparison covers both for all five PDEs. Inspect separate component errors and final residuals as well as the primary metric.

## Analysis

```bash
python scripts/analysis/analyze_pde_guidance_schedules.py --root /tmp/pde_mse
python scripts/analysis/analyze_pde_guidance_schedules.py --root /tmp/pde_rms
python scripts/analysis/compare_pde_loss_forms.py --mse-root /tmp/pde_mse --rms-root /tmp/pde_rms --output-root /tmp/pde_comparison
```

These commands independently aggregate per-sample metrics with SQLite and audit target/mask/seed pairing, loss-form labels, schedule activation and clipping bounds. Outputs include `loss_comparison.csv` (fixed schedule, separately tuned weights), `loss_tune_selected.csv` (schedule and weight chosen on tuning data), raw per-sample metrics, audit records, and canonical report input. Bootstrap intervals are paired across the eight holdout samples, with no multiplicity correction. Results are exploratory ID-only evidence; a difference below 1% is described as near tied, not statistically equivalent.
