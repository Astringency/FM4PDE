# Hybrid geometric grids

The existing `time_grid: geometric` setting now also supports `hybrid_d2s`
and `hybrid_s2d`. `time_grid_eta` is the manuscript geometric growth factor;
the switching experiment uses 0.05. There is no additional stage allocation
parameter.

Start with the full N-step grid `g[0] = 0`, `g[k] = (1 + eta) ** (k - N)`
for k = 1, ..., N. The switch is at flow time `switch_ratio`:

- D→S: retain the geometric nodes strictly before the switch, insert the
  switch exactly, then divide the remaining time uniformly over the remaining
  steps.
- S→D: retain the geometric nodes strictly after the switch, including 1,
  and use the remaining steps uniformly from 0 to the switch.

The step crossing the switch is shortened to end or begin at the switch.
Uniform-grid hybrids retain their original step partition
`round(switch_ratio * num_steps)`; at the three published switching times
and 100 steps, this is also the specified flow time.

| Switch time | Uniform D→S (D/S steps) | Geometric D→S (D/S steps) | Uniform S→D (S/D steps) | Geometric S→D (S/D steps) |
|---|---:|---:|---:|---:|
| 0.2 | 20/80 | 68/32 | 20/80 | 67/33 |
| 0.5 | 50/50 | 86/14 | 50/50 | 85/15 |
| 0.8 | 80/20 | 96/4 | 80/20 | 95/5 |

Each schedule has exactly 100 updates and 100 velocity evaluations. The
guidance weights, clipping, `stochastic_guidance_coeff: 0.1`, and PDE activation
at step k ≥ 80 are unchanged. Thus the flow time at which PDE guidance begins
can change with the grid. A budget that cannot contain both nonempty phases
raises an error instead of inserting extra steps.

Run the complete 28-setting comparison (14 per PDE) using the existing entry
point. Set `DATA_ROOT`, `CHECKPOINT_ROOT` or per-PDE checkpoint paths, and
`PYTHON_BIN` as described in the README.

```bash
bash scripts/sample/ablations/switching_time.sh --device cuda:0 --output outputs/switching
python -m plot.switching_time --inputs outputs/switching --output figures/ablations
python -m unittest discover -s tests -p test_hybrid_time_grid.py -v
```

For sharded runs, pass every shard output directory to `--inputs`. The renderer
requires all 28 settings and reads the saved predictions and terminal PDE
losses. Solid lines and filled markers show uniform hybrid grids; dashed lines
and open markers show geometric D phases. All three vertical axes are
logarithmic, with relative errors displayed as ratios. Actual times and phases
are also saved in each run's `curves.csv`.
