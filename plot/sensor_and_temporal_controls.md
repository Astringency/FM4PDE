# Observation layouts and temporal residual controls

`run_sensor_and_temporal_controls.py` uses the settings in
`sensor_and_temporal_controls.json` and prepared physical inputs under one
subdirectory per PDE.

- Layouts: Helmholtz input 0, four layouts, five paired seeds, shared and
  separate locations (40 runs). Fixed observations occupy the left half;
  separate Grid masks use a (3,3) translation; Columns uses five full columns.
- Temporal residuals: six PDEs, three residual modes, input/seed 0 (18 runs).
  Near-endpoint differences require additional sparse temporal observations.
  Both endpoint errors include every channel.
- Both studies use 100 stochastic Euler steps, batch size one and disabled TF32.

```bash
bash scripts/sampling/ablations/run_study.sh layouts \
  --inputs /path/to/prepared_inputs --output /absolute/path/to/layout_results
```

The ordinary temporal comparison, including the configured light NS model,
uses the shared ablation launcher:

```bash
bash scripts/sampling/ablations/run.sh temporal_residual_mode
```

`plot_sensor_and_temporal_controls.py` reads saved physical predictions and
masks, checks their field errors, and renders layout figures and temporal
comparison tables. `--render-only` uses previously collected numeric arrays.
Its archive-path constants retain the historical result locations so existing
manuscript assets remain readable. The plan's source paths, recorded hashes,
random seeds and numerical settings are preserved.
