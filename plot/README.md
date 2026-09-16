# Experiment and plotting tools

Use the shell entry points under `scripts/sampling/` for main sampling and
ablations. This directory contains their Python implementations, supporting
studies, result calculations and figure exporters. Run a script with `--help`
for its input and output arguments.

Names describe the task: `prepare_` prepares inputs, `run_` executes a study,
`select_` chooses parameters from development results, `collect_` and `export_`
assemble saved results, and `plot_` renders figures.

## Experiment runners

| Task | Python entry | Shell entry |
| --- | --- | --- |
| Single-input ablations and repeated draws | `run_ablation_study.py` | `run_study.sh ensemble` |
| Development-set parameter selection | `select_ablation_parameters.py` | Called by `run_ablation_queue.py` |
| Sequential ablation and ensemble jobs | `run_ablation_queue.py` | Direct Python entry |
| Guidance and sampler comparisons | `run_guidance_comparison.py` | `run_study.sh guidance` |
| Physical-guidance weight selection | `run_guidance_weight_sweep.py` | `run_study.sh weights` |
| Four layouts with shared/separate locations; temporal residuals | `run_sensor_and_temporal_controls.py` | `run_study.sh layouts`; ordinary temporal sweeps use `run.sh temporal_residual_mode` |
| Shared versus independent random locations | `run_shared_observations.py` | Direct Python entry |
| Conditional averages from a fixed pool of predictions | `run_conditional_sample_scaling.py` | `run_study.sh averaging` |
| One unconditional sample | `run_prior.py` | `run_study.sh prior` |
| Unconditional samples and model profiles across PDEs | `run_unconditional_profiles.py` | Direct Python entry |
| Navier–Stokes evaluation with saved observations | `run_ns_evaluation.py` | Direct Python entry |
| Controlled FM4PDE/DiffusionPDE timing | `run_diffusion_fm_timing.py` | Direct Python entry |
| Matched inverse predictions for timing and frequency analysis | `run_matched_timing.py` | Direct Python entry |

Shell names in this table refer to `scripts/sampling/ablations/`. Prepared study
inputs contain physical fields, masks, model weights and the selected settings.
Each runner documents its required format through its arguments and checks.
The model-selection options for repeated studies are described in the root
README. The complete layout and temporal controls are described in
[sensor_and_temporal_controls.md](sensor_and_temporal_controls.md); their
settings are stored in `sensor_and_temporal_controls.json`.

`prepare_diffusion_comparison.py` extracts Smooth inputs and optional metrics
from existing DiffusionPDE results. The timing comparison also requires the
DiffusionPDE repository and its trained models. `run_matched_timing.py` uses the
FM4PDEbaseline repository for the supervised comparison methods.

## Result calculations and figures

| Task | Scripts |
| --- | --- |
| Ablation fields and tables | `collect_paper_ablation_fields.py`, `export_paper_ablation_tables.py`, `plot_paper_ablation_fields.py` |
| Sampling budgets, density, noise and guidance | `plot_ablation_overview.py`, `plot_paper_ablation_sweeps.py` |
| Reconstructed fields, guidance states and switching times | `plot_reconstruction_controls.py` |
| Complete layout and temporal controls | `plot_sensor_and_temporal_controls.py` |
| Repeated-draw statistics | `export_paper_seed_ensemble.py`, `plot_paper_seed_ensemble.py` |
| Conditional sample averages | `export_conditional_sample_scaling.py`, `plot_conditional_sample_scaling.py` |
| Frequency metrics and tables | `export_matched_spectra.py`, `summarize_spectral_evidence.py`, `export_frequency_tables.py` |
| Frequency comparison figures | `plot_frequency_comparison.py` |
| Training histories | `plot_training_curves.py` |
| Unconditional field galleries | `plot_unconditional_samples.py` |
| FM4PDE/DiffusionPDE timing | `export_diffusion_fm_timing.py` |
| Sampling error/time trajectories | `plot_sampling_trajectories.py` |

For the frequency comparison, the study root contains `inputs_v2/` and
`results_v3/` produced by the matched inverse study:

```bash
python plot/export_matched_spectra.py \
  --study /path/to/frequency_study --output /path/to/frequency_report
python plot/summarize_spectral_evidence.py --report /path/to/frequency_report
```

The reconstruction and complete-control renderers retain the paths of their
saved manuscript datasets. Renaming source files does not rename existing
checkpoints, result directories, result keys or manuscript assets. Set the
script's input/output arguments, or its documented archive-path constants,
when using a different installation. The renderers use Times New Roman;
`FM4PDE_FONT_DIR` can point to the installed font files.
