#!/usr/bin/env bash
set -euo pipefail
# Existing study implementations; --help after the study prints its arguments.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"
study="${1:?Choose ensemble, guidance, averaging, layouts, weights, prior or traces}"
shift
case "${study}" in
  ensemble) script=plot/run_ablation_study.py; mode=(ensemble);;
  guidance) script=plot/run_guidance_comparison.py; mode=(run);;
  averaging) script=plot/run_conditional_sample_scaling.py; mode=();;
  layouts) script=plot/run_sensor_and_temporal_controls.py; mode=(layouts);;
  weights) script=plot/run_guidance_weight_sweep.py; mode=();;
  prior) script=plot/run_prior.py; mode=();;
  traces) exec bash scripts/sampling/ablations/run_traces.sh "$@";;
  *) echo "Unknown study: ${study}" >&2; exit 2;;
esac
exec "${PYTHON_BIN:-python}" "${script}" "${mode[@]}" "$@"
