#!/usr/bin/env bash
set -euo pipefail
# Existing study implementations; --help after the study prints its arguments.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"
study="${1:?Choose ensemble, guidance, averaging, layouts, weights, prior or traces}"
shift
case "${study}" in
  ensemble) script=plot/run_paper_ablation_revision.py; mode=(ensemble);;
  guidance) script=plot/run_revision_sampling.py; mode=(run);;
  averaging) script=plot/run_conditional_sample_scaling.py; mode=();;
  layouts) script=plot/run_shared_sensors_0912.py; mode=();;
  weights) script=plot/run_guidance_weight_0910.py; mode=();;
  prior) script=plot/run_prior.py; mode=();;
  traces) exec bash scripts/sampling/ablations/run_traces.sh "$@";;
  *) echo "Unknown study: ${study}" >&2; exit 2;;
esac
exec "${PYTHON_BIN:-python}" "${script}" "${mode[@]}" "$@"
