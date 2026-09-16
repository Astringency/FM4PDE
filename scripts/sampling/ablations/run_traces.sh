#!/usr/bin/env bash
set -euo pipefail
# Paired FM4PDE/DiffusionPDE error and time trajectories, using prepared inputs.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"
command=run
if [[ "${1:-}" =~ ^(run|prepare|verify|plot)$ ]]; then
  command="$1"
  shift
fi
case "${command}" in
  run) exec "${PYTHON_BIN:-python}" experiments/trajectories/run.py \
    --diffusion-root "${DIFFUSION_ROOT:-${ROOT}/../DiffusionPDE}" "$@";;
  prepare) exec "${PYTHON_BIN:-python}" experiments/trajectories/prepare.py "$@";;
  verify) exec "${PYTHON_BIN:-python}" experiments/trajectories/verify.py "$@";;
  plot) exec "${PYTHON_BIN:-python}" plot/plot_sampling_trajectories.py "$@";;
esac
