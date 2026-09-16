#!/usr/bin/env bash
set -euo pipefail
# Main comparisons: four endpoint/static PDEs and the two Burgers layouts.
# DATA_ROOT, CHECKPOINT_ROOT or CHECKPOINT_<PDE> locate external assets.
# PLAN_ONLY=true prints the jobs without sampling.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/main}"
read -r -a DISTRIBUTIONS <<< "${TEST_TYPE_LIST:-id smooth rough}"
for distribution in "${DISTRIBUTIONS[@]}"; do
  TEST_TYPE="${distribution}" TASK_LIST="forward inverse both" NUM_OBS=500 \
    OUTPUT_DIR="${OUTPUT_ROOT}/sparse_${distribution}" \
    bash "${HERE}/run_sample_sweep.sh"
  TEST_TYPE="${distribution}" TASK_LIST="forward inverse" NUM_OBS=16384 \
    OUTPUT_DIR="${OUTPUT_ROOT}/full_${distribution}" \
    bash "${HERE}/run_sample_sweep.sh"
  TEST_TYPE="${distribution}" OUTPUT_DIR="${OUTPUT_ROOT}/burger_${distribution}" \
    bash "${HERE}/run_sample_sweep_burger.sh"
done
