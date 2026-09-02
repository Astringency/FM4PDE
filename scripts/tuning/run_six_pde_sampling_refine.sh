#!/usr/bin/env bash
set -euo pipefail

# Second-round local search for the six newer PDEs.  It starts from the
# currently promoted configs/main/{task}/{pde}.yaml parameters, favors rough
# generalization during selection, and validates the winner on disjoint
# offsets.  All artifacts live in a new directory, so round-one results remain
# untouched.
#
# Typical two-GPU run:
#   PDE_DATA_ROOT=/large_storage/zhangxf/PDEdata \
#   DEVICE_LIST="cuda:0 cuda:1" MAX_PARALLEL_TASKS=2 \
#     bash scripts/tuning/run_six_pde_sampling_refine.sh
#
# Inspect the exact job count without data or GPUs:
#   PLAN_ONLY=true bash scripts/tuning/run_six_pde_sampling_refine.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export OUTPUT_DIR="${OUTPUT_DIR:-outputs/tuning/six_pde_sampling_refine}"
export PROFILE="${PROFILE:-standard}"
export CANDIDATE_SET="refined"
export DISTRIBUTION_WEIGHTS="${DISTRIBUTION_WEIGHTS:-id=1,smooth=1,rough=2}"
export TEST_TYPE_LIST="${TEST_TYPE_LIST:-id smooth rough}"
export TUNE_OFFSET_LIST="${TUNE_OFFSET_LIST:-100}"
export HOLDOUT_OFFSET_LIST="${HOLDOUT_OFFSET_LIST:-7000 9000}"
export BATCH_SIZE="${BATCH_SIZE:-2}"
export NUM_STEPS="${NUM_STEPS:-100}"
export RESUME="${RESUME:-true}"
export SKIP_KNOWN_NONFINITE="${SKIP_KNOWN_NONFINITE:-true}"

exec bash "${SCRIPT_DIR}/run_six_pde_sampling_tuning.sh"
