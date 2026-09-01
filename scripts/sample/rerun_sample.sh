#!/usr/bin/env bash
set -euo pipefail

# Rerun only the six PDE/task settings changed by the round3 configs:
#   poisson inverse, helmholtz inverse, darcy inverse, nsnonbounded inverse,
#   helmholtz both, and darcy forward.
# Each setting is evaluated on id, smooth, and rough data.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

NUM_SAMPLES="${NUM_SAMPLES:-1000}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-50}"
NUM_STEPS="${NUM_STEPS:-100}"
NUM_OBS="${NUM_OBS:-500}"
SAMPLE_SEED="${SAMPLE_SEED:-0}"
CONFIG_DIR="${CONFIG_DIR:-configs/main}"
TEST_TYPE_LIST="${TEST_TYPE_LIST:-id smooth rough}"
OUTPUT_SUFFIX="${OUTPUT_SUFFIX:-tuned1}"
INVERSE_DEVICE_LIST="${INVERSE_DEVICE_LIST:-cuda:0 cuda:1}"
HELMHOLTZ_DEVICE="${HELMHOLTZ_DEVICE:-cuda:0}"
DARCY_DEVICE="${DARCY_DEVICE:-cuda:1}"
MAX_PARALLEL_TASKS="${MAX_PARALLEL_TASKS:-2}"
RESUME="${RESUME:-true}"

run_sweep() {
    local test_type="$1"
    local output_dir="$2"
    local pde_list="$3"
    local task_list="$4"
    local device_list="$5"
    local parallel="$6"
    local max_parallel="$7"
    local aggregate="$8"

    NUM_SAMPLES="${NUM_SAMPLES}" \
    MAX_BATCH_SIZE="${MAX_BATCH_SIZE}" \
    PDE_LIST="${pde_list}" \
    TASK_LIST="${task_list}" \
    SAMPLER_LIST="stochastic" \
    SENSOR_MODE_LIST="random" \
    NUM_STEPS="${NUM_STEPS}" \
    NUM_OBS="${NUM_OBS}" \
    TEST_TYPE="${test_type}" \
    SAMPLE_SEED="${SAMPLE_SEED}" \
    CONFIG_DIR="${CONFIG_DIR}" \
    OUTPUT_DIR="${output_dir}" \
    DEVICE_LIST="${device_list}" \
    PARALLEL="${parallel}" \
    MAX_PARALLEL_TASKS="${max_parallel}" \
    RESUME="${RESUME}" \
    VIS=false \
    AGGREGATE="${aggregate}" \
    bash scripts/sample/run_sample_sweep.sh
}

for test_type in ${TEST_TYPE_LIST}; do
    output_dir="outputs/MAIN1000_100_TEST_${test_type}_${OUTPUT_SUFFIX}"

    echo "[$(date '+%F %T')] Starting ${test_type} inverse settings -> ${output_dir}"
    run_sweep \
        "${test_type}" \
        "${output_dir}" \
        "poisson helmholtz darcy nsnonbounded" \
        "inverse" \
        "${INVERSE_DEVICE_LIST}" \
        true \
        "${MAX_PARALLEL_TASKS}" \
        false

    echo "[$(date '+%F %T')] Starting ${test_type} helmholtz/both"
    run_sweep \
        "${test_type}" \
        "${output_dir}" \
        "helmholtz" \
        "both" \
        "${HELMHOLTZ_DEVICE}" \
        false \
        1 \
        false

    echo "[$(date '+%F %T')] Starting ${test_type} darcy/forward"
    run_sweep \
        "${test_type}" \
        "${output_dir}" \
        "darcy" \
        "forward" \
        "${DARCY_DEVICE}" \
        false \
        1 \
        true

    echo "[$(date '+%F %T')] Completed ${test_type} settings"
done

echo "[$(date '+%F %T')] All tuned reruns completed."
