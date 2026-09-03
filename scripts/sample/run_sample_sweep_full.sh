#!/usr/bin/env bash
set -euo pipefail
# ============================================================================
# Full-observation MAIN1000_100_TEST sweep (non-Burgers PDEs only)
#
# Runs Poisson, Helmholtz, Darcy, and non-bounded Navier--Stokes for forward,
# inverse, and both tasks on the id, rough, and smooth test distributions.
# With the current 128 x 128 fields, NUM_OBS=16384 produces a full spatial
# observation mask.  Burgers is intentionally excluded.
#
# Examples:
#   bash scripts/sample/run_sample_sweep_full.sh
#   PLAN_ONLY=true bash scripts/sample/run_sample_sweep_full.sh
#   TEST_TYPE_LIST="id rough" DEVICE_LIST="cuda:0 cuda:1" \
#     bash scripts/sample/run_sample_sweep_full.sh
#
# Main overrides:
#   TEST_TYPE_LIST       Space-separated test distributions (default: id rough smooth)
#   NUM_SAMPLES          Samples per PDE/task (default: 1000)
#   NUM_OBS              Observations per active field (default: 16384)
#   OUTPUT_ROOT          Parent output directory (default: outputs/main)
#   OUTPUT_TAG           Output directory prefix (default: MAIN1000_100_TEST_FULL)
#   OUTPUT_SUFFIX        Optional suffix after the test type (default: empty)
#   DEVICE_LIST          Devices passed to run_sample_sweep.sh (default: cuda:0 cuda:1)
#   PLAN_ONLY            Print plans without sampling (default: false)
#
# Remaining sweep controls such as MAX_BATCH_SIZE, PARALLEL,
# MAX_PARALLEL_TASKS, RESUME, VIS, DRY_RUN, and AGGREGATE are forwarded to
# run_sample_sweep.sh.
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    sed -n '3,29p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit 0
fi
if (( $# > 0 )); then
    echo "Usage: bash scripts/sample/run_sample_sweep_full.sh" >&2
    exit 2
fi

TEST_TYPE_LIST="${TEST_TYPE_LIST:-id rough smooth}"
NUM_SAMPLES="${NUM_SAMPLES:-1000}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-20}"
PDE_LIST="${PDE_LIST:-poisson helmholtz darcy nsnonbounded}"
TASK_LIST="${TASK_LIST:-forward inverse both}"
SAMPLER_LIST="${SAMPLER_LIST:-stochastic}"
SENSOR_MODE_LIST="${SENSOR_MODE_LIST:-random}"
NUM_STEPS="${NUM_STEPS:-100}"
NUM_OBS="${NUM_OBS:-16384}"
SAMPLE_SEED="${SAMPLE_SEED:-0}"
CONFIG_DIR="${CONFIG_DIR:-configs/main}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/main}"
OUTPUT_TAG="${OUTPUT_TAG:-MAIN1000_100_TEST_FULL}"
OUTPUT_SUFFIX="${OUTPUT_SUFFIX:-}"
DEVICE_LIST="${DEVICE_LIST:-cuda:0 cuda:1}"
PARALLEL="${PARALLEL:-true}"
MAX_PARALLEL_TASKS="${MAX_PARALLEL_TASKS:-2}"
RESUME="${RESUME:-true}"
VIS="${VIS:-false}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-1}"
PLAN_ONLY="${PLAN_ONLY:-false}"
DRY_RUN="${DRY_RUN:-false}"
AGGREGATE="${AGGREGATE:-true}"

read -r -a TEST_TYPES <<< "${TEST_TYPE_LIST}"
if (( ${#TEST_TYPES[@]} == 0 )); then
    echo "TEST_TYPE_LIST must select at least one of: id rough smooth" >&2
    exit 2
fi

for test_type in "${TEST_TYPES[@]}"; do
    case "${test_type}" in
        id|rough|smooth) ;;
        *)
            echo "Unsupported test type: ${test_type}; expected id, rough, or smooth" >&2
            exit 2
            ;;
    esac

    output_dir="${OUTPUT_ROOT}/${OUTPUT_TAG}_${test_type}"
    if [[ -n "${OUTPUT_SUFFIX}" ]]; then
        output_dir="${output_dir}_${OUTPUT_SUFFIX}"
    fi

    echo "[$(date '+%F %T')] Starting full-observation ${test_type} sweep -> ${output_dir}"
    NUM_SAMPLES="${NUM_SAMPLES}" \
    MAX_BATCH_SIZE="${MAX_BATCH_SIZE}" \
    PDE_LIST="${PDE_LIST}" \
    TASK_LIST="${TASK_LIST}" \
    SAMPLER_LIST="${SAMPLER_LIST}" \
    SENSOR_MODE_LIST="${SENSOR_MODE_LIST}" \
    NUM_STEPS="${NUM_STEPS}" \
    NUM_OBS="${NUM_OBS}" \
    TEST_TYPE="${test_type}" \
    SAMPLE_SEED="${SAMPLE_SEED}" \
    CONFIG_DIR="${CONFIG_DIR}" \
    OUTPUT_DIR="${output_dir}" \
    DEVICE_LIST="${DEVICE_LIST}" \
    PARALLEL="${PARALLEL}" \
    MAX_PARALLEL_TASKS="${MAX_PARALLEL_TASKS}" \
    RESUME="${RESUME}" \
    VIS="${VIS}" \
    PROGRESS_INTERVAL="${PROGRESS_INTERVAL}" \
    PLAN_ONLY="${PLAN_ONLY}" \
    DRY_RUN="${DRY_RUN}" \
    AGGREGATE="${AGGREGATE}" \
        bash "${SCRIPT_DIR}/run_sample_sweep.sh"
    echo "[$(date '+%F %T')] Completed full-observation ${test_type} sweep"
done

echo "[$(date '+%F %T')] All non-Burgers full-observation sweeps completed."
