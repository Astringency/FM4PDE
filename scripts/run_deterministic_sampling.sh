#!/usr/bin/env bash
set -euo pipefail

# Server-side validation for the safeguarded deterministic sampler.
# Examples:
#   bash scripts/run_deterministic_sampling.sh
#   PDE_LIST="poisson darcy heat" BATCH_SIZE=4 bash scripts/run_deterministic_sampling.sh
#   INCLUDE_ROLLOUT=false PLAN_ONLY=true bash scripts/run_deterministic_sampling.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
GRID="${GRID:-configs/ablations/main_deterministic_safe.yaml}"
PDE_LIST="${PDE_LIST:-poisson}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/ablations}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-1}"
OFFSET="${OFFSET:-0}"
SAMPLE_SEED="${SAMPLE_SEED:-0}"
TEST_TYPE="${TEST_TYPE:-id}"
INCLUDE_ROLLOUT="${INCLUDE_ROLLOUT:-true}"
PLAN_ONLY="${PLAN_ONLY:-false}"
RESUME="${RESUME:-true}"
AGGREGATE="${AGGREGATE:-true}"

is_true() {
    case "${1,,}" in
        true|1|yes|on) return 0 ;;
        false|0|no|off) return 1 ;;
        *) echo "Invalid boolean value: $1" >&2; exit 2 ;;
    esac
}

read -r -a PDES <<< "${PDE_LIST}"
if [[ ${#PDES[@]} -eq 0 ]]; then
    echo "PDE_LIST must select at least one PDE" >&2
    exit 2
fi

ARGS=(
    --grid "${GRID}"
    --group deterministic_safe_single
    --override "output_dir=${OUTPUT_DIR}"
    --override "device=${DEVICE}"
    --override "batch_size=${BATCH_SIZE}"
    --override "offset=${OFFSET}"
    --override "sample_seed=${SAMPLE_SEED}"
    --override "test_type=${TEST_TYPE}"
    --override save_plots=false
)
if is_true "${INCLUDE_ROLLOUT}"; then
    ARGS+=(--group deterministic_safe_rollout)
fi
for pde in "${PDES[@]}"; do
    ARGS+=(--pde "${pde}")
done
if is_true "${RESUME}"; then
    ARGS+=(--resume)
else
    ARGS+=(--no-resume)
fi
if is_true "${PLAN_ONLY}"; then
    ARGS+=(--list)
fi

"${PYTHON_BIN}" -u -m sampling.sweep "${ARGS[@]}"

if ! is_true "${PLAN_ONLY}" && is_true "${AGGREGATE}"; then
    "${PYTHON_BIN}" -u -m sampling.aggregate "${OUTPUT_DIR}" --output-dir "${OUTPUT_DIR}"
fi
