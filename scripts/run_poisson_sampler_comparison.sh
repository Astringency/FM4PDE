#!/usr/bin/env bash
set -euo pipefail

# Poisson sampler comparison:
#   3 distributions x 3 tasks x 4 sampler phases x 1 sample = 36 jobs.
#
# Usage:
#   PLAN_ONLY=true bash scripts/run_poisson_sampler_comparison.sh
#   bash scripts/run_poisson_sampler_comparison.sh
#   DATA_ROOT=/path/to/PDEdata DEVICE=cuda:0 bash scripts/run_poisson_sampler_comparison.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
GRID="${GRID:-configs/ablations/poisson_sampler_comparison.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/ablations/poisson_sampler_comparison}"
DEVICE="${DEVICE:-cuda}"
DATA_ROOT="${DATA_ROOT:-}"
TEST_TYPES="${TEST_TYPES:-id smooth rough}"
BATCH_SIZE="${BATCH_SIZE:-1}"
OFFSET="${OFFSET:-0}"
SAMPLE_SEED="${SAMPLE_SEED:-0}"
MASK_SEED="${MASK_SEED:-0}"
NOISE_SEED="${NOISE_SEED:-0}"
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

read -r -a DISTRIBUTIONS <<< "${TEST_TYPES}"
if [[ ${#DISTRIBUTIONS[@]} -eq 0 ]]; then
    echo "TEST_TYPES must select at least one distribution" >&2
    exit 2
fi

if ! is_true "${PLAN_ONLY}"; then
    missing_paths=()
    checkpoint_path="$("${PYTHON_BIN}" -c 'from sampling.config import load_config; print(load_config("configs/main/both/poisson.yaml").checkpoint_path)')"
    [[ -r "${checkpoint_path}" ]] || missing_paths+=("${checkpoint_path}")
    for test_type in "${DISTRIBUTIONS[@]}"; do
        if [[ -n "${DATA_ROOT}" ]]; then
            data_path="${DATA_ROOT%/}/poisson/poisson_test_10000-128-128_${test_type}.mat"
        else
            data_path="$("${PYTHON_BIN}" -c 'import sys; from sampling.config import load_config; print(load_config("configs/main/both/poisson.yaml", overrides={"test_type": sys.argv[1]}).data_path)' "${test_type}")"
        fi
        [[ -r "${data_path}" ]] || missing_paths+=("${data_path}")
    done
    if [[ ${#missing_paths[@]} -ne 0 ]]; then
        echo "Required experiment files are missing:" >&2
        printf '  %s\n' "${missing_paths[@]}" >&2
        echo "Set DATA_ROOT if the server stores PDEdata elsewhere." >&2
        exit 2
    fi
fi

for test_type in "${DISTRIBUTIONS[@]}"; do
    case "${test_type}" in
        id|smooth|rough) ;;
        *) echo "Unsupported test type: ${test_type}" >&2; exit 2 ;;
    esac

    ARGS=(
        --grid "${GRID}"
        --group poisson_sampler_comparison
        --pde poisson
        --override "output_dir=${OUTPUT_DIR}"
        --override "device=${DEVICE}"
        --override "test_type=${test_type}"
        --override "batch_size=${BATCH_SIZE}"
        --override "offset=${OFFSET}"
        --override "sample_seed=${SAMPLE_SEED}"
        --override "mask_seed=${MASK_SEED}"
        --override "noise_seed=${NOISE_SEED}"
        --override save_plots=false
    )
    if [[ -n "${DATA_ROOT}" ]]; then
        ARGS+=(
            --override "data_path=${DATA_ROOT%/}/poisson/poisson_test_10000-128-128_${test_type}.mat"
        )
    fi
    if is_true "${RESUME}"; then
        ARGS+=(--resume)
    else
        ARGS+=(--no-resume)
    fi

    echo "[$(date '+%F %T')] Poisson ${test_type}: 12 sampler/task jobs"
    if is_true "${PLAN_ONLY}"; then
        "${PYTHON_BIN}" -u -m sampling.sweep "${ARGS[@]}" --list
    else
        "${PYTHON_BIN}" -u -m sampling.sweep "${ARGS[@]}"
    fi
done

if ! is_true "${PLAN_ONLY}" && is_true "${AGGREGATE}"; then
    "${PYTHON_BIN}" -u -m sampling.aggregate "${OUTPUT_DIR}" --output-dir "${OUTPUT_DIR}"
    "${PYTHON_BIN}" -u scripts/analysis/summarize_poisson_sampler_comparison.py \
        --root "${OUTPUT_DIR}"
    "${PYTHON_BIN}" -u scripts/summary.py --exp ablations
fi

echo "Poisson sampler comparison complete: ${OUTPUT_DIR}"
