#!/usr/bin/env bash
set -euo pipefail

# One-click, resumable sampling-parameter tuning for:
#   advection_diffusion reaction_diffusion steady_heat_conduction
#   heat shallow_water wave
#
# Typical server run:
#   PDE_DATA_ROOT=/path/to/PDEdata DEVICE_LIST="cuda:0 cuda:1" \
#     bash scripts/tuning/run_six_pde_sampling_tuning.sh
#
# Cheap plan (does not require data/checkpoints/GPUs):
#   PLAN_ONLY=true PROFILE=quick \
#     bash scripts/tuning/run_six_pde_sampling_tuning.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
PDE_DATA_ROOT="${PDE_DATA_ROOT:-${HOME}/share/PDEdata}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/tuning/six_pde_sampling}"
PDE_LIST="${PDE_LIST:-advection_diffusion reaction_diffusion steady_heat_conduction heat shallow_water wave}"
TASK_LIST="${TASK_LIST:-both forward inverse}"
DEVICE_LIST="${DEVICE_LIST:-cuda:0}"
PROFILE="${PROFILE:-standard}"
CANDIDATE_SET="${CANDIDATE_SET:-broad}"
DISTRIBUTION_WEIGHTS="${DISTRIBUTION_WEIGHTS:-}"
PHASE="${PHASE:-all}"
NUM_STEPS="${NUM_STEPS:-100}"
SAMPLE_SEED="${SAMPLE_SEED:-0}"
MASK_SEED="${MASK_SEED:-0}"
PLAN_ONLY="${PLAN_ONLY:-false}"
RESUME="${RESUME:-true}"
SKIP_KNOWN_NONFINITE="${SKIP_KNOWN_NONFINITE:-false}"

# Optional space-separated overrides. Empty means use the selected profile.
TEST_TYPE_LIST="${TEST_TYPE_LIST:-}"
TUNE_OFFSET_LIST="${TUNE_OFFSET_LIST:-}"
HOLDOUT_OFFSET_LIST="${HOLDOUT_OFFSET_LIST:-}"
BATCH_SIZE="${BATCH_SIZE:-}"

read -r -a PDES <<< "${PDE_LIST}"
read -r -a TASKS <<< "${TASK_LIST}"
read -r -a DEVICES <<< "${DEVICE_LIST}"
MAX_PARALLEL_TASKS="${MAX_PARALLEL_TASKS:-${#DEVICES[@]}}"

is_true() {
    case "${1,,}" in
        true|1|yes|on) return 0 ;;
        false|0|no|off) return 1 ;;
        *) echo "Invalid boolean value: $1" >&2; exit 2 ;;
    esac
}

join_comma() {
    local IFS=,
    echo "$*"
}

case "${PHASE}" in
    prepare|tune|holdout|all|analyze|analyze-tune) ;;
    *) echo "PHASE must be one of: prepare, tune, holdout, all, analyze, analyze-tune" >&2; exit 2 ;;
esac

if (( ${#PDES[@]} == 0 || ${#TASKS[@]} == 0 || ${#DEVICES[@]} == 0 )); then
    echo "PDE_LIST, TASK_LIST, and DEVICE_LIST must be non-empty" >&2
    exit 2
fi
if (( MAX_PARALLEL_TASKS < 1 || MAX_PARALLEL_TASKS > ${#DEVICES[@]} )); then
    echo "MAX_PARALLEL_TASKS must be between 1 and the number of DEVICE_LIST entries" >&2
    exit 2
fi

COMMON_ARGS=(
    --root "${OUTPUT_DIR}"
    --profile "${PROFILE}"
    --candidate-set "${CANDIDATE_SET}"
    --tasks "$(join_comma "${TASKS[@]}")"
    --data-root "${PDE_DATA_ROOT}"
    --num-steps "${NUM_STEPS}"
    --sample-seed "${SAMPLE_SEED}"
    --mask-seed "${MASK_SEED}"
)

if [[ -n "${DISTRIBUTION_WEIGHTS}" ]]; then
    COMMON_ARGS+=(--distribution-weights "${DISTRIBUTION_WEIGHTS}")
fi

if [[ -n "${TEST_TYPE_LIST}" ]]; then
    read -r -a TEST_TYPES <<< "${TEST_TYPE_LIST}"
    COMMON_ARGS+=(--test-types "$(join_comma "${TEST_TYPES[@]}")")
fi
if [[ -n "${TUNE_OFFSET_LIST}" ]]; then
    read -r -a TUNE_OFFSETS <<< "${TUNE_OFFSET_LIST}"
    COMMON_ARGS+=(--tune-offsets "$(join_comma "${TUNE_OFFSETS[@]}")")
fi
if [[ -n "${HOLDOUT_OFFSET_LIST}" ]]; then
    read -r -a HOLDOUT_OFFSETS <<< "${HOLDOUT_OFFSET_LIST}"
    COMMON_ARGS+=(--holdout-offsets "$(join_comma "${HOLDOUT_OFFSETS[@]}")")
fi
if [[ -n "${BATCH_SIZE}" ]]; then
    COMMON_ARGS+=(--batch-size "${BATCH_SIZE}")
fi
if ! is_true "${RESUME}"; then
    COMMON_ARGS+=(--no-resume)
fi
if is_true "${SKIP_KNOWN_NONFINITE}"; then
    COMMON_ARGS+=(--skip-known-nonfinite)
fi

ALL_PDES="$(join_comma "${PDES[@]}")"

if is_true "${PLAN_ONLY}"; then
    exec "${PYTHON_BIN}" -u scripts/tuning/run_six_pde_sampling_tuning.py \
        "${COMMON_ARGS[@]}" --phase "${PHASE}" --pdes "${ALL_PDES}" \
        --device "${DEVICES[0]}" --plan-only
fi

if [[ "${PHASE}" == "prepare" || "${PHASE}" == "analyze" || "${PHASE}" == "analyze-tune" ]]; then
    exec "${PYTHON_BIN}" -u scripts/tuning/run_six_pde_sampling_tuning.py \
        "${COMMON_ARGS[@]}" --phase "${PHASE}" --pdes "${ALL_PDES}" \
        --device "${DEVICES[0]}"
fi

# Slim large training checkpoints one at a time before GPU workers start. This
# avoids concurrent host-memory spikes, especially for shallow_water and wave.
"${PYTHON_BIN}" -u scripts/tuning/run_six_pde_sampling_tuning.py \
    "${COMMON_ARGS[@]}" --phase prepare --pdes "${ALL_PDES}" \
    --device "${DEVICES[0]}"

mkdir -p "${OUTPUT_DIR}/launcher_logs"
pids=()
labels=()
failed=0
device_index=0

wait_workers() {
    local index
    for index in "${!pids[@]}"; do
        if ! wait "${pids[index]}"; then
            echo "FAIL worker ${labels[index]}; see ${OUTPUT_DIR}/launcher_logs/${labels[index]}.log" >&2
            failed=1
        else
            echo "DONE worker ${labels[index]}"
        fi
    done
    pids=()
    labels=()
}

for pde in "${PDES[@]}"; do
    device="${DEVICES[device_index]}"
    label="${pde}_${PHASE}_${PROFILE}"
    log_path="${OUTPUT_DIR}/launcher_logs/${label}.log"
    echo "START worker ${label} device=${device} log=${log_path}"
    "${PYTHON_BIN}" -u scripts/tuning/run_six_pde_sampling_tuning.py \
        "${COMMON_ARGS[@]}" --phase "${PHASE}" --pdes "${pde}" \
        --device "${device}" >"${log_path}" 2>&1 &
    pids+=("$!")
    labels+=("${label}")
    device_index=$(( (device_index + 1) % ${#DEVICES[@]} ))
    if (( ${#pids[@]} == MAX_PARALLEL_TASKS )); then
        wait_workers
    fi
done
if (( ${#pids[@]} > 0 )); then
    wait_workers
fi

# Rebuild one combined table after the per-PDE workers have written their
# disjoint artifacts. For tune-only runs this intentionally skips holdout.
analysis_phase="analyze"
if [[ "${PHASE}" == "tune" ]]; then
    analysis_phase="analyze-tune"
fi
if ! "${PYTHON_BIN}" -u scripts/tuning/run_six_pde_sampling_tuning.py \
    "${COMMON_ARGS[@]}" --phase "${analysis_phase}" --pdes "${ALL_PDES}" \
    --device "${DEVICES[0]}"; then
    echo "FAIL combined analysis" >&2
    failed=1
fi

if (( failed != 0 )); then
    echo "One or more workers failed. Fix the reported issue and rerun this same command; successful jobs resume." >&2
    exit 1
fi

if [[ "${PHASE}" == "tune" ]]; then
    echo "Tuning screen complete. Selected-parameter tables are under ${OUTPUT_DIR}"
else
    echo "Tuning complete. Recommended configs: ${OUTPUT_DIR}/recommended_configs/${PROFILE}"
fi
