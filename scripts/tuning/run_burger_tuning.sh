#!/usr/bin/env bash
set -euo pipefail

# Resumable Burgers tuning/validation/sampler comparison.
#
# Recommended A100x2 protocol:
#   STAGE=screen DEVICE_LIST="cuda:0 cuda:1" \
#     bash scripts/tuning/run_burger_tuning.sh
#   STAGE=validate ZETA_OBS_U_LIST="<selected>" ZETA_PDE_LIST="<selected>" \
#     DEVICE_LIST="cuda:0 cuda:1" bash scripts/tuning/run_burger_tuning.sh
#   STAGE=samplers ZETA_OBS_U_LIST="<selected>" ZETA_PDE_LIST="<selected>" \
#     DEVICE_LIST="cuda:0 cuda:1" bash scripts/tuning/run_burger_tuning.sh
#
# Set CACHE_DIR to copy the test data and checkpoint off a slow remote mount.
# Existing complete cache files are reused only when their byte sizes match.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

CONFIG_PATH="${CONFIG_PATH:-configs/main/both/burger.yaml}"
PDE_DATA_ROOT="${PDE_DATA_ROOT:-${HOME}/share/PDEdata}"
TEST_TYPE="${TEST_TYPE:-id}"
case "${TEST_TYPE}" in
    id|smooth|rough) ;;
    *) echo "TEST_TYPE must be one of: id, smooth, rough" >&2; exit 2 ;;
esac
DATA_PATH="${DATA_PATH:-${PDE_DATA_ROOT}/burgers/burger_test_10000-128-128_${TEST_TYPE}.mat}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-outputs/pretrained/formal/burger/260625-150439-burger-batch4-epoch300-accum8-float32/fm4burger.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/tuning/burger}"
STAGE="${STAGE:-screen}"
DEVICE_LIST="${DEVICE_LIST:-cuda:0 cuda:1}"
MAX_PARALLEL_TASKS="${MAX_PARALLEL_TASKS:-2}"
NUM_STEPS="${NUM_STEPS:-100}"
NUM_OBS="${NUM_OBS:-500}"
NUM_SENSOR_COLUMNS="${NUM_SENSOR_COLUMNS:-16}"
SAMPLE_SEED="${SAMPLE_SEED:-0}"
MASK_SEED_LIST="${MASK_SEED_LIST:-0}"
CLIP_MODE_LIST="${CLIP_MODE_LIST:-global_norm}"
PLAN_ONLY="${PLAN_ONLY:-false}"
RESUME="${RESUME:-true}"
AGGREGATE="${AGGREGATE:-true}"
CACHE_DIR="${CACHE_DIR:-}"

case "${STAGE}" in
    screen)
        SENSOR_MODE_LIST="${SENSOR_MODE_LIST:-random sensor_column}"
        SAMPLER_LIST="${SAMPLER_LIST:-stochastic}"
        ZETA_OBS_U_LIST="${ZETA_OBS_U_LIST:-51200 102400 204800 409600}"
        ZETA_PDE_LIST="${ZETA_PDE_LIST:-10}"
        CLIP_THRESHOLD_LIST="${CLIP_THRESHOLD_LIST:-20 50 100}"
        OFFSET_LIST="${OFFSET_LIST:-0 1000}"
        BATCH_SIZE="${BATCH_SIZE:-8}"
        ;;
    validate)
        SENSOR_MODE_LIST="${SENSOR_MODE_LIST:-random sensor_column}"
        SAMPLER_LIST="${SAMPLER_LIST:-stochastic}"
        ZETA_OBS_U_LIST="${ZETA_OBS_U_LIST:-409600}"
        ZETA_PDE_LIST="${ZETA_PDE_LIST:-10}"
        CLIP_THRESHOLD_LIST="${CLIP_THRESHOLD_LIST:-50}"
        OFFSET_LIST="${OFFSET_LIST:-3000 5000}"
        BATCH_SIZE="${BATCH_SIZE:-8}"
        ;;
    samplers)
        SENSOR_MODE_LIST="${SENSOR_MODE_LIST:-random sensor_column}"
        SAMPLER_LIST="${SAMPLER_LIST:-stochastic deterministic hybrid_s2d hybrid_d2s}"
        ZETA_OBS_U_LIST="${ZETA_OBS_U_LIST:-409600}"
        ZETA_PDE_LIST="${ZETA_PDE_LIST:-10}"
        CLIP_THRESHOLD_LIST="${CLIP_THRESHOLD_LIST:-50}"
        OFFSET_LIST="${OFFSET_LIST:-7000 8000}"
        BATCH_SIZE="${BATCH_SIZE:-8}"
        ;;
    *)
        echo "STAGE must be one of: screen, validate, samplers" >&2
        exit 2
        ;;
esac

is_true() {
    case "${1,,}" in
        true|1|yes|on) return 0 ;;
        false|0|no|off) return 1 ;;
        *) echo "Invalid boolean value: $1" >&2; exit 2 ;;
    esac
}

read -r -a DEVICES <<< "${DEVICE_LIST}"
read -r -a SENSOR_MODES <<< "${SENSOR_MODE_LIST}"
read -r -a SAMPLERS <<< "${SAMPLER_LIST}"
read -r -a ZETA_OBS_VALUES <<< "${ZETA_OBS_U_LIST}"
read -r -a ZETA_PDE_VALUES <<< "${ZETA_PDE_LIST}"
read -r -a OFFSETS <<< "${OFFSET_LIST}"
read -r -a MASK_SEEDS <<< "${MASK_SEED_LIST}"
read -r -a CLIP_MODES <<< "${CLIP_MODE_LIST}"
read -r -a CLIP_THRESHOLDS <<< "${CLIP_THRESHOLD_LIST}"

if (( ${#DEVICES[@]} == 0 || MAX_PARALLEL_TASKS < 1 )); then
    echo "DEVICE_LIST must be non-empty and MAX_PARALLEL_TASKS must be positive" >&2
    exit 2
fi
if (( MAX_PARALLEL_TASKS > ${#DEVICES[@]} )); then
    echo "MAX_PARALLEL_TASKS cannot exceed the number of devices; this script assigns one job per GPU" >&2
    exit 2
fi

if ! is_true "${PLAN_ONLY}"; then
    if [[ ! -f "${DATA_PATH}" ]]; then
        echo "Burger test data not found: ${DATA_PATH}" >&2
        exit 2
    fi
    if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
        echo "Burger checkpoint not found: ${CHECKPOINT_PATH}" >&2
        exit 2
    fi
fi

cache_file() {
    local source_path="$1"
    local target_path="$2"
    local source_size target_size
    source_size="$(stat -Lc '%s' "${source_path}")"
    target_size="$(stat -Lc '%s' "${target_path}" 2>/dev/null || true)"
    if [[ "${source_size}" != "${target_size}" ]]; then
        mkdir -p "$(dirname "${target_path}")"
        cp "${source_path}" "${target_path}.partial"
        mv "${target_path}.partial" "${target_path}"
    fi
}

if [[ -n "${CACHE_DIR}" ]] && ! is_true "${PLAN_ONLY}"; then
    cached_data="${CACHE_DIR}/burger_test_10000-128-128_${TEST_TYPE}.mat"
    cached_checkpoint_dir="${CACHE_DIR}/checkpoint"
    cached_checkpoint="${cached_checkpoint_dir}/fm4burger.pth"
    cache_file "${DATA_PATH}" "${cached_data}"
    cache_file "${CHECKPOINT_PATH}" "${cached_checkpoint}"
    checkpoint_dir="$(dirname "${CHECKPOINT_PATH}")"
    for metadata_name in normalizer.pt normalization.json data_metadata.json; do
        if [[ -f "${checkpoint_dir}/${metadata_name}" ]]; then
            cache_file "${checkpoint_dir}/${metadata_name}" "${cached_checkpoint_dir}/${metadata_name}"
        fi
    done
    DATA_PATH="${cached_data}"
    CHECKPOINT_PATH="${cached_checkpoint}"
fi

mkdir -p "${OUTPUT_DIR}/logs" "${OUTPUT_DIR}/jobs/${STAGE}"

run_job() {
    local device="$1"
    local sensor_mode="$2"
    local sampler="$3"
    local zeta_obs_u="$4"
    local zeta_pde="$5"
    local clip_mode="$6"
    local clip_threshold="$7"
    local offset="$8"
    local mask_seed="$9"
    local job_id job_root marker log_path
    job_id="mode-${sensor_mode}_obs-${NUM_OBS}_cols-${NUM_SENSOR_COLUMNS}_sampler-${sampler}_zu-${zeta_obs_u}_zp-${zeta_pde}_clip-${clip_mode}-${clip_threshold}_steps-${NUM_STEPS}_offset-${offset}_batch-${BATCH_SIZE}_seed-${SAMPLE_SEED}_mask-${mask_seed}"
    job_root="${OUTPUT_DIR}/jobs/${STAGE}/${job_id}"
    marker="${job_root}/.complete"
    log_path="${OUTPUT_DIR}/logs/${STAGE}_${job_id}.log"

    local command=(
        python -u -m sampling.runner
        --config "${CONFIG_PATH}"
        --override "data_path=${DATA_PATH}"
        --override "test_type=${TEST_TYPE}"
        --override "checkpoint_path=${CHECKPOINT_PATH}"
        --override "output_dir=${job_root}"
        --override "ablation_group=burger_${STAGE}"
        --override "task=both"
        --override "device=${device}"
        --override "batch_size=${BATCH_SIZE}"
        --override "offset=${offset}"
        --override "num_steps=${NUM_STEPS}"
        --override "num_obs=${NUM_OBS}"
        --override "num_sensor_columns=${NUM_SENSOR_COLUMNS}"
        --override "sensor_mode=${sensor_mode}"
        --override "sampler_phase=${sampler}"
        --override "sample_seed=${SAMPLE_SEED}"
        --override "mask_seed=${mask_seed}"
        --override "zeta_obs_a=0"
        --override "zeta_obs_u=${zeta_obs_u}"
        --override "zeta_pde=${zeta_pde}"
        --override "clip_mode=${clip_mode}"
        --override "clip_threshold=${clip_threshold}"
        --override "save_plots=false"
    )

    if is_true "${PLAN_ONLY}"; then
        printf 'PLAN %s device=%s\n' "${job_id}" "${device}"
        return 0
    fi
    if is_true "${RESUME}" && [[ -f "${marker}" ]]; then
        echo "SKIP ${job_id}"
        return 0
    fi

    mkdir -p "${job_root}"
    echo "RUN  ${job_id} device=${device} log=${log_path}"
    if ! "${command[@]}" >"${log_path}" 2>&1; then
        echo "FAIL ${job_id}; see ${log_path}" >&2
        return 1
    fi

    local metrics_path result_path
    metrics_path="$(find "${job_root}" -type f -name metrics_final.json -printf '%T@ %p\n' \
        | sort -nr | sed -n '1{s/^[^ ]* //;p;}')"
    result_path="${metrics_path%/metrics_final.json}/result.pt"
    if [[ -z "${metrics_path}" || ! -f "${result_path}" ]]; then
        echo "FAIL ${job_id}; missing completed artifacts" >&2
        return 1
    fi
    if ! python - "${metrics_path}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    metrics = json.load(handle)
if metrics.get("status") != "ok" or metrics.get("pde_residual_status") == "error":
    raise SystemExit(1)
PY
    then
        echo "FAIL ${job_id}; metrics did not report a successful PDE evaluation" >&2
        return 1
    fi
    touch "${marker}"
    echo "DONE ${job_id}"
}

pids=()
labels=()
failed=0
device_index=0

wait_group() {
    local index
    for index in "${!pids[@]}"; do
        if ! wait "${pids[$index]}"; then
            echo "Job failed: ${labels[$index]}" >&2
            failed=1
        fi
    done
    pids=()
    labels=()
}

for offset in "${OFFSETS[@]}"; do
    for mask_seed in "${MASK_SEEDS[@]}"; do
        for sensor_mode in "${SENSOR_MODES[@]}"; do
            for sampler in "${SAMPLERS[@]}"; do
                for zeta_obs_u in "${ZETA_OBS_VALUES[@]}"; do
                    for zeta_pde in "${ZETA_PDE_VALUES[@]}"; do
                        for clip_mode in "${CLIP_MODES[@]}"; do
                            for clip_threshold in "${CLIP_THRESHOLDS[@]}"; do
                                device="${DEVICES[$((device_index % ${#DEVICES[@]}))]}"
                                if is_true "${PLAN_ONLY}"; then
                                    run_job "${device}" "${sensor_mode}" "${sampler}" "${zeta_obs_u}" "${zeta_pde}" "${clip_mode}" "${clip_threshold}" "${offset}" "${mask_seed}"
                                else
                                    run_job "${device}" "${sensor_mode}" "${sampler}" "${zeta_obs_u}" "${zeta_pde}" "${clip_mode}" "${clip_threshold}" "${offset}" "${mask_seed}" &
                                    pids+=("$!")
                                    labels+=("${sensor_mode}/${sampler}/zu=${zeta_obs_u}/zp=${zeta_pde}/clip=${clip_mode}:${clip_threshold}/offset=${offset}/mask=${mask_seed}")
                                    if (( ${#pids[@]} >= MAX_PARALLEL_TASKS )); then
                                        wait_group
                                    fi
                                fi
                                device_index=$((device_index + 1))
                            done
                        done
                    done
                done
            done
        done
    done
done

if (( ${#pids[@]} > 0 )); then
    wait_group
fi

if ! is_true "${PLAN_ONLY}" && is_true "${AGGREGATE}"; then
    python -u -m sampling.aggregate "${OUTPUT_DIR}/jobs/${STAGE}" --output-dir "${OUTPUT_DIR}/summary/${STAGE}"
    echo "Summary: ${OUTPUT_DIR}/summary/${STAGE}/summary_all_grouped.csv"
fi

exit "${failed}"
