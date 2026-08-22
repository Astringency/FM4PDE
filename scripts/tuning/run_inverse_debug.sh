#!/usr/bin/env bash
set -euo pipefail

# Resumable inverse/random/100-step debug pass for the ten non-Burgers PDEs.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PDE_DATA_ROOT="${PDE_DATA_ROOT:-${HOME}/share/PDEdata}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/tuning/inverse_debug}"
PDE_LIST="${PDE_LIST:-darcy poisson helmholtz nsnonbounded reaction_diffusion shallow_water heat wave advection_diffusion steady_heat_conduction}"
DEVICE_LIST="${DEVICE_LIST:-cuda:0 cuda:1}"
MAX_PARALLEL_TASKS="${MAX_PARALLEL_TASKS:-2}"
NUM_STEPS="${NUM_STEPS:-100}"
NUM_OBS="${NUM_OBS:-500}"
OFFSET="${OFFSET:-0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
SAMPLE_SEED="${SAMPLE_SEED:-0}"
MASK_SEED="${MASK_SEED:-0}"
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

data_path_for() {
    case "$1" in
        darcy) echo "${PDE_DATA_ROOT}/darcy/darcy_test_10000-128-128.mat" ;;
        poisson) echo "${PDE_DATA_ROOT}/poisson/poisson_test_10000-128-128.mat" ;;
        helmholtz) echo "${PDE_DATA_ROOT}/helmholtz/helmholtz_test_10000-128-128.mat" ;;
        nsnonbounded) echo "${PDE_DATA_ROOT}/nsnonbounded/nsnonbounded_test_10000-128-128-10.mat" ;;
        reaction_diffusion) echo "${PDE_DATA_ROOT}/reaction_diffusion/reaction_diffusion_test_grf_10000-128-128-T1-steps10.h5" ;;
        shallow_water) echo "${PDE_DATA_ROOT}/shallow_water/shallow_water_test_10000-128-128-10.h5" ;;
        heat) echo "${PDE_DATA_ROOT}/heat/heat_test_10000-128-128.h5" ;;
        wave) echo "${PDE_DATA_ROOT}/wave/wave_test_10000-128-128.h5" ;;
        advection_diffusion) echo "${PDE_DATA_ROOT}/advection_diffusion/advection_diffusion_test_10000-128-128.h5" ;;
        steady_heat_conduction) echo "${PDE_DATA_ROOT}/steady_heat_conduction/steady_heat_conduction_test_10000-128-128.h5" ;;
        *) echo "Unsupported PDE: $1" >&2; return 2 ;;
    esac
}

read -r -a PDES <<< "${PDE_LIST}"
read -r -a DEVICES <<< "${DEVICE_LIST}"
if (( ${#DEVICES[@]} == 0 || MAX_PARALLEL_TASKS < 1 || MAX_PARALLEL_TASKS > ${#DEVICES[@]} )); then
    echo "Use one or more devices and no more parallel tasks than devices" >&2
    exit 2
fi

if ! is_true "${PLAN_ONLY}"; then
    preflight_failed=0
    for pde in "${PDES[@]}"; do
        config_path="configs/main/${pde}.yaml"
        data_path="$(data_path_for "${pde}")"
        if [[ ! -f "${config_path}" ]]; then
            echo "MISSING CONFIG ${pde}: ${ROOT_DIR}/${config_path}" >&2
            preflight_failed=1
        fi
        if [[ ! -f "${data_path}" ]]; then
            echo "MISSING DATA   ${pde}: ${data_path}" >&2
            preflight_failed=1
        fi
    done
    if (( preflight_failed != 0 )); then
        echo "Preflight failed. Set PDE_DATA_ROOT to the directory containing the per-PDE subdirectories." >&2
        echo "Example: PDE_DATA_ROOT=/absolute/path/to/PDEdata bash scripts/tuning/run_inverse_debug.sh" >&2
        exit 2
    fi
fi

mkdir -p "${OUTPUT_DIR}/logs" "${OUTPUT_DIR}/jobs"

run_job() {
    local device="$1"
    local pde="$2"
    local config_path="configs/main/${pde}.yaml"
    local data_path job_root marker log_path
    data_path="$(data_path_for "${pde}")"
    job_root="${OUTPUT_DIR}/jobs/${pde}_inverse_random_steps-${NUM_STEPS}_offset-${OFFSET}_batch-${BATCH_SIZE}_seed-${SAMPLE_SEED}_mask-${MASK_SEED}"
    marker="${job_root}/.complete"
    log_path="${OUTPUT_DIR}/logs/${pde}_inverse_random.log"

    if is_true "${PLAN_ONLY}"; then
        printf 'PLAN pde=%s device=%s data=%s\n' "${pde}" "${device}" "${data_path}"
        return 0
    fi
    if is_true "${RESUME}" && [[ -f "${marker}" ]]; then
        echo "SKIP ${pde}"
        return 0
    fi
    mkdir -p "${job_root}"
    echo "RUN  ${pde} device=${device} log=${log_path}"
    if ! python -u -m sampling.runner \
        --config "${config_path}" \
        --override "data_path=${data_path}" \
        --override "output_dir=${job_root}" \
        --override "ablation_group=inverse_debug" \
        --override "task=inverse" \
        --override "sensor_mode=random" \
        --override "num_obs=${NUM_OBS}" \
        --override "num_steps=${NUM_STEPS}" \
        --override "offset=${OFFSET}" \
        --override "batch_size=${BATCH_SIZE}" \
        --override "sample_seed=${SAMPLE_SEED}" \
        --override "mask_seed=${MASK_SEED}" \
        --override "device=${device}" \
        --override "save_plots=false" >"${log_path}" 2>&1; then
        echo "FAIL ${pde}; see ${log_path}" >&2
        return 1
    fi

    local metrics_path result_path
    metrics_path="$(find "${job_root}" -type f -name metrics_final.json -printf '%T@ %p\n' \
        | sort -nr | sed -n '1{s/^[^ ]* //;p;}')"
    result_path="${metrics_path%/metrics_final.json}/result.pt"
    if [[ -z "${metrics_path}" || ! -f "${result_path}" ]]; then
        echo "FAIL ${pde}; missing completed artifacts" >&2
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
        echo "FAIL ${pde}; unsuccessful metrics" >&2
        return 1
    fi
    touch "${marker}"
    echo "DONE ${pde}"
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

for pde in "${PDES[@]}"; do
    device="${DEVICES[$((device_index % ${#DEVICES[@]}))]}"
    if is_true "${PLAN_ONLY}"; then
        run_job "${device}" "${pde}"
    else
        run_job "${device}" "${pde}" &
        pids+=("$!")
        labels+=("${pde}")
        if (( ${#pids[@]} >= MAX_PARALLEL_TASKS )); then
            wait_group
        fi
    fi
    device_index=$((device_index + 1))
done

if (( ${#pids[@]} > 0 )); then
    wait_group
fi

if ! is_true "${PLAN_ONLY}" && is_true "${AGGREGATE}"; then
    if find "${OUTPUT_DIR}/jobs" -type f -name metrics_final.json -print -quit | grep -q .; then
        python -u -m sampling.aggregate "${OUTPUT_DIR}/jobs" --output-dir "${OUTPUT_DIR}/summary"
        echo "Summary: ${OUTPUT_DIR}/summary/summary_all_grouped.csv"
    else
        echo "No completed metrics found; skipping aggregation." >&2
    fi
fi

exit "${failed}"
