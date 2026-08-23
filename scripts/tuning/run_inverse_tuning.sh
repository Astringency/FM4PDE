#!/usr/bin/env bash
set -euo pipefail

# Cross-offset inverse tuning for the ten non-Burgers PDEs.
#
# Stage 1 stabilizes the six equations that diverged or produced NaN/Inf:
#   STAGE=stabilize DEVICE_LIST="cuda:0 cuda:1" \
#     bash scripts/tuning/run_inverse_tuning.sh
#
# Stage 2 improves coefficient recovery for the four numerically stable PDEs:
#   STAGE=refine DEVICE_LIST="cuda:0 cuda:1" \
#     bash scripts/tuning/run_inverse_tuning.sh
#
# Each default stage uses offsets 0-1 and 1000-1001. Set PLAN_ONLY=true to
# inspect the jobs without requiring data, checkpoints, or GPUs.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PDE_DATA_ROOT="${PDE_DATA_ROOT:-${HOME}/share/PDEdata}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/tuning/inverse_tuning}"
STAGE="${STAGE:-stabilize}"
DEVICE_LIST="${DEVICE_LIST:-cuda:0 cuda:1}"
MAX_PARALLEL_TASKS="${MAX_PARALLEL_TASKS:-2}"
NUM_STEPS="${NUM_STEPS:-100}"
NUM_OBS="${NUM_OBS:-500}"
OFFSET_LIST="${OFFSET_LIST:-0 1000}"
BATCH_SIZE="${BATCH_SIZE:-2}"
SAMPLE_SEED="${SAMPLE_SEED:-0}"
MASK_SEED_LIST="${MASK_SEED_LIST:-0}"
PLAN_ONLY="${PLAN_ONLY:-false}"
RESUME="${RESUME:-true}"
AGGREGATE="${AGGREGATE:-true}"
CLIP_MODE="${CLIP_MODE:-global_norm}"

case "${STAGE}" in
    stabilize)
        PDE_LIST="${PDE_LIST:-reaction_diffusion shallow_water heat wave advection_diffusion steady_heat_conduction}"
        ;;
    refine)
        PDE_LIST="${PDE_LIST:-darcy poisson helmholtz nsnonbounded}"
        ;;
    *)
        echo "STAGE must be one of: stabilize, refine" >&2
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

emit_grid() {
    local obs_values="$1"
    local pde_values="$2"
    local clip_values="$3"
    local zeta_obs_u zeta_pde clip_threshold
    for zeta_obs_u in ${obs_values}; do
        for zeta_pde in ${pde_values}; do
            for clip_threshold in ${clip_values}; do
                printf '%s %s %s\n' "${zeta_obs_u}" "${zeta_pde}" "${clip_threshold}"
            done
        done
    done
}

grid_for() {
    local pde="$1"
    if [[ -n "${ZETA_OBS_U_LIST:-}" && -n "${ZETA_PDE_LIST:-}" && -n "${CLIP_THRESHOLD_LIST:-}" ]]; then
        emit_grid "${ZETA_OBS_U_LIST}" "${ZETA_PDE_LIST}" "${CLIP_THRESHOLD_LIST}"
        return 0
    fi

    case "${STAGE}:${pde}" in
        stabilize:reaction_diffusion) emit_grid "1000000 4000000 16000000" "0.1 1 10" "50" ;;
        stabilize:shallow_water) emit_grid "50000 200000 800000" "0.1 1 10" "50" ;;
        stabilize:heat) emit_grid "1000 10000 100000" "0.01 0.1 1" "50" ;;
        stabilize:wave) emit_grid "10000 100000 1000000" "0.01 0.1 1" "50" ;;
        stabilize:advection_diffusion) emit_grid "10000 100000 1000000" "0.01 0.1 1" "50" ;;
        stabilize:steady_heat_conduction) emit_grid "10000 100000 1000000" "0.01 0.1 1" "50" ;;
        refine:darcy) emit_grid "250000000 500000000 1000000000 2000000000" "0.03 0.1 0.3" "50" ;;
        refine:poisson) emit_grid "45000000 90000000 180000000 360000000" "0.03 0.1 0.3" "50" ;;
        refine:helmholtz) emit_grid "40000000 80000000 160000000 320000000" "0.03 0.1 0.3" "50" ;;
        refine:nsnonbounded) emit_grid "30000 60000 120000 240000" "0.03 0.1 0.3" "50" ;;
        *) echo "No ${STAGE} grid for ${pde}" >&2; return 2 ;;
    esac
}

read -r -a PDES <<< "${PDE_LIST}"
read -r -a DEVICES <<< "${DEVICE_LIST}"
read -r -a OFFSETS <<< "${OFFSET_LIST}"
read -r -a MASK_SEEDS <<< "${MASK_SEED_LIST}"

if (( ${#DEVICES[@]} == 0 || MAX_PARALLEL_TASKS < 1 || MAX_PARALLEL_TASKS > ${#DEVICES[@]} )); then
    echo "Use one or more devices and no more parallel tasks than devices" >&2
    exit 2
fi
if (( ${#OFFSETS[@]} < 2 )); then
    echo "Use at least two OFFSET values to avoid single-offset tuning" >&2
    exit 2
fi

if ! is_true "${PLAN_ONLY}"; then
    preflight_failed=0
    for pde in "${PDES[@]}"; do
        config_path="configs/main/inverse/${pde}.yaml"
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
        exit 2
    fi
fi

stage_root="${OUTPUT_DIR}/${STAGE}"
mkdir -p "${stage_root}/logs" "${stage_root}/jobs"

run_job() {
    local device="$1"
    local pde="$2"
    local zeta_obs_u="$3"
    local zeta_pde="$4"
    local clip_threshold="$5"
    local offset="$6"
    local mask_seed="$7"
    local config_path="configs/main/inverse/${pde}.yaml"
    local data_path job_id job_root marker log_path
    data_path="$(data_path_for "${pde}")"
    job_id="${pde}_inverse_random_zu-${zeta_obs_u}_zp-${zeta_pde}_clip-${CLIP_MODE}-${clip_threshold}_steps-${NUM_STEPS}_offset-${offset}_batch-${BATCH_SIZE}_seed-${SAMPLE_SEED}_mask-${mask_seed}"
    job_root="${stage_root}/jobs/${job_id}"
    marker="${job_root}/.complete"
    log_path="${stage_root}/logs/${job_id}.log"

    if is_true "${PLAN_ONLY}"; then
        printf 'PLAN pde=%s device=%s zu=%s zp=%s clip=%s offset=%s batch=%s mask=%s\n' \
            "${pde}" "${device}" "${zeta_obs_u}" "${zeta_pde}" \
            "${clip_threshold}" "${offset}" "${BATCH_SIZE}" "${mask_seed}"
        return 0
    fi
    if is_true "${RESUME}" && [[ -f "${marker}" ]]; then
        echo "SKIP ${job_id}"
        return 0
    fi

    mkdir -p "${job_root}"
    echo "RUN  ${job_id} device=${device} log=${log_path}"
    if ! python -u -m sampling.runner \
        --config "${config_path}" \
        --override "data_path=${data_path}" \
        --override "output_dir=${job_root}" \
        --override "ablation_group=inverse_tuning_${STAGE}" \
        --override "task=inverse" \
        --override "sensor_mode=random" \
        --override "sampler_phase=stochastic" \
        --override "num_obs=${NUM_OBS}" \
        --override "num_steps=${NUM_STEPS}" \
        --override "offset=${offset}" \
        --override "batch_size=${BATCH_SIZE}" \
        --override "sample_seed=${SAMPLE_SEED}" \
        --override "mask_seed=${mask_seed}" \
        --override "zeta_obs_a=0" \
        --override "zeta_obs_u=${zeta_obs_u}" \
        --override "zeta_pde=${zeta_pde}" \
        --override "clip_mode=${CLIP_MODE}" \
        --override "clip_threshold=${clip_threshold}" \
        --override "device=${device}" \
        --override "save_plots=false" >"${log_path}" 2>&1; then
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
import math
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    metrics = json.load(handle)
required = (metrics.get("rel_l2_a"), metrics.get("rel_l2_u"), metrics.get("pde_residual_norm"))
if metrics.get("status") != "ok" or not all(isinstance(value, (int, float)) and math.isfinite(value) for value in required):
    raise SystemExit(1)
PY
    then
        echo "FAIL ${job_id}; final metrics are missing or non-finite" >&2
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

for pde in "${PDES[@]}"; do
    while read -r zeta_obs_u zeta_pde clip_threshold; do
        for offset in "${OFFSETS[@]}"; do
            for mask_seed in "${MASK_SEEDS[@]}"; do
                device="${DEVICES[$((device_index % ${#DEVICES[@]}))]}"
                if is_true "${PLAN_ONLY}"; then
                    run_job "${device}" "${pde}" "${zeta_obs_u}" "${zeta_pde}" "${clip_threshold}" "${offset}" "${mask_seed}"
                else
                    run_job "${device}" "${pde}" "${zeta_obs_u}" "${zeta_pde}" "${clip_threshold}" "${offset}" "${mask_seed}" &
                    pids+=("$!")
                    labels+=("${pde}/zu=${zeta_obs_u}/zp=${zeta_pde}/clip=${clip_threshold}/offset=${offset}/mask=${mask_seed}")
                    if (( ${#pids[@]} >= MAX_PARALLEL_TASKS )); then
                        wait_group
                    fi
                fi
                device_index=$((device_index + 1))
            done
        done
    done < <(grid_for "${pde}")
done

if (( ${#pids[@]} > 0 )); then
    wait_group
fi

if ! is_true "${PLAN_ONLY}" && is_true "${AGGREGATE}"; then
    if find "${stage_root}/jobs" -type f -name metrics_final.json -print -quit | grep -q .; then
        summary_dir="${stage_root}/summary"
        python -u -m sampling.aggregate "${stage_root}/jobs" --output-dir "${summary_dir}"
        expected_samples=$(( ${#OFFSETS[@]} * ${#MASK_SEEDS[@]} * BATCH_SIZE ))
        python -u scripts/tuning/select_inverse_params.py \
            "${summary_dir}/summary_all_grouped.csv" \
            --expected-n "${expected_samples}" \
            --output "${summary_dir}/selected_params.csv" \
            | tee "${summary_dir}/selected_params.txt"
        echo "Summary: ${summary_dir}/summary_all_grouped.csv"
        echo "Selection: ${summary_dir}/selected_params.csv"
    else
        echo "No completed metrics found; skipping aggregation and selection." >&2
    fi
fi

exit "${failed}"
