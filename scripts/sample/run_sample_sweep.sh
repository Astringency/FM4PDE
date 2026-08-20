#!/usr/bin/env bash
set -euo pipefail
# ============================================================================
# FM4PDE resumable sampling sweep
#
# Each PDE × task × sensor-mode combination is an independent scheduling unit.
# Samplers and sample chunks within that unit remain ordered so offsets are easy
# to audit/resume.
#
# Examples:
#   bash scripts/sample/run_sample_sweep.sh
#   OUTPUT_DIR=outputs/MAIN1000 NUM_SAMPLES=1000 \
#     TASK_LIST="forward inverse both" bash scripts/sample/run_sample_sweep.sh
#   PARALLEL=true MAX_PARALLEL_TASKS=2 DEVICE_LIST="cuda:0 cuda:1" \
#     bash scripts/sample/run_sample_sweep.sh
#   PDE_LIST="burger" TASK_LIST="both" \
#     SENSOR_MODE_LIST="random sensor_column" bash scripts/sample/run_sample_sweep.sh
#   RESUME=true bash scripts/sample/run_sample_sweep.sh
#
# Environment variables:
#   NUM_SAMPLES          Samples per PDE × task × sampler × sensor mode (default: 1000)
#   MAX_BATCH_SIZE       Maximum samples in one runner process (default: 50)
#   PDE_LIST             Space-separated PDEs (default: four non-Burgers main PDEs)
#   TASK_LIST            Space-separated tasks (default: forward inverse both)
#   SAMPLER_LIST         Space-separated samplers (default: stochastic)
#   SENSOR_MODE_LIST     Space-separated modes (default: random)
#   NUM_STEPS            Sampling steps (default: 100)
#   NUM_OBS              Sparse observations (default: 500)
#   OUTPUT_DIR           Artifact root (default: outputs/MAIN1000_100)
#   CONFIG_DIR           Main config directory (default: configs/main)
#   DEVICE               Device used when DEVICE_LIST is unset (default: cuda)
#   DEVICE_LIST          Devices assigned round-robin to PDE × task × sensor jobs
#   PARALLEL             Run PDE × task × sensor jobs concurrently (default: true)
#   MAX_PARALLEL_TASKS   Maximum concurrent PDE × task × sensor jobs (default: 2)
#   RESUME               Skip samples with matching successful artifacts (default: true)
#   VIS                  Save one plot per completed batch (default: true)
#   PROGRESS_INTERVAL    Live table refresh interval in seconds (default: 1)
#   PLAN_ONLY            Print the resumable plan without sampling (default: false)
#   AGGREGATE            Aggregate successful results after completion (default: true)
# ============================================================================

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

NUM_SAMPLES="${NUM_SAMPLES:-1000}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-50}"
PDE_LIST="${PDE_LIST:-poisson helmholtz darcy nsnonbounded}"
TASK_LIST="${TASK_LIST:-forward inverse both}"
SAMPLER_LIST="${SAMPLER_LIST:-stochastic}" # deterministic hybrid_s2d
SENSOR_MODE_LIST="${SENSOR_MODE_LIST:-random}"
NUM_STEPS="${NUM_STEPS:-100}"
NUM_OBS="${NUM_OBS:-500}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/MAIN1000_100}"
CONFIG_DIR="${CONFIG_DIR:-configs/main}"
DEVICE="${DEVICE:-cuda}"
DEVICE_LIST="${DEVICE_LIST:-${DEVICE}}"
PARALLEL="${PARALLEL:-true}"
MAX_PARALLEL_TASKS="${MAX_PARALLEL_TASKS:-2}"
RESUME="${RESUME:-true}"
VIS="${VIS:-true}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-1}"
PLAN_ONLY="${PLAN_ONLY:-false}"
AGGREGATE="${AGGREGATE:-true}"

read -r -a PDES <<< "${PDE_LIST}"
read -r -a TASKS <<< "${TASK_LIST}"
read -r -a SAMPLERS <<< "${SAMPLER_LIST}"
read -r -a DEVICES <<< "${DEVICE_LIST}"
SENSOR_MODES=()
if [[ -n "${SENSOR_MODE_LIST}" ]]; then
    read -r -a SENSOR_MODES <<< "${SENSOR_MODE_LIST}"
fi

ARGS=(
    --num-samples "${NUM_SAMPLES}"
    --max-batch-size "${MAX_BATCH_SIZE}"
    --num-steps "${NUM_STEPS}"
    --num-obs "${NUM_OBS}"
    --output-dir "${OUTPUT_DIR}"
    --config-dir "${CONFIG_DIR}"
    --max-parallel-tasks "${MAX_PARALLEL_TASKS}"
    --progress-interval "${PROGRESS_INTERVAL}"
    --sample-script "scripts/sample/run_sample.sh"
    --pdes "${PDES[@]}"
    --tasks "${TASKS[@]}"
    --samplers "${SAMPLERS[@]}"
    --devices "${DEVICES[@]}"
)

if (( ${#SENSOR_MODES[@]} > 0 )); then
    ARGS+=(--sensor-modes "${SENSOR_MODES[@]}")
fi

is_true() {
    case "${1,,}" in
        true|1|yes|on) return 0 ;;
        false|0|no|off) return 1 ;;
        *) echo "Invalid boolean value: $1" >&2; exit 2 ;;
    esac
}

is_true "${PARALLEL}" && ARGS+=(--parallel) || ARGS+=(--no-parallel)
is_true "${RESUME}" && ARGS+=(--resume) || ARGS+=(--no-resume)
is_true "${VIS}" && ARGS+=(--vis) || ARGS+=(--no-vis)
is_true "${PLAN_ONLY}" && ARGS+=(--plan-only)
is_true "${AGGREGATE}" && ARGS+=(--aggregate) || ARGS+=(--no-aggregate)
[[ "${DRY_RUN:-false}" == "true" ]] && ARGS+=(--dry-run)
[[ -n "${SAMPLE_SEED:-}" ]] && ARGS+=(--sample-seed "${SAMPLE_SEED}")

exec python -u -m sampling.sample_sweep "${ARGS[@]}"
