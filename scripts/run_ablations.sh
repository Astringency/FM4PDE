#!/usr/bin/env bash
set -euo pipefail
# ============================================================================
# FM4PDE formal ablation sweep
#
# Examples:
#   bash scripts/run_ablations.sh
#   PDE_LIST="poisson heat" bash scripts/run_ablations.sh
#   PDE_LIST="poisson heat" bash scripts/run_ablations.sh \
#     guidance_components time_grid_by_sampler
#   PDE_LIST="poisson" PLAN_ONLY=true bash scripts/run_ablations.sh sampler_phase
#   PARALLEL=true MAX_PARALLEL_TASKS=2 DEVICE_LIST="cuda:0 cuda:1" \
#     bash scripts/run_ablations.sh
#
# Environment variables:
#   PDE_LIST       Space-separated PDEs (default: all 11 formal PDEs)
#   OUTPUT_DIR     Artifact and aggregate output root (default: outputs/ablations)
#   DEVICE         Runtime device override (default: cuda)
#   DEVICE_LIST    Devices assigned round-robin to concurrent tasks (default: DEVICE)
#   PARALLEL       Run independent ablation tasks concurrently (default: false)
#   MAX_PARALLEL_TASKS  Maximum concurrent ablation tasks (default: 2)
#   BATCH_SIZE     Samples in each ablation job (default: 1)
#   OFFSET         Shared dataset offset override (default: 0)
#   VIS            Save plots for completed jobs (default: false)
#   DRY_RUN        Run the sampling runner in dry-run mode (default: false)
#   PLAN_ONLY      List selected jobs without sampling (default: false)
#   RESUME         Skip matching successful jobs already in OUTPUT_DIR (default: true)
#   AGGREGATE      Aggregate successful jobs after sampling (default: true)
# ============================================================================

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

GRID="configs/ablations/all_internal_ablation_grid.yaml"
PDE_LIST="${PDE_LIST:-darcy poisson helmholtz nsnonbounded burger reaction_diffusion shallow_water heat wave advection_diffusion steady_heat_conduction}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/ablations}"
DEVICE="${DEVICE:-cuda}"
DEVICE_LIST="${DEVICE_LIST:-${DEVICE}}"
PARALLEL="${PARALLEL:-false}"
MAX_PARALLEL_TASKS="${MAX_PARALLEL_TASKS:-2}"
BATCH_SIZE="${BATCH_SIZE:-1}"
OFFSET="${OFFSET:-0}"
VIS="${VIS:-false}"
DRY_RUN="${DRY_RUN:-false}"
PLAN_ONLY="${PLAN_ONLY:-false}"
RESUME="${RESUME:-true}"
AGGREGATE="${AGGREGATE:-true}"

ALL_GROUPS=(
    guidance_components
    loss_state_by_sampler
    sampler_phase
    time_grid_by_sampler
    num_steps_by_sampler
    step_method_by_sampler
    sensor_sparsity
    sensor_mode
    noise_robustness
    deterministic_endpoint_bt
    temporal_residual_mode
    statistics_stability
)

is_true() {
    case "${1,,}" in
        true|1|yes|on) return 0 ;;
        false|0|no|off) return 1 ;;
        *) echo "Invalid boolean value: $1" >&2; exit 2 ;;
    esac
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    echo "Usage: bash scripts/run_ablations.sh [group ...]"
    echo ""
    echo "Available ablation groups:"
    for group in "${ALL_GROUPS[@]}"; do
        echo "  ${group}"
    done
    echo ""
    echo "Select PDEs with PDE_LIST, for example:"
    echo '  PDE_LIST="poisson heat" PLAN_ONLY=true bash scripts/run_ablations.sh guidance_components'
    echo ""
    echo "Run tasks concurrently, for example:"
    echo '  PARALLEL=true MAX_PARALLEL_TASKS=2 DEVICE_LIST="cuda:0 cuda:1" bash scripts/run_ablations.sh'
    exit 0
fi

read -r -a PDES <<< "${PDE_LIST}"
read -r -a DEVICES <<< "${DEVICE_LIST}"
if [[ ${#PDES[@]} -eq 0 ]]; then
    echo "PDE_LIST must select at least one PDE" >&2
    exit 2
fi
if [[ ${#DEVICES[@]} -eq 0 ]]; then
    echo "DEVICE_LIST must select at least one device" >&2
    exit 2
fi
if ! [[ "${MAX_PARALLEL_TASKS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "MAX_PARALLEL_TASKS must be a positive integer" >&2
    exit 2
fi

VIS_OVERRIDE=false
if is_true "${VIS}"; then
    VIS_OVERRIDE=true
fi

ARGS=(--grid "${GRID}")
for pde in "${PDES[@]}"; do
    ARGS+=(--pde "${pde}")
done
for group in "$@"; do
    ARGS+=(--group "${group}")
done
ARGS+=(
    --max-parallel-tasks "${MAX_PARALLEL_TASKS}"
    --devices "${DEVICES[@]}"
    --override "output_dir=${OUTPUT_DIR}"
    --override "device=${DEVICE}"
    --override "batch_size=${BATCH_SIZE}"
    --override "offset=${OFFSET}"
    --override "save_plots=${VIS_OVERRIDE}"
)
if is_true "${PARALLEL}"; then
    ARGS+=(--parallel)
else
    ARGS+=(--no-parallel)
fi
if is_true "${RESUME}"; then
    ARGS+=(--resume)
else
    ARGS+=(--no-resume)
fi

if is_true "${PLAN_ONLY}"; then
    exec python -u -m sampling.sweep "${ARGS[@]}" --list
fi
if is_true "${DRY_RUN}"; then
    ARGS+=(--dry-run)
fi

python -u -m sampling.sweep "${ARGS[@]}"

if is_true "${AGGREGATE}"; then
    python -u -m sampling.aggregate "${OUTPUT_DIR}" --output-dir "${OUTPUT_DIR}"
fi

echo "Ablations complete: ${OUTPUT_DIR}/summary_all_raw.csv"
