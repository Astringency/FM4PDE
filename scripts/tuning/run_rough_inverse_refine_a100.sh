#!/usr/bin/env bash
set -euo pipefail

# Rough-priority refinement around the currently promoted tuned1 inverse
# configs. Run one half on each A100 server, or set PDE_LIST explicitly.
#
# Server 0:
#   SERVER_RANK=0 bash scripts/tuning/run_rough_inverse_refine_a100.sh
# Server 1:
#   SERVER_RANK=1 bash scripts/tuning/run_rough_inverse_refine_a100.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

SERVER_RANK="${SERVER_RANK:-}"
SOURCE_ROOT="${SOURCE_ROOT:-outputs/artifacts/balanced_sampling_tuning}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-outputs/artifacts/rough_inverse_refine}"
PHASE="${PHASE:-all}"
PROFILE="${PROFILE:-quick}"
DISTRIBUTION_WEIGHTS="${DISTRIBUTION_WEIGHTS:-id=1,smooth=1,rough=2}"
MAX_PDE_RESIDUAL_RATIO="${MAX_PDE_RESIDUAL_RATIO:-1.05}"
RESIDUAL_PENALTY="${RESIDUAL_PENALTY:-0.10}"
TUNE_SAMPLES_PER_TEST_TYPE="${TUNE_SAMPLES_PER_TEST_TYPE:-20}"
HOLDOUT_SAMPLES_PER_TEST_TYPE="${HOLDOUT_SAMPLES_PER_TEST_TYPE:-40}"
NUM_STEPS="${NUM_STEPS:-100}"
DEVICE="${DEVICE:-cuda:0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
PLAN_ONLY="${PLAN_ONLY:-false}"
BASELINE_ONLY="${BASELINE_ONLY:-false}"
ANALYSIS_LABEL="${ANALYSIS_LABEL:-rough_refine}"

if [[ -n "${PDE_LIST:-}" ]]; then
    selected_pdes="${PDE_LIST}"
elif [[ "${SERVER_RANK}" == "0" ]]; then
    selected_pdes="poisson,helmholtz"
elif [[ "${SERVER_RANK}" == "1" ]]; then
    selected_pdes="darcy,nsnonbounded"
else
    echo "Set SERVER_RANK=0 or SERVER_RANK=1, or provide PDE_LIST explicitly." >&2
    exit 2
fi

case "${PHASE}" in
    tune|holdout|all|analyze) ;;
    *) echo "PHASE must be tune, holdout, all, or analyze" >&2; exit 2 ;;
esac
case "${PROFILE}" in
    quick|standard) ;;
    *) echo "PROFILE must be quick or standard" >&2; exit 2 ;;
esac

IFS=',' read -r -a pdes <<< "${selected_pdes}"
microbatch_args=()
for pde in "${pdes[@]}"; do
    case "${pde}" in
        poisson) microbatch_args+=(--microbatch "poisson=${POISSON_MICROBATCH:-10}") ;;
        helmholtz) microbatch_args+=(--microbatch "helmholtz=${HELMHOLTZ_MICROBATCH:-10}") ;;
        darcy) microbatch_args+=(--microbatch "darcy=${DARCY_MICROBATCH:-10}") ;;
        nsnonbounded) microbatch_args+=(--microbatch "nsnonbounded=${NSNONBOUNDED_MICROBATCH:-4}") ;;
        *) echo "Unsupported PDE in PDE_LIST: ${pde}" >&2; exit 2 ;;
    esac
done

command=(
    "${PYTHON_BIN}" -u scripts/tuning/run_rough_inverse_refine.py
    --root "${ARTIFACT_ROOT}"
    --source-root "${SOURCE_ROOT}"
    --phase "${PHASE}"
    --profile "${PROFILE}"
    --pdes "${selected_pdes}"
    --distribution-weights "${DISTRIBUTION_WEIGHTS}"
    --max-pde-residual-ratio "${MAX_PDE_RESIDUAL_RATIO}"
    --residual-penalty "${RESIDUAL_PENALTY}"
    --tune-samples-per-test-type "${TUNE_SAMPLES_PER_TEST_TYPE}"
    --holdout-samples-per-test-type "${HOLDOUT_SAMPLES_PER_TEST_TYPE}"
    --num-steps "${NUM_STEPS}"
    --device "${DEVICE}"
    --analysis-label "${ANALYSIS_LABEL}"
    "${microbatch_args[@]}"
)

case "${PLAN_ONLY,,}" in
    true|1|yes|on) command+=(--plan-only) ;;
    false|0|no|off) ;;
    *) echo "PLAN_ONLY must be true or false" >&2; exit 2 ;;
esac
case "${BASELINE_ONLY,,}" in
    true|1|yes|on) command+=(--baseline-only) ;;
    false|0|no|off) ;;
    *) echo "BASELINE_ONLY must be true or false" >&2; exit 2 ;;
esac

printf 'Rough inverse refine: rank=%s pdes=%s phase=%s profile=%s weights=%s max_pde_ratio=%s\n' \
    "${SERVER_RANK:-custom}" "${selected_pdes}" "${PHASE}" "${PROFILE}" \
    "${DISTRIBUTION_WEIGHTS}" "${MAX_PDE_RESIDUAL_RATIO}"
printf 'COMMAND'
printf ' %q' "${command[@]}"
printf '\n'

exec "${command[@]}"
