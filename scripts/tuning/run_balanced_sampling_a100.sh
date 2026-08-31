#!/usr/bin/env bash
set -euo pipefail

# Round-three balanced tuning across both/forward/inverse and ID/smooth/rough.
#
# Server 0:
#   SERVER_RANK=0 bash scripts/tuning/run_balanced_sampling_a100.sh
# Server 1:
#   SERVER_RANK=1 bash scripts/tuning/run_balanced_sampling_a100.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

SERVER_RANK="${SERVER_RANK:-}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-outputs/artifacts/balanced_sampling_tuning}"
PHASE="${PHASE:-all}"
TASK_LIST="${TASK_LIST:-both,forward,inverse}"
TUNE_SAMPLES_PER_TEST_TYPE="${TUNE_SAMPLES_PER_TEST_TYPE:-20}"
HOLDOUT_SAMPLES_PER_TEST_TYPE="${HOLDOUT_SAMPLES_PER_TEST_TYPE:-40}"
NUM_STEPS="${NUM_STEPS:-100}"
DEVICE="${DEVICE:-cuda:0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
PLAN_ONLY="${PLAN_ONLY:-false}"
ANALYSIS_LABEL="${ANALYSIS_LABEL:-round3}"

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

manifest="${ARTIFACT_ROOT}/balanced_samples.csv"
if [[ ! -f "${manifest}" ]]; then
    echo "Missing ${manifest}." >&2
    echo "Run scripts/tuning/prepare_balanced_sampling_samples.py first." >&2
    exit 2
fi

IFS=',' read -r -a pdes <<< "${selected_pdes}"
IFS=',' read -r -a tasks <<< "${TASK_LIST}"
microbatch_args=()
for pde in "${pdes[@]}"; do
    for task in "${tasks[@]}"; do
        case "${task}" in
            both|forward|inverse) ;;
            *) echo "Unsupported task in TASK_LIST: ${task}" >&2; exit 2 ;;
        esac
        for test_type in id smooth rough; do
            tune_subset="${ARTIFACT_ROOT}/subsets/tune/${pde}_${task}_${test_type}.mat"
            holdout_subset="${ARTIFACT_ROOT}/subsets/holdout/${pde}_${task}_${test_type}.mat"
            if [[ ! -f "${tune_subset}" || ! -f "${holdout_subset}" ]]; then
                echo "Missing balanced subsets for ${pde}/${task}/${test_type}." >&2
                exit 2
            fi
        done
    done
    case "${pde}" in
        poisson) microbatch_args+=(--microbatch "poisson=${POISSON_MICROBATCH:-10}") ;;
        helmholtz) microbatch_args+=(--microbatch "helmholtz=${HELMHOLTZ_MICROBATCH:-10}") ;;
        darcy) microbatch_args+=(--microbatch "darcy=${DARCY_MICROBATCH:-10}") ;;
        nsnonbounded) microbatch_args+=(--microbatch "nsnonbounded=${NSNONBOUNDED_MICROBATCH:-4}") ;;
        *) echo "Unsupported PDE in PDE_LIST: ${pde}" >&2; exit 2 ;;
    esac
done

command=(
    "${PYTHON_BIN}" -u scripts/tuning/run_balanced_sampling_tuning.py
    --root "${ARTIFACT_ROOT}"
    --phase "${PHASE}"
    --pdes "${selected_pdes}"
    --tasks "${TASK_LIST}"
    --tune-samples-per-test-type "${TUNE_SAMPLES_PER_TEST_TYPE}"
    --holdout-samples-per-test-type "${HOLDOUT_SAMPLES_PER_TEST_TYPE}"
    --num-steps "${NUM_STEPS}"
    --device "${DEVICE}"
    --analysis-label "${ANALYSIS_LABEL}"
    "${microbatch_args[@]}"
)

printf 'A100 balanced tuning: rank=%s pdes=%s tasks=%s phase=%s tune/test=%s holdout/test=%s steps=%s device=%s\n' \
    "${SERVER_RANK:-custom}" "${selected_pdes}" "${TASK_LIST}" "${PHASE}" \
    "${TUNE_SAMPLES_PER_TEST_TYPE}" "${HOLDOUT_SAMPLES_PER_TEST_TYPE}" \
    "${NUM_STEPS}" "${DEVICE}"
printf 'COMMAND'
printf ' %q' "${command[@]}"
printf '\n'

case "${PLAN_ONLY,,}" in
    true|1|yes|on) exit 0 ;;
    false|0|no|off) ;;
    *) echo "PLAN_ONLY must be true or false" >&2; exit 2 ;;
esac

exec "${command[@]}"
