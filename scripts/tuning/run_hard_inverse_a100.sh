#!/usr/bin/env bash
set -euo pipefail

# Split hard-sample inverse tuning across two independent A100 servers.
#
# Server 0:
#   SERVER_RANK=0 bash scripts/tuning/run_hard_inverse_a100.sh
# Server 1:
#   SERVER_RANK=1 bash scripts/tuning/run_hard_inverse_a100.sh
#
# The two servers may use independent copies of the repository. Each copy must
# contain hard_samples.csv plus the tune/holdout subsets prepared by
# prepare_hard_inverse_samples.py. Runs are resumable.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

SERVER_RANK="${SERVER_RANK:-}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-artifacts/inverse_hard_tuning}"
PHASE="${PHASE:-all}"
SAMPLES_PER_TEST_TYPE="${SAMPLES_PER_TEST_TYPE:-10}"
NUM_STEPS="${NUM_STEPS:-100}"
DEVICE="${DEVICE:-cuda:0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
PLAN_ONLY="${PLAN_ONLY:-false}"

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

manifest="${ARTIFACT_ROOT}/hard_samples.csv"
if [[ ! -f "${manifest}" ]]; then
    echo "Missing ${manifest}. Run scripts/tuning/prepare_hard_inverse_samples.py first." >&2
    exit 2
fi

IFS=',' read -r -a pdes <<< "${selected_pdes}"
microbatch_args=()
for pde in "${pdes[@]}"; do
    for test_type in id smooth rough; do
        subset="${ARTIFACT_ROOT}/subsets/tune/${pde}_${test_type}.mat"
        holdout="${ARTIFACT_ROOT}/subsets/holdout/${pde}_${test_type}.mat"
        if [[ ! -f "${subset}" || ! -f "${holdout}" ]]; then
            echo "Missing hard-sample subsets for ${pde}/${test_type}." >&2
            exit 2
        fi
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
    "${PYTHON_BIN}" -u scripts/tuning/run_hard_inverse_tuning.py
    --root "${ARTIFACT_ROOT}"
    --phase "${PHASE}"
    --pdes "${selected_pdes}"
    --samples-per-test-type "${SAMPLES_PER_TEST_TYPE}"
    --num-steps "${NUM_STEPS}"
    --device "${DEVICE}"
    "${microbatch_args[@]}"
)

printf 'A100 hard tuning: rank=%s pdes=%s phase=%s samples/test=%s steps=%s device=%s\n' \
    "${SERVER_RANK:-custom}" "${selected_pdes}" "${PHASE}" \
    "${SAMPLES_PER_TEST_TYPE}" "${NUM_STEPS}" "${DEVICE}"
printf 'COMMAND'
printf ' %q' "${command[@]}"
printf '\n'

case "${PLAN_ONLY,,}" in
    true|1|yes|on) exit 0 ;;
    false|0|no|off) ;;
    *) echo "PLAN_ONLY must be true or false" >&2; exit 2 ;;
esac

exec "${command[@]}"
