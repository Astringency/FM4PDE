#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PDE="${PDE:?请设置 PDE 环境变量，如 PDE=poisson}"
TASK="${TASK:-both}"
SAMPLER_PHASE="${SAMPLER_PHASE:-stochastic}"
BATCH_SIZE="${BATCH_SIZE:-1}"
OFFSET="${OFFSET:-0}"
DEVICE="${DEVICE:-cuda}"
CONFIG_DIR="${CONFIG_DIR:-configs/main}"
TEST_TYPE="${TEST_TYPE:-id}"

case "${TASK}" in
    both|forward|inverse) ;;
    *)
        echo "TASK 必须是 both、forward 或 inverse，当前值: ${TASK}" >&2
        exit 2
        ;;
esac

case "${TEST_TYPE}" in
    id|smooth|rough) ;;
    *)
        echo "TEST_TYPE 必须是 id、smooth 或 rough，当前值: ${TEST_TYPE}" >&2
        exit 2
        ;;
esac

CONFIG="${CONFIG_DIR}/${TASK}/${PDE}.yaml"
if [[ ! -f "${CONFIG}" ]]; then
    echo "配置文件不存在或该 PDE/TASK 组合不受支持: ${CONFIG}" >&2
    exit 2
fi

OVERRIDES=(
    --override "pde=${PDE}"
    --override "task=${TASK}"
    --override "sampler_phase=${SAMPLER_PHASE}"
    --override "batch_size=${BATCH_SIZE}"
    --override "offset=${OFFSET}"
    --override "device=${DEVICE}"
    --override "test_type=${TEST_TYPE}"
)

[[ -n "${NUM_STEPS:-}" ]]   && OVERRIDES+=(--override "num_steps=${NUM_STEPS}")
[[ -n "${NUM_OBS:-}" ]]     && OVERRIDES+=(--override "num_obs=${NUM_OBS}")
[[ -n "${SENSOR_MODE:-}" ]] && OVERRIDES+=(--override "sensor_mode=${SENSOR_MODE}")
[[ -n "${SAMPLE_SEED:-}" ]] && OVERRIDES+=(--override "sample_seed=${SAMPLE_SEED}")
[[ -n "${PDE_GUIDANCE_START_RATIO:-}" ]] && OVERRIDES+=(--override "pde_guidance_start_ratio=${PDE_GUIDANCE_START_RATIO}")
[[ -n "${PDE_GUIDANCE_RAMP_RATIO:-}" ]] && OVERRIDES+=(--override "pde_guidance_ramp_ratio=${PDE_GUIDANCE_RAMP_RATIO}")
[[ -n "${OUTPUT_DIR:-}" ]]  && OVERRIDES+=(--override "output_dir=${OUTPUT_DIR}")

FLAGS=(--config "${CONFIG}" "${OVERRIDES[@]}")
[[ "${DRY_RUN:-false}" == "true" ]] && FLAGS+=(--dry-run)
[[ "${VIS:-false}" == "true" ]]    && FLAGS+=(--vis)

echo "== FM4PDE sampling =="
echo "  config:   ${CONFIG}"
echo "  pde:      ${PDE}"
echo "  task:     ${TASK}"
echo "  sampler:  ${SAMPLER_PHASE}"
echo "  sensor:   ${SENSOR_MODE:-config default}"
echo "  test:     ${TEST_TYPE}"
echo "  device:   ${DEVICE}"
echo "  batch:    ${BATCH_SIZE}"

exec "${PYTHON_BIN:-python}" -u -m sampling.runner "${FLAGS[@]}" "$@"
