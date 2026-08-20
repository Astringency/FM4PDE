#!/usr/bin/env bash
set -euo pipefail
# ============================================================================
# FM4PDE single sampling
#
# 用法:
#   PDE=poisson TASK=both bash scripts/sample/run_sample.sh
#   PDE=poisson TASK=forward SAMPLER_PHASE=deterministic bash scripts/sample/run_sample.sh
#   PDE=helmholtz TASK=inverse NUM_STEPS=1000 NUM_OBS=500 BATCH_SIZE=10 bash scripts/sample/run_sample.sh
#
# 环境变量:
#   PDE              方程名 (必选，如 poisson, helmholtz, darcy, nsnonbounded, burger)
#   TASK             任务: forward / inverse / both (默认 both)
#   SAMPLER_PHASE    采样器: stochastic / deterministic / hybrid_s2d (默认 stochastic)
#   BATCH_SIZE       批量大小 (默认 1)
#   OFFSET           样本偏移 (默认 0)
#   NUM_STEPS        采样步数 (默认从配置读取)
#   NUM_OBS          观测点数 (默认从配置读取)
#   DEVICE           设备 (默认 cuda)
#   OUTPUT_DIR       输出目录 (默认从配置读取)
#   SAMPLE_SEED      随机种子 (默认 42)
#   CONFIG_DIR       配置目录 (默认 configs/main)
#   DRY_RUN          仅校验不运行 (默认 false)
#   VIS              采样后绘图 (默认 false)
# ============================================================================

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PDE="${PDE:?请设置 PDE 环境变量，如 PDE=poisson}"
TASK="${TASK:-both}"
SAMPLER_PHASE="${SAMPLER_PHASE:-stochastic}"
BATCH_SIZE="${BATCH_SIZE:-1}"
OFFSET="${OFFSET:-0}"
DEVICE="${DEVICE:-cuda}"
CONFIG_DIR="${CONFIG_DIR:-configs/main}"

# ── 配置文件: configs/ablations/base/<pde>_<task>.yaml ───────────────────────
# CONFIG="${CONFIG_DIR}/${PDE}_${TASK}.yaml"
CONFIG="${CONFIG_DIR}/${PDE}.yaml"
if [[ ! -f "${CONFIG}" ]]; then
    echo "配置文件不存在: ${CONFIG}" >&2
    exit 2
fi

# ── 构建 override 参数 ────────────────────────────────────────────────────────
OVERRIDES=(
    --override "pde=${PDE}"
    --override "task=${TASK}"
    --override "sampler_phase=${SAMPLER_PHASE}"
    --override "batch_size=${BATCH_SIZE}"
    --override "offset=${OFFSET}"
    --override "device=${DEVICE}"
)

# 仅在显式设置时覆盖 (否则用配置文件的值)
[[ -n "${NUM_STEPS:-}" ]]   && OVERRIDES+=(--override "num_steps=${NUM_STEPS}")
[[ -n "${NUM_OBS:-}" ]]     && OVERRIDES+=(--override "num_obs=${NUM_OBS}")
[[ -n "${SAMPLE_SEED:-}" ]] && OVERRIDES+=(--override "sample_seed=${SAMPLE_SEED}")
[[ -n "${OUTPUT_DIR:-}" ]]  && OVERRIDES+=(--override "output_dir=${OUTPUT_DIR}")

# ── 运行 ──────────────────────────────────────────────────────────────────────
FLAGS=(--config "${CONFIG}" "${OVERRIDES[@]}")
[[ "${DRY_RUN:-false}" == "true" ]] && FLAGS+=(--dry-run)
[[ "${VIS:-false}" == "true" ]]    && FLAGS+=(--vis)

echo "== FM4PDE sampling =="
echo "  config:   ${CONFIG}"
echo "  pde:      ${PDE}"
echo "  task:     ${TASK}"
echo "  sampler:  ${SAMPLER_PHASE}"
echo "  device:   ${DEVICE}"
echo "  batch:    ${BATCH_SIZE}"

python -u -m sampling.runner "${FLAGS[@]}"
