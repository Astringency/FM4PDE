#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# FM4PDE 主实验采样脚本 — 4 PDEs × 4 tasks × N samples
#
# 用法:
#   bash scripts/sample/run_sample_sweep.sh              # 默认 100 样本
#   NUM_SAMPLES=1000 bash scripts/sample/run_sample_sweep.sh
#   PDE_LIST="poisson darcy" NUM_SAMPLES=100 TASK_LIST="forward both" bash scripts/sample/run_sample_sweep.sh
#
# 环境变量:
#   NUM_SAMPLES      每 PDE×task 的样本数 (默认 100)
#   MAX_BATCH_SIZE   单次 GPU 最大 batch (默认 100, 超出的自动分片)
#   PDE_LIST         要运行的 PDE, 空格分隔 (默认 poisson darcy helmholtz nsnonbounded)
#   TASK_LIST        要运行的 task, 空格分隔 (默认 forward inverse both unconditional)
#   CHECKPOINT_ROOT  checkpoint 根目录 (默认 outputs/pretrained/formal)
#   DATA_ROOT        测试数据根目录 (默认 /home/tat512/C01Python/PDEdata)
#   OUTPUT_DIR       输出根目录  (默认 outputs/samples/sweep)
#   DEVICE           设备 (默认自动检测)
#   NUM_STEPS        采样步数 (默认 100)
#   NUM_OBS          观测点数量 (默认 500)
# ============================================================================

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

# ---- 可配置参数 ----
NUM_SAMPLES="${NUM_SAMPLES:-100}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-100}"

DEFAULT_PDES="poisson darcy helmholtz nsnonbounded"
PDE_LIST="${PDE_LIST:-${DEFAULT_PDES}}"
read -r -a PDES <<< "${PDE_LIST}"

DEFAULT_TASKS="forward inverse both unconditional"
TASK_LIST="${TASK_LIST:-${DEFAULT_TASKS}}"
read -r -a TASKS <<< "${TASK_LIST}"

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-outputs/pretrained/formal}"
DATA_ROOT="${DATA_ROOT:-/home/tat512/C01Python/PDEdata}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/samples/sweep}"
DEVICE="${DEVICE:-$(
  python - <<'PY'
try:
    import torch
    print("cuda" if torch.cuda.is_available() else "cpu")
except Exception:
    print("cpu")
PY
)}"
NUM_STEPS="${NUM_STEPS:-100}"
NUM_OBS="${NUM_OBS:-500}"
SAMPLE_SEED="${SAMPLE_SEED:-42}"

echo "============================================================"
echo " FM4PDE Sampling Sweep"
echo "============================================================"
echo " PDEs:       ${PDES[*]}"
echo " Tasks:      ${TASKS[*]}"
echo " Samples:    ${NUM_SAMPLES} per PDE×task"
echo " Max batch:  ${MAX_BATCH_SIZE}"
echo " Checkpoint: ${CHECKPOINT_ROOT}"
echo " Data root:  ${DATA_ROOT}"
echo " Output:     ${OUTPUT_DIR}"
echo " Device:     ${DEVICE}"
echo " Num steps:  ${NUM_STEPS}"
echo " Num obs:    ${NUM_OBS}"
echo "============================================================"

# ---- 计算分片 ----
# 将 NUM_SAMPLES 拆成多个 batch_size ≤ MAX_BATCH_SIZE 的 chunk
compute_chunks() {
    local total="$1"
    local max_batch="$2"
    local remaining="$total"
    local offset=0
    while (( remaining > 0 )); do
        local chunk=$(( remaining > max_batch ? max_batch : remaining ))
        echo "${chunk} ${offset}"
        offset=$(( offset + chunk ))
        remaining=$(( remaining - chunk ))
    done
}

# ---- 单个采样任务 ----
run_single() {
    local pde="$1"
    local task="$2"
    local batch_size="$3"
    local offset="$4"

    local guidance="obs_pde"
    if [[ "${task}" == "unconditional" ]]; then
        guidance="noguide"
    fi

    echo "  [$(date '+%H:%M:%S')] pde=${pde} task=${task} batch=${batch_size} offset=${offset}"

    PDE="${pde}" \
    TASK="${task}" \
    BATCH_SIZE="${batch_size}" \
    OFFSET="${offset}" \
    CHECKPOINT_ROOT="${CHECKPOINT_ROOT}" \
    DATA_ROOT="${DATA_ROOT}" \
    OUTPUT_DIR="${OUTPUT_DIR}" \
    DEVICE="${DEVICE}" \
    NUM_STEPS="${NUM_STEPS}" \
    NUM_OBS="${NUM_OBS}" \
    GUIDANCE_COMPONENTS="${guidance}" \
    SAMPLE_SEED="${SAMPLE_SEED}" \
    bash scripts/sample/run_sample.sh
}

# ---- 主循环 ----
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "${LOG_DIR}"

MAIN_LOG="${LOG_DIR}/sweep_${TIMESTAMP}.log"
exec > >(tee -a "${MAIN_LOG}") 2>&1

START_TIME=$(date +%s)

for pde in "${PDES[@]}"; do
    for task in "${TASKS[@]}"; do
        echo ""
        echo "--- PDE=${pde}  TASK=${task}  SAMPLES=${NUM_SAMPLES} ---"

        chunk_index=0
        while IFS=' ' read -r chunk_size chunk_offset; do
            [[ -z "${chunk_size}" ]] && continue
            run_single "${pde}" "${task}" "${chunk_size}" "${chunk_offset}"
            chunk_index=$(( chunk_index + 1 ))
        done < <(compute_chunks "${NUM_SAMPLES}" "${MAX_BATCH_SIZE}")
    done
done

END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))

echo ""
echo "============================================================"
echo " Sweep complete"
echo " Elapsed:   ${ELAPSED}s ($(( ELAPSED / 60 )) min)"
echo " Output:    ${OUTPUT_DIR}"
echo " Log:       ${MAIN_LOG}"
echo "============================================================"

# ---- 聚合所有结果 ----
echo ""
echo "== Aggregating metrics =="
python -u -m sampling.aggregate "${OUTPUT_DIR}" --output-dir "${OUTPUT_DIR}" \
    2>&1 | tee -a "${LOG_DIR}/aggregate_${TIMESTAMP}.log"

echo ""
echo "== Done =="
echo "summary: ${OUTPUT_DIR}/summary_all_raw.csv"
