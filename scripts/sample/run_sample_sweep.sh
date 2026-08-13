#!/usr/bin/env bash
set -euo pipefail
# ============================================================================
# FM4PDE 批量采样扫描脚本
#
# 对多个 PDE × task × sampler × N 样本自动分片调用 run_sample.sh。
# 跑完后自动聚合结果。
#
# 用法:
#   bash scripts/sample/run_sample_sweep.sh
#   NUM_SAMPLES=1000 bash scripts/sample/run_sample_sweep.sh
#   PDE_LIST="poisson darcy" TASK_LIST="forward both" bash scripts/sample/run_sample_sweep.sh
#
# 环境变量:
#   NUM_SAMPLES      每 PDE×task×sampler 样本数 (默认 1000)
#   MAX_BATCH_SIZE   单次 GPU 最大 batch (默认 50)
#   PDE_LIST         方程名, 空格分隔 (默认全部 5 个)
#   TASK_LIST        任务, 空格分隔 (默认 forward inverse both)
#   SAMPLER_LIST     采样器, 空格分隔 (默认 stochastic deterministic hybrid_s2d)
#   NUM_STEPS        采样步数 (默认 100)
#   NUM_OBS          观测点数 (默认 500)
#   OUTPUT_DIR       输出根目录 (默认 outputs/samples)
#   DEVICE           设备 (默认 cuda)
# ============================================================================

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

NUM_SAMPLES="${NUM_SAMPLES:-1000}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-50}"

DEFAULT_PDES="poisson helmholtz darcy nsnonbounded burger"
PDE_LIST="${PDE_LIST:-${DEFAULT_PDES}}"
read -r -a PDES <<< "${PDE_LIST}"

DEFAULT_TASKS="forward inverse both"
TASK_LIST="${TASK_LIST:-${DEFAULT_TASKS}}"
read -r -a TASKS <<< "${TASK_LIST}"

DEFAULT_SAMPLERS="stochastic deterministic hybrid_s2d"
SAMPLER_LIST="${SAMPLER_LIST:-${DEFAULT_SAMPLERS}}"
read -r -a SAMPLERS <<< "${SAMPLER_LIST}"

NUM_STEPS="${NUM_STEPS:-100}"
NUM_OBS="${NUM_OBS:-500}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/samples}"
DEVICE="${DEVICE:-cuda}"

SAMPLE_SCRIPT="scripts/sample/run_sample.sh"

echo "============================================================"
echo " FM4PDE Sampling Sweep"
echo "============================================================"
echo " PDEs:       ${PDES[*]}"
echo " Tasks:      ${TASKS[*]}"
echo " Samplers:   ${SAMPLERS[*]}"
echo " Samples:    ${NUM_SAMPLES} per config"
echo " Max batch:  ${MAX_BATCH_SIZE}"
echo " Steps:      ${NUM_STEPS}"
echo " Obs:        ${NUM_OBS}"
echo " Output:     ${OUTPUT_DIR}"
echo " Device:     ${DEVICE}"
echo "============================================================"

# ── 计算分片 ──────────────────────────────────────────────────────────────────
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

# ── 单个采样任务 ──────────────────────────────────────────────────────────────
run_one() {
    local pde="$1" task="$2" sampler="$3" batch_size="$4" offset="$5"
    echo "  [$(date '+%H:%M:%S')] pde=${pde} task=${task} sampler=${sampler} batch=${batch_size} offset=${offset}"

    PDE="${pde}" \
    TASK="${task}" \
    SAMPLER_PHASE="${sampler}" \
    BATCH_SIZE="${batch_size}" \
    OFFSET="${offset}" \
    NUM_STEPS="${NUM_STEPS}" \
    NUM_OBS="${NUM_OBS}" \
    OUTPUT_DIR="${OUTPUT_DIR}" \
    DEVICE="${DEVICE}" \
    bash "${SAMPLE_SCRIPT}" 2>&1 | grep -E "rel_l2|status|Error|Step" | tail -1 || true
}

# ── 主循环 ────────────────────────────────────────────────────────────────────
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "${LOG_DIR}"

MAIN_LOG="${LOG_DIR}/sweep_${TIMESTAMP}.log"
exec > >(tee -a "${MAIN_LOG}") 2>&1

START_TIME=$(date +%s)

TOTAL_CONFIGS=$((${#PDES[@]} * ${#TASKS[@]} * ${#SAMPLERS[@]}))
CURRENT=0

for pde in "${PDES[@]}"; do
    for task in "${TASKS[@]}"; do
        for sampler in "${SAMPLERS[@]}"; do
            CURRENT=$((CURRENT + 1))
            echo ""
            echo "--- [${CURRENT}/${TOTAL_CONFIGS}] ${pde}/${task}/${sampler} ---"

            while IFS=' ' read -r chunk_size chunk_offset; do
                [[ -z "${chunk_size}" ]] && continue
                run_one "${pde}" "${task}" "${sampler}" "${chunk_size}" "${chunk_offset}"
            done < <(compute_chunks "${NUM_SAMPLES}" "${MAX_BATCH_SIZE}")
        done
    done
done

END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))

echo ""
echo "============================================================"
echo " Sweep complete"
echo " Elapsed:   ${ELAPSED}s ($(( ELAPSED / 60 )) min)"
echo " Output:    ${OUTPUT_DIR}"
echo "============================================================"

# ── 聚合 ──────────────────────────────────────────────────────────────────────
echo ""
echo "== Aggregating metrics =="
python -u -m sampling.aggregate "${OUTPUT_DIR}" --output-dir "${OUTPUT_DIR}" \
    2>&1 | tee -a "${LOG_DIR}/aggregate_${TIMESTAMP}.log"

echo ""
echo "== Done =="
echo "summary: ${OUTPUT_DIR}/summary_all_raw.csv"
