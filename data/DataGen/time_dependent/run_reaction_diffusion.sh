#!/usr/bin/env bash
set -euo pipefail

# FM4PDE 2D reaction-diffusion generator.
# Train and test use disjoint seed offsets to avoid leakage.

PYTHON_BIN="${PYTHON_BIN:-python}"
SAVE_PATH="${SAVE_PATH:-/large_storage/zhangxf/PDEdata/reaction_diffusion}"
N_TRAIN="${N_TRAIN:-50000}"
N_TEST="${N_TEST:-10000}"
TRAIN_SAMPLES_PER_FILE="${TRAIN_SAMPLES_PER_FILE:-10000}"
TEST_SAMPLES_PER_FILE="${TEST_SAMPLES_PER_FILE:-10000}"
RESOLUTION="${RESOLUTION:-128}"
T_FINAL="${T_FINAL:-1.0}"
N_SAVE_STEPS="${N_SAVE_STEPS:-10}"
INIT_MODE="${INIT_MODE:-grf}"
OVERWRITE_ARGS=()

if [[ "${OVERWRITE:-0}" == "1" ]]; then
  OVERWRITE_ARGS=(--overwrite)
fi

"${PYTHON_BIN}" data/DataGen/time_dependent/gen_rd.py \
  --save-path "${SAVE_PATH}" \
  --total-samples "${N_TRAIN}" \
  --samples-per-file "${TRAIN_SAMPLES_PER_FILE}" \
  --resolution "${RESOLUTION}" \
  --T "${T_FINAL}" \
  --n-save-steps "${N_SAVE_STEPS}" \
  --init-mode "${INIT_MODE}" \
  --split train \
  --seed-offset 0 \
  "${OVERWRITE_ARGS[@]}"

"${PYTHON_BIN}" data/DataGen/time_dependent/gen_rd.py \
  --save-path "${SAVE_PATH}" \
  --total-samples "${N_TEST}" \
  --samples-per-file "${TEST_SAMPLES_PER_FILE}" \
  --resolution "${RESOLUTION}" \
  --T "${T_FINAL}" \
  --n-save-steps "${N_SAVE_STEPS}" \
  --init-mode "${INIT_MODE}" \
  --split test \
  --seed-offset 10000000 \
  "${OVERWRITE_ARGS[@]}"

# iid comparison:
# INIT_MODE=iid bash data/DataGen/time_dependent/run_reaction_diffusion.sh
