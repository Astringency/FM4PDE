#!/usr/bin/env bash
set -euo pipefail

# Formal FM4PDE endpoint-pair HDF5 data generation.
#
# Defaults:
#   - PDEs: heat, wave, advection_diffusion, steady_heat_conduction
#   - train: 50000 samples, split into 5 shards of 10000
#   - test: 10000 samples
#   - spatial resolution: 128 x 128
#   - recorded time nodes: 11 (initial time + 10 intervals)
#   - time-dependent PDEs save full_trajectory
#   - output root: /large_storage/zhangxf/PDEdata
#
# Run from the repository root or from any directory:
#   bash data/DataGen/run_generate_pair_h5s_50k_10k_fulltraj.sh
#
# Useful overrides:
#   PYTHON_BIN=/path/to/python bash data/DataGen/run_generate_pair_h5s_50k_10k_fulltraj.sh
#   DATA_ROOT=/other/path OVERWRITE=1 bash data/DataGen/run_generate_pair_h5s_50k_10k_fulltraj.sh
#   CHUNK_SIZE=64 COMPRESSION=gzip bash data/DataGen/run_generate_pair_h5s_50k_10k_fulltraj.sh
#   SPLIT=test N_TEST=10000 bash data/DataGen/run_generate_pair_h5s_50k_10k_fulltraj.sh
#   DRY_RUN=1 bash data/DataGen/run_generate_pair_h5s_50k_10k_fulltraj.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-/large_storage/zhangxf/PDEdata}"

RESOLUTION="${RESOLUTION:-128}"
N_TRAIN="${N_TRAIN:-50000}"
N_TEST="${N_TEST:-10000}"
TRAIN_SHARDS="${TRAIN_SHARDS:-5}"
SAMPLES_PER_SHARD="${SAMPLES_PER_SHARD:-10000}"
N_TIME="${N_TIME:-11}"
T_FINAL="${T_FINAL:-1.0}"
SPLIT="${SPLIT:-both}"
CHUNK_SIZE="${CHUNK_SIZE:-128}"
COMPRESSION="${COMPRESSION:-lzf}"
COMPRESSION_LEVEL="${COMPRESSION_LEVEL:-4}"
DTYPE="${DTYPE:-float32}"

PDE_LIST=(
  heat
  wave
  advection_diffusion
  steady_heat_conduction
)

OVERWRITE_ARGS=()
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  OVERWRITE_ARGS=(--overwrite)
fi

DRY_RUN_ARGS=()
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  DRY_RUN_ARGS=(--dry-run)
fi

if [[ "${DRY_RUN:-0}" != "1" ]]; then
  mkdir -p "${DATA_ROOT}"
fi

cat <<EOF
FM4PDE formal endpoint-pair HDF5 generation
  repo:              ${REPO_ROOT}
  python:            ${PYTHON_BIN}
  data root:         ${DATA_ROOT}
  PDEs:              ${PDE_LIST[*]}
  split:             ${SPLIT}
  resolution:        ${RESOLUTION}x${RESOLUTION}
  train samples:     ${N_TRAIN} (${TRAIN_SHARDS} shards x ${SAMPLES_PER_SHARD})
  test samples:      ${N_TEST}
  n_time:            ${N_TIME} recorded nodes
  save trajectory:   true
  chunk size:        ${CHUNK_SIZE}
  compression:       ${COMPRESSION}
  overwrite:         ${OVERWRITE:-0}
  dry run:           ${DRY_RUN:-0}

Note: full_trajectory substantially increases disk usage for time-dependent PDEs.
EOF

for PDE in "${PDE_LIST[@]}"; do
  echo
  echo "===== Generating ${PDE} ====="
  "${PYTHON_BIN}" "${REPO_ROOT}/data/DataGen/python/generate_pair_h5s.py" \
    --pde "${PDE}" \
    --out-root "${DATA_ROOT}" \
    --resolution "${RESOLUTION}" \
    --n-train "${N_TRAIN}" \
    --n-test "${N_TEST}" \
    --split "${SPLIT}" \
    --train-shards "${TRAIN_SHARDS}" \
    --samples-per-shard "${SAMPLES_PER_SHARD}" \
    --n-time "${N_TIME}" \
    --T "${T_FINAL}" \
    --chunk-size "${CHUNK_SIZE}" \
    --compression "${COMPRESSION}" \
    --compression-level "${COMPRESSION_LEVEL}" \
    --dtype "${DTYPE}" \
    --save-trajectory \
    "${DRY_RUN_ARGS[@]}" \
    "${OVERWRITE_ARGS[@]}"
done

echo
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "Dry run completed. No files were written."
else
  echo "All requested PDE datasets have been generated under ${DATA_ROOT}."
fi
