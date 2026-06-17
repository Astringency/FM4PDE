#!/usr/bin/env bash
set -euo pipefail

DATA_PATH="${DATA_PATH:-/large_storage/zhangxf/PDEdata/}"
OUTPUT_DIR="${OUTPUT_DIR:-/research_data/users/zhangxifeng/C01Python/FM4PDE/output/pretrained/}"
EPOCHS="${EPOCHS:-500}"
SAVE_FULL_PDE_PARAMS="${SAVE_FULL_PDE_PARAMS:-0}"

extra_args=()
if [[ "${SAVE_FULL_PDE_PARAMS}" == "1" ]]; then
  extra_args+=(--save_full_pde_params)
fi

run_train() {
  local port="$1"
  local nproc="$2"
  local batch_size="$3"
  local accum_iter="$4"
  local dataset="$5"

  torchrun \
    --master_addr=127.0.0.1 \
    --master_port="${port}" \
    --nproc_per_node="${nproc}" \
    train.py \
    --batch_size="${batch_size}" \
    --epochs="${EPOCHS}" \
    --accum_iter="${accum_iter}" \
    --decay_lr \
    --dataset="${dataset}" \
    --data_path="${DATA_PATH}" \
    --output_dir="${OUTPUT_DIR}" \
    "${extra_args[@]}"
}

run_train 12356 "${NPROC_DARCY:-4}" 10 256 darcy
run_train 12355 "${NPROC_POISSON:-4}" 10 256 poisson
run_train 12354 "${NPROC_HELMHOLTZ:-2}" 10 256 helmholtz
run_train 12353 "${NPROC_NS:-4}" 10 256 nsnonbounded
run_train 12352 "${NPROC_SHALLOW_WATER:-4}" 10 256 shallow_water
run_train 12351 "${NPROC_REACTION_DIFFUSION:-2}" 2 512 reaction_diffusion
run_train 12350 "${NPROC_HEAT:-4}" 10 256 heat
run_train 12349 "${NPROC_WAVE:-4}" 10 256 wave
run_train 12348 "${NPROC_ADVECTION_DIFFUSION:-4}" 10 256 advection_diffusion
run_train 12347 "${NPROC_STEADY_HEAT_CONDUCTION:-4}" 10 256 steady_heat_conduction
