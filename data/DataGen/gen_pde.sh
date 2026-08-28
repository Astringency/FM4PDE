#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATIC_DIR="${SCRIPT_DIR}/static"
PYTHON_DIR="${SCRIPT_DIR}/python"
TIME_DIR="${SCRIPT_DIR}/time_dependent"

usage() {
  cat <<'EOF'
Generate FM4PDE datasets for any of the 11 supported PDEs.

Usage:
  PDE=poisson bash data/DataGen/gen_pde.sh
  bash data/DataGen/gen_pde.sh PDE=poisson TYPE=smooth
  PDE=poisson TYPE=smooth bash data/DataGen/gen_pde.sh
  PDE="heat wave" TYPE=train bash data/DataGen/gen_pde.sh
  PDE=all TYPE=all bash data/DataGen/gen_pde.sh

Required environment variable:
  PDE                   One or more PDE names, or all

Optional environment variables:
  TYPE=all              all, train, id, smooth, or rough
  TRAIN_SHARDS=5        Number of training files
  SAMPLES_PER_SHARD=10000
  TEST_SAMPLES=10000
  RESOLUTION=128
  OUT_ROOT=/large_storage/zhangxf/PDEdata
  DEVICE=cuda:0         Device used by nsnonbounded
  OVERWRITE=false       true replaces existing files; false skips them
  DRY_RUN=false         true prints commands without generating data
  PYTHON_BIN=python
  MATLAB_BIN=matlab

Supported PDEs:
  poisson helmholtz darcy nsnonbounded burger heat wave
  advection_diffusion reaction_diffusion shallow_water
  steady_heat_conduction

Without TYPE, one PDE invocation creates five training shards plus id, smooth,
and rough test files. Existing files are skipped unless OVERWRITE=true.
EOF
}

canonical_type() {
  case "${1,,}" in
    all) printf '%s' 'all' ;;
    train) printf '%s' 'train' ;;
    id|test) printf '%s' 'id' ;;
    smooth|easy|easytest) printf '%s' 'smooth' ;;
    rough|hard|hardtest) printf '%s' 'rough' ;;
    *)
      printf 'Unsupported TYPE: %s\n' "$1" >&2
      exit 2
      ;;
  esac
}

canonical_pde() {
  case "${1,,}" in
    poisson|helmholtz|darcy|heat|wave|advection_diffusion|reaction_diffusion|shallow_water|steady_heat_conduction)
      printf '%s' "${1,,}"
      ;;
    ns|nsnonbounded) printf '%s' 'nsnonbounded' ;;
    burger|burgers) printf '%s' 'burger' ;;
    *)
      printf 'Unsupported PDE: %s\n' "$1" >&2
      exit 2
      ;;
  esac
}

seed_for_type() {
  case "$1" in
    train) printf '%s' '0' ;;
    id) printf '%s' '10000000' ;;
    smooth) printf '%s' '20000000' ;;
    rough) printf '%s' '30000000' ;;
  esac
}

require_positive_integer() {
  local name="$1"
  local value="$2"
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    printf '%s must be a positive integer, got %s\n' "$name" "$value" >&2
    exit 2
  fi
}

matlab_quote() {
  local value="$1"
  value="${value//\'/\'\'}"
  printf '%s' "$value"
}

run_command() {
  if [[ "$DRY_RUN" == true ]]; then
    printf 'DRY-RUN:'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

run_matlab_static() {
  local pde="$1"
  local dataset_type="$2"
  local samples="$3"
  local seed="$4"
  local shard_id="$5"
  local command_text

  case "$pde" in
    poisson)
      command_text="addpath('${STATIC_DIR_M}'); generate_poisson('${dataset_type}',${samples},${RESOLUTION},'${OUT_ROOT_M}',${seed},${MATLAB_OVERWRITE},${shard_id});"
      ;;
    helmholtz)
      command_text="addpath('${STATIC_DIR_M}'); generate_inhom_helmholtz('${dataset_type}',${samples},${RESOLUTION},1,'${OUT_ROOT_M}',${seed},${MATLAB_OVERWRITE},${shard_id});"
      ;;
    darcy)
      command_text="addpath('${STATIC_DIR_M}'); generate_darcy('${dataset_type}',${samples},${RESOLUTION},'${OUT_ROOT_M}',${seed},${MATLAB_OVERWRITE},${shard_id});"
      ;;
    burger)
      command_text="addpath('${STATIC_DIR_M}'); gen_burgers1('${dataset_type}',${samples},${RESOLUTION},${BURGERS_STEPS},'${OUT_ROOT_M}',${seed},${MATLAB_OVERWRITE},${shard_id});"
      ;;
  esac
  run_command "$MATLAB_BIN" -batch "$command_text"
}

run_nsnonbounded() {
  local dataset_type="$1"
  local seed="$2"
  local total_samples
  local command=(
    "$PYTHON_BIN" "${TIME_DIR}/gen_nbns.py"
    --dataset-type "$dataset_type"
    --resolution "$RESOLUTION"
    --record-steps "$NS_RECORD_STEPS"
    --device "$DEVICE"
    --seed-offset "$seed"
    --out-dir "${OUT_ROOT}/nsnonbounded"
  )
  if [[ "$dataset_type" == train ]]; then
    total_samples=$((TRAIN_SHARDS * SAMPLES_PER_SHARD))
    command+=(--total-samples "$total_samples" --samples-per-file "$SAMPLES_PER_SHARD")
  else
    command+=(--total-samples "$TEST_SAMPLES" --samples-per-file "$TEST_SAMPLES")
  fi
  if [[ "$OVERWRITE" == true ]]; then
    command+=(--overwrite)
  fi
  run_command "${command[@]}"
}

run_pair_h5() {
  local pde="$1"
  local dataset_type="$2"
  local seed="$3"
  local total_train=$((TRAIN_SHARDS * SAMPLES_PER_SHARD))
  local command=(
    "$PYTHON_BIN" "${PYTHON_DIR}/generate_pair_h5s.py"
    --pde "$pde"
    --out-root "$OUT_ROOT"
    --resolution "$RESOLUTION"
    --dataset-type "$dataset_type"
  )
  if [[ "$dataset_type" == train ]]; then
    command+=(
      --split train
      --n-train "$total_train"
      --train-shards "$TRAIN_SHARDS"
      --samples-per-shard "$SAMPLES_PER_SHARD"
      --base-seed-train "$seed"
    )
  else
    command+=(--split test --n-test "$TEST_SAMPLES" --base-seed-test "$seed")
  fi
  if [[ "$OVERWRITE" == true ]]; then
    command+=(--overwrite)
  fi
  run_command "${command[@]}"
}

run_reaction_diffusion() {
  local dataset_type="$1"
  local seed="$2"
  local total_train=$((TRAIN_SHARDS * SAMPLES_PER_SHARD))
  local command=(
    "$PYTHON_BIN" "${TIME_DIR}/gen_rd.py"
    --save-path "${OUT_ROOT}/reaction_diffusion"
    --resolution "$RESOLUTION"
    --n-save-steps "$RD_SAVE_STEPS"
    --init-mode grf
    --dataset-type "$dataset_type"
    --seed-offset "$seed"
  )
  if [[ "$dataset_type" == train ]]; then
    command+=(--split train --total-samples "$total_train" --samples-per-file "$SAMPLES_PER_SHARD")
  else
    command+=(--split test --total-samples "$TEST_SAMPLES" --samples-per-file "$TEST_SAMPLES")
  fi
  if [[ "$OVERWRITE" == true ]]; then
    command+=(--overwrite)
  fi
  run_command "${command[@]}"
}

run_shallow_water() {
  local dataset_type="$1"
  local seed="$2"
  local total_train=$((TRAIN_SHARDS * SAMPLES_PER_SHARD))
  local command=(
    "$PYTHON_BIN" "${TIME_DIR}/gen_swe.py"
    --out-dir "${OUT_ROOT}/shallow_water"
    --resolution "$RESOLUTION"
    --tsteps "$SWE_STEPS"
    --dataset-type "$dataset_type"
    --base-seed "$seed"
  )
  if [[ "$dataset_type" == train ]]; then
    command+=(--split train --total-samples "$total_train" --samples-per-file "$SAMPLES_PER_SHARD")
  else
    command+=(--split test --total-samples "$TEST_SAMPLES" --samples-per-file "$TEST_SAMPLES")
  fi
  if [[ "$OVERWRITE" == true ]]; then
    command+=(--overwrite)
  fi
  run_command "${command[@]}"
}

generate_one() {
  local pde="$1"
  local dataset_type="$2"
  local seed
  seed="$(seed_for_type "$dataset_type")"
  printf '\n[%s] type=%s seed=%s\n' "$pde" "$dataset_type" "$seed"

  case "$pde" in
    poisson|helmholtz|darcy|burger)
      if [[ "$dataset_type" == train ]]; then
        local shard
        for ((shard = 1; shard <= TRAIN_SHARDS; shard++)); do
          local shard_seed=$(((shard - 1) * SAMPLES_PER_SHARD))
          run_matlab_static "$pde" train "$SAMPLES_PER_SHARD" "$shard_seed" "$shard"
        done
      else
        run_matlab_static "$pde" "$dataset_type" "$TEST_SAMPLES" "$seed" 0
      fi
      ;;
    nsnonbounded)
      run_nsnonbounded "$dataset_type" "$seed"
      ;;
    heat|wave|advection_diffusion|steady_heat_conduction)
      run_pair_h5 "$pde" "$dataset_type" "$seed"
      ;;
    reaction_diffusion)
      run_reaction_diffusion "$dataset_type" "$seed"
      ;;
    shallow_water)
      run_shallow_water "$dataset_type" "$seed"
      ;;
  esac
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    PDE=*) PDE="${1#PDE=}" ;;
    TYPE=*) TYPE="${1#TYPE=}" ;;
    TRAIN_SHARDS=*) TRAIN_SHARDS="${1#TRAIN_SHARDS=}" ;;
    SAMPLES_PER_SHARD=*) SAMPLES_PER_SHARD="${1#SAMPLES_PER_SHARD=}" ;;
    TEST_SAMPLES=*) TEST_SAMPLES="${1#TEST_SAMPLES=}" ;;
    RESOLUTION=*) RESOLUTION="${1#RESOLUTION=}" ;;
    OUT_ROOT=*) OUT_ROOT="${1#OUT_ROOT=}" ;;
    DEVICE=*) DEVICE="${1#DEVICE=}" ;;
    OVERWRITE=*) OVERWRITE="${1#OVERWRITE=}" ;;
    DRY_RUN=*) DRY_RUN="${1#DRY_RUN=}" ;;
    PYTHON_BIN=*) PYTHON_BIN="${1#PYTHON_BIN=}" ;;
    MATLAB_BIN=*) MATLAB_BIN="${1#MATLAB_BIN=}" ;;
    NS_RECORD_STEPS=*) NS_RECORD_STEPS="${1#NS_RECORD_STEPS=}" ;;
    BURGERS_STEPS=*) BURGERS_STEPS="${1#BURGERS_STEPS=}" ;;
    RD_SAVE_STEPS=*) RD_SAVE_STEPS="${1#RD_SAVE_STEPS=}" ;;
    SWE_STEPS=*) SWE_STEPS="${1#SWE_STEPS=}" ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

PDE_RAW="${PDE:-}"
if [[ -z "$PDE_RAW" ]]; then
  printf 'PDE is required, for example: PDE=poisson bash data/DataGen/gen_pde.sh\n' >&2
  usage >&2
  exit 2
fi

TYPE="$(canonical_type "${TYPE:-all}")"
TRAIN_SHARDS="${TRAIN_SHARDS:-5}"
SAMPLES_PER_SHARD="${SAMPLES_PER_SHARD:-10000}"
TEST_SAMPLES="${TEST_SAMPLES:-10000}"
RESOLUTION="${RESOLUTION:-128}"
OUT_ROOT="${OUT_ROOT:-${PDE_DATA_ROOT:-/large_storage/zhangxf/PDEdata}}"
DEVICE="${DEVICE:-cuda:0}"
OVERWRITE="${OVERWRITE:-false}"
DRY_RUN="${DRY_RUN:-false}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MATLAB_BIN="${MATLAB_BIN:-matlab}"
NS_RECORD_STEPS="${NS_RECORD_STEPS:-10}"
BURGERS_STEPS="${BURGERS_STEPS:-127}"
RD_SAVE_STEPS="${RD_SAVE_STEPS:-10}"
SWE_STEPS="${SWE_STEPS:-10}"

require_positive_integer TRAIN_SHARDS "$TRAIN_SHARDS"
require_positive_integer SAMPLES_PER_SHARD "$SAMPLES_PER_SHARD"
require_positive_integer TEST_SAMPLES "$TEST_SAMPLES"
require_positive_integer RESOLUTION "$RESOLUTION"
require_positive_integer NS_RECORD_STEPS "$NS_RECORD_STEPS"
require_positive_integer BURGERS_STEPS "$BURGERS_STEPS"
require_positive_integer RD_SAVE_STEPS "$RD_SAVE_STEPS"
require_positive_integer SWE_STEPS "$SWE_STEPS"
if [[ "$OVERWRITE" != true && "$OVERWRITE" != false ]]; then
  printf 'OVERWRITE must be true or false\n' >&2
  exit 2
fi
if [[ "$DRY_RUN" != true && "$DRY_RUN" != false ]]; then
  printf 'DRY_RUN must be true or false\n' >&2
  exit 2
fi

STATIC_DIR_M="$(matlab_quote "$STATIC_DIR")"
OUT_ROOT_M="$(matlab_quote "$OUT_ROOT")"
MATLAB_OVERWRITE="$OVERWRITE"

PDE_NORMALIZED="${PDE_RAW//,/ }"
if [[ "${PDE_NORMALIZED,,}" == all ]]; then
  PDE_NORMALIZED="poisson helmholtz darcy nsnonbounded burger heat wave advection_diffusion reaction_diffusion shallow_water steady_heat_conduction"
fi
read -r -a RAW_PDES <<< "$PDE_NORMALIZED"
PDES=()
for raw_pde in "${RAW_PDES[@]}"; do
  PDES+=("$(canonical_pde "$raw_pde")")
done

if [[ "$TYPE" == all ]]; then
  TYPES=(train id smooth rough)
else
  TYPES=("$TYPE")
fi

printf 'PDEs: %s\n' "${PDES[*]}"
printf 'Types: %s\n' "${TYPES[*]}"
printf 'Train: %s shards x %s samples\n' "$TRAIN_SHARDS" "$SAMPLES_PER_SHARD"
printf 'Tests: %s samples per type\n' "$TEST_SAMPLES"
printf 'Resolution: %s\n' "$RESOLUTION"
printf 'Output root: %s\n' "$OUT_ROOT"
printf 'Overwrite: %s\n' "$OVERWRITE"

for selected_pde in "${PDES[@]}"; do
  for selected_type in "${TYPES[@]}"; do
    generate_one "$selected_pde" "$selected_type"
  done
done
