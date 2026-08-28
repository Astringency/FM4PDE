#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATAGEN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

usage() {
  cat <<'EOF'
Generate one of the explicit train/easytest/hardtest distributions.

Usage:
  bash data/DataGen/static/gen_pde.sh TYPE [options]
  bash data/DataGen/static/gen_pde.sh --type TYPE [options]

TYPE:
  train       Training distribution
  easytest    Smooth out-of-distribution test
  hardtest    Rough out-of-distribution test

Aliases: easy/smooth/test -> easytest; hard/rough -> hardtest

Options:
  --pdes LIST             Comma/space-separated PDEs (default: all five)
  --samples N             Samples per static/test file (default: 10000)
  --resolution N          Spatial resolution (default: 128)
  --out-root PATH         Dataset root (default: $PDE_DATA_ROOT or project default)
  --device DEVICE         NS device (default: cuda:0)
  --seed N                Override the profile's disjoint seed offset
  --helmholtz-k VALUE     Helmholtz wave number (default: 1)
  --ns-record-steps N     Saved positive-time NS frames (default: 10)
  --burgers-steps N       Burgers time intervals (default: 127)
  --overwrite             Replace an existing output file
  --dry-run               Print resolved profiles and commands only
  -h, --help              Show this help

Examples:
  bash data/DataGen/static/gen_pde.sh train
  bash data/DataGen/static/gen_pde.sh easytest --pdes poisson,helmholtz,darcy
  bash data/DataGen/static/gen_pde.sh hardtest --pdes nsnonbounded,burgers --samples 1000
EOF
}

canonical_type() {
  case "${1,,}" in
    train) printf '%s' 'train' ;;
    easytest|easy|smooth|test) printf '%s' 'easytest' ;;
    hardtest|hard|rough) printf '%s' 'hardtest' ;;
    *)
      printf 'Invalid dataset type: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
}

require_positive_integer() {
  local name="$1"
  local value="$2"
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    printf '%s must be a positive integer, got: %s\n' "$name" "$value" >&2
    exit 2
  fi
}

require_nonnegative_integer() {
  local name="$1"
  local value="$2"
  if [[ ! "$value" =~ ^[0-9]+$ ]]; then
    printf '%s must be a non-negative integer, got: %s\n' "$name" "$value" >&2
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

DATASET_TYPE=""
PDES_RAW="poisson,helmholtz,darcy,nsnonbounded,burgers"
SAMPLES=10000
RESOLUTION=128
OUT_ROOT="${PDE_DATA_ROOT:-/large_storage/zhangxf/PDEdata}"
DEVICE="cuda:0"
SEED=""
HELMHOLTZ_K=1
NS_RECORD_STEPS=10
BURGERS_STEPS=127
OVERWRITE=false
DRY_RUN=false

if [[ $# -gt 0 && "${1:0:1}" != "-" ]]; then
  DATASET_TYPE="$1"
  shift
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --type)
      DATASET_TYPE="${2:?--type requires a value}"
      shift 2
      ;;
    --pdes)
      PDES_RAW="${2:?--pdes requires a value}"
      shift 2
      ;;
    --samples)
      SAMPLES="${2:?--samples requires a value}"
      shift 2
      ;;
    --resolution)
      RESOLUTION="${2:?--resolution requires a value}"
      shift 2
      ;;
    --out-root)
      OUT_ROOT="${2:?--out-root requires a value}"
      shift 2
      ;;
    --device)
      DEVICE="${2:?--device requires a value}"
      shift 2
      ;;
    --seed)
      SEED="${2:?--seed requires a value}"
      shift 2
      ;;
    --helmholtz-k)
      HELMHOLTZ_K="${2:?--helmholtz-k requires a value}"
      shift 2
      ;;
    --ns-record-steps)
      NS_RECORD_STEPS="${2:?--ns-record-steps requires a value}"
      shift 2
      ;;
    --burgers-steps)
      BURGERS_STEPS="${2:?--burgers-steps requires a value}"
      shift 2
      ;;
    --overwrite)
      OVERWRITE=true
      shift
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'Unknown argument: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$DATASET_TYPE" ]]; then
  printf 'A dataset type is required.\n' >&2
  usage >&2
  exit 2
fi

DATASET_TYPE="$(canonical_type "$DATASET_TYPE")"
require_positive_integer '--samples' "$SAMPLES"
require_positive_integer '--resolution' "$RESOLUTION"
require_positive_integer '--ns-record-steps' "$NS_RECORD_STEPS"
require_positive_integer '--burgers-steps' "$BURGERS_STEPS"
if [[ ! "$HELMHOLTZ_K" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  printf '%s\n' '--helmholtz-k must be a non-negative number' >&2
  exit 2
fi

if [[ -z "$SEED" ]]; then
  case "$DATASET_TYPE" in
    train) SEED=0 ;;
    easytest) SEED=10000000 ;;
    hardtest) SEED=20000000 ;;
  esac
fi
require_nonnegative_integer '--seed' "$SEED"

case "$DATASET_TYPE" in
  train)
    STATIC_PROFILE='alpha=2.0 tau=3.0'
    TEMPORAL_PROFILE='alpha/gamma=2.5 tau=7.0'
    ;;
  easytest)
    STATIC_PROFILE='alpha=3.0 tau=4.0'
    TEMPORAL_PROFILE='alpha/gamma=3.0 tau=6.5'
    ;;
  hardtest)
    STATIC_PROFILE='alpha=1.5 tau=5.0'
    TEMPORAL_PROFILE='alpha/gamma=1.5 tau=5.0'
    ;;
esac

printf 'dataset type:      %s\n' "$DATASET_TYPE"
printf 'static profile:    %s\n' "$STATIC_PROFILE"
printf 'NS/Burgers profile:%s\n' " $TEMPORAL_PROFILE"
printf 'seed:              %s\n' "$SEED"
printf 'samples/resolution:%s / %s\n' " $SAMPLES" "$RESOLUTION"
printf 'output root:       %s\n' "$OUT_ROOT"

STATIC_DIR_M="$(matlab_quote "$SCRIPT_DIR")"
OUT_ROOT_M="$(matlab_quote "$OUT_ROOT")"
if [[ "$OVERWRITE" == true ]]; then
  MATLAB_OVERWRITE=true
else
  MATLAB_OVERWRITE=false
fi

PDES_NORMALIZED="${PDES_RAW//,/ }"
if [[ "${PDES_NORMALIZED,,}" == "all" ]]; then
  PDES_NORMALIZED="poisson helmholtz darcy nsnonbounded burgers"
fi
read -r -a PDE_ITEMS <<< "$PDES_NORMALIZED"
if [[ ${#PDE_ITEMS[@]} -eq 0 ]]; then
  printf '%s\n' '--pdes selected no equations' >&2
  exit 2
fi

for requested_pde in "${PDE_ITEMS[@]}"; do
  case "${requested_pde,,}" in
    poisson)
      command_text="addpath('${STATIC_DIR_M}'); generate_poisson('${DATASET_TYPE}',${SAMPLES},${RESOLUTION},'${OUT_ROOT_M}',${SEED},${MATLAB_OVERWRITE});"
      run_command "${MATLAB_BIN:-matlab}" -batch "$command_text"
      ;;
    helmholtz)
      command_text="addpath('${STATIC_DIR_M}'); generate_inhom_helmholtz('${DATASET_TYPE}',${SAMPLES},${RESOLUTION},${HELMHOLTZ_K},'${OUT_ROOT_M}',${SEED},${MATLAB_OVERWRITE});"
      run_command "${MATLAB_BIN:-matlab}" -batch "$command_text"
      ;;
    darcy)
      command_text="addpath('${STATIC_DIR_M}'); generate_darcy('${DATASET_TYPE}',${SAMPLES},${RESOLUTION},'${OUT_ROOT_M}',${SEED},${MATLAB_OVERWRITE});"
      run_command "${MATLAB_BIN:-matlab}" -batch "$command_text"
      ;;
    nsnonbounded|ns)
      if (( (RESOLUTION & (RESOLUTION - 1)) != 0 )); then
        printf 'NS resolution must be a power of two, got: %s\n' "$RESOLUTION" >&2
        exit 2
      fi
      ns_command=(
        "${PYTHON_BIN:-python}"
        "${DATAGEN_DIR}/time_dependent/gen_nbns.py"
        --dataset-type "$DATASET_TYPE"
        --total-samples "$SAMPLES"
        --resolution "$RESOLUTION"
        --record-steps "$NS_RECORD_STEPS"
        --device "$DEVICE"
        --seed-offset "$SEED"
        --out-dir "${OUT_ROOT}/nsnonbounded"
      )
      if [[ "$OVERWRITE" == true ]]; then
        ns_command+=(--overwrite)
      fi
      run_command "${ns_command[@]}"
      ;;
    burger|burgers)
      if (( RESOLUTION % 2 != 0 )); then
        printf 'Burgers resolution must be even, got: %s\n' "$RESOLUTION" >&2
        exit 2
      fi
      command_text="addpath('${STATIC_DIR_M}'); gen_burgers1('${DATASET_TYPE}',${SAMPLES},${RESOLUTION},${BURGERS_STEPS},'${OUT_ROOT_M}',${SEED},${MATLAB_OVERWRITE});"
      run_command "${MATLAB_BIN:-matlab}" -batch "$command_text"
      ;;
    *)
      printf 'Unsupported PDE: %s\n' "$requested_pde" >&2
      exit 2
      ;;
  esac
done
