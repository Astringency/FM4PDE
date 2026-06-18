#!/usr/bin/env bash
set -euo pipefail

# Formal FM4PDE sampling launcher.
#
# Defaults:
#   - sample all PDEs
#   - use configs/ablations/base/<pde>.yaml
#   - recursively find the newest checkpoint under CHECKPOINT_ROOT
#   - write outputs under outputs/samples/FM4PDE
#
# Examples:
#   bash scripts/sample/run_sample.sh
#   PDE=heat bash scripts/sample/run_sample.sh
#   PDE_LIST="heat wave nsnonbounded" CHECKPOINT_ROOT=outputs/pretrained/formal bash scripts/sample/run_sample.sh
#   PDE=heat CHECKPOINT_PATH=outputs/pretrained/formal/heat/<run>/fm4heat.pth bash scripts/sample/run_sample.sh
#   PDE=heat TEST_DATA_PATH=/path/to/heat_test.h5 NUM_STEPS=200 NUM_OBS=500 bash scripts/sample/run_sample.sh
#   CHECKPOINT_HEAT=/path/fm4heat.pth TEST_DATA_HEAT=/path/heat_test.h5 PDE=heat bash scripts/sample/run_sample.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

DEFAULT_DATA_ROOT="/large_storage/zhangxf/PDEdata"
DATA_ROOT="${DATA_ROOT:-${DEFAULT_DATA_ROOT}}"
DATA_ROOT="${DATA_ROOT%/}"
CONFIG_DIR="${CONFIG_DIR:-configs/ablations/base}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-outputs/pretrained}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/samples/FM4PDE}"
LOG_DIR="${LOG_DIR:-${OUTPUT_DIR}/logs}"

TASK="${TASK:-both}"
DEVICE="${DEVICE:-$(
  python - <<'PY'
try:
    import torch
    print("cuda" if torch.cuda.is_available() else "cpu")
except Exception:
    print("cpu")
PY
)}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_STEPS="${NUM_STEPS:-100}"
NUM_OBS="${NUM_OBS:-500}"
OFFSET="${OFFSET:-0}"
SAMPLER_PHASE="${SAMPLER_PHASE:-stochastic}"
LOSS_STATE="${LOSS_STATE:-endpoint}"
GUIDANCE_COMPONENTS="${GUIDANCE_COMPONENTS:-obs_pde}"
RESIDUAL_MODE="${RESIDUAL_MODE:-auto}"
MODEL_PROFILE="${MODEL_PROFILE:-recommended}"
SENSOR_MODE="${SENSOR_MODE:-}"
NOISE_LEVEL="${NOISE_LEVEL:-}"
SAMPLE_SEED="${SAMPLE_SEED:-}"
MASK_SEED="${MASK_SEED:-}"

DEFAULT_PDE_LIST=(
  darcy
  poisson
  helmholtz
  nsnonbounded
  burger
  reaction_diffusion
  shallow_water
  heat
  wave
  advection_diffusion
  steady_heat_conduction
)

if [[ -z "${PDE_LIST:-}" ]]; then
  if [[ -n "${PDE:-}" ]]; then
    PDE_LIST="${PDE}"
  else
    PDE_LIST="${DEFAULT_PDE_LIST[*]}"
  fi
fi
read -r -a PDES <<< "${PDE_LIST}"
if (( ${#PDES[@]} == 0 )); then
  echo "PDE_LIST resolved to an empty list." >&2
  exit 2
fi

upper_name() {
  local value="$1"
  value="${value^^}"
  printf '%s\n' "${value//[^A-Z0-9]/_}"
}

value_for_pde() {
  local prefix="$1"
  local pde="$2"
  local fallback="$3"
  local key
  key="${prefix}_$(upper_name "${pde}")"
  printf '%s\n' "${!key:-${fallback}}"
}

checkpoint_names_for_pde() {
  local pde="$1"
  printf 'fm4%s.pth\n' "${pde}"
  if [[ "${pde}" == "burger" ]]; then
    printf 'fm4burgers.pth\n'
  fi
}

latest_checkpoint_for_pde() {
  local pde="$1"
  local root="$2"
  local checkpoint_path
  local names=()
  if [[ ! -d "${root}" ]]; then
    printf '\n'
    return
  fi
  while IFS= read -r name; do
    names+=(-o -name "${name}")
  done < <(checkpoint_names_for_pde "${pde}")
  unset 'names[0]'
  checkpoint_path="$(
    find "${root}" -type f \( "${names[@]}" \) -printf '%T@ %p\n' 2>/dev/null \
      | sort -nr \
      | head -n 1 \
      | cut -d' ' -f2-
  )"
  printf '%s\n' "${checkpoint_path}"
}

checkpoint_for_pde() {
  local pde="$1"
  local per_pde
  per_pde="$(value_for_pde CHECKPOINT "${pde}" "")"
  if [[ -n "${per_pde}" ]]; then
    printf '%s\n' "${per_pde}"
  elif [[ -n "${CHECKPOINT_PATH:-}" && ${#PDES[@]} -eq 1 ]]; then
    printf '%s\n' "${CHECKPOINT_PATH}"
  else
    latest_checkpoint_for_pde "${pde}" "${CHECKPOINT_ROOT}"
  fi
}

config_for_pde() {
  local pde="$1"
  if [[ ${#PDES[@]} -eq 1 && -n "${CONFIG:-}" ]]; then
    printf '%s\n' "${CONFIG}"
  else
    printf '%s\n' "${CONFIG_DIR}/${pde}.yaml"
  fi
}

test_data_patterns_for_pde() {
  case "$1" in
    darcy)
      printf '%s\n' "darcy_test_*.mat" "*darcy*test*.mat"
      ;;
    poisson)
      printf '%s\n' "poisson_test_*.mat" "*poisson*test*.mat"
      ;;
    helmholtz)
      printf '%s\n' "helmholtz_test_*.mat" "*helmholtz*test*.mat"
      ;;
    nsnonbounded)
      printf '%s\n' "nsnonbounded_test_*.mat" "*nsnonbounded*test*.mat" "nsnonbounded_*-*-*-*.mat"
      ;;
    burger)
      printf '%s\n' "burger_test_*.mat" "burgers_test_*.mat" "*burger*test*.mat"
      ;;
    reaction_diffusion)
      printf '%s\n' "reaction_diffusion_test_*.h5" "*reaction_diffusion*test*.h5"
      ;;
    shallow_water)
      printf '%s\n' "shallow_water_test_*.h5" "swe_test_*.h5" "2d_swe_test_*.h5" "*shallow*water*test*.h5" "*swe*test*.h5"
      ;;
    heat)
      printf '%s\n' "heat_test_*.h5" "heat_fixed_test_*.h5" "*heat*test*.h5"
      ;;
    wave)
      printf '%s\n' "wave_test_*.h5" "*wave*test*.h5"
      ;;
    advection_diffusion)
      printf '%s\n' "advection_diffusion_test_*.h5" "*advection*diffusion*test*.h5"
      ;;
    steady_heat_conduction)
      printf '%s\n' "steady_heat_conduction_test_*.h5" "*steady*heat*conduction*test*.h5"
      ;;
    *)
      printf '%s\n' "*${1}*test*"
      ;;
  esac
}

pde_data_dirs() {
  local pde="$1"
  printf '%s\n' "${DATA_ROOT}/${pde}"
  if [[ "${pde}" == "burger" ]]; then
    printf '%s\n' "${DATA_ROOT}/burgers"
  fi
}

find_test_data_for_pde() {
  local pde="$1"
  local pde_dir
  local pattern
  local found
  while IFS= read -r pde_dir; do
    [[ -d "${pde_dir}" ]] || continue
    while IFS= read -r pattern; do
      found="$(
        find "${pde_dir}" -maxdepth 1 -type f -name "${pattern}" -printf '%T@ %p\n' \
          | sort -nr \
          | head -n 1 \
          | cut -d' ' -f2-
      )"
      if [[ -n "${found}" ]]; then
        printf '%s\n' "${found}"
        return
      fi
    done < <(test_data_patterns_for_pde "${pde}")
  done < <(pde_data_dirs "${pde}")
  printf '\n'
}

test_data_for_pde() {
  local pde="$1"
  local config="$2"
  local per_pde
  local discovered
  per_pde="$(value_for_pde TEST_DATA "${pde}" "")"
  if [[ -n "${per_pde}" ]]; then
    printf '%s\n' "${per_pde}"
    return
  fi
  if [[ -n "${TEST_DATA_PATH:-}" && ${#PDES[@]} -eq 1 ]]; then
    printf '%s\n' "${TEST_DATA_PATH}"
    return
  fi
  discovered="$(find_test_data_for_pde "${pde}")"
  if [[ -n "${discovered}" ]]; then
    printf '%s\n' "${discovered}"
    return
  fi
  local config_data_path
  config_data_path="$(awk -F': ' '/^data_path:/ {print $2; exit}' "${config}" | tr -d '"' || true)"
  if [[ -n "${config_data_path}" && "${DATA_ROOT}" != "${DEFAULT_DATA_ROOT}" && "${config_data_path}" == "${DEFAULT_DATA_ROOT}"* ]]; then
    printf '%s\n' "${config_data_path/${DEFAULT_DATA_ROOT}/${DATA_ROOT}}"
    return
  fi
  if [[ -n "${config_data_path}" ]]; then
    printf '%s\n' "${config_data_path}"
  fi
}

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

echo "== FM4PDE formal sampling =="
echo "repo: ${ROOT_DIR}"
echo "pdes: ${PDES[*]}"
echo "checkpoint_root: ${CHECKPOINT_ROOT}"
echo "data_root: ${DATA_ROOT}"
echo "output_dir: ${OUTPUT_DIR}"
echo "model_profile: ${MODEL_PROFILE}"

run_sample() {
  local pde="$1"
  local config
  local checkpoint_path
  local test_data_path
  local log_path
  local overrides=()

  config="$(config_for_pde "${pde}")"
  if [[ ! -f "${config}" ]]; then
    echo "Sampling config does not exist for ${pde}: ${config}" >&2
    exit 2
  fi

  checkpoint_path="$(checkpoint_for_pde "${pde}")"
  if [[ -z "${checkpoint_path}" || ! -s "${checkpoint_path}" ]]; then
    echo "Checkpoint for ${pde} was not found." >&2
    echo "Set CHECKPOINT_PATH for a single PDE, CHECKPOINT_${pde^^}, or CHECKPOINT_ROOT." >&2
    exit 2
  fi

  overrides+=(
    --override "pde=${pde}"
    --override "task=${TASK}"
    --override "checkpoint_path=${checkpoint_path}"
    --override "output_dir=${OUTPUT_DIR}"
    --override "device=${DEVICE}"
    --override "batch_size=${BATCH_SIZE}"
    --override "num_steps=${NUM_STEPS}"
    --override "num_obs=${NUM_OBS}"
    --override "offset=${OFFSET}"
    --override "sampler_phase=${SAMPLER_PHASE}"
    --override "loss_state=${LOSS_STATE}"
    --override "guidance_components=${GUIDANCE_COMPONENTS}"
    --override "model_profile=${MODEL_PROFILE}"
    --override "allow_synthetic_data=false"
  )
  if [[ "${RESIDUAL_MODE}" != "auto" ]]; then
    overrides+=(--override "residual_mode=${RESIDUAL_MODE}")
  fi
  if [[ -n "${SENSOR_MODE}" ]]; then
    overrides+=(--override "sensor_mode=${SENSOR_MODE}")
  fi
  if [[ -n "${NOISE_LEVEL}" ]]; then
    overrides+=(--override "noise_level=${NOISE_LEVEL}")
  fi
  if [[ -n "${SAMPLE_SEED}" ]]; then
    overrides+=(--override "sample_seed=${SAMPLE_SEED}")
  fi
  if [[ -n "${MASK_SEED}" ]]; then
    overrides+=(--override "mask_seed=${MASK_SEED}")
  fi

  test_data_path="$(test_data_for_pde "${pde}" "${config}")"
  if [[ -n "${test_data_path}" ]]; then
    if [[ ! -e "${test_data_path}" ]]; then
      echo "Test data path for ${pde} does not exist: ${test_data_path}" >&2
      echo "Expected formal test data under these directories:" >&2
      pde_data_dirs "${pde}" >&2
      echo "Matching one of:" >&2
      test_data_patterns_for_pde "${pde}" >&2
      exit 2
    fi
    overrides+=(--override "data_path=${test_data_path}")
  fi

  for item in ${EXTRA_OVERRIDES:-}; do
    overrides+=(--override "${item}")
  done

  log_path="${LOG_DIR}/sample_${pde}.log"
  echo
  echo "============================================================"
  echo "PDE: ${pde}"
  echo "config: ${config}"
  echo "checkpoint: ${checkpoint_path}"
  if [[ -n "${test_data_path}" ]]; then
    echo "test_data: ${test_data_path}"
  fi
  echo "log: ${log_path}"
  echo "============================================================"

  python -u -m sampling.runner \
    --config "${config}" \
    "${overrides[@]}" 2>&1 | tee "${log_path}"
}

for pde in "${PDES[@]}"; do
  run_sample "${pde}"
done

echo
echo "== Aggregating sampling metrics =="
python -u -m sampling.aggregate "${OUTPUT_DIR}" --output-dir "${OUTPUT_DIR}" \
  2>&1 | tee "${LOG_DIR}/aggregate.log"

echo
echo "== Sampling launcher complete =="
echo "output_dir: ${OUTPUT_DIR}"
echo "summary_all_raw: ${OUTPUT_DIR}/summary_all_raw.csv"
echo "logs: ${LOG_DIR}"
