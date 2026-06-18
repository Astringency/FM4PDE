#!/usr/bin/env bash
set -euo pipefail

# End-to-end FM4PDE check run over all PDE datasets by default:
#   1. load training data from DATA_ROOT
#   2. train each PDE for 10-20 epochs
#   3. save checkpoints under outputs/checkrun/train/<pde>
#   4. run sampling on each PDE's formal test data
#   5. write per-PDE metrics plus an all-PDE summary under outputs/checkrun
#
# Default command:
#   bash scripts/checkrun.sh
#
# Common overrides:
#   EPOCHS=20 bash scripts/checkrun.sh
#   PDE=heat bash scripts/checkrun.sh
#   PDE_LIST="heat wave nsnonbounded" bash scripts/checkrun.sh
#   MAX_TRAIN_SAMPLES=32 SAMPLE_NUM_STEPS=20 bash scripts/checkrun.sh
#   DATA_ROOT=/large_storage/zhangxf/PDEdata OUTPUT_ROOT=outputs/checkrun bash scripts/checkrun.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

DEFAULT_DATA_ROOT="/large_storage/zhangxf/PDEdata"
DATA_ROOT="${DATA_ROOT:-${DEFAULT_DATA_ROOT}}"
DATA_ROOT="${DATA_ROOT%/}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/checkrun}"
SAMPLE_OUTPUT_DIR="${SAMPLE_OUTPUT_DIR:-${OUTPUT_ROOT}/sampling}"
LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/logs}"
CONFIG_DIR="${CONFIG_DIR:-configs/ablations/base}"

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

EPOCHS="${EPOCHS:-10}"
if (( EPOCHS < 10 || EPOCHS > 20 )); then
  echo "EPOCHS must be between 10 and 20 for this check run; got ${EPOCHS}" >&2
  exit 2
fi

DATA_SIZE="${DATA_SIZE:-1}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-16}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
ACCUM_ITER="${ACCUM_ITER:-1}"
NUM_WORKERS="${NUM_WORKERS:-0}"
LR="${LR:-0.0001}"
SEED="${SEED:-0}"

SAMPLE_BATCH_SIZE="${SAMPLE_BATCH_SIZE:-1}"
SAMPLE_NUM_STEPS="${SAMPLE_NUM_STEPS:-10}"
NUM_OBS="${NUM_OBS:-128}"
OFFSET="${OFFSET:-0}"
GUIDANCE_COMPONENTS="${GUIDANCE_COMPONENTS:-obs_pde}"
RESIDUAL_MODE="${RESIDUAL_MODE:-auto}"
MODEL_PROFILE="${MODEL_PROFILE:-recommended}"

CONDA_ENV="${CONDA_ENV:-fm4pde}"
PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  if command -v conda >/dev/null 2>&1; then
    CONDA_PREFIX_FOR_ENV="$(conda env list | awk -v env="${CONDA_ENV}" '$1 == env {print $NF; exit}')"
    if [[ -n "${CONDA_PREFIX_FOR_ENV}" && -x "${CONDA_PREFIX_FOR_ENV}/bin/python" ]]; then
      PYTHON_BIN="${CONDA_PREFIX_FOR_ENV}/bin/python"
    fi
  fi
fi
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "DATA_ROOT does not exist: ${DATA_ROOT}" >&2
  exit 2
fi

DEVICE="${DEVICE:-$("${PYTHON_BIN}" - <<'PY'
try:
    import torch
    print("cuda" if torch.cuda.is_available() else "cpu")
except Exception:
    print("cpu")
PY
)}"

mkdir -p "${OUTPUT_ROOT}" "${SAMPLE_OUTPUT_DIR}" "${LOG_DIR}"

echo "== FM4PDE all-PDE check run =="
echo "repo: ${ROOT_DIR}"
echo "python: ${PYTHON_BIN}"
echo "pdes: ${PDES[*]}"
echo "data_root: ${DATA_ROOT}"
echo "device: ${DEVICE}"
echo "epochs: ${EPOCHS}"
echo "model_profile: ${MODEL_PROFILE}"
echo "output_root: ${OUTPUT_ROOT}"

# Force a single-process check run by default. This avoids accidental SLURM or
# torchrun environment variables changing train.py into distributed mode.
unset RANK WORLD_SIZE LOCAL_RANK MASTER_ADDR MASTER_PORT
unset SLURM_PROCID SLURM_LOCALID SLURM_NTASKS

infer_train_data_path() {
  if [[ -n "${TRAIN_DATA_PATH:-}" ]]; then
    printf '%s\n' "${TRAIN_DATA_PATH}"
  else
    printf '%s\n' "${DATA_ROOT}"
  fi
}

upper_name() {
  local value="$1"
  value="${value^^}"
  printf '%s\n' "${value//[^A-Z0-9]/_}"
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

pde_data_dirs() {
  local pde="$1"
  printf '%s\n' "${DATA_ROOT}/${pde}"
  if [[ "${pde}" == "burger" ]]; then
    printf '%s\n' "${DATA_ROOT}/burgers"
  fi
}

test_data_override_for_pde() {
  local pde="$1"
  local config="$2"
  local per_pde_key
  local discovered
  local config_data_path
  per_pde_key="TEST_DATA_$(upper_name "${pde}")"
  if [[ -n "${!per_pde_key:-}" ]]; then
    printf '%s\n' "${!per_pde_key}"
    return
  fi
  if [[ -n "${TEST_DATA_PATH:-}" ]]; then
    printf '%s\n' "${TEST_DATA_PATH}"
    return
  fi
  discovered="$(find_test_data_for_pde "${pde}")"
  if [[ -n "${discovered}" ]]; then
    printf '%s\n' "${discovered}"
    return
  fi
  config_data_path="$(awk -F': ' '/^data_path:/ {print $2; exit}' "${config}" | tr -d '"' || true)"
  if [[ -n "${config_data_path}" && "${DATA_ROOT}" != "${DEFAULT_DATA_ROOT}" && "${config_data_path}" == "${DEFAULT_DATA_ROOT}"* ]]; then
    printf '%s\n' "${config_data_path/${DEFAULT_DATA_ROOT}/${DATA_ROOT}}"
    return
  fi
  if [[ -n "${config_data_path}" ]]; then
    printf '%s\n' "${config_data_path}"
  fi
}

latest_checkpoint_for_pde() {
  local pde="$1"
  local train_output_dir="$2"
  local checkpoint_path
  local train_parent
  local train_prefix
  train_parent="$(dirname "${train_output_dir}")"
  train_prefix="$(basename "${train_output_dir}")"
  checkpoint_path="$(
    {
      find "${train_output_dir}" -type f -name "fm4${pde}.pth" -printf '%T@ %p\n'
      find "${train_parent}" -maxdepth 2 -path "${train_parent}/${train_prefix}*" -type f -name "fm4${pde}.pth" -printf '%T@ %p\n'
    } | sort -nr | head -n 1 | cut -d' ' -f2-
  )"
  if [[ -z "${checkpoint_path}" ]]; then
    checkpoint_path="$(
      {
        find "${train_output_dir}" -type f -name "fm4${pde}-checkpoint.pth" -printf '%T@ %p\n'
        find "${train_parent}" -maxdepth 2 -path "${train_parent}/${train_prefix}*" -type f -name "fm4${pde}-checkpoint.pth" -printf '%T@ %p\n'
      } | sort -nr | head -n 1 | cut -d' ' -f2-
    )"
  fi
  printf '%s\n' "${checkpoint_path}"
}

run_one_pde() {
  local pde="$1"
  local config
  local train_data_path
  local train_output_dir
  local checkpoint_path
  local test_data_path
  local metrics_path

  config="$(config_for_pde "${pde}")"
  if [[ ! -f "${config}" ]]; then
    echo "Sampling config does not exist for ${pde}: ${config}" >&2
    exit 2
  fi

  train_data_path="$(infer_train_data_path "${pde}")"
  train_output_dir="${OUTPUT_ROOT}/train/${pde}"
  mkdir -p "${train_output_dir}" "${SAMPLE_OUTPUT_DIR}" "${LOG_DIR}"

  echo
  echo "============================================================"
  echo "PDE: ${pde}"
  echo "train_data_path: ${train_data_path}"
  echo "sampling_config: ${config}"
  echo "train_output_dir: ${train_output_dir}"
  echo "============================================================"

  local train_args=(
    -u train.py
    --dataset "${pde}"
    --data_path "${train_data_path}"
    --output_dir "${train_output_dir}/"
    --epochs "${EPOCHS}"
    --batch_size "${TRAIN_BATCH_SIZE}"
    --accum_iter "${ACCUM_ITER}"
    --data_size "${DATA_SIZE}"
    --max_train_samples "${MAX_TRAIN_SAMPLES}"
    --num_workers "${NUM_WORKERS}"
    --device "${DEVICE}"
    --lr "${LR}"
    --seed "${SEED}"
    --model_profile "${MODEL_PROFILE}"
    --eval_frequency -1
  )
  if [[ "${pde}" == "reaction_diffusion" ]]; then
    train_args+=(--rd_init_mode_filter "${RD_INIT_MODE_FILTER:-grf}")
  fi
  if [[ "${USE_EMA:-0}" == "1" ]]; then
    train_args+=(--use_ema)
  fi

  echo
  echo "== Training ${pde} =="
  "${PYTHON_BIN}" "${train_args[@]}" 2>&1 | tee "${LOG_DIR}/train_${pde}.log"

  checkpoint_path="$(latest_checkpoint_for_pde "${pde}" "${train_output_dir}")"
  if [[ ! -s "${checkpoint_path}" ]]; then
    echo "Expected trained checkpoint was not created under: ${train_output_dir}" >&2
    echo "Looked for fm4${pde}.pth or fm4${pde}-checkpoint.pth recursively, including old sibling dirs like ${train_output_dir}260618-..." >&2
    exit 1
  fi
  echo "checkpoint: ${checkpoint_path}"

  local sample_overrides=(
    --override "pde=${pde}"
    --override "checkpoint_path=${checkpoint_path}"
    --override "output_dir=${SAMPLE_OUTPUT_DIR}"
    --override "device=${DEVICE}"
    --override "batch_size=${SAMPLE_BATCH_SIZE}"
    --override "num_steps=${SAMPLE_NUM_STEPS}"
    --override "num_obs=${NUM_OBS}"
    --override "offset=${OFFSET}"
    --override "guidance_components=${GUIDANCE_COMPONENTS}"
    --override "model_profile=${MODEL_PROFILE}"
    --override "allow_synthetic_data=false"
  )
  if [[ "${RESIDUAL_MODE}" != "auto" ]]; then
    sample_overrides+=(--override "residual_mode=${RESIDUAL_MODE}")
  fi
  test_data_path="$(test_data_override_for_pde "${pde}" "${config}")"
  if [[ -n "${test_data_path}" ]]; then
    if [[ ! -e "${test_data_path}" ]]; then
      echo "Test data path does not exist for ${pde}: ${test_data_path}" >&2
      echo "Expected formal test data under these directories:" >&2
      pde_data_dirs "${pde}" >&2
      echo "Matching one of:" >&2
      test_data_patterns_for_pde "${pde}" >&2
      exit 2
    fi
    sample_overrides+=(--override "data_path=${test_data_path}")
  fi

  echo
  echo "== Sampling ${pde} =="
  "${PYTHON_BIN}" -u -m sampling.runner \
    --config "${config}" \
    "${sample_overrides[@]}" 2>&1 | tee "${LOG_DIR}/sampling_${pde}.log"

  metrics_path="$(
    find "${SAMPLE_OUTPUT_DIR}/${pde}" -name metrics_final.json -printf '%T@ %p\n' \
      | sort -nr \
      | head -n 1 \
      | cut -d' ' -f2-
  )"
  if [[ -z "${metrics_path}" || ! -f "${metrics_path}" ]]; then
    echo "Could not find sampling metrics_final.json under ${SAMPLE_OUTPUT_DIR}/${pde}" >&2
    exit 1
  fi

  cp "${metrics_path}" "${OUTPUT_ROOT}/metrics_final_${pde}.json"
  "${PYTHON_BIN}" - "${pde}" "${metrics_path}" "${OUTPUT_ROOT}/checkrun_summary_${pde}.json" <<'PY'
import json
import sys
from pathlib import Path

pde = sys.argv[1]
metrics_path = Path(sys.argv[2])
summary_path = Path(sys.argv[3])
metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
keys = [
    "status",
    "run_dir",
    "wall_clock_time",
    "rel_l2_a",
    "rel_l2_u",
    "obs_rel_l2_a",
    "obs_rel_l2_u",
    "L_pde",
    "pde_residual_norm",
    "pde_residual_status",
    "num_recorded_steps",
]
summary = {"pde": pde}
summary.update({key: metrics.get(key) for key in keys if key in metrics})
summary["metrics_path"] = str(metrics_path)
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2, sort_keys=True))
PY
}

for pde in "${PDES[@]}"; do
  run_one_pde "${pde}"
done

echo
echo "== Aggregating all PDE sampling metrics =="
"${PYTHON_BIN}" -u -m sampling.aggregate "${SAMPLE_OUTPUT_DIR}" --output-dir "${OUTPUT_ROOT}" \
  2>&1 | tee "${LOG_DIR}/aggregate_all.log"

"${PYTHON_BIN}" - "${OUTPUT_ROOT}" "${PDES[@]}" <<'PY'
import json
import sys
from pathlib import Path

output_root = Path(sys.argv[1])
pdes = sys.argv[2:]
summaries = []
missing = []
for pde in pdes:
    path = output_root / f"checkrun_summary_{pde}.json"
    if not path.exists():
        missing.append(pde)
        continue
    summaries.append(json.loads(path.read_text(encoding="utf-8")))
payload = {
    "status": "ok" if not missing and all(item.get("status") == "ok" for item in summaries) else "incomplete",
    "num_pdes": len(pdes),
    "num_summaries": len(summaries),
    "missing": missing,
    "summaries": summaries,
}
(output_root / "checkrun_summary_all.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(payload, indent=2, sort_keys=True))
PY

echo
echo "== Done =="
echo "all summary: ${OUTPUT_ROOT}/checkrun_summary_all.json"
echo "aggregate raw csv: ${OUTPUT_ROOT}/summary_all_raw.csv"
echo "logs: ${LOG_DIR}"
