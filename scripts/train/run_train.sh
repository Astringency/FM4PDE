#!/usr/bin/env bash
set -euo pipefail

# Formal FM4PDE training launcher.
#
# Defaults:
#   - train all PDEs
#   - 300 epochs
#   - torchrun multi-GPU, auto-detecting 4 GPUs when available, otherwise 2 or 1
#   - small per-GPU batch with gradient accumulation to TARGET_EFFECTIVE_BATCH
#   - data root: /large_storage/zhangxf/PDEdata/
#   - checkpoints/logs under outputs/pretrained/formal/<pde>/<timestamp...>/
#
# Examples:
#   bash scripts/train/run_train.sh
#   PDE=heat bash scripts/train/run_train.sh
#   PDE=heat SCALAR_CONDITIONING_PARAMS="alpha T" bash scripts/train/run_train.sh
#   PDE_LIST="heat wave nsnonbounded" NPROC_PER_NODE=2 bash scripts/train/run_train.sh
#   EPOCHS=300 TARGET_EFFECTIVE_BATCH=64 OUTPUT_DIR=outputs/pretrained/formal bash scripts/train/run_train.sh
#   MAX_TRAIN_SAMPLES=1024 DATA_SIZE=1 bash scripts/train/run_train.sh
#   PDE=heat TRAIN_DATA_CONFIG=configs/training_data.yaml bash scripts/train/run_train.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

DATA_PATH="${DATA_PATH:-/large_storage/zhangxf/PDEdata/}"
DATA_PATH="${DATA_PATH%/}/"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/pretrained/formal}"
OUTPUT_DIR="${OUTPUT_DIR%/}"
LOG_DIR="${LOG_DIR:-${OUTPUT_DIR}/logs}"
EPOCHS="${EPOCHS:-300}"
DATA_SIZE="${DATA_SIZE:-5}"
TRAIN_DATA_CONFIG="${TRAIN_DATA_CONFIG:-}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-}"
TARGET_EFFECTIVE_BATCH="${TARGET_EFFECTIVE_BATCH:-64}"
NUM_WORKERS="${NUM_WORKERS:-8}"
LR="${LR:-0.0001}"
SEED="${SEED:-0}"
SAMPLING_DTYPE="${SAMPLING_DTYPE:-float32}"
MODEL_PROFILE="${MODEL_PROFILE:-recommended}"
LR_SCHEDULER="${LR_SCHEDULER:-warmup_cosine}"
MIN_LR="${MIN_LR:-0.000001}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"
EVAL_FREQUENCY="${EVAL_FREQUENCY:-50}"
SCALAR_CONDITIONING_PARAMS="${SCALAR_CONDITIONING_PARAMS:-}"
RD_INIT_MODE_FILTER="${RD_INIT_MODE_FILTER:-grf}"
SAVE_FULL_PDE_PARAMS="${SAVE_FULL_PDE_PARAMS:-0}"
USE_EMA="${USE_EMA:-0}"
FUSED_ADAMW="${FUSED_ADAMW:-1}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-1}"
DDP_STATIC_GRAPH="${DDP_STATIC_GRAPH:-1}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-29500}"

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

detect_nproc() {
  if [[ -n "${NPROC_PER_NODE:-}" ]]; then
    printf '%s\n' "${NPROC_PER_NODE}"
    return
  fi
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    local visible="${CUDA_VISIBLE_DEVICES// /}"
    if [[ -n "${visible}" ]]; then
      local count
      count="$(awk -F',' '{print NF}' <<< "${visible}")"
      if (( count >= 4 )); then
        printf '4\n'
      elif (( count >= 2 )); then
        printf '2\n'
      else
        printf '1\n'
      fi
      return
    fi
  fi
  python - <<'PY'
try:
    import torch
    count = torch.cuda.device_count() if torch.cuda.is_available() else 0
except Exception:
    count = 0
if count >= 4:
    print(4)
elif count >= 2:
    print(2)
else:
    print(1)
PY
}

NPROC_DEFAULT="$(detect_nproc)"
DEVICE_DEFAULT="$(
  python - <<'PY'
try:
    import torch
    print("cuda" if torch.cuda.is_available() else "cpu")
except Exception:
    print("cpu")
PY
)"

upper_name() {
  local value="$1"
  value="${value^^}"
  printf '%s\n' "${value//[^A-Z0-9]/_}"
}

per_gpu_batch_default() {
  case "$1" in
    reaction_diffusion|shallow_water|wave)
      printf '2\n'
      ;;
    *)
      printf '4\n'
      ;;
  esac
}

ceil_div() {
  local numerator="$1"
  local denominator="$2"
  printf '%s\n' $(( (numerator + denominator - 1) / denominator ))
}

value_for_pde() {
  local prefix="$1"
  local pde="$2"
  local fallback="$3"
  local key
  key="${prefix}_$(upper_name "${pde}")"
  printf '%s\n' "${!key:-${fallback}}"
}

device_for_nproc() {
  local nproc="$1"
  if (( nproc > 1 )); then
    printf 'cuda\n'
    return
  fi
  printf '%s\n' "${DEVICE:-${DEVICE_DEFAULT}}"
}

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

echo "== FM4PDE formal training =="
echo "repo: ${ROOT_DIR}"
echo "pdes: ${PDES[*]}"
echo "data_path: ${DATA_PATH}"
echo "output_dir: ${OUTPUT_DIR}"
echo "epochs: ${EPOCHS}"
echo "model_profile: ${MODEL_PROFILE}"
echo "lr_scheduler: ${LR_SCHEDULER}"
echo "min_lr: ${MIN_LR}"
echo "warmup_epochs: ${WARMUP_EPOCHS}"
echo "eval_frequency: ${EVAL_FREQUENCY}"
echo "scalar_conditioning_params: ${SCALAR_CONDITIONING_PARAMS:-<disabled>}"
echo "default_nproc_per_node: ${NPROC_DEFAULT}"
echo "target_effective_batch: ${TARGET_EFFECTIVE_BATCH}"
echo "fused_adamw: ${FUSED_ADAMW}"
echo "persistent_workers: ${PERSISTENT_WORKERS}"
echo "ddp_static_graph: ${DDP_STATIC_GRAPH}"
echo "train_data_config: ${TRAIN_DATA_CONFIG:-<automatic discovery>}"

run_train() {
  local index="$1"
  local pde="$2"
  local nproc
  local batch_size
  local accum_iter
  local target_batch
  local effective_batch
  local port
  local pde_output_dir
  local device
  local log_path
  local eval_frequency
  local scalar_params
  local scalar_param_array
  local extra_args=()

  nproc="$(value_for_pde NPROC "${pde}" "${NPROC_DEFAULT}")"
  batch_size="$(value_for_pde BATCH "${pde}" "$(per_gpu_batch_default "${pde}")")"
  target_batch="$(value_for_pde TARGET_EFFECTIVE_BATCH "${pde}" "${TARGET_EFFECTIVE_BATCH}")"
  accum_iter="$(value_for_pde ACCUM "${pde}" "")"
  if [[ -z "${accum_iter}" ]]; then
    accum_iter="$(ceil_div "${target_batch}" $(( batch_size * nproc )))"
    if (( accum_iter < 1 )); then
      accum_iter=1
    fi
  fi
  effective_batch=$(( batch_size * nproc * accum_iter ))
  port=$(( MASTER_PORT_BASE + index ))
  pde_output_dir="${OUTPUT_DIR}/${pde}/"
  device="$(device_for_nproc "${nproc}")"
  log_path="${LOG_DIR}/train_${pde}.log"
  eval_frequency="${EVAL_FREQUENCY}"
  scalar_params="$(value_for_pde SCALAR_CONDITIONING_PARAMS "${pde}" "${SCALAR_CONDITIONING_PARAMS}")"

  if [[ "${pde}" == "reaction_diffusion" ]]; then
    extra_args+=(--rd_init_mode_filter "${RD_INIT_MODE_FILTER}")
  fi
  if [[ -n "${scalar_params}" ]]; then
    read -r -a scalar_param_array <<< "${scalar_params}"
    extra_args+=(--scalar_conditioning_params "${scalar_param_array[@]}")
  fi
  if [[ -n "${MAX_TRAIN_SAMPLES}" ]]; then
    extra_args+=(--max_train_samples "${MAX_TRAIN_SAMPLES}")
  fi
  if [[ -n "${TRAIN_DATA_CONFIG}" ]]; then
    extra_args+=(--train_data_config "${TRAIN_DATA_CONFIG}")
  fi
  if [[ "${SAVE_FULL_PDE_PARAMS}" == "1" ]]; then
    extra_args+=(--save_full_pde_params)
  fi
  if [[ "${USE_EMA}" == "1" ]]; then
    extra_args+=(--use_ema)
  fi
  if [[ "${FUSED_ADAMW}" == "0" ]]; then
    extra_args+=(--no-fused_adamw)
  fi
  if [[ "${PERSISTENT_WORKERS}" == "0" ]]; then
    extra_args+=(--no-persistent_workers)
  fi
  if [[ "${DDP_STATIC_GRAPH}" == "0" ]]; then
    extra_args+=(--no-ddp_static_graph)
  fi
  if [[ -n "${RESUME:-}" && ${#PDES[@]} -eq 1 ]]; then
    extra_args+=(--resume "${RESUME}")
  fi

  echo
  echo "============================================================"
  echo "PDE: ${pde}"
  echo "nproc_per_node: ${nproc}"
  echo "batch_size_per_gpu: ${batch_size}"
  echo "accum_iter: ${accum_iter}"
  echo "effective_batch: ${effective_batch}"
  echo "scalar_conditioning_params: ${scalar_params:-<disabled>}"
  echo "output_dir: ${pde_output_dir}<timestamp>/"
  echo "log: ${log_path}"
  echo "============================================================"

  if (( nproc > 1 )); then
    torchrun \
      --master_addr=127.0.0.1 \
      --master_port="${port}" \
      --nproc_per_node="${nproc}" \
      train.py \
      --batch_size="${batch_size}" \
      --epochs="${EPOCHS}" \
      --accum_iter="${accum_iter}" \
      --dataset="${pde}" \
      --data_path="${DATA_PATH}" \
      --data_size="${DATA_SIZE}" \
      --output_dir="${pde_output_dir}" \
      --device="${device}" \
      --num_workers="${NUM_WORKERS}" \
      --lr="${LR}" \
      --min_lr="${MIN_LR}" \
      --seed="${SEED}" \
      --sampling_dtype="${SAMPLING_DTYPE}" \
      --model_profile="${MODEL_PROFILE}" \
      --lr_scheduler="${LR_SCHEDULER}" \
      --warmup_epochs="${WARMUP_EPOCHS}" \
      --eval_frequency="${eval_frequency}" \
      "${extra_args[@]}" 2>&1 | tee "${log_path}"
  else
    python -u train.py \
      --batch_size="${batch_size}" \
      --epochs="${EPOCHS}" \
      --accum_iter="${accum_iter}" \
      --dataset="${pde}" \
      --data_path="${DATA_PATH}" \
      --data_size="${DATA_SIZE}" \
      --output_dir="${pde_output_dir}" \
      --device="${device}" \
      --num_workers="${NUM_WORKERS}" \
      --lr="${LR}" \
      --min_lr="${MIN_LR}" \
      --seed="${SEED}" \
      --sampling_dtype="${SAMPLING_DTYPE}" \
      --model_profile="${MODEL_PROFILE}" \
      --lr_scheduler="${LR_SCHEDULER}" \
      --warmup_epochs="${WARMUP_EPOCHS}" \
      --eval_frequency="${eval_frequency}" \
      "${extra_args[@]}" 2>&1 | tee "${log_path}"
  fi
}

index=0
for pde in "${PDES[@]}"; do
  run_train "${index}" "${pde}"
  index=$(( index + 1 ))
done

echo
echo "== Training launcher complete =="
echo "output_dir: ${OUTPUT_DIR}"
echo "logs: ${LOG_DIR}"
