#!/usr/bin/env bash
set -euo pipefail

PDE="${PDE:-poisson}"
TASK="${TASK:-both}"
CONFIG="${CONFIG:-configs/${PDE}.yaml}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_STEPS="${NUM_STEPS:-100}"
NUM_OBS="${NUM_OBS:-500}"
SAMPLER_PHASE="${SAMPLER_PHASE:-stochastic}"
GUIDANCE_COMPONENTS="${GUIDANCE_COMPONENTS:-obs_pde}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/samples/FM4PDE}"

python -u -m sampling.runner \
  --config "$CONFIG" \
  --override "pde=${PDE}" \
  --override "task=${TASK}" \
  --override "device=${DEVICE}" \
  --override "batch_size=${BATCH_SIZE}" \
  --override "num_steps=${NUM_STEPS}" \
  --override "num_obs=${NUM_OBS}" \
  --override "sampler_phase=${SAMPLER_PHASE}" \
  --override "guidance_components=${GUIDANCE_COMPONENTS}" \
  --override "output_dir=${OUTPUT_DIR}"
