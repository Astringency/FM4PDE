#!/usr/bin/env bash
set -euo pipefail
# ============================================================================
# Burgers-only resumable sampling sweep
#
# This wrapper fixes the PDE/task selection to burger/both and, by default,
# runs both observation layouts configured for Burgers:
#   random         -> uses NUM_OBS (default: 500)
#   sensor_column  -> uses num_sensor_columns from configs/main/burger.yaml
#
# All common controls are inherited by run_sample_sweep.sh, including
# NUM_SAMPLES, MAX_BATCH_SIZE, SAMPLER_LIST, DEVICE_LIST, RESUME, VIS,
# PLAN_ONLY, and AGGREGATE. SENSOR_MODE_LIST may be overridden explicitly.
#
# Examples:
#   bash scripts/sample/run_sample_sweep_burger.sh
#   OUTPUT_DIR=outputs/MAIN1000 SAMPLER_LIST="stochastic" \
#     DEVICE_LIST="cuda:0 cuda:1" bash scripts/sample/run_sample_sweep_burger.sh
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PDE_LIST="burger"
export TASK_LIST="both"
export SENSOR_MODE_LIST="${SENSOR_MODE_LIST:-random sensor_column}"

exec bash "${SCRIPT_DIR}/run_sample_sweep.sh"
