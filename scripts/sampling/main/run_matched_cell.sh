#!/usr/bin/env bash
set -euo pipefail
# Validate and replay one saved comparison cell, preserving all existing results.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY_ROOT="${STUDY_ROOT:?Set STUDY_ROOT to the saved aligned study containing protocol.json and inputs/}"
CELL="${CELL:?Set CELL, for example supervised/poisson/id/sparse_joint}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/reproductions/aligned}"
JOB_ID="${JOB_ID:-$(date +%Y%m%dT%H%M%S)-${CELL//\//-}}"
common=(--protocol "${STUDY_ROOT}/protocol.json" --input-root "${STUDY_ROOT}"
        --output-root "${OUTPUT_ROOT}" --cell "${CELL}"
        --batch-size "${BATCH_SIZE:-8}" --device "${DEVICE:-cuda:0}")
bash "${HERE}/run_matched.sh" --mode pilot --job-id "${JOB_ID}-pilot" "${common[@]}"
exec bash "${HERE}/run_matched.sh" --mode run --job-id "${JOB_ID}-run" \
  --pilot-certificate "${OUTPUT_ROOT}/jobs/${JOB_ID}-pilot" "${common[@]}"
