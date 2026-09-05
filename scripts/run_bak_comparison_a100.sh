#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# All 36 cells x 1000 held-out samples = 36,000 legacy reconstructions.
# Override SAMPLES/PRIORITY_SAMPLES for a shorter prespecified pilot.
BAK_PYTHON="${BAK_PYTHON:-python}"
BAK_OUTPUT="${BAK_OUTPUT:-outputs/bak_comparison_full}"
SAMPLES="${SAMPLES:-1000}"
PRIORITY_SAMPLES="${PRIORITY_SAMPLES:-1000}"
BAK_BATCH_SIZE="${BAK_BATCH_SIZE:-4}"
BAK_GPU0="${BAK_GPU0:-0}"
BAK_GPU1="${BAK_GPU1:-1}"
mkdir -p "$BAK_OUTPUT"
common=(--output "$BAK_OUTPUT" --samples "$SAMPLES" --priority-samples "$PRIORITY_SAMPLES")
"$BAK_PYTHON" -m scripts.compare_bak "${common[@]}" --prepare-only

pids=()
stop_workers() { for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done; }
trap stop_workers INT TERM EXIT
"$BAK_PYTHON" -u -m scripts.compare_bak "${common[@]}" --device "cuda:$BAK_GPU0" \
  --num-workers 2 --worker-index 0 --batch-size "$BAK_BATCH_SIZE" "$@" \
  > "$BAK_OUTPUT/worker0.log" 2>&1 &
pids+=("$!")
"$BAK_PYTHON" -u -m scripts.compare_bak "${common[@]}" --device "cuda:$BAK_GPU1" \
  --num-workers 2 --worker-index 1 --batch-size "$BAK_BATCH_SIZE" "$@" \
  > "$BAK_OUTPUT/worker1.log" 2>&1 &
pids+=("$!")
failed=0
while (( ${#pids[@]} )); do
  finished_pid=""
  if ! wait -n -p finished_pid "${pids[@]}"; then
    failed=1
    stop_workers
    break
  fi
  remaining=()
  for pid in "${pids[@]}"; do
    if [[ "$pid" != "$finished_pid" ]]; then remaining+=("$pid"); fi
  done
  pids=("${remaining[@]}")
done
pids=()
trap - INT TERM EXIT
"$BAK_PYTHON" -m scripts.compare_bak --output "$BAK_OUTPUT" --summarize-only
if (( failed )); then
  echo "A worker failed. Inspect worker*.log and paired_results.worker*.jsonl before retrying." >&2
  exit 1
fi
echo "Completed requested samples. Summary: $BAK_OUTPUT/summary.csv"
