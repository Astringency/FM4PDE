#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:-/tmp/fm4pde_future_pdes}"

python data/DataGen/python/generate_future_pdes.py \
  --pde all \
  --out-root "${DATA_ROOT}" \
  --resolution 128 \
  --n-train 50000 \
  --n-test 1000 \
  --train-shards 5 \
  --n-time 11 \
  --overwrite

