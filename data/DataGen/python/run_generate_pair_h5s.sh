#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:-/tmp/fm4pde_pair_h5}"

python data/DataGen/python/generate_pair_h5s.py \
  --pde all \
  --out-root "${DATA_ROOT}" \
  --resolution 128 \
  --n-train 50000 \
  --n-test 1000 \
  --train-shards 5 \
  --n-time 11 \
  --overwrite

