#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-outputs/ablations}"
OUT_DIR="${2:-$ROOT}"

python -m sampling.aggregate "$ROOT" --output-dir "$OUT_DIR"
