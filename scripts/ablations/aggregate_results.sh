#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-outputs/ablations}"
OUT_DIR="${2:-$ROOT}"

python -m fm4pde_ablation.aggregate "$ROOT" --output-dir "$OUT_DIR"
