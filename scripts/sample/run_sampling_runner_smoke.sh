#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

python -m sampling.sweep \
  --grid configs/ablations/all_internal_ablation_grid.yaml \
  --list

python -m sampling.runner \
  --config configs/ablations/smoke.yaml \
  --dry-run \
  --override num_steps=2 \
  --override device=cpu

python -m sampling.runner \
  --config configs/reaction_diffusion.yaml \
  --override num_steps=2 \
  --override batch_size=1 \
  --override device=cpu \
  --override allow_synthetic_data=false

python -m sampling.aggregate outputs/ablations
