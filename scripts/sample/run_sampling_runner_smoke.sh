#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

python -m sampling.runner \
  --config configs/reaction_diffusion.yaml \
  --dry-run \
  --override num_steps=2 \
  --override device=cpu

python -m sampling.runner \
  --config configs/reaction_diffusion.yaml \
  --dry-run \
  --override batch_size=1 \
  --override num_steps=2 \
  --override device=cpu
