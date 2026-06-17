#!/usr/bin/env bash
set -euo pipefail

python -m fm4pde_ablation.runner --config configs/ablations/smoke.yaml --dry-run
