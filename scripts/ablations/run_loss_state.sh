#!/usr/bin/env bash
set -euo pipefail

python -m fm4pde_ablation.sweep --grid configs/ablations/main_loss_state.yaml "$@"
