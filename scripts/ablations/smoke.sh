#!/usr/bin/env bash
set -euo pipefail

python -m sampling.runner --config configs/ablations/smoke.yaml --dry-run
