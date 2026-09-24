#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
exec "${PYTHON_BIN:-python}" -m experiments.paper.run configs/experiments/ablations/noise_level.yaml "$@"
