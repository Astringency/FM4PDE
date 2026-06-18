#!/usr/bin/env bash
set -euo pipefail

python -m sampling.sweep --grid configs/ablations/main_sampler_phase.yaml "$@"
