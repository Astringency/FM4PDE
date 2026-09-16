#!/usr/bin/env bash
set -euo pipefail
# Replay the aligned main study using its saved inputs, masks and protocol.
# Use --help for pilot/production arguments and an explicit output directory.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"
exec "${PYTHON_BIN:-python}" -m experiments.aligned_sampling.run_inference "$@"
