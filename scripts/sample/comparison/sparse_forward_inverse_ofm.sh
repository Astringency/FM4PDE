#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
# The OFM branch uses the same cases and guidance in the main and architecture studies.
exec bash "$ROOT/scripts/sample/ablations/velocity_architecture.sh" --methods FM4PDE-OFM "$@"
