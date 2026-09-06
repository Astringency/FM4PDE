#!/usr/bin/env bash
set -euo pipefail

# ID/Smooth/Rough x random500/sensor_column = 6,000 paired reconstructions.
# Each GPU handles one observation layout, with Rough first. Reuses archived
# sensor columns (currently 5), checkpoint seed/batch rows, and ground truth.
export BAK_SUITE=burgers
export BAK_OUTPUT="${BAK_OUTPUT:-outputs/bak_comparison_burgers_full}"
exec bash "$(dirname "${BASH_SOURCE[0]}")/run_bak_comparison_a100.sh" "$@"
