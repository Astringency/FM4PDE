#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export FM4PDE_ROOT="$ROOT"
COCOGEN_REPO="${COCOGEN_ROOT:-$ROOT/../CoCoGen}"
exec bash "$COCOGEN_REPO/scripts/sample/run.sh" "$@"
