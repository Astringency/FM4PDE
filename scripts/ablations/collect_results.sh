#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-outputs/ablations}"
OUT="${2:-outputs/ablations/summary_all.csv}"

python - "$ROOT" "$OUT" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
out = Path(sys.argv[2])
rows = []
for path in root.rglob("metrics_final.json"):
    with path.open() as handle:
        row = json.load(handle)
    row["metrics_path"] = str(path)
    rows.append(row)

out.parent.mkdir(parents=True, exist_ok=True)
if not rows:
    out.write_text("")
    raise SystemExit(0)
keys = sorted({key for row in rows for key in row})
with out.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=keys)
    writer.writeheader()
    writer.writerows(rows)
print(f"wrote {len(rows)} rows to {out}")
PY
