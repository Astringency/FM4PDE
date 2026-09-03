from __future__ import annotations

import argparse
import json
from typing import Any

from sampling.sweep import run_job


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one isolated FM4PDE ablation job.")
    parser.add_argument("--payload", required=True, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        payload: dict[str, Any] = json.loads(args.payload)
        config_path = str(payload["config_path"])
        overrides = payload["overrides"]
        if not isinstance(overrides, dict):
            raise ValueError("worker overrides must be a mapping")
        result = run_job(
            config_path,
            overrides,
            dry_run=bool(payload.get("dry_run", False)),
            resume=bool(payload.get("resume", True)),
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
