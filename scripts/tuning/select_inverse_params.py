#!/usr/bin/env python3
"""Rank inverse tuning configurations using coefficient error across offsets."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


OUTPUT_FIELDS = (
    "rank",
    "pde",
    "zeta_obs_u",
    "zeta_pde",
    "clip_mode",
    "clip_threshold",
    "rel_l2_a_n",
    "rel_l2_a_mean",
    "rel_l2_a_median",
    "rel_l2_a_p90",
    "rel_l2_a_max",
    "rel_l2_u_mean",
    "pde_residual_norm_mean",
    "robust_score",
)


def _number(row: dict[str, str], name: str) -> float:
    try:
        value = float(row.get(name, ""))
    except (TypeError, ValueError):
        return math.nan
    return value


def select_rows(
    rows: list[dict[str, str]], expected_n: int, top_k: int
) -> list[dict[str, str | int | float]]:
    by_pde: dict[str, list[tuple[float, dict[str, str]]]] = {}
    for row in rows:
        pde = row.get("pde", "")
        n = _number(row, "rel_l2_a_n")
        mean = _number(row, "rel_l2_a_mean")
        p90 = _number(row, "rel_l2_a_p90")
        maximum = _number(row, "rel_l2_a_max")
        if not pde or n < expected_n or not all(math.isfinite(value) for value in (mean, p90, maximum)):
            continue
        robust_score = 0.5 * mean + 0.3 * p90 + 0.2 * maximum
        by_pde.setdefault(pde, []).append((robust_score, row))

    selected: list[dict[str, str | int | float]] = []
    for pde in sorted(by_pde):
        ranked = sorted(
            by_pde[pde],
            key=lambda item: (
                item[0],
                _number(item[1], "rel_l2_a_p90"),
                _number(item[1], "rel_l2_a_mean"),
            ),
        )
        for rank, (score, row) in enumerate(ranked[:top_k], start=1):
            selected.append(
                {
                    "rank": rank,
                    "pde": pde,
                    "zeta_obs_u": row.get("zeta_obs_u", ""),
                    "zeta_pde": row.get("zeta_pde", ""),
                    "clip_mode": row.get("clip_mode", ""),
                    "clip_threshold": row.get("clip_threshold", ""),
                    "rel_l2_a_n": int(_number(row, "rel_l2_a_n")),
                    "rel_l2_a_mean": _number(row, "rel_l2_a_mean"),
                    "rel_l2_a_median": _number(row, "rel_l2_a_median"),
                    "rel_l2_a_p90": _number(row, "rel_l2_a_p90"),
                    "rel_l2_a_max": _number(row, "rel_l2_a_max"),
                    "rel_l2_u_mean": _number(row, "rel_l2_u_mean"),
                    "pde_residual_norm_mean": _number(row, "pde_residual_norm_mean"),
                    "robust_score": score,
                }
            )
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("--expected-n", type=int, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.expected_n < 1 or args.top_k < 1:
        parser.error("--expected-n and --top-k must be positive")
    with args.summary.open(newline="", encoding="utf-8") as handle:
        selected = select_rows(list(csv.DictReader(handle)), args.expected_n, args.top_k)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(selected)

    if not selected:
        print(f"No configuration completed all {args.expected_n} expected samples.")
        return
    print("pde rank zeta_obs_u zeta_pde clip mean_a p90_a max_a score")
    for row in selected:
        print(
            f"{row['pde']} {row['rank']} {row['zeta_obs_u']} {row['zeta_pde']} "
            f"{row['clip_threshold']} {100 * float(row['rel_l2_a_mean']):.2f}% "
            f"{100 * float(row['rel_l2_a_p90']):.2f}% "
            f"{100 * float(row['rel_l2_a_max']):.2f}% "
            f"{100 * float(row['robust_score']):.2f}%"
        )


if __name__ == "__main__":
    main()
