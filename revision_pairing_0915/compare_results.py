"""Join independently audited FM results to frozen historical summary references.

No predictions, baseline metrics, significance tests, or joint-field aggregates
are computed. Only the new per-input audit errors are checked against its summary.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


IDENTITY = ("cohort", "pde", "distribution", "setting")


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def number(value, *, nullable=False):
    if value is None or value == "":
        require(nullable, "Missing required numeric value")
        return None
    parsed = float(value)
    require(math.isfinite(parsed), f"Nonfinite numeric value: {value}")
    return parsed


def close(actual, expected, name: str) -> None:
    if actual is None or expected is None:
        require(actual is expected, f"Null mismatch: {name}")
    else:
        require(math.isclose(actual, expected, rel_tol=2e-12, abs_tol=2e-12),
                f"Summary/per-input or unit mismatch: {name}: {actual} vs {expected}")


def target_fields(cell: dict) -> list[str]:
    if cell["pde"] == "burger" or cell["task"] == "forward":
        return ["u"]
    return ["a"] if cell["task"] == "inverse" else ["a", "u"]


def expected_baselines(cell: dict) -> set[tuple[str, int]]:
    if cell["cohort"] == "diffusion":
        return {("DiffusionPDE", 100), ("DiffusionPDE", 1000)}
    if cell["pde"] == "burger":
        return {(name, 0) for name in ("RecFNO", "Senseiver", "VoronoiCNN", "4D-Var", "VIVID")}
    if cell["setting"] == "full_forward":
        return {(name, 0) for name in ("FNO", "DeepONet", "IFNO")}
    if cell["setting"] == "full_inverse":
        return {("IFNO", 0)}
    return {(name, 0) for name in ("RecFNO", "Senseiver", "VoronoiCNN")}


def write_csv(path: Path, rows: list[dict]) -> None:
    require(bool(rows), f"No comparison rows for {path}")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def compare_results(protocol_path: Path, audit_dir: Path, references_path: Path,
                    output: Path, *, allow_partial=False) -> dict:
    protocol_path, audit_dir, references_path, output = [
        p.resolve() for p in (protocol_path, audit_dir, references_path, output)]
    require(output.is_relative_to(protocol_path.parent), "Comparison output must be inside this isolated task directory")
    paths = {"protocol": protocol_path, "audit": audit_dir / "audit.json",
             "new_summary": audit_dir / "summary.csv", "new_per_sample": audit_dir / "per_sample.csv",
             "references": references_path}
    sources = {key: {"path": str(path), "sha256": sha256_file(path)} for key, path in paths.items()}
    protocol = json.loads(protocol_path.read_text())
    audit = json.loads(paths["audit"].read_text())
    references = json.loads(references_path.read_text())
    protocol_sha = sources["protocol"]["sha256"]
    require(protocol["status"] == "frozen", "Protocol is not frozen")
    content = dict(protocol)
    recorded_content_sha = content.pop("content_sha256")
    # Same canonical JSON format used by build_protocol.json_hash.
    content_sha = hashlib.sha256(json.dumps(content, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    require(content_sha == recorded_content_sha, "Protocol content hash mismatch")
    require(audit["protocol_sha256"] == protocol_sha, "Audit/protocol SHA mismatch")
    require(references["sources"]["protocol"]["sha256"] == protocol_sha, "References/protocol SHA mismatch")
    require(audit["status"] in ("pass", "partial") and not audit["failed_batches"], "Cannot compare an audit with failures")
    require(allow_partial or (audit["status"] == "pass" and audit["complete"]),
            "Formal comparison requires a complete passing audit; use --allow-partial only for a snapshot")
    cells = {c["cell_id"]: c for c in protocol["cells"]}
    require(len(cells) == len(protocol["cells"]) == 57, "Unexpected or duplicate protocol cells")
    require(all(c["count"] == 1000 and c["config"]["num_steps"] == 100 for c in cells.values()),
            "This comparison requires 1000-input, 100-step FM cells")
    require(set(audit["coverage"]) == set(cells), "Audit coverage does not match protocol cells")

    summaries = {}
    for row in read_csv(paths["new_summary"]):
        cell_id = row["cell_id"]
        require(cell_id in cells and cell_id not in summaries, f"Unknown or duplicate summary cell: {cell_id}")
        require(all(row[k] == cells[cell_id][k] for k in IDENTITY), f"Summary identity mismatch: {cell_id}")
        summaries[cell_id] = row
    per_cell = defaultdict(list)
    for row in read_csv(paths["new_per_sample"]):
        cell_id = row["cell_id"]
        require(cell_id in cells, f"Unknown per-input cell: {cell_id}")
        require(all(row[k] == cells[cell_id][k] for k in (*IDENTITY, "task")), f"Per-input identity mismatch: {cell_id}")
        per_cell[cell_id].append(row)
    verified_new = {}
    for cell_id, cell in cells.items():
        rows = per_cell[cell_id]
        n = len(rows)
        indices = [int(r["index"]) for r in rows]
        source_indices = [int(r["source_index"]) for r in rows]
        ids = [r["sample_id"] for r in rows]
        require(len(set(indices)) == len(set(source_indices)) == len(set(ids)) == n,
                f"Duplicate per-input index/source ID: {cell_id}")
        require(all(ids) and all(0 <= i < 1000 for i in indices), f"Invalid input indices/IDs: {cell_id}")
        require(source_indices == indices, f"Frozen source indices are not 0..999 input row indices: {cell_id}")
        coverage = audit["coverage"][cell_id]
        require(coverage["expected"] == 1000 and coverage["found"] == n and
                coverage["missing"] == sorted(set(range(1000)) - set(indices)), f"Audit coverage mismatch: {cell_id}")
        require(allow_partial or n == 1000, f"Formal comparison requires new n=1000: {cell_id}, n={n}")
        require((cell_id in summaries) == bool(n), f"Summary presence does not match per-input rows: {cell_id}")
        if n:
            require(int(summaries[cell_id]["n"]) == n, f"Summary n mismatch: {cell_id}")
        for field in ("a", "u"):
            key = f"error_{field}_percent"
            values = [number(r[key], nullable=cell["pde"] == "burger" and field == "a") for r in rows]
            if cell["pde"] == "burger" and field == "a":
                require(all(v is None for v in values), "Burgers must have only a scientific u field")
                values = []
            require(all(v >= 0 for v in values), f"Negative relative error: {cell_id}/{field}")
            mean = statistics.fmean(values) if values else None
            sd = statistics.stdev(values) if len(values) > 1 else None
            if n:
                close(number(summaries[cell_id][key + "_mean"], nullable=True), mean, cell_id + "/" + field + "/mean")
                close(number(summaries[cell_id][key + "_sd"], nullable=True), sd, cell_id + "/" + field + "/sd")
            if field in target_fields(cell):
                verified_new[cell_id, field] = {"n": n, "mean_percent": mean, "sd_percent": sd}
    total = sum(len(rows) for rows in per_cell.values())
    require(audit["verified_predictions"] == total, "Audited total differs from per-input CSV")
    require(audit["complete"] == (total == 57000), "Audit completion flag disagrees with coverage")
    require((audit["status"] == "pass") == audit["complete"], "Audit status disagrees with completion")
    require(audit["predictions_by_cohort"] == dict(Counter(row["cohort"] for rows in per_cell.values() for row in rows)),
            "Audit cohort totals disagree with per-input CSV")

    refs = defaultdict(list)
    seen = set()
    for ref in references["records"]:
        cell_id, field = ref["cell_id"], ref["field"]
        key = (cell_id, field, ref["method"], int(ref["steps"]))
        require(key not in seen, f"Duplicate reference key: {key}")
        seen.add(key)
        require(cell_id in cells and field in target_fields(cells[cell_id]), f"Unexpected reference cell/field: {key}")
        cell = cells[cell_id]
        require(all(ref[k] == cell[k] for k in (*IDENTITY, "task")), f"Reference identity mismatch: {key}")
        require(ref["new_protocol_truth_sha256"] == cell["truth_sha256"] and
                ref["new_protocol_masks_sha256"] == cell["masks_sha256"], f"Reference input binding mismatch: {key}")
        require(ref["status"] == "available" and int(ref["n"]) == 1000, f"Reference not available at n=1000: {key}")
        expected_metric = "relative_L2_full_trajectory" if cell["pde"] == "burger" else f"relative_L2_{field}"
        expected_domain = "full_128x128_trajectory" if cell["pde"] == "burger" else "full_128x128_field"
        require(ref["metric"] == expected_metric and ref["evaluation_domain"] == expected_domain and
                ref["aggregation"] == "arithmetic_mean_of_per_input_relative_L2" and
                ref["sampling_unit"] == "physical_input", f"Reference metric/domain/unit mismatch: {key}")
        require(ref["source_unit"] in ("ratio", "percent"), f"Unknown reference unit: {key}")
        for metric in ("mean", "sd"):
            ratio = number(ref[metric + "_ratio"])
            require(ratio >= 0, f"Negative reference metric: {key}")
            close(number(ref[metric + "_percent"]), 100 * ratio, str(key) + "/unit")
            close(number(ref[metric + "_source"]), ratio * (100 if ref["source_unit"] == "percent" else 1), str(key) + "/source_unit")
        refs[cell_id, field].append(ref)
    require(set(refs) == set(verified_new), "Reference and new target fields do not match")
    historical, baselines = [], []
    for (cell_id, field), new in sorted(verified_new.items()):
        cell = cells[cell_id]
        old_fm = [r for r in refs[cell_id, field] if r["reference_role"] == "historical_FM"]
        controls = [r for r in refs[cell_id, field] if r["reference_role"] == "paired_baseline"]
        require(len(old_fm) == 1 and old_fm[0]["method"] == "FM4PDE" and old_fm[0]["steps"] == 100,
                f"Historical FM field must be unique at 100 steps: {cell_id}/{field}")
        require({(r["method"], int(r["steps"])) for r in controls} == expected_baselines(cell),
                f"Missing or unexpected comparator methods/steps: {cell_id}/{field}")
        require(len(old_fm) + len(controls) == len(refs[cell_id, field]), "Unknown reference role")
        for ref in old_fm + controls:
            is_old_fm = ref["reference_role"] == "historical_FM"
            require(ref["reference_inputs_and_masks_paired_to_new_protocol"] == (not is_old_fm), "Reference pairing role mismatch")
            formal = audit["complete"] and new["n"] == 1000
            row = {"cell_id": cell_id, **{k: cell[k] for k in (*IDENTITY, "task")}, "field": field,
                   "metric": ref["metric"], "evaluation_domain": ref["evaluation_domain"],
                   "comparison_status": "formal" if formal else "partial_snapshot",
                   "new_cell_complete": new["n"] == 1000, "formal_comparison": formal,
                   "new_method": "FM4PDE", "new_steps": int(cell["config"]["num_steps"]),
                   "new_n": new["n"], "new_expected_n": 1000,
                   "new_mean_percent": new["mean_percent"], "new_sd_percent": new["sd_percent"],
                   "new_mean_ratio": None if new["mean_percent"] is None else new["mean_percent"] / 100,
                   "new_sd_ratio": None if new["sd_percent"] is None else new["sd_percent"] / 100,
                   "new_sd_definition": "sample_standard_deviation_ddof1_across_inputs",
                   "reference_role": ref["reference_role"], "reference_method": ref["method"],
                   "reference_steps": int(ref["steps"]), "reference_n": int(ref["n"]),
                   "reference_mean_percent": ref["mean_percent"], "reference_sd_percent": ref["sd_percent"],
                   "reference_mean_ratio": ref["mean_ratio"], "reference_sd_ratio": ref["sd_ratio"],
                   "reference_source_unit": ref["source_unit"], "reference_mean_source": ref["mean_source"],
                   "reference_sd_source": ref["sd_source"], "reference_sd_definition": ref["sd_definition"],
                   "delta_mean_percentage_points_new_minus_reference": None if new["mean_percent"] is None else new["mean_percent"] - ref["mean_percent"],
                   "delta_interpretation": "descriptive_difference_of_means; lower_error_is_better; no_causal_attribution",
                   "paired_test_performed": False, "reference_inputs_and_masks_aligned": not is_old_fm,
                   "same_complete_inputs_and_masks_in_comparison": (not is_old_fm) and new["n"] == 1000,
                   "pairing_note": ref["pairing_note"],
                   "rng_note": "New FM uses a canonical per-input RNG realization that differs from historical batched FM RNG; historical differences cannot isolate the effect of mask/input alignment." if is_old_fm else "New FM inference uses the frozen canonical per-input RNG protocol.",
                   "partial_note": "" if formal else "Running snapshot. New n is explicit; reference summaries retain all 1000 inputs. This is not a formal or paired subset comparison.",
                   "new_protocol_sha256": protocol_sha, "new_audit_sha256": sources["audit"]["sha256"],
                   "new_summary_path": str(paths["new_summary"]), "new_summary_sha256": sources["new_summary"]["sha256"],
                   "new_per_sample_path": str(paths["new_per_sample"]), "new_per_sample_sha256": sources["new_per_sample"]["sha256"],
                   "reference_manifest_path": str(references_path), "reference_manifest_sha256": sources["references"]["sha256"],
                   "reference_source_path": ref["source_path"], "reference_source_sha256": ref["source_sha256"],
                   "reference_source_csv_line": ref.get("source_csv_line"),
                   "reference_original_source_path": ref.get("original_source_path"),
                   "reference_original_source_sha256": ref.get("original_source_sha256"),
                   "reference_original_row_locator": ref.get("original_row_locator")}
            (historical if is_old_fm else baselines).append(row)
    # Reject an audit being rewritten while this comparison was reading it.
    require(all(sha256_file(paths[k]) == s["sha256"] for k, s in sources.items()), "An input artifact changed during comparison")
    output.mkdir(parents=True, exist_ok=True)
    files = {"historical": output / "new_vs_historical.csv", "baselines": output / "new_vs_baselines.csv"}
    require(not set(files.values()) & set(paths.values()), "Comparison output would overwrite an input")
    write_csv(files["historical"], historical)
    write_csv(files["baselines"], baselines)
    report = {"version": 1, "status": "complete" if audit["complete"] else "partial_snapshot", "allow_partial": allow_partial,
              "study": protocol["study"], "new_predictions": total, "expected_predictions": 57000,
              "complete_cells": sum(len(per_cell[c]) == 1000 for c in cells), "expected_cells": 57,
              "historical_comparison_rows": len(historical), "baseline_comparison_rows": len(baselines),
              "formal_historical_rows": sum(r["formal_comparison"] for r in historical),
              "formal_baseline_rows": sum(r["formal_comparison"] for r in baselines),
              "sources": sources, "script_sha256": sha256_file(Path(__file__)),
              "outputs": {k: {"path": str(p), "sha256": sha256_file(p)} for k, p in files.items()},
              "cells": [{"cell_id": c, "new_n": len(per_cell[c]), "expected_n": 1000,
                         "status": "complete_cell" if len(per_cell[c]) == 1000 else "partial_cell"} for c in sorted(cells)],
              "limitations": [
                  "Historical FM comparisons are unpaired: masks and the canonical RNG realization change, and NS physical inputs change from 2000..2999 to 0..999.",
                  "Baseline comparisons use the independently frozen cohort alignment, but only differences of existing means are reported; no paired test or confidence interval is inferred from aggregate summaries.",
                  "During a partial snapshot, reference n stays 1000; it is not restricted to the available new subset.",
                  "Forward reports u, inverse reports a, joint reports a and u separately; Burgers reports only full-trajectory u. No joint aggregate is invented.",
                  "new_steps comes only from the FM protocol and stays 100; 1000-step DiffusionPDE is a separately labelled reference_steps value.",
                  "NS Smooth retains the primary 1000-input means. The separate 996-input selection-overlap sensitivity is outside these primary comparison CSVs."
              ]}
    (output / "comparison_summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    report = compare_results(args.protocol, args.audit_dir, args.references, args.output, allow_partial=args.allow_partial)
    print(json.dumps({k: report[k] for k in ("status", "new_predictions", "complete_cells", "historical_comparison_rows", "baseline_comparison_rows")}))


if __name__ == "__main__":
    main()
