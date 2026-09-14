"""Independently verify stored paired predictions; never run model inference.

Checks complete input coverage, the tensors actually passed to inference, frozen
weights/configuration, prediction hashes, finiteness, and float64 field errors.
Only audit artifacts under the explicitly supplied output directory are written.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
import sys

import numpy as np
import torch

try:
    from .input_sources import load_cell
    from .build_protocol import RUNTIME_FIELDS, json_hash, sha256_file
except ImportError:
    from input_sources import load_cell
    from build_protocol import RUNTIME_FIELDS, json_hash, sha256_file


def tensor_digest(value: torch.Tensor) -> str:
    """Execution digest: dtype string, compact JSON shape, contiguous raw bytes.

    This implementation is intentionally independent of the sampling runner.
    It is distinct from the historical source-array digest in input_sources.
    """
    array = value.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(str(array.dtype).encode())
    h.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
    h.update(array.numpy().tobytes(order="C"))
    return h.hexdigest()


def resolve(root: Path, filename: str) -> Path:
    path = Path(filename)
    return path if path.is_absolute() else root / path


def verify_actual_inputs(data: dict, indices: list[int], recorded: dict, dtype: str) -> dict:
    if dtype != "float32":
        raise ValueError(f"Unapproved execution precision: {dtype}")
    index = torch.as_tensor(indices, dtype=torch.int64)
    a = data["coef_ground_truth"].index_select(0, index).float()
    u = data["sol_ground_truth"].index_select(0, index).float()
    ma = data["masks"]["coef"].index_select(0, index).float()
    mu = data["masks"]["sol"].index_select(0, index).float()
    actual = dict(coef_truth=a, sol_truth=u, coef_mask=ma, sol_mask=mu,
                  observed_coef=a * ma, observed_sol=u * mu)
    expected = {key: tensor_digest(value) for key, value in actual.items()}
    if recorded != expected:
        bad = sorted(key for key in expected if recorded.get(key) != expected[key])
        raise ValueError(f"Executed input tensors do not match frozen comparator inputs: {bad}")
    return actual


def relative_error(prediction: torch.Tensor, truth: torch.Tensor) -> np.ndarray:
    # Convert both operands before subtraction; never derive this result from
    # a metric reported by the inference process.
    predicted = prediction.detach().cpu().double().numpy().reshape(len(prediction), -1)
    target = truth.detach().cpu().double().numpy().reshape(len(truth), -1)
    if not np.isfinite(predicted).all() or not np.isfinite(target).all():
        raise ValueError("Nonfinite prediction or truth")
    denominator = np.sqrt(np.einsum("ij,ij->i", target, target))
    if np.any(denominator <= 0):
        raise ValueError("Zero truth norm: relative error is undefined")
    difference = predicted - target
    error = 100 * np.sqrt(np.einsum("ij,ij->i", difference, difference)) / denominator
    if not np.isfinite(error).all():
        raise ValueError("Nonfinite independently recomputed error")
    return error


def check_configuration(cell: dict, config: dict, receipt: dict) -> None:
    # External masks and explicit runtime choices may change only these fields.
    # All other values supplied by the historical configuration are retained.
    allowed = RUNTIME_FIELDS | {"sensor_mode", "shared_mask"}
    for key, value in cell["config"].items():
        if key not in allowed and config.get(key) != value:
            raise ValueError(f"Scientific configuration changed: {cell['cell_id']} {key}: {value!r} -> {config.get(key)!r}")
    recorded_hash = receipt.get("effective_config_sha256", receipt.get("config_sha256"))
    if recorded_hash != json_hash(config):
        raise ValueError("Effective configuration hash mismatch")
    if config.get("num_steps") != 100 or config.get("noise_level") != 0:
        raise ValueError("Unexpected steps or observation noise")


def audit_batch(path: Path, cell: dict, data: dict, protocol_sha: str) -> tuple[list[dict], dict]:
    receipt_path = path.parent / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    bound_hash = receipt.get("prediction_sha256", receipt.get("tensor_sha256", receipt.get("result_sha256")))
    if bound_hash != sha256_file(path):
        raise ValueError(f"Prediction file hash mismatch: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    for key, expected in dict(cell_id=cell["cell_id"], protocol_sha256=protocol_sha,
                              checkpoint_sha256=cell["checkpoint"]["sha256"],
                              input_truth_sha256=cell["truth_sha256"], input_masks_sha256=cell["masks_sha256"]).items():
        if payload.get(key) != expected:
            raise ValueError(f"Payload identity mismatch: {key}: {path}")
        if key in receipt and receipt[key] != expected:
            raise ValueError(f"Receipt identity mismatch: {key}: {path}")
    indices = payload["indices"]
    if not indices or any(type(i) is not int or i < 0 or i >= cell["count"] for i in indices):
        raise ValueError("Invalid frozen input row indices")
    if len(indices) != len(set(indices)):
        raise ValueError("Repeated input within a batch")
    ids = [str(data["sample_ids"][i]) for i in indices]
    source_indices = [int(data["source_indices"][i]) for i in indices]
    if list(map(str, payload["sample_ids"])) != ids or list(map(int, payload["source_indices"])) != source_indices:
        raise ValueError("Physical source IDs do not match frozen inputs")
    config = payload["effective_config"]
    check_configuration(cell, config, receipt)
    actual = verify_actual_inputs(data, indices, payload["runtime_input_hashes"], config["dtype"])
    if "runtime_input_hashes" in receipt and receipt["runtime_input_hashes"] != payload["runtime_input_hashes"]:
        raise ValueError("Receipt/payload executed-input hash mismatch")
    prediction = payload["prediction"]
    channels = 1 if cell["pde"] == "burger" else 2
    if tuple(prediction.shape) != (len(indices), channels, 128, 128):
        raise ValueError(f"Unexpected prediction shape: {tuple(prediction.shape)}")
    if not torch.isfinite(prediction).all():
        raise ValueError("Nonfinite stored prediction")
    # Explicit final tensors are cross-checked against the unified prediction.
    a_pred = prediction[:, :1]
    u_pred = prediction[:, -1:]
    for key, expected in (("coef_final", a_pred), ("sol_final", u_pred)):
        if key in payload and not torch.equal(payload[key], expected):
            raise ValueError(f"Duplicated prediction fields disagree: {key}")
    u_error = relative_error(u_pred, actual["sol_truth"])
    a_error = None if channels == 1 else relative_error(a_pred, actual["coef_truth"])
    rows = []
    for position, index in enumerate(indices):
        row = dict(cell_id=cell["cell_id"], cohort=cell["cohort"], pde=cell["pde"], distribution=cell["distribution"],
                   setting=cell["setting"], task=cell["task"], index=index, sample_id=ids[position], source_index=source_indices[position],
                   error_a_percent=None if a_error is None else float(a_error[position]), error_u_percent=float(u_error[position]),
                   prediction_file=str(path), prediction_sha256=bound_hash)
        rows.append(row)
    seconds = float(receipt["seconds"])
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("Invalid elapsed time")
    if int(receipt["nfe"]) != 100:
        raise ValueError("Unexpected number of model evaluations")
    source_hashes = receipt.get("source_hashes", {})
    if not source_hashes or not all(re.fullmatch(r"[0-9a-f]{64}", value) for value in source_hashes.values()):
        raise ValueError("Missing or malformed executed source hashes")
    audit = dict(cell_id=cell["cell_id"], prediction_file=str(path), prediction_sha256=bound_hash,
                 receipt_sha256=sha256_file(receipt_path), n=len(indices), seconds=seconds,
                 config_sha256=receipt.get("effective_config_sha256", receipt.get("config_sha256")), runtime_input_hashes=payload["runtime_input_hashes"],
                 source_hashes=source_hashes)
    return rows, audit


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        if rows:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


NS_SMOOTH_EXCLUSIONS = [197, 251, 364, 878]
NS_SMOOTH_CELLS = {
    f"{cohort}/nsnonbounded/smooth/{setting}"
    for cohort, settings in (
        ("supervised", ("sparse_forward", "sparse_inverse", "sparse_joint", "full_forward", "full_inverse")),
        ("diffusion", ("sparse_forward", "sparse_inverse", "sparse_joint")),
    )
    for setting in settings
}


def load_sensitivity_spec(root: Path, cells: dict) -> tuple[dict, dict]:
    """Bind the pre-existing four-input selection overlap without changing scope."""
    path = root / "provenance" / "ns_existing_selection_overlap.json"
    source_hash = sha256_file(path)
    spec = json.loads(path.read_text())
    expected = dict(version=1, task_name="baseline_pairing_20260915_57k",
                    pde="nsnonbounded", distribution="smooth", overlap_count=4,
                    primary_count=1000, sensitivity_count=996)
    for key, value in expected.items():
        if spec.get(key) != value:
            raise ValueError(f"Unexpected selection-overlap specification: {key}")
    excluded = spec.get("exclude_source_indices")
    if excluded != NS_SMOOTH_EXCLUSIONS or any(type(i) is not int for i in excluded):
        raise ValueError("Sensitivity exclusion must be exactly the four existing Smooth development source indices")
    applicable = spec.get("applicable_cells", [])
    frozen_smooth = {key for key, cell in cells.items()
                     if cell["pde"] == "nsnonbounded" and cell["distribution"] == "smooth"}
    if len(applicable) != 8 or set(applicable) != NS_SMOOTH_CELLS or frozen_smooth != NS_SMOOTH_CELLS:
        raise ValueError("Sensitivity must apply to exactly the eight frozen NS Smooth cells")
    if any(cells[key]["count"] != 1000 for key in applicable):
        raise ValueError("Sensitivity requires the unchanged primary 1000-input cells")
    proof = spec.get("physical_overlap_proof", [])
    if len(proof) != 4 or {item["index"] for item in proof} != set(excluded):
        raise ValueError("Missing or duplicated physical selection-overlap proof")
    for item in proof:
        digest = item.get("paired_input_truth_sha256", "")
        if (not re.fullmatch(r"[0-9a-f]{64}", digest)
                or item.get("historical_development_truth_sha256") != digest
                or item.get("exact_float32_physical_fields_equal") is not True):
            raise ValueError("Physical selection-overlap proof does not establish equal fields")
    if sha256_file(path) != source_hash:
        raise ValueError("Selection-overlap specification changed while being read")
    return spec, dict(path=str(path), sha256=source_hash)


def verify_sensitivity_inputs(cell: dict, data: dict, spec: dict) -> list[dict]:
    """Rehash both frozen physical fields using the historical proof convention."""
    source_indices = [int(i) for i in data["source_indices"]]
    if len(source_indices) != 1000 or len(set(source_indices)) != 1000:
        raise ValueError(f"Sensitivity source coverage is not unique: {cell['cell_id']}")
    locations = {source: i for i, source in enumerate(source_indices)}
    anchors = []
    for proof in spec["physical_overlap_proof"]:
        source = proof["index"]
        if source not in locations:
            raise ValueError(f"Sensitivity source index absent from frozen input: {source}")
        index = locations[source]
        pair = torch.cat((data["coef_ground_truth"][index:index + 1],
                          data["sol_ground_truth"][index:index + 1]), dim=1)
        array = np.asarray(pair.detach().cpu().numpy(), dtype="<f4")
        # Independent implementation of the historical float32/shape digest.
        actual = hashlib.sha256(str(array.shape).encode() + array.tobytes(order="C")).hexdigest()
        if actual != proof["paired_input_truth_sha256"]:
            raise ValueError(f"Frozen NS Smooth truth differs from selection-overlap proof: {cell['cell_id']} source {source}")
        anchors.append(dict(cell_id=cell["cell_id"], source_index=source,
                            index=index, physical_truth_sha256=actual))
    return anchors


def sensitivity_summary(cell: dict, rows: list[dict], spec: dict) -> dict:
    """Supplementary mean/SD from verified rows only, excluding by physical ID."""
    if cell["cell_id"] not in NS_SMOOTH_CELLS:
        raise ValueError("Sensitivity cannot exclude an ID/Rough or non-NS physical input")
    excluded = set(spec["exclude_source_indices"])
    removed = sorted(row["source_index"] for row in rows if row["source_index"] in excluded)
    retained = [row for row in rows if row["source_index"] not in excluded]
    if len({row["source_index"] for row in rows}) != len(rows):
        raise ValueError("Repeated physical source in sensitivity rows")
    complete = len(rows) == 1000 and len(retained) == 996 and removed == NS_SMOOTH_EXCLUSIONS
    result = dict(cell_id=cell["cell_id"], cohort=cell["cohort"], pde=cell["pde"],
                  distribution=cell["distribution"], setting=cell["setting"],
                  status="complete" if complete else "partial_snapshot", primary_n=len(rows),
                  primary_expected_n=1000, n=len(retained), expected_n=996,
                  excluded_verified_n=len(removed), excluded_source_indices=json.dumps(sorted(excluded)),
                  excluded_verified_source_indices=json.dumps(removed))
    for key in ("error_a_percent", "error_u_percent"):
        values = np.asarray([row[key] for row in retained], dtype=np.float64)
        primary = np.asarray([row[key] for row in rows], dtype=np.float64)
        if not np.isfinite(values).all() or not np.isfinite(primary).all():
            raise ValueError("Nonfinite NS sensitivity metric")
        result[key + "_mean"] = float(values.mean()) if len(values) else None
        result[key + "_sd"] = float(values.std(ddof=1)) if len(values) > 1 else None
        result["primary_" + key + "_mean"] = float(primary.mean()) if len(primary) else None
        result["primary_" + key + "_sd"] = float(primary.std(ddof=1)) if len(primary) > 1 else None
        result[key + "_delta_mean_percentage_points_sensitivity_minus_primary"] = (
            result[key + "_mean"] - result["primary_" + key + "_mean"] if len(values) else None)
    return result


def audit_outputs(protocol_path: Path, results_root: Path, output: Path, *, allow_partial=False) -> dict:
    torch.set_num_threads(2)
    protocol = json.loads(protocol_path.read_text())
    if protocol["status"] != "frozen":
        raise ValueError("Only a frozen protocol can be audited")
    content = dict(protocol)
    recorded_content_hash = content.pop("content_sha256")
    if json_hash(content) != recorded_content_hash:
        raise ValueError("Protocol content hash mismatch")
    root = protocol_path.parent.resolve()
    if sha256_file(resolve(root, protocol["inputs_manifest"])) != protocol["inputs_manifest_sha256"]:
        raise ValueError("Input manifest changed after protocol freeze")
    if not output.resolve().is_relative_to(root):
        raise ValueError("Audit output must remain inside this task's protocol directory")
    protocol_sha = sha256_file(protocol_path)
    cells = {c["cell_id"]: c for c in protocol["cells"]}
    if len(cells) != 57 or sum(c["count"] for c in cells.values()) != 57000:
        raise ValueError("Unexpected formal study scope")
    sensitivity_spec, sensitivity_source = load_sensitivity_spec(root, cells)
    weight_checks = {}
    original_weight_checks = {}
    for cell in cells.values():
        model = cell["checkpoint"]
        if model["path"] not in weight_checks:
            actual = sha256_file(resolve(root, model["path"]))
            if actual != model["sha256"]:
                raise ValueError(f"Model changed: {cell['pde']}")
            weight_checks[model["path"]] = actual
            if model.get("source") and Path(model["source"]).is_file():
                original = sha256_file(Path(model["source"]))
                if original != model["sha256"]:
                    raise ValueError(f"Original checkpoint source changed: {cell['pde']}")
                original_weight_checks[model["source"]] = original
    by_cell = defaultdict(list)
    failures = []
    ignored_partial_files = []
    pending_predictions = []
    for path in sorted(results_root.rglob("prediction.pt")):
        # Both runner staging directories and rsync's hidden transfer directories
        # are uncommitted. Never count or inspect their copied payloads.
        if any(part.startswith(".") for part in path.relative_to(results_root).parts[:-1]):
            ignored_partial_files.append(str(path))
            continue
        if not path.parent.name.startswith("batch_"):
            failures.append(dict(file=str(path), error="Prediction is outside a committed batch directory"))
            continue
        if not path.resolve().is_relative_to(results_root.resolve()):
            raise ValueError("Result symlink points outside the isolated result directory")
        receipt_path = path.parent / "receipt.json"
        if not receipt_path.exists():
            if allow_partial:
                pending_predictions.append(dict(file=str(path), reason="Prediction transfer has no committed receipt yet"))
            else:
                failures.append(dict(file=str(path), error="Missing committed receipt"))
            continue
        try:
            receipt = json.loads(receipt_path.read_text())
        except Exception as exc:
            failures.append(dict(file=str(path), error=f"Invalid committed receipt: {exc}"))
            continue
        cell_id = receipt.get("cell_id")
        if cell_id not in cells:
            failures.append(dict(file=str(path), error=f"Unexpected cell {cell_id}"))
            continue
        by_cell[cell_id].append(path)
    per_sample, batches, summaries = [], [], []
    sensitivity_rows, sensitivity_anchors = [], []
    coverage = {}
    for cell_id, cell in sorted(cells.items()):
        data = load_cell(root, cell, verify_hashes=True)
        if cell_id in NS_SMOOTH_CELLS:
            sensitivity_anchors.extend(verify_sensitivity_inputs(cell, data, sensitivity_spec))
        seen = set()
        cell_rows = []
        for path in by_cell[cell_id]:
            try:
                rows, receipt = audit_batch(path, cell, data, protocol_sha)
                indices = {row["index"] for row in rows}
                if seen & indices:
                    raise ValueError(f"Repeated predictions for input rows {sorted(seen & indices)}")
                seen.update(indices)
                cell_rows.extend(rows)
                batches.append(receipt)
            except Exception as exc:
                failures.append(dict(file=str(path), error=str(exc)))
        coverage[cell_id] = dict(expected=cell["count"], found=len(seen), missing=sorted(set(range(cell["count"])) - seen))
        per_sample.extend(cell_rows)
        if cell_id in NS_SMOOTH_CELLS:
            sensitivity_rows.append(sensitivity_summary(cell, cell_rows, sensitivity_spec))
        if cell_rows:
            summary = dict(cell_id=cell_id, cohort=cell["cohort"], pde=cell["pde"], distribution=cell["distribution"], setting=cell["setting"], n=len(cell_rows))
            for key in ("error_a_percent", "error_u_percent"):
                values = np.asarray([row[key] for row in cell_rows if row[key] is not None], dtype=np.float64)
                summary[key + "_mean"] = None if not len(values) else float(values.mean())
                summary[key + "_sd"] = None if len(values) < 2 else float(values.std(ddof=1))
            summaries.append(summary)
    source_versions = sorted({json_hash(b["source_hashes"]) for b in batches})
    if len(source_versions) > 1:
        failures.append(dict(file=None, error="Multiple executed source snapshots are mixed in formal results"))
    complete = not failures and all(v["found"] == v["expected"] for v in coverage.values())
    counts = dict(Counter(row["cohort"] for row in per_sample))
    if complete and counts != protocol["expected_predictions_by_cohort"]:
        raise ValueError("Cohort totals do not match frozen scope")
    report = dict(status="pass" if complete else "partial" if allow_partial and not failures else "fail", complete=complete,
                  expected_predictions=57000, verified_predictions=len(per_sample), predictions_by_cohort=counts,
                  verified_batches=len(batches), failed_batches=failures, coverage=coverage,
                  ignored_incomplete_temporary_predictions=ignored_partial_files,
                  pending_unreceipted_predictions=pending_predictions,
                  protocol_sha256=protocol_sha, checkpoint_hashes_verified=weight_checks,
                  available_original_checkpoint_sources_rehashed=original_weight_checks,
                  executed_source_snapshot_hashes=source_versions,
                  field_errors="CPU float64, independently recomputed from stored physical predictions and paired source truths",
                  executed_inputs="Six dtype/shape/byte hashes per batch checked against frozen comparator truth, masks and observations",
                  historical_file_protection_scope="Five frozen checkpoint copies are rehashed. This auditor writes only the explicitly designated task audit directory; it does not assert a bytewise scan of unrelated historical output trees.",
                  audit_script_sha256=sha256_file(Path(__file__)), batch_evidence=batches)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "per_sample.csv", sorted(per_sample, key=lambda r: (r["cell_id"], r["index"])))
    write_csv(output / "summary.csv", summaries)
    if sha256_file(Path(sensitivity_source["path"])) != sensitivity_source["sha256"]:
        raise ValueError("Selection-overlap specification changed during audit")
    sensitivity_path = output / "ns_smooth_996_summary.csv"
    sensitivity_json = output / "ns_smooth_996_sensitivity.json"
    write_csv(sensitivity_path, sensitivity_rows)
    sensitivity_complete = all(row["status"] == "complete" for row in sensitivity_rows)
    sensitivity_report = dict(
        version=1, status="fail" if failures else "complete" if sensitivity_complete else "partial_snapshot",
        complete=sensitivity_complete and not failures, primary_audit_status=report["status"],
        primary_audit_complete=complete, protocol_sha256=protocol_sha,
        selection_overlap_source=sensitivity_source, applicable_cells=sorted(NS_SMOOTH_CELLS),
        exclude_source_indices=NS_SMOOTH_EXCLUSIONS, expected_cells=8,
        primary_expected_n_per_cell=1000, sensitivity_expected_n_per_cell=996,
        verified_primary_predictions=sum(row["primary_n"] for row in sensitivity_rows),
        verified_sensitivity_predictions=sum(row["n"] for row in sensitivity_rows),
        definition="Arithmetic mean and sample standard deviation (ddof=1) of independently recomputed per-input relative L2 errors, in percent.",
        scope="Supplementary NS Smooth analysis only. Exclude physical source indices 197,251,364,878 in the eight named cells; retain all 1000 inputs in the primary per_sample.csv and summary.csv. ID/Rough are unaffected.",
        partial_note="An incomplete cell reports only its verified retained rows with the actual n; it is not a completed 996-input estimate.",
        interpretation="Descriptive sensitivity to the existing development-input overlap; no retuning, significance test, or causal attribution.",
        frozen_physical_overlap_checks=sensitivity_anchors, cells=sensitivity_rows,
        csv=dict(path=str(sensitivity_path), sha256=sha256_file(sensitivity_path)),
        audit_script_sha256=report["audit_script_sha256"])
    sensitivity_json.write_text(json.dumps(sensitivity_report, indent=2, allow_nan=False) + "\n")
    report["ns_smooth_selection_overlap_sensitivity"] = dict(
        status=sensitivity_report["status"], complete=sensitivity_report["complete"],
        csv=sensitivity_report["csv"], json=dict(path=str(sensitivity_json), sha256=sha256_file(sensitivity_json)),
        selection_overlap_source=sensitivity_source)
    (output / "audit.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    report = audit_outputs(args.protocol, args.results, args.output, allow_partial=args.allow_partial)
    print(json.dumps({k: report[k] for k in ("status", "complete", "verified_predictions", "verified_batches", "predictions_by_cohort")}))
    if report["status"] == "fail":
        sys.exit(1)


if __name__ == "__main__":
    main()
