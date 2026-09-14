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
    audit = dict(cell_id=cell["cell_id"], prediction_file=str(path), prediction_sha256=bound_hash,
                 receipt_sha256=sha256_file(receipt_path), n=len(indices), seconds=seconds,
                 config_sha256=receipt.get("effective_config_sha256", receipt.get("config_sha256")), runtime_input_hashes=payload["runtime_input_hashes"],
                 source_hashes=receipt.get("source_hashes", {}))
    return rows, audit


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        if rows:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


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
    if not output.resolve().is_relative_to(root):
        raise ValueError("Audit output must remain inside this task's protocol directory")
    protocol_sha = sha256_file(protocol_path)
    cells = {c["cell_id"]: c for c in protocol["cells"]}
    if len(cells) != 57 or sum(c["count"] for c in cells.values()) != 57000:
        raise ValueError("Unexpected formal study scope")
    weight_checks = {}
    for cell in cells.values():
        model = cell["checkpoint"]
        if model["path"] not in weight_checks:
            actual = sha256_file(resolve(root, model["path"]))
            if actual != model["sha256"]:
                raise ValueError(f"Model changed: {cell['pde']}")
            weight_checks[model["path"]] = actual
    by_cell = defaultdict(list)
    failures = []
    for path in sorted(results_root.rglob("prediction.pt")):
        if not path.resolve().is_relative_to(results_root.resolve()):
            raise ValueError("Result symlink points outside the isolated result directory")
        receipt_path = path.parent / "receipt.json"
        if not receipt_path.exists():
            failures.append(dict(file=str(path), error="Missing committed receipt"))
            continue
        receipt = json.loads(receipt_path.read_text())
        cell_id = receipt.get("cell_id")
        if cell_id not in cells:
            failures.append(dict(file=str(path), error=f"Unexpected cell {cell_id}"))
            continue
        by_cell[cell_id].append(path)
    per_sample, batches, summaries = [], [], []
    coverage = {}
    for cell_id, cell in sorted(cells.items()):
        data = load_cell(root, cell, verify_hashes=True)
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
        if cell_rows:
            summary = dict(cell_id=cell_id, cohort=cell["cohort"], pde=cell["pde"], distribution=cell["distribution"], setting=cell["setting"], n=len(cell_rows))
            for key in ("error_a_percent", "error_u_percent"):
                values = np.asarray([row[key] for row in cell_rows if row[key] is not None], dtype=np.float64)
                summary[key + "_mean"] = None if not len(values) else float(values.mean())
                summary[key + "_sd"] = None if len(values) < 2 else float(values.std(ddof=1))
            summaries.append(summary)
    complete = not failures and all(v["found"] == v["expected"] for v in coverage.values())
    counts = dict(Counter(row["cohort"] for row in per_sample))
    if complete and counts != protocol["expected_predictions_by_cohort"]:
        raise ValueError("Cohort totals do not match frozen scope")
    report = dict(status="pass" if complete else "partial" if allow_partial and not failures else "fail", complete=complete,
                  expected_predictions=57000, verified_predictions=len(per_sample), predictions_by_cohort=counts,
                  verified_batches=len(batches), failed_batches=failures, coverage=coverage,
                  protocol_sha256=protocol_sha, checkpoint_hashes_verified=weight_checks,
                  field_errors="CPU float64, independently recomputed from stored physical predictions and paired source truths",
                  executed_inputs="Six dtype/shape/byte hashes per batch checked against frozen comparator truth, masks and observations",
                  historical_file_protection_scope="Five frozen checkpoint copies are rehashed. This auditor writes only the explicitly designated task audit directory; it does not assert a bytewise scan of unrelated historical output trees.",
                  audit_script_sha256=sha256_file(Path(__file__)), batch_evidence=batches)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "per_sample.csv", sorted(per_sample, key=lambda r: (r["cell_id"], r["index"])))
    write_csv(output / "summary.csv", summaries)
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
