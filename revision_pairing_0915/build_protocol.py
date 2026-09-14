"""Freeze historical FM configurations and join independently prepared paired inputs.

This script does not sample, train, or modify any historical artifact.  A draft
configuration inventory can be produced before inputs and checkpoints are copied.
The executable protocol requires all 57 input entries and five checkpoint hashes.
"""
from __future__ import annotations

import argparse
import copy
import csv
import gzip
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


VERSION = 1
MAIN_PDES = {"poisson", "helmholtz", "darcy"}
RUNTIME_FIELDS = {
    "batch_size", "offset", "device", "output_dir", "checkpoint_path",
    "save_intermediate", "save_plots", "save_per_sample_curves",
    "ablation_name", "ablation_group", "initial_noise_source_batch_size",
    "initial_noise_source_indices", "data_path", "data_paths", "data_config_path",
}
# These values describe the recovered legacy implementation, not inferred tuning.
# Other absent modern fields are left absent; the runner records its resolved
# configuration and its source commit, and must not replace any supplied value.
LEGACY_IMPLEMENTATION = {
    "pde_guidance_reduction": "mse",
    "guidance_operator": "current",
    "legacy_obs_multiplier": 1.0,
    "model_gradient_checkpointing": False,
}
LEGACY_INACTIVE_ELLIPTIC = {
    "enforce_initial_conditions": True,
    "initial_condition_mode": "none",
    "ic_weight": 1.0,
    "legacy_ignore_boundary": False,
    "extra": {},
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def scientific_config(config: dict) -> dict:
    return {key: value for key, value in config.items() if key not in RUNTIME_FIELDS}


def _read_sources(source_data: Path):
    csv_path = source_data / "main_hyperparameters_verified.csv"
    archive_path = source_data / "main_configs_archive.json.gz"
    ns_path = source_data / "ns_main_effective_configurations_0909.json"
    with csv_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    with gzip.open(archive_path, "rt") as stream:
        archive = json.load(stream)
    ns = json.loads(ns_path.read_text())
    sources = [dict(path=str(p.resolve()), sha256=sha256_file(p))
               for p in (csv_path, archive_path, ns_path)]
    return rows, archive, ns, sources


def configuration_cells(source_data: Path) -> tuple[list[dict], list[dict]]:
    rows, archive, ns, sources = _read_sources(source_data)
    groups = defaultdict(list)
    for record in archive["records"]:
        groups[record["sheet"], str(record["workbook_row"])].append(record)
    ns_records = {(r["distribution"], r["setting"]): r for r in ns["records"]}
    cells = []
    for row in rows:
        pde, task, dist = row["pde"], row["task"], row["distribution"].lower()
        wanted = (pde == "nsnonbounded" or
                  (pde in MAIN_PDES | {"burger"} and row["observations"] == "random"))
        if not wanted:
            continue
        assert int(row["n"]) == 1000, row
        setting = (row["revision_setting"] if pde == "nsnonbounded" else
                   "sparse_joint" if task == "both" else f"sparse_{task}")
        if pde == "nsnonbounded":
            record = ns_records[dist, setting]
            archived_config = copy.deepcopy(record["configuration_without_offset"])
            assert archived_config["residual_mode"] == "endpoint_secant"
            assert archived_config["model_profile"] == "auto"
            provenance = dict(kind="ns_current_main_revision_0909", source=str(source_data / "ns_main_effective_configurations_0909.json"),
                              source_record=dict(distribution=dist, setting=setting),
                              producer_commit=ns["producer_commit"], checkpoint_sha256=row["checkpoint_sha256"],
                              historical_input_indices=row["test_index_range"],
                              historical_batches=record["batches"],
                              historical_runtime_batch_size_counts=record["runtime_batch_size_counts"])
        else:
            records = groups[row["sheet"], row["workbook_row"]]
            assert records, (pde, task, dist)
            archived_config = copy.deepcopy(records[0]["config"])
            canonical = {k: v for k, v in archived_config.items() if k not in {"offset", "batch_size"}}
            assert all({k: v for k, v in r["config"].items() if k not in {"offset", "batch_size"}} == canonical for r in records), (pde, task, dist, "historical configuration drift")
            assert sum(r["config"]["batch_size"] for r in records) == 1000
            provenance = dict(kind="main_archived_resolved_config", sheet=row["sheet"], workbook_row=row["workbook_row"],
                              representative_source=records[0]["source"], representative_sha256=records[0]["sha256"],
                              archived_batches=len(records), archive_record_hashes=[r["sha256"] for r in records],
                              historical_runtime_batch_sizes=sorted({r["config"]["batch_size"] for r in records}),
                              historical_input_offsets=[r["config"]["offset"] for r in records])
        config = copy.deepcopy(archived_config)
        # Eight archived Smooth rows carry a removed dataset-loader option.
        # Physical inputs are supplied by the independently frozen paired cache;
        # retain the old loader path as provenance instead of passing an unknown
        # constructor argument to the current sampling dataclass.
        loader_metadata = {}
        if "data_config_path" in config:
            loader_metadata["data_config_path"] = config.pop("data_config_path")
        inactive_metadata = {}
        for key, expected in LEGACY_INACTIVE_ELLIPTIC.items():
            if key in config:
                assert pde in MAIN_PDES and config[key] == expected, (pde, dist, task, key, config[key])
                inactive_metadata[key] = config.pop(key)
        completion = {k: v for k, v in LEGACY_IMPLEMENTATION.items() if k not in config}
        config.update(completion)
        # These are the scientifically active entries exposed by the verified
        # source CSV.  Preserve and cross-check their values individually.
        for key in ("zeta_obs_a", "zeta_obs_u", "zeta_pde", "clip_threshold", "num_steps", "num_obs", "stochastic_guidance_coeff"):
            assert float(config[key]) == float(row[key]), (pde, dist, task, key)
        for key in ("clip_mode", "residual_mode", "sampler_phase", "step_method", "loss_state"):
            assert config[key] == row[key], (pde, dist, task, key)
        assert config["task"] == task and config["pde"] == pde
        assert config["num_steps"] == 100 and config["noise_level"] == 0
        cell = dict(cell_id=f"supervised/{pde}/{dist}/{setting}", cohort="supervised",
                    pde=pde, distribution=dist, setting=setting, task=task, count=1000,
                    historical_config=archived_config, config=config,
                    config_sha256=json_hash(config), scientific_config_sha256=json_hash(scientific_config(config)),
                    config_provenance=provenance, legacy_implementation_completion=completion,
                    historical_loader_metadata=loader_metadata,
                    historical_inactive_metadata=inactive_metadata,
                    historical_inactive_metadata_reason=("These elliptic tasks have no initial-condition term; the archived mode is none, boundary ignoring is false, and extra is empty. Removed dataclass metadata is retained here and in historical_config." if inactive_metadata else None),
                    recorded_checkpoint_path=row.get("recorded_checkpoint_path") or archived_config["checkpoint_path"],
                    historical_data_path=archived_config.get("data_path"))
        cells.append(cell)
    assert len(cells) == 45, len(cells)
    for original in list(cells):
        if original["pde"] in MAIN_PDES | {"nsnonbounded"} and original["distribution"] == "smooth" and original["setting"].startswith("sparse_"):
            cell = copy.deepcopy(original)
            cell["cohort"] = "diffusion"
            cell["cell_id"] = cell["cell_id"].replace("supervised/", "diffusion/", 1)
            cell["config_provenance"]["reuse_scope"] = "Same current FM model and task hyperparameters; physical inputs and sensor positions are independently paired to the archived DiffusionPDE Smooth cohort."
            cells.append(cell)
    cells.sort(key=lambda r: r["cell_id"])
    assert len(cells) == len({c["cell_id"] for c in cells}) == 57
    assert Counter(c["cohort"] for c in cells) == {"supervised": 45, "diffusion": 12}
    return cells, sources


def _rebase(path: str, original_root: Path, output_root: Path) -> str:
    resolved = Path(path) if Path(path).is_absolute() else original_root / path
    # Use absolute paths only when the input directory differs from the protocol
    # directory; normal deployment keeps both in one portable task directory.
    try:
        return str(resolved.resolve().relative_to(output_root.resolve()))
    except ValueError:
        return str(resolved.resolve())


def inventory_cells(path: Path) -> tuple[list[dict], list[dict]]:
    inventory = json.loads(path.read_text())
    unhashed = dict(inventory)
    expected = unhashed.pop("content_sha256")
    assert json_hash(unhashed) == expected, "Configuration inventory content hash mismatch"
    cells = copy.deepcopy(inventory["cells"])
    assert len(cells) == len({c["cell_id"] for c in cells}) == 57
    assert Counter(c["cohort"] for c in cells) == {"supervised": 45, "diffusion": 12}
    for cell in cells:
        assert json_hash(cell["config"]) == cell["config_sha256"]
        assert json_hash(scientific_config(cell["config"])) == cell["scientific_config_sha256"]
        assert cell["count"] == 1000
    sources = copy.deepcopy(inventory["sources"])
    sources.append(dict(path=str(path.resolve()), sha256=sha256_file(path), role="verified_configuration_inventory"))
    return cells, sources


def completed_inputs_snapshot(input_root: Path, output: Path) -> Path:
    """Revalidate completed datasets and freeze a portable pilot input subset."""
    try:
        from .input_sources import load_cell
    except ImportError:
        from input_sources import load_cell
    import torch
    torch.set_num_threads(1)
    entries, receipts, pending, catalogs = [], [], [], set()
    for path in sorted((input_root / "receipts").glob("*.json")):
        try:
            record = json.loads(path.read_text())
        except json.JSONDecodeError:
            pending.append(dict(receipt=path.name, reason="Receipt write not yet complete"))
            continue
        if record.get("status") != "complete":
            pending.append(dict(receipt=path.name, reason=record.get("status", "unknown")))
            continue
        catalogs.add(record["catalog_sha256"])
        assert sha256_file(input_root / record["per_input_file"]) == record["per_input_sha256"]
        for entry in record["cells"]:
            load_cell(input_root, entry, verify_hashes=True)
            row = copy.deepcopy(entry)
            for key in ("truth_file", "masks_file"):
                row[key] = _rebase(row[key], input_root, output.parent)
            entries.append(row)
        receipts.append(dict(dataset_id=record["dataset_id"],
                             receipt=_rebase(str(path.resolve()), input_root, output.parent),
                             receipt_sha256=sha256_file(path),
                             per_input_file=_rebase(record["per_input_file"], input_root, output.parent),
                             per_input_sha256=record["per_input_sha256"]))
    assert entries, "No fully verified completed input dataset is available"
    assert len(catalogs) == 1, "Completed input datasets came from different catalogs"
    assert len(entries) == len({e["cell_id"] for e in entries})
    snapshot = dict(version=VERSION, task_name="baseline_pairing_20260915_57k", status="pilot_only",
                    cells=sorted(entries, key=lambda e: e["cell_id"]), datasets=receipts,
                    catalog_sha256=next(iter(catalogs)),
                    validation=dict(all_selected_input_files_rehashed=True,
                                    all_selected_per_input_records_rehashed=True,
                                    no_model_inference_performed=True))
    # Pending receipts are intentionally not part of the immutable snapshot:
    # their progress cannot change the identity of already completed inputs.
    if pending:
        print(json.dumps(dict(pending_datasets_ignored=pending)))
    path = output.with_name(output.stem + ".inputs.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert json.loads(path.read_text()) == snapshot, "Pilot input snapshot already exists with a different completed subset; use a new output name"
    else:
        path.write_text(json.dumps(snapshot, indent=2, allow_nan=False) + "\n")
    return path


def build_protocol(source_data: Path | None, inputs_manifest: Path | None,
                   checkpoint_map: Path | None, output: Path, *, draft=False,
                   pilot_only=False, config_inventory: Path | None = None,
                   inputs_root: Path | None = None) -> dict:
    assert not (draft and pilot_only)
    if config_inventory:
        cells, sources = inventory_cells(config_inventory)
    else:
        assert source_data, "Supply --source-data or --config-inventory"
        cells, sources = configuration_cells(source_data)
    if pilot_only and inputs_root:
        assert inputs_manifest is None, "Choose completed receipts or an input manifest, not both"
        inputs_manifest = completed_inputs_snapshot(inputs_root, output)
    if pilot_only:
        assert inputs_manifest, "Pilot mode needs --inputs-root or --inputs-manifest"
        partial_inputs = json.loads(inputs_manifest.read_text())
        selected = {c["cell_id"] for c in partial_inputs["cells"]}
        assert selected and selected <= {c["cell_id"] for c in cells}
        cells = [c for c in cells if c["cell_id"] in selected]
    protocol = dict(version=VERSION, status="configuration_draft" if draft else "pilot_only" if pilot_only else "frozen",
                    study="baseline_pairing_20260915_57k", cells=cells, sources=sources,
                    expected_cells=len(cells), expected_predictions=sum(c["count"] for c in cells),
                    expected_predictions_by_cohort={cohort: sum(c["count"] for c in cells if c["cohort"] == cohort) for cohort in ("supervised", "diffusion")},
                    formal_scope=dict(cells=57, predictions=57000, supervised_predictions=45000, diffusion_predictions=12000),
                    production_allowed=not draft and not pilot_only,
                    requested_changes=["Use physical input and sensor tensors of the corresponding comparator cohort", "Canonical per-input sampling randomness and audited batch execution"],
                    preserved_scientific_settings="Each cell retains its current manuscript checkpoint, guidance weights, physical residual, clipping threshold, and 100-step sampler configuration. No training or tuning.",
                    runtime_override_allowlist=sorted(RUNTIME_FIELDS | {"sensor_mode", "shared_mask"}),
                    runtime_override_semantics={"sensor_mode/shared_mask": "External masks from the paired cache override mask generation; executed tensors are hashed.",
                                               "initial_noise_source": "Seed from each saved config; canonical 1000 input rows provide partition-independent draws as documented by the runner.",
                                               "batch_size": "Chosen only after measured device-specific memory and numerical pilots; historical batch sizes are preserved in provenance."})
    if not draft:
        assert inputs_manifest and checkpoint_map
        inputs = json.loads(inputs_manifest.read_text())
        if not pilot_only:
            assert inputs.get("status") == "complete", "Final production protocol requires the complete merged input manifest"
            assert len(cells) == 57 and sum(c["count"] for c in cells) == 57000
        model_map = json.loads(checkpoint_map.read_text())
        model_map = model_map.get("checkpoints", model_map)
        entries = {c["cell_id"]: c for c in inputs["cells"]}
        assert set(entries) == {c["cell_id"] for c in cells}, "Input/config cell inventories differ"
        protocol["inputs_manifest"] = _rebase(str(inputs_manifest.resolve()), output.parent, output.parent)
        protocol["inputs_manifest_sha256"] = sha256_file(inputs_manifest)
        protocol["checkpoint_map_sha256"] = sha256_file(checkpoint_map)
        for cell in cells:
            entry = entries[cell["cell_id"]]
            for key in ("cohort", "pde", "distribution", "setting", "task", "count"):
                assert entry[key] == cell[key], (cell["cell_id"], key)
            for key in ("truth_file", "masks_file"):
                cell[key] = _rebase(entry[key], inputs_manifest.parent, output.parent)
            for key in ("truth_sha256", "masks_sha256", "observed_fields", "num_obs"):
                cell[key] = entry[key]
            assert cell["num_obs"] == cell["config"]["num_obs"], (cell["cell_id"], "observation count changed")
            cell["input_provenance"] = {k: copy.deepcopy(v) for k, v in entry.items() if k not in cell}
            model = copy.deepcopy(model_map[cell["pde"]])
            assert re.fullmatch(r"[0-9a-f]{64}", model["sha256"])
            # Checkpoint-map deployment paths are relative to the final protocol.
            cell["checkpoint"] = model
            if cell["pde"] == "nsnonbounded":
                assert model["sha256"] == cell["config_provenance"]["checkpoint_sha256"]
        if pilot_only:
            try:
                from .input_sources import load_cell
            except ImportError:
                from input_sources import load_cell
            checked_models = set()
            for cell in cells:
                load_cell(output.parent, cell, verify_hashes=True)
                model = cell["checkpoint"]
                model_path = Path(model["path"])
                if not model_path.is_absolute():
                    model_path = output.parent / model_path
                if str(model_path) not in checked_models:
                    assert sha256_file(model_path) == model["sha256"], "Pilot checkpoint hash mismatch"
                    checked_models.add(str(model_path))
    protocol["content_sha256"] = json_hash(protocol)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        assert json.loads(output.read_text()) == protocol, "Refusing to overwrite a different protocol"
    else:
        output.write_text(json.dumps(protocol, indent=2, allow_nan=False) + "\n")
    return protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-data", type=Path)
    parser.add_argument("--config-inventory", type=Path)
    parser.add_argument("--inputs-manifest", type=Path)
    parser.add_argument("--checkpoint-map", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draft", action="store_true")
    parser.add_argument("--pilot-only", action="store_true")
    parser.add_argument("--inputs-root", type=Path, help="In pilot-only mode, collect and verify completed receipts under this prepared-input directory")
    args = parser.parse_args()
    if not args.source_data and not args.config_inventory:
        parser.error("--source-data or --config-inventory is required")
    if args.source_data and args.config_inventory:
        parser.error("Choose --source-data or --config-inventory")
    if args.inputs_root and not args.pilot_only:
        parser.error("--inputs-root is only valid with --pilot-only")
    protocol = build_protocol(args.source_data, args.inputs_manifest, args.checkpoint_map, args.output,
                              draft=args.draft, pilot_only=args.pilot_only,
                              config_inventory=args.config_inventory, inputs_root=args.inputs_root)
    print(json.dumps({"status": protocol["status"], "cells": len(protocol["cells"]), "predictions": protocol["expected_predictions"], "sha256": sha256_file(args.output)}))


if __name__ == "__main__":
    main()
