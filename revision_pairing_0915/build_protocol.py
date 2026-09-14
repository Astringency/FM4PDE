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
    "initial_noise_source_indices", "data_path", "data_paths",
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


def build_protocol(source_data: Path, inputs_manifest: Path | None,
                   checkpoint_map: Path | None, output: Path, *, draft=False) -> dict:
    cells, sources = configuration_cells(source_data)
    protocol = dict(version=VERSION, status="configuration_draft" if draft else "frozen",
                    study="baseline_pairing_20260915_57k", cells=cells, sources=sources,
                    expected_cells=57, expected_predictions=57000,
                    expected_predictions_by_cohort={"supervised": 45000, "diffusion": 12000},
                    requested_changes=["Use physical input and sensor tensors of the corresponding comparator cohort", "Canonical per-input sampling randomness and audited batch execution"],
                    preserved_scientific_settings="Each cell retains its current manuscript checkpoint, guidance weights, physical residual, clipping threshold, and 100-step sampler configuration. No training or tuning.",
                    runtime_override_allowlist=sorted(RUNTIME_FIELDS | {"sensor_mode", "shared_mask"}),
                    runtime_override_semantics={"sensor_mode/shared_mask": "External masks from the paired cache override mask generation; executed tensors are hashed.",
                                               "initial_noise_source": "Seed from each saved config; canonical 1000 input rows provide partition-independent draws as documented by the runner.",
                                               "batch_size": "Chosen only after measured device-specific memory and numerical pilots; historical batch sizes are preserved in provenance."})
    if not draft:
        assert inputs_manifest and checkpoint_map
        inputs = json.loads(inputs_manifest.read_text())
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
    protocol["content_sha256"] = json_hash(protocol)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        assert json.loads(output.read_text()) == protocol, "Refusing to overwrite a different protocol"
    else:
        output.write_text(json.dumps(protocol, indent=2, allow_nan=False) + "\n")
    return protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-data", type=Path, required=True)
    parser.add_argument("--inputs-manifest", type=Path)
    parser.add_argument("--checkpoint-map", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draft", action="store_true")
    args = parser.parse_args()
    protocol = build_protocol(args.source_data, args.inputs_manifest, args.checkpoint_map, args.output, draft=args.draft)
    print(json.dumps({"status": protocol["status"], "cells": len(protocol["cells"]), "predictions": protocol["expected_predictions"], "sha256": sha256_file(args.output)}))


if __name__ == "__main__":
    main()
