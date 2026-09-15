"""Prepare an additive eight-cell correction; never run inference or alter v1.

The output bundle is destined for V1_ROOT/corrections/v2.  Its protocol keeps
all 57 cells for compatibility with the unchanged runner; correction_slices.json
selects only the eight replacements.  Drafts cannot authorize production.
The result selection is by complete cell, fixed before corrected results exist.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from revision_pairing_0915.build_protocol import json_hash, scientific_config, sha256_file

AFFECTED = frozenset(
    f"{cohort}/{pde}/smooth/{setting}"
    for cohort in ("supervised", "diffusion")
    for pde, setting in (("poisson", "sparse_forward"), ("poisson", "sparse_joint"),
                         ("helmholtz", "sparse_forward"), ("darcy", "sparse_joint"))
)
DEPLOYMENT = Path("corrections/v2")
GATE = {"pde_guidance_start_ratio": 0.0, "pde_guidance_ramp_ratio": 0.0}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def encoded(value):
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()


def rebase(name, v1_root):
    source = Path(name)
    source = source if source.is_absolute() else v1_root / source
    # Keep portable relative paths when the bundle is copied between hosts.
    require(source.resolve().is_relative_to(v1_root.resolve()),
            f"Execution artifact is outside the frozen task root: {name}")
    return os.path.relpath(source.resolve(), v1_root / DEPLOYMENT)


def validate_effective(old, new, v1_root):
    """Resolve the real dataclass and evaluate its actual 100-step CPU schedule."""
    import torch
    from revision_pairing_0915.run_inference import effective_config
    from sampling.guidance import make_zeta_schedule
    from sampling.time_grid import make_time_grid
    torch.set_num_threads(2)
    records = []
    for before, after in zip(old["cells"], new["cells"]):
        cid = before["cell_id"]
        require(cid == after["cell_id"], "Cell order changed")
        checkpoint = v1_root / before["checkpoint"]["path"]
        kwargs = dict(checkpoint=checkpoint, device="cpu", indices=[0], count=1000,
                      output_dir=v1_root / "unwritten_validation")
        a, b = (effective_config(cell, **kwargs) for cell in (before, after))
        da, db = dataclasses.asdict(a), dataclasses.asdict(b)
        changes = {key: {"v1": da[key], "v2": db[key]} for key in da if da[key] != db[key]}
        expected = {"pde_guidance_start_ratio": {"v1": 0.8, "v2": 0.0}} if cid in AFFECTED else {}
        require(changes == expected, f"Unexpected effective configuration difference: {cid}: {changes}")
        if cid not in AFFECTED:
            require(before["config"] == after["config"], f"Inherited cell config changed: {cid}")
            continue
        grid = make_time_grid(a.time_grid, a.num_steps, device="cpu", eta=a.time_grid_eta)
        schedules = []
        for cfg in (a, b):
            # These eight cells use constant weights, so bt does not enter them.
            schedules.append([make_zeta_schedule(cfg, grid[k], grid[k + 1], torch.ones(()))
                              for k in range(100)])
        old_z, new_z = (torch.stack([s.zeta_pde_t for s in seq]) for seq in schedules)
        require(torch.equal(old_z, torch.cat([torch.zeros(80), torch.full((20,), .1)])),
                f"Unexpected v1 gate behavior: {cid}")
        require(torch.equal(new_z, torch.full((100,), .1)), f"V2 is not all-step .1: {cid}")
        require(all(torch.equal(getattr(x, key), getattr(y, key))
                    for x, y in zip(*schedules) for key in ("zeta_obs_a_t", "zeta_obs_u_t")),
                f"Observation schedule changed: {cid}")
        records.append(dict(cell_id=cid, effective_changes=changes, v1_active_pde_steps=20,
                            v2_active_pde_steps=100, v2_zeta_pde_float32=float(new_z[0]),
                            observation_schedules_identical=True))
    return dict(status="pass", resolved_cells=57, unchanged_cells=49,
                corrected_cells=records, cuda_used=False, inference_run=False,
                input_model_reuse="Paths and expected hashes unchanged; runner/auditor rehash actual artifacts at execution.")


def prepare(v1_protocol, evidence_path, output, *, freeze=False):
    from revision_pairing_0915.run_inference import validate_protocol
    v1_protocol, evidence_path, output = [p.resolve() for p in (v1_protocol, evidence_path, output)]
    require(not output.exists(), f"Output already exists; no overwrites: {output}")
    old_bytes = v1_protocol.read_bytes()
    old = json.loads(old_bytes)
    validate_protocol(old, "run")
    evidence = json.loads(evidence_path.read_text())
    old_sha = hashlib.sha256(old_bytes).hexdigest()
    require(evidence["status"] == "confirmed_historical_gate_mismatch", "Unconfirmed gate evidence")
    require(set(evidence["affected_cell_ids"]) == AFFECTED, "Evidence does not identify exactly eight affected cells")
    require(evidence["v1_protocol"]["file_sha256"] == old_sha, "Evidence binds a different v1 protocol")
    require(all(evidence["old_gate_equivalent"][k] == v for k, v in GATE.items()),
            "Evidence does not support start=0, ramp=0")
    require(evidence["old_gate_equivalent"]["nonzero_weight_steps"] == 100,
            "Evidence does not establish full-step guidance")
    v1_root = v1_protocol.parent
    new = copy.deepcopy(old)
    new.pop("content_sha256")
    new.update(status="frozen" if freeze else "draft_correction", production_allowed=freeze,
               correction_revision="v2", supersedes_scope="Exactly eight complete cells; all v1 artifacts remain unchanged")
    old_binding = {"path": rebase(v1_protocol, v1_root), "sha256": old_sha}
    evidence_binding = {"path": rebase(evidence_path, v1_root), "sha256": sha256_file(evidence_path)}
    new["correction"] = dict(reason="Restore independently verified historical all-step PDE guidance",
                             source_protocol=old_binding, evidence=evidence_binding,
                             explicit_config=GATE, affected_cell_ids=sorted(AFFECTED),
                             new_predictions=8000, inherited_predictions=49000,
                             result_selection="selection.json", execution_slices="correction_slices.json")
    new["inputs_manifest"] = rebase(new["inputs_manifest"], v1_root)
    changed = []
    for before, cell in zip(old["cells"], new["cells"]):
        cid = cell["cell_id"]
        require(cell["config_sha256"] == json_hash(cell["config"]), f"Config SHA mismatch: {cid}")
        require(cell["scientific_config_sha256"] == json_hash(scientific_config(cell["config"])),
                f"Scientific config SHA mismatch: {cid}")
        if cid in AFFECTED:
            cfg = cell["config"]
            require(all(key not in cfg for key in GATE), f"Gate was already explicit: {cid}")
            require(cfg["zeta_pde"] == .1 and cfg["guidance_schedule"] == "constant"
                    and cfg["num_steps"] == 100 and cfg["time_grid"] == "uniform",
                    f"Unexpected affected configuration: {cid}")
            cfg.update(GATE)
            cell["config_sha256"] = json_hash(cfg)
            cell["scientific_config_sha256"] = json_hash(scientific_config(cfg))
            cell["correction_provenance"] = dict(source_protocol=old_binding,
                    source_config_sha256=before["config_sha256"], evidence=evidence_binding,
                    explicit_historical_completion=GATE)
            changed.append(cid)
        for key in ("truth_file", "masks_file"):
            cell[key] = rebase(cell[key], v1_root)
        cell["checkpoint"]["path"] = rebase(cell["checkpoint"]["path"], v1_root)
        require(before["historical_config"] == cell["historical_config"], "Historical record changed")
        require(all(before[k] == cell[k] for k in ("truth_sha256", "masks_sha256", "count")),
                "Input hash/count changed")
        require(before["checkpoint"]["sha256"] == cell["checkpoint"]["sha256"], "Model hash changed")
    require(set(changed) == AFFECTED, "Missing or excess corrected cell")
    validation = validate_effective(old, new, v1_root)
    new["content_sha256"] = json_hash(new)
    if freeze:
        validate_protocol(new, "run")
    new_sha = hashlib.sha256(encoded(new)).hexdigest()
    selection = dict(schema_version=1, status="frozen" if freeze else "draft",
        task=old["study"], target_protocol={"path": "protocol.json", "sha256": new_sha},
        reason=evidence_binding, expected_cells=57, expected_predictions=57000,
        revision_counts={"v1": {"cells": 49, "predictions": 49000}, "v2": {"cells": 8, "predictions": 8000}},
        policy="Select all 1000 rows of each declared revision; never fallback or choose using observed errors",
        cells=[dict(cell_id=c["cell_id"], count=1000, selected_revision="v2" if c["cell_id"] in AFFECTED else "v1",
                    source_protocol={"path": "protocol.json", "sha256": new_sha} if c["cell_id"] in AFFECTED else old_binding,
                    source_results="jobs" if c["cell_id"] in AFFECTED else "../../jobs",
                    scientific_config_sha256=c["scientific_config_sha256"])
               for c in new["cells"]])
    slices = dict(schema_version=1, status="draft_execution_plan", automatic_launch=False,
        protocol={"path": "protocol.json", "sha256": new_sha}, output_root=".",
        expected_cells=8, expected_predictions=8000, expected_slices=16,
        executor="Unchanged run_inference.py --mode run; wrap each CLI in run_stage.py after GPU allocation and corrected pilots",
        jobs=[dict(job_id="corr_v2_" + cid.replace("/", "_") + f"_{start:04d}_{start+500:04d}",
                   cell_id=cid, start=start, stop=start+500,
                   batch_size=None, device=None, pilot_certificate=None)
              for cid in sorted(AFFECTED) for start in (0, 500)])
    validation.update(source_protocol=old_binding, correction_protocol_sha256=new_sha,
                      evidence=evidence_binding, runner_source_sha256=sha256_file(Path(__file__).with_name("run_inference.py")))
    require(v1_protocol.read_bytes() == old_bytes, "V1 protocol changed during preparation")
    output.mkdir(parents=True, exist_ok=False)
    for name, value in (("protocol.json", new), ("selection.json", selection),
                        ("correction_slices.json", slices), ("validation.json", validation)):
        with (output / name).open("xb") as stream:
            stream.write(encoded(value))
    return {"status": "prepared", "production_allowed": freeze, "output": str(output),
            "deployment_relative_to_v1_root": str(DEPLOYMENT), "protocol_sha256": new_sha,
            "selection_sha256": sha256_file(output / "selection.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1-protocol", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--freeze", action="store_true", help="Emit runner-accepted status only after review; does not execute anything")
    args = parser.parse_args()
    print(json.dumps(prepare(args.v1_protocol, args.evidence, args.output_dir, freeze=args.freeze), indent=2))


if __name__ == "__main__":
    main()
