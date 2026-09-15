"""Repeat the original seven-candidate NS weight selection, then confirmation."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from revision_ns44m_0915.run_ablations import (atomic_json, errors, file_hash,
    json_hash, load_json, tensor_hash, tree_hash)
from revision_ns44m_0915.prepare_inputs import freeze_job

MULTIPLIERS = [0, 1, 10, 100, 1000, 10000, 1000000]


def choose(development):
    baseline = next(r for r in development if r["multiplier"] == 0)
    eligible = [r for r in development if r["multiplier"] > 1 and
                all(r["mean_errors"][f] <= 1.02 * baseline["mean_errors"][f] for f in ("a", "u"))]
    selected = min(eligible, key=lambda r: r["mean_pde"]) if eligible else None
    if selected and selected["mean_pde"] >= baseline["mean_pde"]:
        selected = None
    return dict(selected_multiplier=selected["multiplier"] if selected else None,
                eligible=[r["multiplier"] for r in eligible], development=development,
                selected_on_development_only=True,
                selection="min mean physical MSE; each field mean RelL2 <= 1.02 times Obs-only; multipliers > 1")


def frozen_protocol(main, path, jobs, stage, main_hash):
    protocol = {k:copy.deepcopy(v) for k,v in main.items() if k not in ("jobs", "family_counts", "scope")}
    protocol.update(jobs=jobs, expected_jobs=len(jobs), expected_predictions=sum(len(j["sample_ids"]) for j in jobs),
                    family_counts={stage:len(jobs)}, scope="Physical-weight "+stage,
                    pilot_protocol_sha256=main_hash)
    if path.exists():
        assert load_json(path) == protocol
    else:
        atomic_json(path, protocol)
    return protocol


def run_stage(args, path, output, label):
    runner = ROOT / "revision_ns44m_0915/run_ablations.py"
    for mode in ("run", "audit"):
        argv = [sys.executable, str(runner), mode, "--protocol", str(path), "--output", str(output)]
        if mode == "run":
            argv += ["--checkpoint", str(args.checkpoint), "--worker", args.worker+"_"+label,
                     "--pilot-certificate", str(args.pilot_certificate)]
        start = time.time()
        result = subprocess.run(argv)
        record = dict(argv=argv, returncode=result.returncode, seconds=time.time()-start)
        atomic_json(args.output / "weight_study" / (label+"_"+mode+"_command.json"), record)
        if result.returncode:
            raise RuntimeError(str(record))


def physical_losses(payload):
    import numpy as np
    import torch
    from sampling.config import AblationConfig
    from sampling.losses import _pde_params_with_residual_options
    from sampling.pde_residuals import compute_pde_residual
    cfg = AblationConfig(**payload["config"])
    params = _pde_params_with_residual_options(payload["pde_params"], cfg)
    with torch.no_grad():
        result = compute_pde_residual("nsnonbounded", payload["coef_final"].double(),
            payload["sol_final"].double(), pde_params=params, residual_mode=cfg.residual_mode, k=cfg.k)
        values = np.zeros(payload["coef_final"].shape[0])
        for key, value in result.components.items():
            if value is not None:
                weight = cfg.bc_weight if key == "boundary" else cfg.endpoint_bc_weight if key == "endpoint" else 1.
                values += weight * value.square().flatten(1).mean(1).numpy()
    assert np.isfinite(values).all()
    return values.tolist()


def prepare(args):
    inventory = load_json(args.inventory)["jobs"]
    main = load_json(args.output / "protocol.json")
    main_hash = file_hash(args.output / "protocol.json")
    development = [r for r in inventory if r["family"] == "weight_development"]
    assert len(development) == 7
    devroot = args.output / "weight_study/development"
    jobs = [freeze_job(r, devroot) for r in development]
    for j in jobs:
        assert j["sample_ids"] == [1100,1101,1102,1103]
    observed = sorted(j["config"]["zeta_pde"] / .1 for j in jobs)
    assert observed == MULTIPLIERS, observed
    devprotocol = devroot / "protocol.json"
    frozen_protocol(main, devprotocol, jobs, "weight_development", main_hash)
    templates = [r for r in inventory if r["family"] == "weight_confirmation" and r["source_config"]["zeta_pde"] == 0]
    assert len(templates) == 2
    template_jobs = [freeze_job(r, args.output) for r in templates]
    assert sorted(i for j in template_jobs for i in j["sample_ids"]) == list(range(1500,1508))
    template_path = args.output / "weight_study/confirmation_templates.json"
    template_value = dict(jobs=template_jobs)
    if template_path.exists():
        assert load_json(template_path) == template_value
    else:
        atomic_json(template_path, template_value)
    return main, main_hash, inventory, devroot, devprotocol


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["prepare", "run"])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--inventory", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--pilot-certificate", type=Path)
    p.add_argument("--worker", default="g7")
    args = p.parse_args()
    args.output = args.output.resolve()
    if args.mode == "prepare":
        prepare(args); return
    import torch
    torch.set_num_threads(2)
    # Preparation is local. Runtime only reads the already frozen packets.
    main_protocol = load_json(args.output / "protocol.json")
    main_hash = file_hash(args.output / "protocol.json")
    devroot = args.output / "weight_study/development"
    devprotocol = devroot / "protocol.json"
    dev = load_json(devprotocol)
    run_stage(args, devprotocol, devroot, "development")
    summaries = []
    hashes = []
    for j in dev["jobs"]:
        folder = devroot / "jobs" / j["job_id"]
        receipt = load_json(folder / "receipt.json")
        payload = torch.load(folder / receipt["artifacts"]["result"]["path"], map_location="cpu", weights_only=False)
        losses = physical_losses(payload)
        summaries.append(dict(multiplier=int(round(j["config"]["zeta_pde"]/.1)),
            mean_pde=sum(losses)/len(losses), pde_mse=losses,
            mean_errors={f:sum(receipt["errors"][f])/4 for f in ("a", "u")}))
        hashes.append((receipt["initial_noise_sha256"], receipt["initial_rng_state_sha256"],
                       json_hash(tree_hash(payload["masks"])) ))
    assert len(set(hashes)) == 1, "Development candidates must share observations and random initialization"
    summaries.sort(key=lambda r: r["multiplier"])
    assert [r["multiplier"] for r in summaries] == MULTIPLIERS
    selected = choose(summaries)
    selected["development_protocol_sha256"] = file_hash(devprotocol)
    selection_path = args.output / "weight_study/selection.json"
    if selection_path.exists():
        assert load_json(selection_path) == selected
    else:
        atomic_json(selection_path, selected)
    candidates = [0,1] + ([selected["selected_multiplier"]] if selected["selected_multiplier"] is not None else [])
    # The two frozen Obs-only batches supply exactly the historical eight
    # confirmation fields, masks, random seeds and all non-weight parameters.
    templates = load_json(args.output / "weight_study/confirmation_templates.json")["jobs"]
    confirmroot = args.output / "weight_study/confirmation"
    jobs = []
    for template in templates:
        packet = torch.load(args.output / template["input_file"], map_location="cpu", weights_only=False)
        assert file_hash(args.output / template["input_file"]) == template["input_sha256"]
        for multiplier in candidates:
            value = copy.deepcopy(packet)
            value["source"]["original_config"] = copy.deepcopy(value["config"])
            value["config"].update(zeta_pde=.1*multiplier,
                                    guidance_components="obs_only" if multiplier == 0 else "obs_pde")
            value["source"]["selection_sha256"] = file_hash(selection_path)
            value["source"]["derived_weight_multiplier"] = multiplier
            jid = f"m{multiplier}_id{template['sample_ids'][0]}"
            path = confirmroot / "inputs" / (jid+".pt")
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                assert tree_hash(torch.load(path,map_location="cpu",weights_only=False)) == tree_hash(value)
            else:
                tmp = path.with_suffix(".partial.pt"); torch.save(value,tmp);tmp.replace(path)
            jobs.append(dict(job_id=jid, family="weight_confirmation", sample_ids=value["sample_ids"],
                config=value["config"], source=value["source"], input_file=str(path.relative_to(confirmroot)),
                input_sha256=file_hash(path), input_tensor_sha256=json_hash(tree_hash(value["inputs"])),
                expected_initial_noise_sha256=template["expected_initial_noise_sha256"]))
    cp = confirmroot / "protocol.json"
    frozen_protocol(main_protocol, cp, jobs, "weight_confirmation", main_hash)
    run_stage(args, cp, confirmroot, "confirmation")
    atomic_json(args.output / "weight_study/completion.json", dict(status="complete",
        development_predictions=28, confirmation_predictions=8*len(candidates),
        selected_multiplier=selected["selected_multiplier"], selection_sha256=file_hash(selection_path),
        development_audit_sha256=file_hash(devroot/"audit/audit.json"),
        confirmation_audit_sha256=file_hash(confirmroot/"audit/audit.json")))


if __name__ == "__main__":
    main()
