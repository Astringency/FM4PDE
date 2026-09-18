"""Import existing controls only after input, model, code and replay checks."""
from __future__ import annotations
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from experiments.aligned_sampling.input_sources import load_cell
from experiments.aligned_sampling.run_inference import effective_config, observation_batch, tensor_digest
from experiments.rough_stress.run import evaluate_errors, git_commit, sha256, tensor_hash, write_json, ROOT
from sampling.model_io import load_fm4pde_checkpoint_bundle


def require(value, message):
    if not value:
        raise ValueError(message)


def model_state(path):
    net, normalizer, _ = load_fm4pde_checkpoint_bundle(str(path), "poisson", "cpu", model_profile="recommended")
    state = getattr(net, "model", net).state_dict()
    norm = {k: v.tolist() if torch.is_tensor(v) else v for k,v in vars(normalizer).items()}
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return state, norm, digest.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", required=True)
    p.add_argument("--input-root", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--publish", action="store_true")
    args = p.parse_args()
    torch.set_num_threads(2)
    source, inputs, output = map(Path, (args.source_root, args.input_root, args.output_root))
    require((output/"logs/pilot_reuse_controls.exit").read_text().strip() == "0", "Control pilots are not complete")
    protocol = json.loads((source/"protocol.json").read_text())
    require(sha256(source/"protocol.json") == sha256(output/"setup/reference_protocol.json"), "Reference protocol changed")
    source_audit = json.loads((source/"audits/final/audit.json").read_text())
    require(source_audit["complete"], "Original study is not complete")
    old_weight, new_weight = source/"inputs/weights/poisson.pth", output/"setup/fm4poisson.pth"
    a, na, da = model_state(old_weight)
    b, nb, db = model_state(new_weight)
    require(set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a), "Loaded model weights differ")
    require(na == nb and da == db, "Normalizer or model differs")
    old_sha, new_sha = sha256(old_weight), sha256(new_weight)
    del a, b
    proof = dict(status="verified", source_checkpoint_sha256=old_sha,
        equivalent_checkpoint_sha256=new_sha, model_parameters_exact=True,
        normalizer_exact=True, loaded_model_sha256=da, normalizer=na,
        source_checkpoint=str(old_weight), equivalent_checkpoint=str(new_weight),
        source_protocol_sha256=sha256(source/"protocol.json"), cells=[])
    noise_manifest = json.loads((output/"setup/noise_a100_seed0/manifest.json").read_text())
    noise_maps = [(c,np.load(output/"setup/noise_a100_seed0"/c["file"], mmap_mode="r")) for c in noise_manifest["chunks"]]
    ignored_config = {"checkpoint_path", "output_dir", "device", "data_path", "data_paths", "shared_mask"}
    unused_changed_sources = {"sampling/config.py", "sampling/sample_sweep.py"}
    plans = []
    for dist in ("id", "rough"):
        pack_path = inputs/"inputs"/f"poisson_{dist}.pt"
        receipt = json.loads(pack_path.with_suffix(".json").read_text())
        require(sha256(pack_path) == receipt["pack_sha256"], "Input pack changed")
        pack = torch.load(pack_path, map_location="cpu", weights_only=False)
        for task, setting in (("forward","sparse_forward"),("inverse","sparse_inverse"),("both","sparse_joint")):
            cell = next(c for c in protocol["cells"] if c["cell_id"] == f"supervised/poisson/{dist}/{setting}")
            require(cell["checkpoint"]["sha256"] == old_sha, "Wrong archived checkpoint")
            data = load_cell(source, cell, verify_hashes=True)
            require(torch.equal(pack["raw"]["full_tensor"][:,:1], data["coef_ground_truth"]), "Source fields differ")
            require(torch.equal(pack["raw"]["full_tensor"][:,1:2], data["sol_ground_truth"]), "Solution fields differ")
            require(pack["sample_ids"] == data["sample_ids"], "Sample IDs differ")
            require(pack["raw"].get("pde_params",{}) == data["pde_params"], "PDE parameters differ")
            for field, active in (("coef", task in {"forward","both"}), ("sol", task in {"inverse","both"})):
                expected = pack["masks"][500] if active else torch.zeros_like(pack["masks"][500])
                require(torch.equal(expected, data["masks"][field]), "Observation locations differ")
            reference = copy.deepcopy(cell)
            reference["config"].update(num_obs=500, test_type=dist, data_paths={}, data_path=receipt["source"], shared_mask=True)
            records, coverage, first = [], [], {}
            for old_receipt in (source/"jobs").glob(f"prod_*/supervised/poisson/{dist}/{setting}/batch_*/receipt.json"):
                record = json.loads(old_receipt.read_text())
                prediction = old_receipt.parent/"prediction.pt"
                require(sha256(prediction) == record["prediction_sha256"], "Original prediction changed")
                saved = torch.load(prediction, map_location="cpu", weights_only=False)
                ids = record["indices"]
                require(ids == list(range(ids[0], ids[-1]+1)), "Noncontiguous original batch")
                require(saved["indices"] == ids and saved["source_indices"] == ids, "Original indices differ")
                require(saved["sample_ids"] == [pack["sample_ids"][i] for i in ids], "Original IDs differ")
                require(record["checkpoint_sha256"] == old_sha, "Original inference checkpoint differs")
                require(record["identity"]["torch"] == "2.8.0+cu128" and not record["identity"]["tf32"], "Original numerical environment differs")
                for name, digest in record["source_hashes"].items():
                    if name.startswith("revision_pairing_0915/"):
                        old_code = source/"code_v4"/name
                        new_code = ROOT/"experiments/aligned_sampling"/Path(name).name
                        require(sha256(old_code) == digest, "Archived inference source changed")
                        def functions(path):
                            return {node.name:ast.dump(node, include_attributes=False)
                                for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef)}
                        old_functions, new_functions = functions(old_code), functions(new_code)
                        names = ("infer","effective_config","observation_batch","configure_runtime","_slice_params","tensor_digest") if Path(name).name == "run_inference.py" else tuple(old_functions)
                        require(all(old_functions[k] == new_functions.get(k) for k in names), "Executed inference functions changed")
                    elif name not in unused_changed_sources:
                        require(sha256(ROOT/name) == digest, f"Inference source changed: {name}")
                cfg = effective_config(reference, new_weight, "cuda:0", ids, 1000, output/"unwritten_reuse_check")
                current, previous = cfg.asdict(), saved["effective_config"]
                differences = [k for k in set(current)|set(previous) if k not in ignored_config and current.get(k)!=previous.get(k)]
                require(not differences, f"Scientific configuration changed: {differences}")
                _, _, hashes = observation_batch(data, cfg, ids, "cpu")
                require(hashes == record["runtime_input_hashes"], "Actual inference inputs differ")
                initial = torch.from_numpy(np.stack([next(np.array(array[i-c["start"],0],copy=True) for c,array in noise_maps if c["start"]<=i<c["stop"]) for i in ids]))
                require(tensor_digest(initial) == record["initial_noise_sha256"], "Original canonical noise differs")
                require(record["steps"] == record["nfe"] == 100 and record["random_policy"] == "canonical_pool_archived_seed_fixed_input_row_v1", "Original sampler differs")
                pred = saved["prediction"]
                rows = evaluate_errors(pred, pack["raw"]["full_tensor"][ids], task, "fm4pde")
                for i,row,old_row in zip(ids,rows,record["metrics"]):
                    for field in ("a","u"):
                        require(abs(row[f"rel_l2_{field}"]-old_row[f"rel_l2_{field}"]) < 1e-10, "Original score differs")
                    row.update(index=i, sample_id=pack["sample_ids"][i])
                for j,i in enumerate(ids):
                    if i<4:first[i] = pred[j]
                records.append(dict(source=str(prediction), source_receipt=str(old_receipt),
                    original_receipt=record, rows=rows, mask_sha256=tensor_hash(pack["masks"][500][ids])))
                coverage.extend(ids)
            require(sorted(coverage) == list(range(1000)), "Original coverage is not exactly 1000")
            pilot = output/"pilots/reuse_controls/results/poisson/fm4pde"/task/dist/"obs500/batch_0000_0004.pt"
            pred = torch.load(pilot, map_location="cpu", weights_only=False)["prediction"].double()
            original = torch.stack([first[i] for i in range(4)]).double()
            error = ((pred-original).flatten(2).norm(dim=2)/original.flatten(2).norm(dim=2).clamp_min(1e-12)).max().item()
            require(error < 1e-4, f"Control replay differs: {error}")
            proof["cells"].append(dict(task=task, distribution=dist, count=1000,
                truth_masks_ids_and_runtime_inputs_exact=True, all_initial_noise_rows_exact=True,
                scientific_config_exact=True, replay_max_relative_difference=error,
                current_input_receipt=receipt))
            plans.append(dict(task=task, distribution=dist, cell=cell, input_receipt=receipt, records=records))
            print("CONTROL VERIFIED", dist, task, "1000 inputs; pilot difference", error, flush=True)
        del pack
    proof_path = output/"setup/control_reuse_equivalence.json"
    write_json(proof_path, proof)
    if not args.publish:
        return
    for plan in plans:
        task, dist = plan["task"], plan["distribution"]
        key = f"fm4pde_{task}_{dist}_obs500"
        state = output/"queue_state"
        if (state/f"{key}.json").exists():
            print("SKIP ALREADY CLAIMED", key, flush=True)
            continue
        try:
            (state/f"{key}.claim").mkdir()
        except FileExistsError:
            print("SKIP ALREADY CLAIMED", key, flush=True)
            continue
        write_json(state/f"{key}.json", dict(status="claimed", origin="verified_prior_paired_evaluation", cell=dict(method="fm4pde",task=task,distribution=dist,num_obs=500)))
        folder = output/"results/poisson/fm4pde"/task/dist/"obs500"
        folder.mkdir(parents=True, exist_ok=False)
        write_json(folder/"identity.json", dict(pde="poisson",method="fm4pde",task=task,distribution=dist,
            num_obs=500,count=1000,tf32=False,input_receipt=plan["input_receipt"],batch_size=64,
            checkpoint=dict(path=str(old_weight),sha256=old_sha),
            equivalent_current_checkpoint=dict(path=str(new_weight),sha256=new_sha),
            checkpoint_equivalence_file="setup/control_reuse_equivalence.json",checkpoint_equivalence_sha256=sha256(proof_path),
            origin="verified_prior_paired_evaluation",source_archive=str(source),
            import_code_commit=git_commit(ROOT),fm_reference_cell=plan["cell"]["cell_id"],
            sensor_policy="baseline_v3_seed1_nested_400_in_500_shared_across_fields",
            missing_policy="native_variable_observations_zero_grid_missing_mask"))
        for record in plan["records"]:
            old = record["original_receipt"]
            ids = old["indices"]
            destination = folder/f"batch_{ids[0]:04d}_{ids[-1]+1:04d}.pt"
            shutil.copyfile(record["source"], destination)
            require(sha256(destination) == old["prediction_sha256"], "Imported prediction changed")
            runtime_keys = ("seconds","nfe","steps","batch_size","initial_noise_sha256","peak_allocated_bytes","peak_reserved_bytes","loss_gradient_reduction","clip_scope","fused_weighted_gradient","gradient_operator","random_policy")
            write_json(destination.with_suffix(".json"), dict(start=ids[0],stop=ids[-1]+1,
                rows=record["rows"],runtime={k:old[k] for k in runtime_keys},
                observation_mask_sha256=record["mask_sha256"],prediction_sha256=old["prediction_sha256"],
                original_prediction=record["source"],original_receipt=old,
                original_receipt_path=record["source_receipt"],original_receipt_sha256=sha256(record["source_receipt"])))
        write_json(state/f"{key}.json", dict(status="complete",origin="verified_prior_paired_evaluation",
            cell=dict(method="fm4pde",task=task,distribution=dist,num_obs=500),exit_code=0,finished_at=time.time(),
            source_archive=str(source),verification_proof_sha256=sha256(proof_path)))
        print("CONTROL IMPORTED", key, flush=True)


if __name__ == "__main__":
    main()
