"""Freeze exact historical NS inputs/configurations for a model-only replay."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from revision_ns44m_0915.run_ablations import (MODEL_SHA256, PARAMETER_COUNT,
    atomic_json, file_hash, json_hash, load_json, tensor_hash, tree_hash)

PUBLISHED_MAIN_IDS = {f"archive_{i:04d}" for i in (
    463,464,465,466,505,506,507,508,509,510,511,512,
    477,478,481,482,485,486,489,490,493,494,497,498,501,502,
    473,474,475,476,518,519,520,521,522)}


def one_result(folder):
    paths = list(Path(folder).rglob("result.pt"))
    assert len(paths) == 1, (folder, paths)
    return paths[0]


def known_sources(args):
    rows = []
    published = args.paper_archive / "ablation_publication_snapshot/records.json"
    for r in load_json(published):
        if r["pde"] != "nsnonbounded" or r["id"] not in PUBLISHED_MAIN_IDS:
            continue
        if not r["result_path"]:
            continue  # Actual original temporal sources are supplied explicitly.
        receipt = load_json(r["receipt"])
        rows.append(dict(job_id="main/"+r["id"], family="main", source_result=r["result_path"],
                         source_receipt=r["receipt"], expected_result_sha256=receipt["result_sha256"],
                         expected_mask_sha256=receipt["mask_sha256"], sample_ids=receipt["sample_ids"],
                         source_config=r["config"], publication_source=str(published),
                         publication_sha256=file_hash(published)))
    for receipt_path in sorted((args.paper_archive / "output/nsnonbounded/ensemble").rglob("receipt.json")):
        receipt = load_json(receipt_path)
        suffix = str(receipt_path.parent.relative_to(args.paper_archive / "output/nsnonbounded/ensemble"))
        rows.append(dict(job_id="ensemble/"+suffix, family="ensemble", source_result=str(one_result(receipt_path.parent)),
                         source_receipt=str(receipt_path), expected_result_sha256=receipt["result_sha256"],
                         expected_mask_sha256=receipt["mask_sha256"], sample_ids=receipt["sample_ids"],
                         source_config=receipt["config"]))
    evaluation = args.controls_archive / "results/nsnonbounded/evaluation"
    for guidance in ("noguide", "pde_only", "obs_only", "obs_pde"):
        for receipt_path in sorted((evaluation / ("guidance_"+guidance)).rglob("receipt.json")):
            receipt = load_json(receipt_path)
            assert receipt["status"] == "ok"
            suffix = str(receipt_path.parent.relative_to(evaluation))
            rows.append(dict(job_id="guidance_controls/"+suffix, family="guidance_controls",
                             source_result=str(one_result(receipt_path.parent)),
                             source_receipt=str(receipt_path), sample_ids=receipt["sample_ids"],
                             expected_mask_sha256=receipt["mask_tensor_sha256"],
                             expected_initial_noise_sha256=receipt["initial_noise_sha256"]))
    if args.extra_sources:
        extra = load_json(args.extra_sources)
        extra = extra["jobs"] if isinstance(extra, dict) else extra
        rows.extend(r for r in extra if r["family"] == "current_temporal")
    return rows


def freeze_job(row, output):
    import hashlib
    import torch
    path = Path(row["source_result"])
    payload = torch.load(path, map_location="cpu", weights_only=False)
    sha = file_hash(path)
    for key in ("expected_result_sha256", "source_result_sha256"):
        if row.get(key):
            assert sha == row[key], path
    cfg = payload["config"]
    if row.get("source_config"):
        assert cfg == row["source_config"], (row["job_id"], "published configuration differs from actual payload")
    assert cfg["pde"] == "nsnonbounded"
    ids = row["sample_ids"]
    assert cfg["batch_size"] == len(ids)
    assert payload["coef_ground_truth"].shape == (len(ids), 1, 128, 128)
    assert payload["sol_ground_truth"].shape == (len(ids), 1, 128, 128)
    metadata = payload["ground_truth_metadata"]
    actual_ids = metadata.get("sample_offsets", metadata.get("sample_ids"))
    assert actual_ids is not None and [int(i) for i in actual_ids] == ids
    masks = payload["masks"]
    old_mask_sha = hashlib.sha256(masks["coef"].contiguous().numpy().tobytes() +
                                 masks["sol"].contiguous().numpy().tobytes()).hexdigest()
    for key in ("expected_mask_sha256", "expected_mask_tensor_sha256", "source_mask_tensor_sha256"):
        if row.get(key):
            assert old_mask_sha == row[key]
    if row.get("source_mask"):
        assert file_hash(row["source_mask"]) == row["source_mask_file_sha256"]
    # Check the source masks against the archived seed/batch semantics before
    # explicitly injecting the saved tensors into the replay.
    from sampling.masks import make_pair_masks
    expected = make_pair_masks(payload["coef_ground_truth"].shape, payload["sol_ground_truth"].shape,
                              cfg["num_obs"], cfg["sensor_mode"], cfg["shared_mask"], cfg["mask_seed"],
                              device="cpu", dtype=payload["coef_ground_truth"].dtype,
                              num_sensor_columns=cfg["num_sensor_columns"])
    assert torch.equal(masks["coef"], expected.coef)
    assert torch.equal(masks["sol"], expected.sol)
    inputs = {k: copy.deepcopy(payload[k]) for k in
              ("coef_ground_truth", "sol_ground_truth", "masks", "pde_params", "ground_truth_metadata")}
    if cfg["residual_mode"] == "near_endpoint_temporal":
        assert "near_endpoint_temporal" in inputs["pde_params"]
    source = copy.deepcopy(row)
    source["actual_result_sha256"] = sha
    if row.get("source_receipt"):
        source["actual_receipt_sha256"] = file_hash(row["source_receipt"])
    packet = dict(sample_ids=ids, config=cfg, inputs=inputs, source=source)
    target = output / "inputs" / (row["job_id"] + ".pt")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        previous = torch.load(target, map_location="cpu", weights_only=False)
        assert tree_hash(previous) == tree_hash(packet), target
    else:
        tmp = target.with_name(".partial_"+uuid.uuid4().hex+".pt")
        torch.save(packet, tmp)
        tmp.replace(target)
    return dict(job_id=row["job_id"], family=row["family"], sample_ids=ids, config=cfg,
                source=source, input_file=str(target.relative_to(output)), input_sha256=file_hash(target),
                input_tensor_sha256=json_hash(tree_hash(inputs)),
                expected_initial_noise_sha256=row.get("expected_initial_noise_sha256"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--paper-archive", type=Path, required=True)
    p.add_argument("--controls-archive", type=Path, required=True)
    p.add_argument("--extra-sources", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--freeze", action="store_true")
    args = p.parse_args()
    args.output = args.output.resolve()
    rows = known_sources(args)
    assert len({r["job_id"] for r in rows}) == len(rows)
    jobs = [freeze_job(r, args.output) for r in rows]
    counts = {f: sum(j["family"] == f for j in jobs) for f in sorted({j["family"] for j in jobs})}
    if args.freeze:
        assert counts == {"main": 35, "ensemble": 96, "guidance_controls": 96,
                          "current_temporal": 3}, counts
    import torch
    model = args.output / "inputs/weights/nsnonbounded.pth"
    assert file_hash(model) == MODEL_SHA256
    normalizer = torch.load(model, map_location="cpu", weights_only=False)["normalizer"]
    result = dict(status="frozen" if args.freeze else "draft", version=1,
                  task="ns44m_ablations_20260915", parameter_count=PARAMETER_COUNT,
                  model_sha256=MODEL_SHA256, model_profile="light",
                  normalizer_sha256=json_hash(tree_hash(normalizer)),
                  expected_jobs=len(jobs), expected_predictions=sum(len(j["sample_ids"]) for j in jobs),
                  family_counts=counts, jobs=jobs,
                  scope="Published fixed-configuration experiments; the separately repeated development/confirmation weight selection is audited independently",
                  scientific_change="44M checkpoint and its saved normalizer; other scientific settings and historical batches retained",
                  timing_scope="Synchronized stock sampler including diagnostics and artifact writes, excluding model and frozen-input loading",
                  excluded_prior_gallery="The published prior_ns_main gallery already uses the same 44M checkpoint")
    target = args.output / ("protocol.json" if args.freeze else "protocol.draft.json")
    if target.exists() and args.freeze:
        assert load_json(target) == result, "Cannot replace an existing frozen protocol"
    else:
        atomic_json(target, result)
    print(json.dumps({k:v for k,v in result.items() if k != "jobs"}, indent=2))


if __name__ == "__main__":
    main()
