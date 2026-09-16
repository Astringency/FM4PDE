"""Read and batch saved physical inputs for repeat-sampling studies."""
import copy
import csv
import hashlib
import json
from pathlib import Path

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)

def slice_params(value, index, size):
    import torch
    if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == size:
        return value[index:index + 1].clone()
    if isinstance(value, dict):
        return {k: slice_params(v, index, size) for k, v in value.items()}
    return copy.deepcopy(value)

def prepare_samples(args, pdes, ids):
    """Trace each reused ground truth to its original real test-data artifact."""
    import torch
    from sampling.data import PDEGroundTruth
    from data.specs import get_pde_spec
    source_root = Path(args.source_root)
    index = {}
    with (source_root / "metrics_per_sample_all.csv").open() as handle:
        for row in csv.DictReader(handle):
            if row["pde"] in pdes and row["task"] == "both" and row["sensor_mode"] == "random":
                sample_id = int(row["sample_id"])
                if sample_id in ids:
                    index[(row["pde"], sample_id)] = row
    manifest = []
    for pde in pdes:
        cache = args.root / "cache" / f"{pde}_ground_truth.pt"
        if cache.exists():
            stored = torch.load(cache, map_location="cpu", weights_only=False)
            if stored["sample_ids"] != ids:
                raise ValueError(f"Cache sample mismatch: {cache}")
            manifest.extend(stored["sources"])
            continue
        truths, sources = {}, []
        loaded_path, payload = None, None
        for sample_id in ids:
            row = index[(pde, sample_id)]
            old = Path(row["run_dir"])
            relative = Path(*old.parts[old.parts.index(source_root.name) + 1:])
            path = source_root / relative / "result.pt"
            if path != loaded_path:
                payload = torch.load(path, map_location="cpu", weights_only=False)
                loaded_path = path
            meta = payload["ground_truth_metadata"]
            if meta.get("synthetic") is not False or payload["metrics"].get("synthetic_data") is not False:
                raise ValueError(f"Not verified real data: {path}")
            if "_id.mat" not in Path(meta["data_path"]).name:
                raise ValueError(f"Not the ID test set: {path}")
            local_index = int(row["sample_index"])
            if int(meta["sample_ids"][local_index]) != sample_id:
                raise ValueError(f"Source row mismatch: {path}")
            coef = payload["coef_ground_truth"][local_index:local_index + 1].clone()
            sol = payload["sol_ground_truth"][local_index:local_index + 1].clone()
            count = len(payload["coef_ground_truth"])
            params = slice_params(payload["pde_params"], local_index, count)
            spec = get_pde_spec(pde)
            sample_meta = copy.deepcopy(meta)
            sample_meta.update(offset=sample_id, sample_offsets=[sample_id], sample_ids=[str(sample_id)], batch_size=1)
            sample_meta["comparison_source_artifact"] = str(path)
            truths[sample_id] = PDEGroundTruth(
                pde, coef, sol, coef if pde == "burger" else torch.cat([coef, sol], dim=1), params,
                list(spec.coef_channel_names), list(spec.sol_channel_names), sample_meta,
            )
            digest = hashlib.sha256(coef.numpy().tobytes() + sol.numpy().tobytes()).hexdigest()
            sources.append(dict(pde=pde, sample_id=sample_id, source_artifact=str(path),
                                data_path=meta["data_path"], tensor_sha256=digest))
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(sample_ids=ids, truths=truths, sources=sources), cache)
        manifest.extend(sources)
        print(f"CACHED {pde}: {len(ids)} real ID test samples", flush=True)
    write_json(args.root / "sample_manifest.json", manifest)
