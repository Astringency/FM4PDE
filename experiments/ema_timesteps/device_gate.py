"""Check frozen original predictions before moving sampling to another GPU.

Uses the specified scientific checkout, preserving its inference implementation.
Checks the first difficult and ordinary batch for each frozen sampling seed.
"""
import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path


def run(root, pde, checkout, output):
    sys.path.insert(0, str(checkout.resolve()))
    import torch
    from experiments.aligned_sampling.input_sources import load_cell
    from experiments.aligned_sampling.run_inference import (
        configure_runtime, effective_config, observation_batch,
        relative_differences, runtime_environment,
    )
    from experiments.optimizer_diagnostics.hard_sampling import paired_infer
    from experiments.optimizer_diagnostics.study import sha, write
    from sampling.model_io import load_fm4pde_checkpoint_bundle

    output.mkdir(parents=True, exist_ok=True)
    configure_runtime("cuda:0", False, 4)
    environment = runtime_environment("cuda:0", False)
    selection_path = root / "evaluation_inputs" / pde / "selection.json"
    selection = json.loads(selection_path.read_text())
    original = root / "evaluation" / pde / "variants" / "original"
    identity = json.loads((original / "identity.json").read_text())
    assert environment["gpu"] == identity["environment"]["gpu"]
    assert environment["torch"] == identity["environment"]["torch"]
    assert environment["cuda"] == identity["environment"]["cuda"]
    assert environment["capability"] == identity["environment"]["capability"]
    assert identity["selection_sha256"] == sha(selection_path)
    checkpoint = root / "evaluation_inputs" / pde / "source.pth"
    assert sha(checkpoint) == selection["original_checkpoint_sha256"] == identity["checkpoint_sha256"]
    data = load_cell(root, selection["cell"])
    torch.cuda.reset_peak_memory_stats()
    bundle = load_fm4pde_checkpoint_bundle(str(checkpoint), pde, "cuda:0", prefer_ema=False)
    assert bundle[2]["selected_inference_weight"] == "raw"
    batch = selection["batch_size"]
    starts = [0, len(selection["hard"])]
    assert all(start % batch == 0 for start in starts)
    records = []
    for seed in selection["seeds"]:
        for start in starts:
            ids = selection["indices"][start:start + batch]
            name = f"seed{seed}_batch{start:02d}.pt"
            reference_path = original / name
            receipt = json.loads(reference_path.with_suffix(".json").read_text())
            assert sha(reference_path) == receipt["sha256"]
            reference = torch.load(reference_path, map_location="cpu", weights_only=False)
            cell = copy.deepcopy(selection["cell"])
            cell["config"]["sample_seed"] = seed
            cfg = effective_config(cell, checkpoint, "cuda:0", ids, 1000, output)
            gt, masks, hashes = observation_batch(data, cfg, ids, "cuda:0")
            pred, meta = paired_infer(cfg, bundle, gt, masks, ids)
            assert reference["indices"] == ids and reference["input_hashes"] == hashes
            for key in ["initial_noise_sha256", "bridge_noise_sha256"]:
                assert reference["runtime"][key] == meta[key]
            assert len(meta["bridge_noise_sha256"]) == 100
            configs = [dict(reference["effective_config"]), cfg.asdict()]
            for config in configs:
                for key in ["checkpoint_path", "output_dir"]:
                    config.pop(key, None)
            assert configs[0] == configs[1]
            difference = relative_differences(pred, reference["prediction"])
            torch.save(dict(indices=ids, prediction=pred, input_hashes=hashes,
                            runtime=meta, effective_config=cfg.asdict()), output / name)
            record = dict(seed=seed, start=start, indices=ids, difference=difference,
                          reference_sha256=receipt["sha256"], prediction_sha256=sha(output / name))
            records.append(record)
            print("DEVICE_GATE_BATCH", json.dumps(record), flush=True)
            del pred, gt, masks
    result = dict(
        passed=all(row["difference"]["max_relative"] < 1e-6 for row in records),
        pde=pde, selection_sha256=sha(selection_path), checkpoint_sha256=identity["checkpoint_sha256"],
        environment=environment, reference_environment=identity["environment"],
        scientific_checkout=str(checkout),
        gate_code_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2], text=True).strip(),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(), batches=records,
        scope="First difficult and ordinary frozen batch for each sampling seed; this is not an all-input or all-checkpoint device-equivalence proof.",
    )
    write(output / "result.json", result)
    assert result["passed"], result
    print("DEVICE_GATE_PASSED", json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--pde", required=True)
    parser.add_argument("--scientific-checkout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.pde, args.scientific_checkout, args.output)
