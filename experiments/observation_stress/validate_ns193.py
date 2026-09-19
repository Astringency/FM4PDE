"""Validate RTX 4090 NS predictions against A100 at identical batch sizes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from experiments.rough_stress.run import sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fm-root", required=True)
    args = parser.parse_args()
    root = Path(args.fm_root)
    torch.set_num_threads(2)
    results = []
    suffix = Path("results/nsnonbounded/fm4pde/both/rough2/obs500")
    for size in [4, 16]:
        state = root/"ns_pilot193_queue/queue_state"/f"ns_both_rough2_b{size}.json"
        queue = json.loads(state.read_text())
        assert queue["status"] == "complete" and queue["exit_code"] == 0
        reference = root/("pilots" if size == 4 else "distribution")/suffix
        candidate = root/f"ns_pilots193_b{size}"/suffix
        name = f"batch_0000_{size:04d}"
        records = [json.loads((folder/(name+".json")).read_text()) for folder in [reference, candidate]]
        identities = [json.loads((folder/"identity.json").read_text()) for folder in [reference, candidate]]
        for identity in identities:
            assert identity["batch_size"] == size and identity["tf32"] is False
        for key in ["checkpoint", "fm_protocol_sha256", "input_receipt", "noise_replay"]:
            if key == "checkpoint":
                assert identities[0][key]["sha256"] == identities[1][key]["sha256"]
            elif key == "noise_replay":
                assert identities[0][key]["manifest_sha256"] == identities[1][key]["manifest_sha256"]
            else:
                assert identities[0][key] == identities[1][key], key
        for record, folder in zip(records, [reference, candidate]):
            assert record["prediction_sha256"] == sha256(folder/(name+".pt"))
            assert record["runtime"]["steps"] == record["runtime"]["nfe"] == 100
        assert records[0]["observation_mask_sha256"] == records[1]["observation_mask_sha256"]
        assert records[0]["runtime"]["initial_noise_sha256"] == records[1]["runtime"]["initial_noise_sha256"]
        values = [torch.load(folder/(name+".pt"), map_location="cpu", weights_only=False)
                  for folder in [reference, candidate]]
        assert values[0]["indices"] == values[1]["indices"] == list(range(size))
        assert values[0]["sample_ids"] == values[1]["sample_ids"]
        a, b = [value["prediction"].double() for value in values]
        assert a.shape == b.shape == (size, 2, 128, 128)
        assert torch.isfinite(a).all() and torch.isfinite(b).all()
        differences = (a-b).flatten(2).norm(dim=2)/a.flatten(2).norm(dim=2).clamp_min(1e-12)
        maximum = differences.max(dim=0).values.tolist()
        results.append(dict(batch_size=size, max_per_sample_prediction_relative_l2_by_field=maximum,
            mean_relative_l2_metric_change_pp_by_field={
                field: 100*sum(right[f"rel_l2_{field}"]-left[f"rel_l2_{field}"]
                               for left, right in zip(records[0]["rows"], records[1]["rows"]))/size
                for field in ["a", "u"]},
            reference=str(reference), candidate=str(candidate),
            reference_prediction_sha256=records[0]["prediction_sha256"],
            candidate_prediction_sha256=records[1]["prediction_sha256"],
            candidate_runtime=records[1]["runtime"]))
    passed = all(max(item["max_per_sample_prediction_relative_l2_by_field"]) <= 2e-5 for item in results)
    receipt = dict(status="passed" if passed else "failed", batch_sizes=[4, 16], pde="nsnonbounded", task="both",
        distribution="rough2", tolerance=2e-5,
        criterion="Maximum per-sample physical prediction relative L2 for each field; same batch partition",
        results=results)
    write_json(root/"setup/rtx4090_ns_pilot_equivalence.json", receipt)
    print(json.dumps(receipt, indent=2), flush=True)
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
