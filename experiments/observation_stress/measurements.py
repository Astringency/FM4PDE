"""Freeze identical layouts and physical sensor readings for all four models."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from experiments.rough_stress.run import sha256, slice_tree, tensor_hash, write_json
from sampling.masks import make_mask
from sampling.noise import add_observation_noise
from sampling.observation_operators import BoxAverageObservation


CASES = {
    "random": dict(family="layout", layout="random", window=1, noise=0.),
    "grid": dict(family="layout", layout="grid", window=1, noise=0.),
    "columns": dict(family="layout", layout="columns", window=1, noise=0.),
    "hole10": dict(family="layout", layout="hole10", window=1, noise=0.),
    "hole25": dict(family="layout", layout="hole25", window=1, noise=0.),
    "strip25": dict(family="layout", layout="strip25", window=1, noise=0.),
    **{f"noise{int(v*100):02d}": dict(family="noise", layout="random", window=1, noise=v)
       for v in [.01, .05, .1]},
    **{f"average{k}": dict(family="average", layout="average", window=k, noise=0.)
       for k in [1, 4, 8, 16]},
}


def make_layout(sample_ids, layout, reference_masks):
    n, _, h, w = reference_masks.shape
    assert (h, w) == (128, 128)
    excluded = torch.zeros((1, 1, h, w), dtype=torch.bool)
    metadata = dict(layout=layout, mask_seed=1, shared_fields=True)
    if layout == "random":
        return reference_masks.float().clone(), excluded, metadata
    if layout in {"grid", "columns"}:
        mask = make_mask((1, 1, h, w), 500, "grid" if layout == "grid" else "sensor_column",
                         seed=0, num_sensor_columns=5)
        metadata.update(mask_seed=0, same_layout_across_samples=True,
                        grid_implementation="sampling.masks.make_mask",
                        num_sensor_columns=5 if layout == "columns" else None)
        return mask.repeat(n, 1, 1, 1), excluded, metadata
    yy, xx = torch.meshgrid(torch.linspace(0, 1, h), torch.linspace(0, 1, w), indexing="ij")
    if layout.startswith("hole"):
        area = .1 if layout == "hole10" else .25
        half_side = area ** .5 / 2
        excluded[0, 0] = (xx.sub(.5).abs() < half_side) & (yy.sub(.5).abs() < half_side)
        metadata.update(continuous_excluded_area=area, region_center=[.5, .5],
                        square_side=2*half_side)
    elif layout == "strip25":
        excluded[0, 0] = xx.sub(.5).abs() < .125
        metadata.update(continuous_excluded_area=.25, strip_axis="second_spatial_axis",
                        strip_interval=[.375, .625])
    elif layout == "average":
        # All window sizes, including point control, use the same anchors.
        excluded.fill_(True)
        excluded[..., 8:121, 8:121] = False
        metadata.update(max_window=16, anchor_range_inclusive=[8, 120])
    else:
        raise ValueError(layout)
    mask = torch.zeros_like(reference_masks, dtype=torch.float32)
    eligible = ~excluded.reshape(-1)
    for i, sample_id in enumerate(sample_ids):
        seed = int.from_bytes(hashlib.sha256(f"mask|1|test|{sample_id}|0".encode()).digest()[:8], "big") % (2**63-1)
        order = torch.randperm(h*w, generator=torch.Generator().manual_seed(seed))
        chosen = order[eligible[order]][:500]
        mask[i, 0].view(-1)[chosen] = 1
    metadata["excluded_grid_points"] = int(excluded.sum())
    metadata["excluded_grid_fraction"] = float(excluded.float().mean())
    return mask, excluded, metadata


def build_measurements(full, ids, reference_masks, case):
    spec = CASES[case]
    mask, excluded, meta = make_layout(ids, spec["layout"], reference_masks)
    operator = BoxAverageObservation(spec["window"])
    operator.validate_mask(mask)
    clean = operator(full) * mask
    noisy = clean.clone()
    noise_meta = []
    for i in range(len(full)):
        # Same Gaussian draws at every noise level, separately for f and u.
        for c in range(2):
            seed = i * 2 + c
            noise = add_observation_noise(clean[i:i+1, c:c+1], mask[i:i+1],
                                          spec["noise"], relative=True, seed=seed)
            noisy[i:i+1, c:c+1] = noise.noisy
            noise_meta.append(dict(index=i, channel=c, **noise.metadata))
    counts = mask.flatten(1).sum(1)
    assert bool((counts == (640 if case == "columns" else 500)).all())
    assert not bool((mask.bool() & excluded).any())
    assert torch.isfinite(noisy).all()
    return dict(mask=mask, clean=clean, noisy=noisy, excluded=excluded,
                metadata=dict(**spec, **{k:v for k,v in meta.items() if k not in spec},
                    operator=operator.metadata(), readings_per_field=int(counts[0]),
                    noise_definition="noise_level * per_sample_per_field_observed_population_std * N(0,1)",
                    noise_realization=noise_meta))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", required=True)
    p.add_argument("--output-root", required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    source = Path(args.source_root)
    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    checkpoints = source / "poisson_checkpoints.json"
    write_json(output / checkpoints.name, json.loads(checkpoints.read_text()))
    for dist in ["id", "rough"]:
        path = source / "inputs" / f"poisson_{dist}.pt"
        receipt = json.loads(path.with_suffix(".json").read_text())
        if sha256(path) != receipt["pack_sha256"]:
            raise ValueError("Frozen Poisson source checksum mismatch")
        pack = torch.load(path, map_location="cpu", weights_only=False)
        raw = slice_tree(pack["raw"], 0, 100, pack["count"])
        ids = pack["sample_ids"][:100]
        for case in CASES:
            destination = output / f"poisson_{dist}_{case}.pt"
            if destination.exists():
                existing = json.loads(destination.with_suffix(".json").read_text())
                if sha256(destination) != existing["sha256"]:
                    raise ValueError(f"Corrupt existing measurement pack: {destination}")
                continue
            observations = build_measurements(raw["full_tensor"], ids, pack["masks"][500][:100], case)
            torch.save(dict(raw=raw, sample_ids=ids, source_indices=list(range(100)),
                            distribution=dist, case=case, count=100, **observations), destination)
            write_json(destination.with_suffix(".json"), dict(sha256=sha256(destination),
                source_pack_sha256=receipt["pack_sha256"], metadata=observations["metadata"],
                mask_sha256=tensor_hash(observations["mask"]),
                clean_sha256=tensor_hash(observations["clean"]), noisy_sha256=tensor_hash(observations["noisy"])))
            print("PREPARED", dist, case, flush=True)


if __name__ == "__main__":
    main()
