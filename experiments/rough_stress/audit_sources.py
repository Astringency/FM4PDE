"""Check Poisson packs directly against MAT arrays and the discrete equation."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat
import torch

from experiments.rough_stress.run import sha256, source_name, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-root", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    results = []
    for distribution in ("id", "smooth", "rough", "rough2", "rough3"):
        path = Path(args.data_root)/source_name("poisson", distribution)
        pack_path = Path(args.input_root)/"inputs"/f"poisson_{distribution}.pt"
        receipt = json.loads(pack_path.with_suffix(".json").read_text())
        assert path.name == Path(receipt["source"]).name
        assert sha256(path) == receipt["source_sha256"], "Original source changed"
        assert sha256(pack_path) == receipt["pack_sha256"], "Pack changed"
        pack = torch.load(pack_path, map_location="cpu", weights_only=False)
        raw = loadmat(path, variable_names=["f_data", "phi_data", "dataset_type", "grf_alpha", "grf_tau", "generation_seed"])
        fields = pack["raw"]["full_tensor"].numpy()
        equation, boundary = [], []
        for lo in range(0, 1000, 32):
            hi = min(lo+32, 1000)
            a, u = raw["f_data"][lo:hi], raw["phi_data"][lo:hi]
            assert np.array_equal(fields[lo:hi, 0], a.astype(np.float32))
            assert np.array_equal(fields[lo:hi, 1], u.astype(np.float32))
            laplacian = (u[:, 2:, 1:-1]+u[:, :-2, 1:-1]+u[:, 1:-1, 2:]+u[:, 1:-1, :-2]-4*u[:, 1:-1, 1:-1])*127**2
            norm = np.sqrt(np.sum(a[:, 1:-1, 1:-1]**2, axis=(1, 2)))
            residual = np.sqrt(np.sum((laplacian-a[:, 1:-1, 1:-1])**2, axis=(1, 2)))/np.maximum(norm, 1e-12)
            equation.extend(residual.tolist())
            boundary.append(float(max(np.abs(u[:, [0, -1], :]).max(), np.abs(u[:, :, [0, -1]]).max())))
        assert max(equation) < 1e-8, "Generated fields do not solve the discrete Poisson equation"
        assert max(boundary) < 1e-8, "Nonzero Dirichlet boundary"
        assert pack["sample_ids"] == [f"{path.name}:{i}" for i in range(1000)], "Sample ID/source mismatch"
        metadata = {key:np.asarray(raw[key]).tolist() for key in ("dataset_type", "grf_alpha", "grf_tau", "generation_seed") if key in raw}
        results.append(dict(distribution=distribution, source=str(path), source_sha256=receipt["source_sha256"],
            first_1000_float32_fields_exact=True, discrete_relative_residual_max=max(equation),
            discrete_relative_residual_mean=float(np.mean(equation)), boundary_max=max(boundary), metadata=metadata))
        print("SOURCE VERIFIED", distribution, "max relative PDE residual", max(equation), flush=True)
        del raw, pack, fields
    write_json(args.output, dict(status="complete", count_per_distribution=1000, distributions=results))


if __name__ == "__main__":
    main()
