"""Same-Gaussian-prior controls on real full-resolution Poisson inputs."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from experiments.gaussian_poisson import PoissonPrior, GaussianTilt, self_test
from plot.run_paper_ablation_revision import digest, write


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--ids", type=int, nargs="+", default=[0])
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--name", default="probe")
    p.add_argument("--variants", nargs="+", default=["posterior", "plugin_stochastic100", "exact_flow100", "exact_flow200"])
    args = p.parse_args()
    assert args.output.is_absolute()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    from models.legacy_checkpoint import read_checkpoint
    from data.transform import PDEStandardizer
    from sampling.masks import make_pair_masks
    dest = args.output / "gaussian_tilt" / args.name
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / "protocol.json").exists():
        raise FileExistsError(dest)
    source = args.inputs / "poisson"
    ip = json.loads((source / "protocol.json").read_text())
    assert digest(source / "weights.pth") == ip["weights_sha256"]
    assert digest(source / "truths.pt") == ip["truth_sha256"]
    cache = torch.load(source / "truths.pt", map_location="cpu", weights_only=False)
    data = np.concatenate([cache[i].pair.numpy() for i in args.ids], axis=0).astype(np.float64)
    payload = read_checkpoint(source / "weights.pth", "poisson")
    normalizer = PDEStandardizer.from_state_dict(payload["normalizer"])
    prior = PoissonPrior(mean=normalizer.mean.numpy(), scale=normalizer.std.numpy())
    cfg = json.loads(args.plan.read_text())["pdes"]["poisson"]["random"]["config"]
    masks = make_pair_masks((1, 1, 128, 128), (1, 1, 128, 128), cfg["num_obs"], "random", False, cfg["mask_seed"])
    mask = np.concatenate((masks.coef.numpy(), masks.sol.numpy()), axis=1)
    guide = GaussianTilt(prior, mask, data, [cfg["zeta_obs_a"], cfg["zeta_obs_u"]])
    reproduced_u = prior.solve(data[:, 0])
    solver_error = np.linalg.norm((reproduced_u - data[:, 1]).reshape(len(data), -1), axis=1) / np.linalg.norm(data[:, 1].reshape(len(data), -1), axis=1)
    assert solver_error.max() < 1e-5, solver_error
    protocol = dict(input_ids=args.ids, seeds=args.seeds, n=128, pde="poisson", variants=args.variants,
        declared_prior="Full DCT GRF(alpha=2,tau=3), zero DC; full five-point Dirichlet Poisson solve; no mode truncation",
        limitation="This prescribed Gaussian prior is not the trained FM prior. Comparisons within this experiment isolate guidance/sampler approximation for the Gaussian prior only.",
        source_weights_sha256=ip["weights_sha256"], source_truth_sha256=ip["truth_sha256"],
        normalizer=normalizer.to_json_dict(), config=cfg, mask_seed=cfg["mask_seed"],
        energy="Same finite observation MSE weights. PDE and boundary energies vanish identically under this exact Poisson prior.",
        precision="Full fields and all solves FP64; true relative CG residual <=1e-9; no hard observations",
        estimator_distinction="posterior_mean is scored separately; posterior_draws are exact Gaussian draws up to CG error; exact_flowN has finite Euler discretization error",
        data_solver_relative_errors=solver_error.tolist(), operator_tests=self_test(),
        code_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        start_unix=time.time())
    write(dest / "protocol.json", protocol)
    np.savez_compressed(dest / "inputs.npz", physical_truth=data, masks=mask)
    def save(name, standard, start, numerical):
        physical = prior.decode(standard)
        assert np.isfinite(physical).all()
        errors = {}
        for f, channel in [("a", 0), ("u", 1)]:
            errors[f] = (np.linalg.norm((physical[:, channel] - data[:, channel]).reshape(len(data), -1), axis=1)
                / np.linalg.norm(data[:, channel].reshape(len(data), -1), axis=1)).tolist()
        np.savez_compressed(dest / (name + ".npz"), physical_prediction=physical)
        receipt = dict(name=name, input_ids=args.ids, errors=errors, seconds=time.perf_counter() - start, numerical=numerical)
        write(dest / (name + ".json"), receipt)
        print("DONE", name, receipt, flush=True)
    if "posterior" in args.variants:
        start = time.perf_counter()
        mean, _, rep = guide.moment(np.zeros_like(data), 0.)
        save("posterior_mean", mean, start, rep)
        for seed in args.seeds:
            start = time.perf_counter()
            draw, rep = guide.endpoint_draw(np.random.default_rng(20260911 + seed))
            save(f"posterior_draw_seed{seed}", draw, start, rep)
    for variant in [v for v in args.variants if v != "posterior"]:
        exact = variant.startswith("exact_flow")
        assert exact or variant == "plugin_stochastic100"
        steps = int(variant[len("exact_flow"):]) if exact else 100
        for seed in args.seeds:
            rng = np.random.default_rng(20260911 + seed)
            x = rng.standard_normal(data.shape)
            initial = None
            reports = []
            start = time.perf_counter()
            for k in range(steps):
                t, tn = k / steps, (k + 1) / steps
                endpoint, initial, rep = guide.moment(x, t, tilted=exact, initial=initial)
                reports.append(rep)
                if exact:
                    x += (endpoint - x) / ((1 - t) * steps)
                else:
                    grad, rep = guide.plugin_gradient(endpoint, t)
                    reports.append(rep)
                    norms = np.linalg.norm(grad.reshape(len(data), -1), axis=1)
                    scales = np.minimum(1., cfg["clip_threshold"] / np.maximum(norms, 1e-30))
                    grad *= scales[:, None, None, None]
                    x = tn * endpoint + (1 - tn) * rng.standard_normal(data.shape)
                    x -= cfg["stochastic_guidance_coeff"] * (1 - t) * grad
                if (k + 1) % 20 == 0:
                    print("STEP", variant, seed, k + 1, "seconds", time.perf_counter() - start, "CG", rep, flush=True)
            save(f"{variant}_seed{seed}", x, start,
                 dict(max_relative_residual=max(r["max_relative_residual"] for r in reports),
                      max_cg_iterations=max(r["iterations"] for r in reports), total_cg_iterations=sum(r["iterations"] for r in reports)))
    write(dest / "complete.json", dict(end_unix=time.time(), variants=args.variants, seeds=args.seeds, ids=args.ids))


if __name__ == "__main__":
    main()
