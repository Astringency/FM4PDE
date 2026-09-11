"""Finite-chain diagnostic of the exact terminal tilt of one frozen FM ODE.

This is a target-correct pCN reference, NOT a claim of converged exact sampling
or a replacement for the production stepwise guided sampler.
"""
from __future__ import annotations
import argparse
import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from experiments.exact_tilt_reference import make_ode_generator, pcn_endpoint_mean
from plot.run_paper_ablation_revision import digest, write, physical_errors


def diagnostics(trace):
    """Classical split R-hat and positive-sequence, within-chain ESS estimates."""
    x = np.asarray(trace, dtype=np.float64)  # draw, chain, observable
    half = x.shape[0] // 2
    split = np.concatenate((x[:half], x[-half:]), axis=1)
    within = split.var(axis=0, ddof=1).mean(axis=0)
    between = half * split.mean(axis=0).var(axis=0, ddof=1)
    varhat = ((half - 1) * within + between) / half
    rhat = np.sqrt(varhat / np.maximum(within, 1e-300))
    centered = split - split.mean(axis=0, keepdims=True)
    rho = []
    for lag in range(1, half):
        acov = (centered[:-lag] * centered[lag:]).sum(axis=0) / half
        rho.append(1 - (within - acov.mean(axis=0)) / np.maximum(varhat, 1e-300))
    ess = []
    for k in range(x.shape[-1]):
        total, previous = 0., float("inf")
        for lag in range(0, len(rho) - 1, 2):
            pair = float(rho[lag][k] + rho[lag + 1][k])
            if pair < 0:
                break
            previous = min(previous, pair)
            total += previous
        ess.append(min(split.shape[0] * split.shape[1], split.shape[0] * split.shape[1] / max(1., 1 + 2 * total)))
    return dict(classical_split_rhat=rhat.tolist(), approximate_ess=ess,
                adequate_by_pilot_thresholds=bool(np.all(rhat < 1.05) and min(ess) >= 100),
                caveat="Finite monitored observables only; classical split R-hat, no rank normalization. Passing is not proof of stationarity.")


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--pde", choices=["poisson", "helmholtz"], default="poisson")
    p.add_argument("--input-id", type=int, default=0)
    p.add_argument("--chains", type=int, default=4)
    p.add_argument("--warmup", type=int, default=64)
    p.add_argument("--keep", type=int, default=128)
    p.add_argument("--beta", type=float, default=.2)
    p.add_argument("--seed", type=int, default=20260911)
    args = p.parse_args()
    assert args.output.is_absolute() and args.chains >= 4 and args.keep % 32 == 0
    from sampling.config import AblationConfig
    from sampling.model_io import load_fm4pde_checkpoint_bundle
    from sampling.masks import make_pair_masks
    from sampling.state import standardized_to_physical_state, SplitState
    from sampling.losses import compute_guidance_losses
    from sampling.runner import run_single_ablation, _scalar_conditioning_for_sampling, _class_conditioning_for_sampling
    from scripts.tuning.compare_pde_guidance_schedules import combine_truths
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    source = args.inputs / args.pde
    ip = json.loads((source / "protocol.json").read_text())
    assert digest(source / "weights.pth") == ip["weights_sha256"]
    assert digest(source / "truths.pt") == ip["truth_sha256"]
    plan = json.loads(args.plan.read_text())
    target = args.output / "learned_tilt_pilot" / args.pde / f"input{args.input_id}_beta{args.beta:g}"
    target.mkdir(parents=True, exist_ok=True)
    if (target / "protocol.json").exists():
        raise FileExistsError(f"A pilot already exists: {target}; inspect before resuming")
    conf = copy.deepcopy(plan["pdes"][args.pde]["random"]["config"])
    conf.update(checkpoint_path=str(source / "weights.pth"), output_dir=str(target), device="cuda:0",
                batch_size=1, offset=args.input_id, save_plots=False, allow_synthetic_data=False)
    cfg = AblationConfig(**conf)
    cfg.validate()
    assert cfg.noise_level == 0 and cfg.obs_guidance_reduction == "mse" and cfg.pde_guidance_reduction == "mse"
    truths = torch.load(source / "truths.pt", map_location="cpu", weights_only=False)
    gt = combine_truths(truths, [args.input_id], "cuda:0")
    bundle = load_fm4pde_checkpoint_bundle(str(source / "weights.pth"), args.pde, "cuda:0", model_profile="recommended")
    net, normalizer, payload = bundle
    masks = make_pair_masks(gt.coef.shape, gt.sol.shape, cfg.num_obs, cfg.sensor_mode,
                            cfg.shared_mask, cfg.mask_seed, device="cuda:0")
    extras_cache = {}
    def velocity(x, times):
        if len(x) not in extras_cache:
            c = AblationConfig(**dict(conf, batch_size=len(x)))
            g = combine_truths(truths, [args.input_id] * len(x), "cuda:0")
            se, sm = _scalar_conditioning_for_sampling(checkpoint_payload=payload, gt=g, config=c, device="cuda:0")
            ce, cm = _class_conditioning_for_sampling(checkpoint_payload=payload, pde=args.pde,
                batch_size=len(x), device="cuda:0", cfg_scale=cfg.cfg_scale)
            extras_cache[len(x)] = {**(se or {}), **ce}
        return net(x, times, **extras_cache[len(x)])
    G = make_ode_generator(velocity, steps=100, method="euler")
    def physical(r):
        return standardized_to_physical_state(r, args.pde, cfg.img_channels, normalizer)
    def energy(r):
        # Each chain is evaluated separately with the unchanged PDE/loss code.
        # Physical FP64 arithmetic reduces cancellation in spatial derivatives.
        phys = physical(r.double())
        values = []
        for i in range(len(r)):
            losses = compute_guidance_losses(SplitState(phys.coef[i:i+1], phys.sol[i:i+1]), gt, masks, cfg)
            values.append(cfg.zeta_obs_a * losses.guidance_L_obs_a + cfg.zeta_obs_u * losses.guidance_L_obs_u
                          + cfg.zeta_pde * losses.guidance_L_pde)
        return torch.stack(values)
    # Monitor energy and signed low-frequency Fourier components of both fields.
    def monitor(r):
        spectrum = torch.fft.rfft2(r.double(), norm="ortho")
        values = [energy(r)]
        for channel in range(2):
            for row, col in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                values.append(spectrum[:, channel, row, col].real)
                if (row, col) != (0, 0):
                    values.append(spectrum[:, channel, row, col].imag)
        return torch.stack(values, dim=1)
    names = ["terminal_energy"]
    for channel in ["a", "u"]:
        for row, col in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            names.append(f"{channel}_fft_{row}_{col}_real")
            if (row, col) != (0, 0):
                names.append(f"{channel}_fft_{row}_{col}_imag")
    protocol = dict(pde=args.pde, input_ids=[args.input_id], chains=args.chains, warmup=args.warmup,
        keep=args.keep, beta=args.beta, seed=args.seed, generator="frozen checkpoint numerical Euler ODE, 100 steps, no reinjection",
        config=conf, weights_sha256=ip["weights_sha256"], input_protocol_sha256=digest(source / "protocol.json"),
        code_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        target="density proportional to exp(-fixed terminal energy) relative to G#N(0,I)",
        energy="published zeta_a MSE_a + zeta_u MSE_u + zeta_pde componentwise PDE MSE; no time gate or gradient clipping",
        inference_precision="generator FP32, TF32 off; physical potential FP64",
        scope="One-input mixing pilot; direct terminal pCN, not the nested moment-guided flow. No population or exact finite-chain accuracy claim.",
        continuation_rule="Do not label finite pCN outputs exact; inspect all monitored R-hat/ESS and chain stability before any confirmatory cohort.",
        monitored_observables=names, selected_weight=payload["selected_inference_weight"], start_unix=time.time())
    write(target / "protocol.json", protocol)
    rng = torch.Generator(device="cuda:0").manual_seed(args.seed)
    z = torch.randn((args.chains, 2, 128, 128), generator=rng, device="cuda:0")
    with torch.no_grad():
        torch.cuda.reset_peak_memory_stats()
        a, b = G(z[:1]), G(z[:1])
        assert torch.equal(a, b), "Frozen numerical generator is not deterministic"
        assert torch.cuda.max_memory_allocated() < 12 * 2**30
        r0 = G(z)
        assert torch.allclose(energy(r0), torch.cat([energy(r0[i:i+1]) for i in range(args.chains)]), rtol=1e-12, atol=1e-12)
        torch.save(dict(latents=z.cpu(), endpoints=r0.cpu(), physical=normalizer.inverse_transform(r0).cpu(),
                        truth=gt.pair.cpu(), masks=dict(coef=masks.coef.cpu(), sol=masks.sol.cpu())), target / "initial.pt")
        traces, chunk_means, accepts = [], [], []
        start = time.perf_counter()
        for chunk in range(args.keep // 32):
            result = pcn_endpoint_mean(G, energy, z, beta=args.beta,
                warmup=args.warmup if chunk == 0 else 0, keep=32, monitor=monitor, rng=rng)
            z = result.last_latents
            traces.append(result.trace)
            chunk_means.append(result.mean.cpu())
            accepts.append(result.acceptance.cpu())
            fulltrace = torch.cat(traces).numpy()
            status = dict(chunks_complete=chunk + 1, seconds=time.perf_counter() - start,
                acceptance=torch.stack(accepts).mean(0).tolist(), diagnostics=diagnostics(fulltrace),
                energy_last=energy(result.last_endpoints).tolist(), peak_bytes=torch.cuda.max_memory_allocated())
            torch.save(dict(last_latents=z.cpu(), last_endpoints=result.last_endpoints.cpu(),
                rng_state=rng.get_state().cpu(), trace=torch.cat(traces), chunk_means=torch.stack(chunk_means),
                acceptance=torch.stack(accepts)), target / "chain_state.pt")
            write(target / "progress.json", status)
            print("PCN", args.pde, args.input_id, chunk + 1, status, flush=True)
        mean = torch.stack(chunk_means).mean(0, keepdim=True).to("cuda:0")
        estimates = {"finite_chain_mean": mean, "last_draws": result.last_endpoints}
        errors = {}
        for label, estimate in estimates.items():
            pred = physical(estimate)
            errors[label] = {f: physical_errors(p, t.expand(len(p), -1, -1, -1))
                for f, p, t in [("a", pred.coef, gt.coef), ("u", pred.sol, gt.sol)]}
        torch.save(dict(mean_standardized=mean.cpu(), mean_physical=normalizer.inverse_transform(mean).cpu(),
                        last_draws_physical=normalizer.inverse_transform(result.last_endpoints).cpu(), truth=gt.pair.cpu()),
                   target / "estimates.pt")
        write(target / "complete.json", dict(**status, errors=errors, end_unix=time.time()))


if __name__ == "__main__":
    main()
