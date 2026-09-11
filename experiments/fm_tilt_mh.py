"""Metropolis-corrected latent Langevin proposals for a frozen FM generator.

The drift may use a cheaper, deterministic surrogate. Acceptance MUST use the
full generator potential and both proposal densities. No surrogate target is
substituted. Finite chains are not asserted to be stationary.
"""
from __future__ import annotations
import math
import torch


def flat_sum(x):
    return x.double().flatten(1).sum(1)


def broadcast(x, like):
    return x.reshape((-1,) + (1,) * (like.ndim - 1))


def proposal(z, grad, beta, noise):
    b = broadcast(beta, z)
    rho = torch.sqrt(1 - b.square())
    # Avoid cancellation when beta is small.
    drift = b.square() / (1 + rho)
    return rho * z - drift * grad + b * noise


def log_acceptance(z, y, gz, gy, energy_z, energy_y, beta):
    """Stable MH ratio; cancels the exactly prior-reversible pCN part."""
    b = beta.double()
    rho = torch.sqrt(1 - b.square())
    d = b.square() / (1 + rho)
    z, y, gz, gy = [v.double() for v in (z, y, gz, gy)]
    forward = y - broadcast(rho, z) * z
    reverse = z - broadcast(rho, y) * y
    correction = (d / b.square()) * flat_sum(forward * gz - reverse * gy)
    correction += (d.square() / (2 * b.square())) * flat_sum(gz.square() - gy.square())
    return energy_z.double() - energy_y.double() + correction


def transition(z, endpoints, values, grad, beta, *, target, force, rng):
    noise = torch.randn(z.shape, dtype=z.dtype, device=z.device, generator=rng)
    y = proposal(z, grad, beta, noise)
    proposed_endpoints, proposed_values = target(y)
    proposed_grad = force(y)
    loga = log_acceptance(z, y, grad, proposed_grad, values, proposed_values, beta)
    finite = (torch.isfinite(loga) & torch.isfinite(proposed_values)
              & torch.isfinite(proposed_grad).flatten(1).all(1))
    probability = torch.where(finite, torch.exp(loga.clamp(max=0)), torch.zeros_like(loga))
    accepted = torch.rand(len(z), device=z.device, generator=rng) < probability
    choose = lambda new, old: torch.where(broadcast(accepted, new), new, old)
    return (choose(y, z), choose(proposed_endpoints, endpoints),
            torch.where(accepted, proposed_values, values), choose(proposed_grad, grad),
            accepted, probability)


def adapt_beta(beta, probability, iteration, target_accept=.574):
    """Warm-up only. Freeze beta before any retained posterior samples."""
    rate = min(.1, (iteration + 10) ** (-.6))
    return torch.exp(torch.log(beta) + rate * (probability.to(beta) - target_accept)).clamp(1e-5, .95)


def chain_diagnostics(trace):
    """Rank-normalized and folded split R-hat; simple positive-sequence ESS.

    trace: [draw, chain, observable]. A necessary diagnostic, never a proof.
    """
    import numpy as np
    from scipy.stats import rankdata, norm
    from experiments.run_learned_tilt_pilot import diagnostics
    a = np.asarray(trace, dtype=np.float64)
    half = a.shape[0] // 2
    if half < 4 or a.shape[1] < 4 or not np.isfinite(a).all():
        return dict(passed=False, reason="Need >= 8 draws, >= 4 chains and finite monitors")
    if np.any(a.var(axis=(0, 1)) == 0):
        return dict(passed=False, reason="At least one monitored quantity is constant; mixing cannot be assessed")
    def rank_normalize(x):
        flat = x.reshape(-1, x.shape[-1])
        ranks = rankdata(flat, axis=0)
        return norm.ppf((ranks - .375) / (len(flat) + .25)).reshape(x.shape)
    ranked = diagnostics(rank_normalize(a))
    folded = diagnostics(rank_normalize(abs(a - np.median(a, axis=(0, 1)))))
    raw = diagnostics(a)
    rhat = np.maximum(ranked['classical_split_rhat'], folded['classical_split_rhat'])
    ess = np.minimum(ranked['approximate_ess'], raw['approximate_ess'])
    return dict(rank_folded_split_rhat=rhat.tolist(), approximate_ess=ess.tolist(),
                passed=bool(np.all(rhat < 1.01) and np.all(ess >= 100)),
                thresholds=dict(rhat=1.01, ess=100),
                caveat="Finite monitored quantities; ESS uses a positive-sequence estimate. Passing is not proof of stationarity.")
