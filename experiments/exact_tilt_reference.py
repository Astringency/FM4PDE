"""Reference algorithms for endpoint-tilted flow matching.

This is NOT a claim that finite MCMC or a finite ODE grid is exact.

1. pcn_endpoint_mean: invariant-target-correct latent pCN; asymptotically
   estimates E[R | t R + (1-t) E = x, energy tilt], R = G(Z), Z,E iid N(0,I).
2. moment_guided_sample: an outer independent-bridge flow whose full velocity
   is estimated using (1), with a final conditional endpoint draw.
3. linear_gaussian_endpoint: matrix-free, conjugate-gradient computation of
   the EXACT conditional endpoint mean/velocity for G(Z) = mu + A Z and a
   linear-Gaussian observation energy (up to linear-solve/roundoff error).

G is a fixed DETERMINISTIC generator. Its law, not an unspecified true data
law, defines the prior. An FM ODE generator can be constructed below. The
inner artificial bridge noise is independent of the generator's latent Z.
Do not add moment_guided_sample's full velocity to the original network.

Run: python fm4pde_exact_guidance_algorithms.py --self-test
Dependency: torch. All tests are small CPU examples, NOT PDE checkpoint tests.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from typing import Callable, Optional

import torch
from torch import Tensor

Map = Callable[[Tensor], Tensor]
Velocity = Callable[[Tensor, Tensor], Tensor]


@dataclass
class PCNResult:
    mean: Tensor                         # full endpoint conditional mean estimate
    last_latents: Tensor                 # [chains, *latent_shape]
    last_endpoints: Tensor                # [chains, *endpoint_shape]
    acceptance: Tensor                   # retained-phase acceptance, per chain
    trace: Tensor                        # [keep, chains, monitor_dim]
    samples: Optional[Tensor]            # [keep, chains, *endpoint_shape]


def _randn_like(x: Tensor, rng: Optional[torch.Generator]) -> Tensor:
    return torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=rng)


@torch.no_grad()
def make_ode_generator(velocity: Velocity, *, steps: int = 100,
                       method: str = "heun") -> Map:
    """Wrap an UNGUIDED velocity into a deterministic numerical endpoint map.

    velocity(x, t_vector): batch-first x; t_vector has shape [batch].
    Caller must put the checkpoint in eval mode; disable dropout and all
    reinjection noise. Close over known physical parameters in `velocity`.
    Fixed solver/settings are part of the numerical prior definition.
    """
    if steps < 1 or method not in {"euler", "heun"}:
        raise ValueError("Use steps >= 1 and method='euler' or 'heun'.")

    @torch.no_grad()
    def G(z: Tensor) -> Tensor:
        if z.ndim < 2:
            raise ValueError("Expected [batch, *latent_shape].")
        x = z.clone()
        h = 1.0 / steps
        for k in range(steps):
            t = torch.full((x.shape[0],), k / steps,
                           dtype=x.dtype, device=x.device)
            f = velocity(x, t)
            if f.shape != x.shape:
                raise ValueError("velocity output shape must match x.")
            xp = x + h * f
            if method == "heun":
                tp = torch.full_like(t, (k + 1) / steps)
                x = x + 0.5 * h * (f + velocity(xp, tp))
            else:
                x = xp
        if not torch.isfinite(x).all():
            raise FloatingPointError("Unguided generator produced nonfinite fields.")
        return x
    return G


@torch.no_grad()
def pcn_endpoint_mean(
    G: Map,
    energy: Map,
    z_init: Tensor,
    *,
    x: Optional[Tensor] = None,
    t: Optional[float] = None,
    beta: float = 0.2,
    warmup: int = 200,
    keep: int = 500,
    store_samples: bool = False,
    monitor: Optional[Map] = None,
    rng: Optional[torch.Generator] = None,
) -> PCNResult:
    """Compute conditional endpoint moments by latent pCN-MH.

    Target density relative to standard Gaussian latent measure:
       exp(-energy(G(z)) - ||x - t G(z)||^2 / (2 (1-t)^2)).
    With x=t=None, omit the bridge and target the FINAL tilted posterior.

    G(z): [chains, *latent_shape] -> [chains, *endpoint_shape].
    energy(r): [chains, *endpoint_shape] -> [chains]; NO batch reduction.
    x: ONE outer flow state, with shape [*endpoint_shape], standardized.
    energy may decode r to physical units internally. The bridge MUST stay
    standardized and uses a SUM, not a mean, over endpoint coordinates.

    beta is fixed throughout each call. Rejected states MUST be retained;
    they are included in this implementation's mean and samples.
    Acceptance/trace do not certify mixing. Increase warmup/keep and check
    multi-chain, observable-specific ESS/R-hat externally.
    """
    if z_init.ndim < 2 or not z_init.is_floating_point():
        raise ValueError("z_init must be floating [chains, *latent_shape].")
    if not (0.0 < beta <= 1.0) or warmup < 0 or keep < 1:
        raise ValueError("Require 0 < beta <= 1, warmup >= 0, keep >= 1.")
    if (x is None) != (t is None):
        raise ValueError("Provide BOTH x and t, or neither.")
    if t is not None and not (0.0 <= float(t) < 1.0):
        raise ValueError("Bridge requires 0 <= t < 1.")
    n = z_init.shape[0]
    if n < 1:
        raise ValueError("At least one chain is required.")
    z = z_init.clone()

    def evaluate(latents: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        r = G(latents)
        if r.ndim < 2 or r.shape[0] != n or r.device != z.device:
            raise ValueError("G must preserve chain batch/device and return fields.")
        if not torch.isfinite(r).all():
            raise FloatingPointError("Nonfinite generator output.")
        e = energy(r)
        if e.shape != (n,):
            raise ValueError("energy must return one scalar per chain, shape [chains].")
        e = e.to(device=z.device, dtype=torch.float64)
        V = e.clone()
        if x is not None:
            if x.shape != r.shape[1:] or x.device != r.device:
                raise ValueError("x shape/device must match one generated endpoint.")
            diff = float(t) * r.double() - x.double().unsqueeze(0)
            V = V + diff.flatten(1).square().sum(1) / (2.0 * (1.0-float(t))**2)
        if not torch.isfinite(V).all():
            raise FloatingPointError("Nonfinite potential; inspect scales/energy.")
        return r, V, e

    r, V, e = evaluate(z)
    total = torch.zeros_like(r[0], dtype=torch.float64)
    accepted = torch.zeros(n, dtype=torch.float64, device=z.device)
    traces, saved = [], []
    rho = math.sqrt(1.0 - beta * beta)
    for j in range(warmup + keep):
        zp = rho * z + beta * _randn_like(z, rng)
        rp, Vp, ep = evaluate(zp)
        u = torch.rand((n,), device=z.device, dtype=torch.float64, generator=rng)
        logu = u.clamp_min(torch.finfo(torch.float64).tiny).log()
        accept = logu < torch.minimum(V - Vp, torch.zeros_like(V))
        z = torch.where(accept.reshape(n, *([1] * (z.ndim-1))), zp, z)
        r = torch.where(accept.reshape(n, *([1] * (r.ndim-1))), rp, r)
        V, e = torch.where(accept, Vp, V), torch.where(accept, ep, e)
        if j >= warmup:
            accepted += accept.double()
            total += r.double().sum(dim=0)
            obs = monitor(r) if monitor is not None else torch.stack((e, V), dim=1)
            if obs.ndim != 2 or obs.shape[0] != n:
                raise ValueError("monitor must return [chains, monitor_dim].")
            traces.append(obs.detach().double().cpu())
            if store_samples:
                saved.append(r.clone())
    return PCNResult(
        mean=(total / (n * keep)).to(r.dtype),
        last_latents=z,
        last_endpoints=r,
        acceptance=accepted / keep,
        trace=torch.stack(traces),
        samples=torch.stack(saved) if store_samples else None,
    )


@dataclass
class GuidedResult:
    sample: Tensor
    last_flow_state: Tensor
    last_local_mean: Tensor               # NOT the overall posterior mean
    diagnostics: list[dict]


@torch.no_grad()
def moment_guided_sample(
    G: Map, energy: Map, z_init: Tensor, x_init: Tensor, *,
    outer_steps: int = 32, stop_t: float = 0.95,
    beta: float = 0.2, first_warmup: int = 500,
    warmup: int = 100, keep: int = 200,
    rng: Optional[torch.Generator] = None,
) -> GuidedResult:
    """EXPENSIVE reference guided sampler for ONE output field.

    x_init must be an independent N(0,I) draw in endpoint coordinates.
    Each inner solve estimates the FULL tilted velocity (mean - x)/(1-t).
    Do not add the original velocity again. This defines the independent
    affine bridge of G#N(0,I), which need not equal a finite trained network's
    original intermediate marginals.

    Euler integrates only to stop_t < 1. The last operation samples an endpoint
    conditionally instead of replacing it by its conditional mean. With exact
    inner sampling and correct stop_t marginal this terminal operation is exact;
    finite MCMC and the outer Euler grid still cause approximation errors.
    """
    if outer_steps < 1 or not (0.0 < stop_t < 1.0):
        raise ValueError("Require outer_steps >= 1 and 0 < stop_t < 1.")
    x, z = x_init.clone(), z_init.clone()
    diagnostics = []
    dt = stop_t / outer_steps
    for k in range(outer_steps):
        t = k * dt
        result = pcn_endpoint_mean(
            G, energy, z, x=x, t=t, beta=beta,
            warmup=first_warmup if k == 0 else warmup,
            keep=keep, rng=rng,
        )
        x = x + dt * (result.mean - x) / (1.0 - t)
        if not torch.isfinite(x).all():
            raise FloatingPointError("Nonfinite outer state; inspect inner mixing/grid.")
        z = result.last_latents
        diagnostics.append({"t": t, "acceptance": result.acceptance.cpu().tolist()})
    terminal = pcn_endpoint_mean(
        G, energy, z, x=x, t=stop_t, beta=beta,
        warmup=warmup, keep=keep, rng=rng,
    )
    idx = int(torch.randint(z.shape[0], (), device=z.device, generator=rng).item())
    diagnostics.append({"t": stop_t, "acceptance": terminal.acceptance.cpu().tolist()})
    return GuidedResult(terminal.last_endpoints[idx].clone(), x,
                        terminal.mean, diagnostics)


@dataclass
class CGResult:
    solution: Tensor
    iterations: int
    residual_norm: float


@torch.no_grad()
def conjugate_gradient(
    matvec: Map, rhs: Tensor, *, initial: Optional[Tensor] = None,
    precondition: Optional[Map] = None,
    rtol: float = 1e-10, atol: float = 0.0, maxiter: int = 2000,
) -> CGResult:
    """Real, matrix-free SPD conjugate gradients; raises on nonconvergence.

    A practical preconditioner must also be symmetric positive definite.
    Use float64 for the Gaussian reference. `matvec` and adjoints must be
    deterministic and consistent with the same discretization/inner product.
    """
    if rhs.dtype not in (torch.float32, torch.float64):
        raise ValueError("CG requires a real float32/float64 right-hand side.")
    if rtol < 0 or atol < 0 or rtol + atol <= 0 or maxiter < 1:
        raise ValueError("Invalid CG tolerance/budget.")
    dot = lambda a, b: torch.dot(a.reshape(-1), b.reshape(-1))
    x = torch.zeros_like(rhs) if initial is None else initial.clone()
    r = rhs - matvec(x)
    threshold = max(atol, rtol * float(torch.linalg.vector_norm(rhs)))
    rn = float(torch.linalg.vector_norm(r))
    if rn <= threshold:
        return CGResult(x, 0, rn)
    M = precondition if precondition is not None else lambda v: v
    z = M(r)
    rz = dot(r, z)
    if not torch.isfinite(rz) or float(rz) <= 0:
        raise RuntimeError("Invalid preconditioner/residual in CG.")
    p = z.clone()
    for it in range(1, maxiter + 1):
        Ap = matvec(p)
        pAp = dot(p, Ap)
        if not torch.isfinite(pAp) or float(pAp) <= 0:
            raise RuntimeError("CG lost SPD curvature; check A/AT, B/BT and Rinv.")
        alpha = rz / pAp
        x, r = x + alpha * p, r - alpha * Ap
        rn = float(torch.linalg.vector_norm(r))
        if rn <= threshold:
            true_r = rhs - matvec(x)
            true_rn = float(torch.linalg.vector_norm(true_r))
            if true_rn <= threshold:
                return CGResult(x, it, true_rn)
            r = true_r  # restart if recursive residual drifted
            z = M(r)
            rz, p = dot(r, z), z.clone()
            continue
        z = M(r)
        rz_new = dot(r, z)
        if not torch.isfinite(rz_new) or float(rz_new) <= 0:
            raise RuntimeError("CG preconditioner must be positive definite.")
        p = z + (rz_new / rz) * p
        rz = rz_new
    raise RuntimeError(f"CG did not converge in {maxiter} steps; residual={rn:.3e}, "
                       f"threshold={threshold:.3e}.")


@dataclass
class GaussianGuideResult:
    mean: Tensor
    velocity: Tensor                     # FULL tilted velocity, not a correction
    latent_mean: Tensor
    iterations: int
    residual_norm: float


@torch.no_grad()
def linear_gaussian_endpoint(
    x: Tensor, t: float, mu: Tensor,
    A: Map, AT: Map, B: Map, BT: Map, Rinv: Map, observation: Tensor, *,
    latent_initial: Optional[Tensor] = None,
    precondition: Optional[Map] = None,
    rtol: float = 1e-10, atol: float = 0.0, maxiter: int = 2000,
) -> GaussianGuideResult:
    """Exact-model linear Gaussian endpoint mean, computed using matrix-free CG.

    Prior: endpoint R = mu + A Z, Z ~ N(0,I).
    Energy: 0.5 (B R - observation)^T Rcov^{-1}(B R - observation).
    A,AT and B,BT must be linear maps and exact discrete adjoint pairs.
    Rinv applies the SPD observation precision, not a matrix inverse call.
    All quantities must be in consistent standardized endpoint coordinates.
    Linear physical residuals can be stacked with observation residuals.
    A must use the full prescribed prior; rank truncation changes that prior.

    Matrix solved (scaled by sigma^2 to avoid division by small sigma):
      [sigma^2 I + t^2 A^T A + sigma^2 A^T B^T Rinv B A] zbar
          = t A^T(x-t mu) + sigma^2 A^T B^T Rinv(y-B mu).
    """
    if not (0.0 <= t < 1.0) or x.shape != mu.shape:
        raise ValueError("Require 0 <= t < 1 and x.shape == mu.shape.")
    s2 = (1.0 - t)**2
    obs_rhs = AT(BT(Rinv(observation - B(mu))))
    rhs = t * AT(x - t * mu) + s2 * obs_rhs

    def matvec(z: Tensor) -> Tensor:
        az = A(z)
        return s2 * z + t*t * AT(az) + s2 * AT(BT(Rinv(B(az))))

    result = conjugate_gradient(matvec, rhs, initial=latent_initial,
                                precondition=precondition, rtol=rtol,
                                atol=atol, maxiter=maxiter)
    mean = mu + A(result.solution)
    return GaussianGuideResult(mean, (mean-x)/(1.0-t), result.solution,
                               result.iterations, result.residual_norm)


def self_test() -> dict:
    """Deterministic small tests and Monte Carlo checks; no trained PDE model."""
    torch.set_num_threads(1)
    dtype = torch.float64
    rng = torch.Generator(device="cpu").manual_seed(20260911)
    rand = lambda *shape: torch.randn(shape, dtype=dtype, generator=rng)
    G = lambda z: z
    energy = lambda r: 0.5 * (r[:, 0]-1.0).square()
    scalar = pcn_endpoint_mean(G, energy, rand(16, 1), beta=0.6,
                              warmup=1000, keep=6000,
                              store_samples=True, rng=rng)
    draws = scalar.samples.reshape(-1)
    scalar_mean, scalar_var = float(draws.mean()), float(draws.var())
    assert abs(scalar_mean-0.5) < 0.025
    assert abs(scalar_var-0.5) < 0.04

    x, t = torch.tensor([0.2], dtype=dtype), 0.6
    conditional = pcn_endpoint_mean(G, energy, rand(16, 1), x=x, t=t,
                                   beta=0.5, warmup=1000, keep=6000,
                                   store_samples=True, rng=rng)
    q = 2.0 + t*t/(1.0-t)**2
    m = (1.0+t*float(x[0])/(1.0-t)**2)/q
    c = 1.0/q
    cd = conditional.samples.reshape(-1)
    assert abs(float(cd.mean())-m) < 0.025
    assert abs(float(cd.var())-c) < 0.03

    # Non-Gaussian endpoint law: (z,z^2), compared with 1D numerical quadrature.
    Gnon = lambda z: torch.cat((z, z.square()), dim=1)
    Enon = lambda r: 0.5*((r[:,0]-0.8)/0.7)**2 + 0.5*((r[:,1]-1.4)/0.8)**2
    xn, tn = torch.tensor([0.1, 0.2], dtype=dtype), 0.4
    nonlinear = pcn_endpoint_mean(Gnon, Enon, rand(16,1), x=xn, t=tn,
                                 beta=0.65, warmup=1500, keep=8000, rng=rng)
    zz = torch.linspace(-8.0, 8.0, 40001, dtype=dtype).unsqueeze(1)
    rr = Gnon(zz)
    logw = -0.5*zz[:,0]**2-Enon(rr)-(xn-tn*rr).square().sum(1)/(2*(1-tn)**2)
    w = torch.softmax(logw, dim=0)
    exact_non = (w[:,None]*rr).sum(0)
    nonlinear_error = float((nonlinear.mean-exact_non).abs().max())
    assert nonlinear_error < 0.035

    # Matrix-free solver vs dense independent Gaussian calculation.
    Am, Bm = rand(6,4), rand(3,6)
    mu, xx, y = rand(6), rand(6), rand(3)
    variance = torch.tensor([0.5,0.8,1.2], dtype=dtype)
    A, AT = lambda z: Am@z, lambda r: Am.T@r
    B, BT = lambda r: Bm@r, lambda v: Bm.T@v
    Rinv = lambda v: v/variance
    max_cg_error = 0.0
    K = Bm@Am
    for tt in [0.0,0.03,0.6,0.97]:
        out = linear_gaussian_endpoint(xx,tt,mu,A,AT,B,BT,Rinv,y,
                                       rtol=1e-12, atol=1e-13,maxiter=100)
        sig2 = (1.0-tt)**2
        precision = torch.eye(4,dtype=dtype)+tt*tt/sig2*(Am.T@Am)+K.T@(K/variance[:,None])
        b = tt/sig2*Am.T@(xx-tt*mu)+K.T@((y-Bm@mu)/variance)
        truth = mu+Am@torch.linalg.solve(precision,b)
        max_cg_error = max(max_cg_error,float((out.mean-truth).abs().max()))
    assert max_cg_error < 1e-9

    # Adaptor and outer loop smoke test; NOT an outer-flow accuracy certificate.
    zero_v = lambda a, times: torch.zeros_like(a)
    adaptor = make_ode_generator(zero_v,steps=4,method="heun")
    test_z = rand(2,3)
    assert torch.equal(adaptor(test_z),test_z)
    smoke = moment_guided_sample(G,energy,rand(8,1),rand(1),outer_steps=4,
                                stop_t=0.7,beta=0.5,first_warmup=100,
                                warmup=50,keep=100,rng=rng)
    assert torch.isfinite(smoke.sample).all()
    return {
        "torch_version": torch.__version__, "device": "cpu", "seed": 20260911,
        "scope": "Small scalar/linear/nonlinear tests only; no FM4PDE checkpoint or PDE benchmark.",
        "scalar_posterior": {"target_mean":0.5,"target_variance":0.5,
            "pcn_mean":scalar_mean,"pcn_variance":scalar_var,
            "mean_acceptance":float(scalar.acceptance.mean())},
        "scalar_bridge": {"t":t,"x":0.2,"target_mean":m,"target_variance":c,
            "pcn_mean":float(cd.mean()),"pcn_variance":float(cd.var())},
        "nonlinear_bridge": {"quadrature_mean":exact_non.tolist(),
            "pcn_mean":nonlinear.mean.tolist(),"max_abs_error":nonlinear_error},
        "linear_gaussian_cg_max_abs_error_vs_dense":max_cg_error,
        "outer_sampler_smoke_test":"passed (no accuracy claim)",
        "all_assertions_passed": True,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="Run small CPU tests.")
    parser.add_argument("--report", type=str, help="Optional path for JSON test report.")
    args = parser.parse_args()
    if not args.self_test:
        parser.error("Use --self-test, or import the module functions.")
    report = self_test()
    text = json.dumps(report,indent=2,ensure_ascii=False)
    print(text)
    if args.report:
        with open(args.report,"w",encoding="utf-8") as f:
            f.write(text+"\n")
