"""Fixed Gaussian proposal geometry; the full learned FM target is unchanged.

Only a proposal uses the finite-difference linearization. All latent coordinates
remain stochastic, including the orthogonal complement of the fitted subspace.
"""
from __future__ import annotations
import math
import torch


def full_trajectory_vjp(adapter, endpoint_seed):
    """VJP of the complete cached discrete G, for an arbitrary endpoint seed."""
    adjoint = endpoint_seed
    for i in reversed(range(adapter.full_steps)):
        with torch.enable_grad():
            x = adapter.cached_states[i].detach().requires_grad_(True)
            t = torch.full((len(x),), i/adapter.full_steps, device=x.device, dtype=x.dtype)
            velocity = adapter.net(x,t,**adapter.extras)
            vjp, = torch.autograd.grad(velocity,x,grad_outputs=adjoint/adapter.full_steps)
        adjoint = (adjoint+vjp).detach()
    return adjoint


def cosine_basis(n=128, width=8, channels=2, *, device='cpu', dtype=torch.float32):
    j = torch.arange(n, device=device, dtype=torch.float64)
    k = torch.arange(width, device=device, dtype=torch.float64)
    c = torch.cos(math.pi * k[:, None] * (j[None, :] + .5) / n) * math.sqrt(2 / n)
    c[0] /= math.sqrt(2)
    spatial = torch.einsum('ai,bj->abij', c, c).reshape(width * width, n, n)
    basis = torch.zeros(channels * width * width, channels, n, n, device=device, dtype=torch.float64)
    for channel in range(channels):
        basis[channel * width * width:(channel + 1) * width * width, channel] = spatial
    return basis.flatten(1).to(dtype)


class GaussianReference:
    def __init__(self, basis, root, mean):
        self.basis, self.root, self.mean = basis, root, mean
        self.delta = root - torch.eye(len(root), device=root.device, dtype=root.dtype)

    def linear(self, value):
        flat = value.flatten(1)
        return (flat + ((flat @ self.basis.T) @ self.delta) @ self.basis).reshape_as(value)

    def decode(self, value):
        return self.mean + self.linear(value)

    def residual_energy(self, white, latent, energy):
        return energy.double() + .5 * (latent.double().square() - white.double().square()).flatten(1).sum(1)

    def residual_force(self, white, latent, gradient):
        return self.linear(latent + gradient) - white


def fit_reference(basis, jacobian, center, endpoint, adapter):
    """Fit the exact quadratic Poisson endpoint energy to a surrogate G Jacobian.

    The Gaussian is a proposal, never a replacement likelihood or FM prior.
    Compute the positive semidefinite Gauss--Newton matrix without a dense
    full-field Hessian. The adapter has one observation case repeated four times.
    """
    count = len(adapter.singles)
    gradients = []
    with torch.enable_grad():
        zero = torch.zeros_like(endpoint).repeat(count, 1, 1, 1).double().requires_grad_(True)
        gzero, = torch.autograd.grad(adapter.potential(zero).sum(), zero)
        r = endpoint.repeat(count, 1, 1, 1).double().requires_grad_(True)
        gbase, = torch.autograd.grad(adapter.potential(r).sum(), r)
    for start in range(0, len(jacobian), count):
        block = jacobian[start:start + count].to(endpoint).double()
        size = len(block)
        if size < count:
            block = torch.cat([block, torch.zeros_like(block[:1]).repeat(count - size, 1, 1, 1)])
        with torch.enable_grad():
            block.requires_grad_(True)
            g, = torch.autograd.grad(adapter.potential(block).sum(), block)
        gradients.append((g - gzero)[:size].detach().flatten(1).cpu())
    j = jacobian.double().flatten(1).cpu()
    h = j @ torch.cat(gradients).T
    asymmetry = (h - h.T).norm() / h.norm()
    assert asymmetry < 1e-8, float(asymmetry)
    h = .5 * (h + h.T)
    eigenvalues, vectors = torch.linalg.eigh(h)
    assert float(eigenvalues.min()) > -1e-6
    eigenvalues = eigenvalues.clamp(min=0)
    precision = torch.eye(len(h), dtype=h.dtype) + h
    c = center.double().flatten(1).cpu() @ basis.double().cpu().T
    g = j @ gbase[0].detach().flatten().cpu()
    m = c[0] - torch.linalg.solve(precision, c[0] + g)
    root = (vectors * (1 + eigenvalues).rsqrt()) @ vectors.T
    device, dtype = endpoint.device, endpoint.dtype
    b = basis.to(device=device, dtype=dtype)
    mean = (m.to(b) @ b).reshape_as(endpoint)
    return GaussianReference(b, root.to(b), mean), dict(
        eigenvalues=eigenvalues.tolist(), gram_relative_asymmetry=float(asymmetry),
        mean_coefficients=m.tolist(), root=root, hessian=h)
