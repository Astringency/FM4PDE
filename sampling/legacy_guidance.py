"""Historical guidance residuals, isolated from the current physical metrics.

These reproduce FM4PDE_bak/sampling/generate_pde.py. In particular its NS
surrogate is a spatial derivative sum, not the Navier-Stokes residual.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def legacy_residual(pde, a, u, k=1):
    a, u = a.double(), u.double()
    if pde in {"poisson", "helmholtz"}:
        h = 1 / (u.shape[-1] - 1)
        pad = F.pad(u, (1, 1, 1, 1))
        residual = (pad[..., :-2, 1:-1] + pad[..., 2:, 1:-1]
                    + pad[..., 1:-1, :-2] + pad[..., 1:-1, 2:] - 4 * u) / h**2 - a
        if pde == "helmholtz":
            return residual + k**2 * u
    elif pde in {"darcy", "nsnonbounded", "burger"}:
        dx = u.new_tensor([-1, 0, 1]).view(1, 1, 1, 3) / 2
        dy = dx.transpose(-1, -2)
        ux, uy = F.conv2d(u, dx, padding=(0, 1)), F.conv2d(u, dy, padding=(1, 0))
        if pde == "burger":
            # Historical time-space stencil: no dt/dx scaling or boundary mask.
            return uy + u * ux - 0.01 * F.conv2d(ux, dx, padding=(0, 1))
        if pde == "darcy":
            return F.conv2d(a * ux, dx, padding=(0, 1)) + F.conv2d(a * uy, dy, padding=(1, 0)) + 1
        residual = -ux - uy
    else:
        raise ValueError(f"Unsupported legacy guidance PDE: {pde}")
    interior = torch.ones_like(residual)
    interior[..., (0, -1), :] = 0
    interior[..., :, (0, -1)] = 0
    return residual * interior


def legacy_pde_loss(pde, a, u, k=1, *, sensor_columns=None):
    field = legacy_residual(pde, a, u, k)
    flat = field.flatten(1)
    # Old sample.py reused img_resolution_x=sensor_size in the PDE denominator.
    width = sensor_columns if pde == "burger" and sensor_columns is not None else u.shape[-1]
    if width <= 0:
        raise ValueError("Legacy PDE loss requires a positive sensor/grid width")
    return (torch.linalg.vector_norm(flat, dim=1) / (u.shape[-2] * width)).mean()
