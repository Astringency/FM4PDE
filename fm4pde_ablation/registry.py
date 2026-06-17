from __future__ import annotations

from fm4pde_ablation.pde_residuals import residual_status


PDE_REGISTRY = {
    "darcy": {"coef_channels": 1, "sol_channels": 1, "residual_status": residual_status("darcy")},
    "poisson": {"coef_channels": 1, "sol_channels": 1, "residual_status": residual_status("poisson")},
    "helmholtz": {"coef_channels": 1, "sol_channels": 1, "residual_status": residual_status("helmholtz")},
    "nsnonbounded": {"coef_channels": 1, "sol_channels": 1, "residual_status": residual_status("nsnonbounded")},
    "burger": {"coef_channels": 1, "sol_channels": 1, "residual_status": residual_status("burger")},
    "reaction_diffusion": {"coef_channels": 2, "sol_channels": 2, "residual_status": residual_status("reaction_diffusion")},
    "shallow_water": {"coef_channels": 3, "sol_channels": 3, "residual_status": residual_status("shallow_water")},
}


def get_pde_info(pde: str) -> dict[str, object]:
    if pde not in PDE_REGISTRY:
        raise ValueError(f"Unknown PDE {pde!r}")
    return PDE_REGISTRY[pde]
