from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PDEDataSpec:
    name: str
    label_id: int
    coef_channels: int
    sol_channels: int
    coef_channel_names: tuple[str, ...]
    sol_channel_names: tuple[str, ...]
    scalar_param_names: tuple[str, ...] = ()
    optional_scalar_param_names: frozenset[str] = frozenset()
    param_aliases: dict[str, tuple[str, ...]] = field(default_factory=dict)
    residual_family: str = "static"
    default_loadby: str = "pair_h5"

    @property
    def channel_names(self) -> tuple[str, ...]:
        if self.name == "burger":
            return self.coef_channel_names
        return self.coef_channel_names + self.sol_channel_names

    @property
    def img_channels(self) -> int:
        if self.name == "burger":
            return self.coef_channels
        return self.coef_channels + self.sol_channels

    def to_metadata(self) -> dict[str, object]:
        return {
            "name": self.name,
            "label_id": self.label_id,
            "coef_channels": self.coef_channels,
            "sol_channels": self.sol_channels,
            "coef_channel_names": list(self.coef_channel_names),
            "sol_channel_names": list(self.sol_channel_names),
            "channel_names": list(self.channel_names),
            "scalar_param_names": list(self.scalar_param_names),
            "optional_scalar_param_names": sorted(self.optional_scalar_param_names),
            "param_aliases": {key: list(value) for key, value in self.param_aliases.items()},
            "residual_family": self.residual_family,
            "default_loadby": self.default_loadby,
            "img_channels": self.img_channels,
        }


PAIR_H5_DIAGNOSTIC_PARAMS = (
    "residual_norm",
    "picard_iters",
    "converged",
    "n_sources",
    "source_x",
    "source_y",
    "source_amp",
    "source_sigma",
)


PDE_DATA_SPECS: dict[str, PDEDataSpec] = {
    "darcy": PDEDataSpec(
        name="darcy",
        label_id=0,
        coef_channels=1,
        sol_channels=1,
        coef_channel_names=("coef",),
        sol_channel_names=("pressure",),
        residual_family="static",
        default_loadby="h5py",
    ),
    "poisson": PDEDataSpec(
        name="poisson",
        label_id=1,
        coef_channels=1,
        sol_channels=1,
        coef_channel_names=("source",),
        sol_channel_names=("phi",),
        residual_family="static",
        default_loadby="scipy",
    ),
    "helmholtz": PDEDataSpec(
        name="helmholtz",
        label_id=2,
        coef_channels=1,
        sol_channels=1,
        coef_channel_names=("source",),
        sol_channel_names=("psi",),
        residual_family="static",
        default_loadby="scipy",
    ),
    "nsnonbounded": PDEDataSpec(
        name="nsnonbounded",
        label_id=3,
        coef_channels=1,
        sol_channels=1,
        coef_channel_names=("w0",),
        sol_channel_names=("wT",),
        scalar_param_names=("nu", "T", "solver_dt"),
        optional_scalar_param_names=frozenset({"nu", "T", "solver_dt"}),
        param_aliases={"nu": ("nu", "viscosity"), "T": ("T", "total_time"), "solver_dt": ("solver_dt", "dt")},
        residual_family="temporal_endpoint",
        default_loadby="h5py",
    ),
    "burger": PDEDataSpec(
        name="burger",
        label_id=4,
        coef_channels=1,
        sol_channels=1,
        coef_channel_names=("u",),
        sol_channel_names=("u",),
        scalar_param_names=("nu",),
        optional_scalar_param_names=frozenset({"nu"}),
        residual_family="full_time_space",
        default_loadby="scipy",
    ),
    "reaction_diffusion": PDEDataSpec(
        name="reaction_diffusion",
        label_id=5,
        coef_channels=2,
        sol_channels=2,
        coef_channel_names=("u0", "v0"),
        sol_channel_names=("uT", "vT"),
        scalar_param_names=(
            "T",
            "D_u",
            "D_v",
            "k",
            "n_save_steps",
            "tdim",
            "x_left",
            "x_right",
            "y_bottom",
            "y_top",
            "dx",
            "dy",
        ),
        optional_scalar_param_names=frozenset(
            {
                "T",
                "D_u",
                "D_v",
                "k",
                "n_save_steps",
                "tdim",
                "x_left",
                "x_right",
                "y_bottom",
                "y_top",
                "dx",
                "dy",
            }
        ),
        param_aliases={"D_u": ("D_u", "Du"), "D_v": ("D_v", "Dv"), "T": ("T", "total_time")},
        residual_family="temporal_endpoint",
        default_loadby="rd",
    ),
    "shallow_water": PDEDataSpec(
        name="shallow_water",
        label_id=6,
        coef_channels=3,
        sol_channels=3,
        coef_channel_names=("h0", "hu0", "hv0"),
        sol_channel_names=("hT", "huT", "hvT"),
        scalar_param_names=("g", "T", "x_left", "x_right", "y_bottom", "y_top", "dx", "dy"),
        optional_scalar_param_names=frozenset({"g", "T", "x_left", "x_right", "y_bottom", "y_top", "dx", "dy"}),
        param_aliases={"g": ("g", "grav"), "T": ("T", "total_time")},
        residual_family="temporal_endpoint",
        default_loadby="swe",
    ),
    "heat": PDEDataSpec(
        name="heat",
        label_id=7,
        coef_channels=1,
        sol_channels=1,
        coef_channel_names=("u0",),
        sol_channel_names=("uT",),
        scalar_param_names=("alpha", "T", "total_time", "dt"),
        optional_scalar_param_names=frozenset({"alpha", "T", "total_time", "dt"}),
        param_aliases={"alpha": ("alpha", "fixed_alpha")},
        residual_family="temporal_endpoint",
        default_loadby="pair_h5",
    ),
    "wave": PDEDataSpec(
        name="wave",
        label_id=8,
        coef_channels=2,
        sol_channels=2,
        coef_channel_names=("u0", "v0"),
        sol_channel_names=("uT", "vT"),
        scalar_param_names=("c", "T", "total_time", "dt"),
        optional_scalar_param_names=frozenset({"c", "T", "total_time", "dt"}),
        param_aliases={"c": ("c", "fixed_c")},
        residual_family="temporal_endpoint",
        default_loadby="pair_h5",
    ),
    "advection_diffusion": PDEDataSpec(
        name="advection_diffusion",
        label_id=9,
        coef_channels=1,
        sol_channels=1,
        coef_channel_names=("u0",),
        sol_channel_names=("uT",),
        scalar_param_names=("b_x", "b_y", "kappa", "T", "total_time", "dt"),
        optional_scalar_param_names=frozenset({"T", "total_time", "dt"}),
        residual_family="temporal_endpoint",
        default_loadby="pair_h5",
    ),
    "steady_heat_conduction": PDEDataSpec(
        name="steady_heat_conduction",
        label_id=10,
        coef_channels=1,
        sol_channels=1,
        coef_channel_names=("f",),
        sol_channel_names=("u",),
        scalar_param_names=("u_D",) + PAIR_H5_DIAGNOSTIC_PARAMS,
        optional_scalar_param_names=frozenset(PAIR_H5_DIAGNOSTIC_PARAMS),
        residual_family="static",
        default_loadby="pair_h5",
    ),
}


STATIC_PDES = frozenset(name for name, spec in PDE_DATA_SPECS.items() if spec.residual_family == "static")
FULL_TIME_SPACE_PDES = frozenset(name for name, spec in PDE_DATA_SPECS.items() if spec.residual_family == "full_time_space")
TEMPORAL_ENDPOINT_PDES = frozenset(name for name, spec in PDE_DATA_SPECS.items() if spec.residual_family == "temporal_endpoint")


def get_pde_spec(pde: str) -> PDEDataSpec:
    key = str(pde).lower()
    try:
        return PDE_DATA_SPECS[key]
    except KeyError as exc:
        raise ValueError(f"Unsupported PDE {pde!r}; expected one of {sorted(PDE_DATA_SPECS)}") from exc
