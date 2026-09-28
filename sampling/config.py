from __future__ import annotations

import argparse
import ast
import dataclasses
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from data.specs import TEMPORAL_ENDPOINT_PDES


VALID_PDES = {
    "darcy",
    "poisson",
    "helmholtz",
    "nsnonbounded",
    "burger",
    "reaction_diffusion",
    "shallow_water",
    "heat",
    "wave",
    "advection_diffusion",
    "steady_heat_conduction",
}

VALID_TASKS = {"forward", "inverse", "both", "unconditional"}
VALID_GUIDANCE_COMPONENTS = {
    "noguide",
    "obs_only",
    "pde_only",
    "obs_pde",
    "coef_obs_only",
    "sol_obs_only",
    "both_obs",
}
VALID_LOSS_STATES = {"xt", "x_next", "endpoint"}
VALID_SAMPLER_PHASES = {"deterministic", "stochastic", "hybrid_d2s", "hybrid_s2d"}
VALID_SENSOR_MODES = {"random", "fixed", "grid", "sensor_column", "per_sample_random", "time_slices"}
VALID_TIME_GRIDS = {"uniform", "geometric"}
VALID_RESIDUAL_MODES = {
    "auto",
    "hermite_bridge",
    "near_endpoint_temporal",
    "endpoint_secant",
    "full_time_space",
    "disabled",
}
VALID_MODEL_PROFILES = {"auto", "recommended", "light", "base", "heavy", "legacy"}
VALID_BOUNDARY_CONDITION_MODES = {"auto", "dirichlet_zero", "neumann_zero", "periodic", "mixed", "none", "wall", "open"}
VALID_BOUNDARY_RESIDUAL_NORMALIZATION = {"mean", "sqrt_grid_over_mask", "mask_mean"}
VALID_TEST_TYPES = {"id", "smooth", "rough", "rough2", "rough3", "joint_ood"}


@dataclass
class AblationConfig:
    pde: str = "poisson"
    task: str = "forward"
    checkpoint_path: str = "outputs/pretrained/fm4poisson.pth"
    output_dir: str = "outputs/ablations"
    model_profile: str = "recommended"

    guidance_components: str = "obs_pde"
    model_gradient_checkpointing: bool = False
    loss_state: str = "endpoint"
    sampler_phase: str = "stochastic"
    switch_ratio: float = 0.5
    clip_threshold: float = 1e10

    num_obs: int = 500
    num_sensor_columns: int | None = None
    sensor_mode: str = "random"
    shared_mask: bool = False
    mask_seed: int = 0
    noise_level: float = 0.0
    noise_seed: int = 0
    noise_level_coef: float | None = None
    noise_level_sol: float | None = None

    time_grid: str = "uniform"
    num_steps: int = 100

    batch_size: int = 1
    sample_seed: int = 42
    device: str = "cpu"
    dtype: str = "float32"
    save_intermediate: bool = False
    save_plots: bool = False
    save_per_sample_curves: bool = False
    dry_run: bool = False

    zeta_obs_a: float = 1.0
    zeta_obs_u: float = 1.0
    zeta_pde: float = 1.0
    pde_guidance_start_ratio: float = 0.8
    stochastic_guidance_coeff: float = 0.1
    time_grid_eta: float = 0.4
    pde_residual_status: str = "auto"
    residual_mode: str = "auto"
    enforce_boundary_conditions: bool = True
    boundary_condition_mode: str = "auto"
    bc_weight: float = 1.0
    endpoint_bc_weight: float = 1.0
    boundary_residual_normalization: str = "sqrt_grid_over_mask"
    allow_unknown_boundary_conditions: bool = False
    data_path: str = ""
    data_paths: dict[str, str] = field(default_factory=dict)
    test_type: str = "id"
    offset: int = 0
    img_channels: int = 2
    img_resolution: int = 128
    coef_name: str = ""
    solution_name: str = ""
    loadby: str = ""
    ablation_name: str = ""
    ablation_group: str = ""
    allow_synthetic_data: bool = True
    empty_cache_each_step: bool = False
    initial_noise_source_batch_size: int | None = None
    initial_noise_source_indices: list[int] = field(default_factory=list)
    k: int = 1
    runtime_metadata: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def validate(self) -> None:
        self.resolve_test_data_path()
        self.residual_mode = normalize_residual_mode(self.residual_mode)
        checks = [
            ("pde", self.pde, VALID_PDES),
            ("task", self.task, VALID_TASKS),
            ("guidance_components", self.guidance_components, VALID_GUIDANCE_COMPONENTS),
            ("loss_state", self.loss_state, VALID_LOSS_STATES),
            ("sampler_phase", self.sampler_phase, VALID_SAMPLER_PHASES),
            ("sensor_mode", self.sensor_mode, VALID_SENSOR_MODES),
            ("time_grid", self.time_grid, VALID_TIME_GRIDS),
            ("residual_mode", self.residual_mode, VALID_RESIDUAL_MODES),
            ("model_profile", self.model_profile, VALID_MODEL_PROFILES),
            ("boundary_condition_mode", self.boundary_condition_mode, VALID_BOUNDARY_CONDITION_MODES),
            ("boundary_residual_normalization", self.boundary_residual_normalization, VALID_BOUNDARY_RESIDUAL_NORMALIZATION),
        ]
        for name, value, allowed in checks:
            if value not in allowed:
                raise ValueError(f"{name}={value!r} is invalid; expected one of {sorted(allowed)}")
        if self.test_type not in VALID_TEST_TYPES:
            raise ValueError(
                f"test_type={self.test_type!r} is invalid; expected one of {sorted(VALID_TEST_TYPES)}"
            )
        if not 0.0 <= self.switch_ratio <= 1.0:
            raise ValueError("switch_ratio must be in [0, 1]")
        if not 0.0 <= self.pde_guidance_start_ratio <= 1.0:
            raise ValueError("pde_guidance_start_ratio must be in [0, 1]")
        if self.time_grid == "geometric" and self.sampler_phase == "stochastic":
            raise ValueError("geometric time grids require a deterministic sampling phase")
        if self.time_grid_eta <= 0:
            raise ValueError("time_grid_eta must be positive")
        if self.stochastic_guidance_coeff < 0:
            raise ValueError("stochastic_guidance_coeff must be non-negative")
        if self.num_steps < 1:
            raise ValueError("num_steps must be positive")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.num_obs < 0:
            raise ValueError("num_obs must be non-negative")
        if self.sensor_mode == "time_slices" and self.pde != "burger":
            raise ValueError("time_slices is defined for Burgers [B,C,T,X] trajectories")
        if self.sensor_mode in {"sensor_column", "time_slices"}:
            if self.num_sensor_columns is None or int(self.num_sensor_columns) <= 0:
                raise ValueError(
                    "sensor_mode='sensor_column' requires an explicit positive num_sensor_columns"
                )
        elif self.num_sensor_columns is not None and int(self.num_sensor_columns) <= 0:
            raise ValueError("num_sensor_columns must be positive when specified")
        if self.residual_mode == "near_endpoint_temporal":
            has_observations = (
                self.num_sensor_columns is not None and int(self.num_sensor_columns) > 0
                if self.sensor_mode == "sensor_column"
                else self.num_obs > 0
            )
            if not has_observations:
                raise ValueError(
                    "residual_mode='near_endpoint_temporal' requires a positive main endpoint sensor budget"
                )
        if self.residual_mode == "near_endpoint_temporal" and self.pde not in TEMPORAL_ENDPOINT_PDES:
            raise ValueError(
                f"residual_mode={self.residual_mode!r} is only supported for temporal endpoint PDEs "
                f"{sorted(TEMPORAL_ENDPOINT_PDES)}; got pde={self.pde!r}"
            )
        if self.pde != "burger" and self.residual_mode in {"full_time_space"}:
            raise ValueError(
                f"residual_mode={self.residual_mode!r} requires a model-predicted full time-space field, "
                "but the current FM4PDE model outputs only a/u endpoint fields. "
                "Only Burgers outputs a full predicted time-space field."
            )
        if self.pde == "burger" and self.residual_mode in {
            "hermite_bridge",
            "endpoint_secant",
        }:
            raise ValueError(
                f"residual_mode={self.residual_mode!r} is an endpoint approximation, but Burgers already "
                "outputs the full predicted time-space field. Use auto or full_time_space."
            )
        if self.bc_weight < 0 or self.endpoint_bc_weight < 0:
            raise ValueError("bc_weight and endpoint_bc_weight must be non-negative")
        if self.clip_threshold <= 0:
            raise ValueError("clip_threshold must be positive")
        validate_task_guidance(self.task, self.guidance_components)

    def resolve_test_data_path(self) -> str:
        if not self.data_paths:
            return self.data_path
        unknown = sorted(set(self.data_paths).difference(VALID_TEST_TYPES))
        if unknown:
            raise ValueError(f"Unknown data_paths test types: {', '.join(unknown)}")
        missing = sorted(({"id", "smooth", "rough"} | {self.test_type}).difference(self.data_paths))
        if missing:
            raise ValueError(f"data_paths is missing test types: {', '.join(missing)}")
        if self.test_type not in VALID_TEST_TYPES:
            raise ValueError(
                f"test_type={self.test_type!r} is invalid; expected one of {sorted(VALID_TEST_TYPES)}"
            )
        self.data_path = str(self.data_paths[self.test_type])
        return self.data_path

    def resolved_ablation_name(self) -> str:
        if self.ablation_name:
            return self.ablation_name
        phase = self.sampler_phase
        if phase.startswith("hybrid"):
            phase = f"{phase}_{self.switch_ratio:g}"
        return (
            f"{self.guidance_components}_{self.loss_state}_{phase}_"
            f"pdegate-s{self.pde_guidance_start_ratio:g}_clip{self.clip_threshold:g}_"
            f"{self.sensor_mode}{self._sensor_budget()}_noise{self.noise_level:g}_"
            f"{self.time_grid}{self.num_steps}_"
            f"{self._short_residual_fragment()}_{self._short_bc_ic_fragment()}"
        )

    def _short_residual_fragment(self) -> str:
        if self.residual_mode == "near_endpoint_temporal":
            return f"res-near-aligned-{self.sensor_mode}{self._sensor_budget()}"
        return {
            "auto": "res-auto", "hermite_bridge": "res-hermite",
            "endpoint_secant": "res-secant", "full_time_space": "res-fulltime",
            "disabled": "res-off",
        }[self.residual_mode]

    def _sensor_budget(self) -> int:
        if self.sensor_mode in {"sensor_column", "time_slices"}:
            return int(self.num_sensor_columns or 0)
        return int(self.num_obs)

    def _short_bc_ic_fragment(self) -> str:
        return (
            f"bc-{self.boundary_condition_mode}-bcw{self.bc_weight:g}_"
            f"epw{self.endpoint_bc_weight:g}"
        )

    def asdict(self) -> dict[str, Any]:
        data = dataclasses.asdict(self)
        data.pop("runtime_metadata", None)
        data["ablation_name"] = self.resolved_ablation_name()
        return data


def validate_task_guidance(task: str, guidance_components: str) -> None:
    if task == "unconditional" and guidance_components != "noguide":
        raise ValueError("task='unconditional' requires guidance_components='noguide'")
    if task == "forward" and guidance_components in {"sol_obs_only", "both_obs"}:
        raise ValueError("task='forward' cannot request solution-side observation-only guidance")
    if task == "inverse" and guidance_components in {"coef_obs_only", "both_obs"}:
        raise ValueError("task='inverse' cannot request coefficient-side observation-only guidance")


def normalize_residual_mode(mode: Any) -> str:
    normalized = str(mode)
    if normalized not in VALID_RESIDUAL_MODES:
        raise ValueError(f"residual_mode={mode!r} is invalid; expected one of {sorted(VALID_RESIDUAL_MODES)}")
    return normalized


def load_config(path: str | os.PathLike[str], overrides: dict[str, Any] | None = None) -> AblationConfig:
    raw = load_yaml_file(path)
    unknown = sorted(set(raw).difference(_field_names()))
    if unknown:
        raise ValueError(f"Unknown config fields: {', '.join(unknown)}")
    cfg = AblationConfig(**raw)
    normalized_overrides = {
        key.replace("-", "_"): value for key, value in (overrides or {}).items()
    }
    for key, value in normalized_overrides.items():
        set_config_value(cfg, key, value)
    if "data_path" in normalized_overrides and "data_paths" not in normalized_overrides:
        # An explicit file path is a complete override.  Do not let the
        # per-test-type mapping replace it during validation.
        cfg.data_paths = {}
    resolve_resource_paths(cfg, use_checkpoint_environment='checkpoint_path' not in normalized_overrides)
    cfg.validate()
    return cfg


def resolve_resource_paths(cfg: AblationConfig, *, use_checkpoint_environment: bool = True) -> None:
    """Resolve public data/checkpoint locations without changing scientific settings.

    Absolute paths and explicit CLI overrides retain their meaning. DATA_ROOT
    replaces the ``datasets/`` prefix; CHECKPOINT_ROOT replaces
    ``outputs/pretrained/``. CHECKPOINT_<PDE> selects one equation's weights.
    """
    root = Path(__file__).resolve().parents[1]

    def resolve(value: str, prefix: str, environment: str) -> str:
        if not value:
            return value
        value = os.path.expandvars(os.path.expanduser(str(value)))
        p = Path(value)
        if p.is_absolute():
            return str(p)
        if value.startswith(prefix + '/') and os.environ.get(environment):
            return str(Path(os.environ[environment]).expanduser().resolve() / value[len(prefix)+1:])
        return str(root / p)

    cfg.data_path = resolve(cfg.data_path, 'datasets', 'DATA_ROOT')
    cfg.data_paths = {k: resolve(v, 'datasets', 'DATA_ROOT') for k, v in cfg.data_paths.items()}
    checkpoint = os.environ.get('CHECKPOINT_' + cfg.pde.upper()) if use_checkpoint_environment else None
    cfg.checkpoint_path = resolve(checkpoint or cfg.checkpoint_path, 'outputs/pretrained', 'CHECKPOINT_ROOT')

def save_resolved_config(cfg: AblationConfig, output_dir: str | os.PathLike[str]) -> Path:
    path = Path(output_dir) / "resolved_config.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dump_yaml(cfg.asdict()), encoding="utf-8")
    return path


def parse_cli_overrides(items: list[str] | None) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for item in items or []:
        item = item[2:] if item.startswith("--") else item
        if "=" not in item:
            raise ValueError(f"Override {item!r} must use key=value syntax")
        key, raw_value = item.split("=", 1)
        overrides[key.replace("-", "_")] = _parse_scalar(raw_value)
    return overrides


def set_config_value(cfg: AblationConfig, key: str, value: Any) -> None:
    key = key.replace("-", "_")
    if key not in _field_names():
        raise ValueError(f"Unknown config override: {key}")
    current = getattr(cfg, key)
    if isinstance(current, bool):
        value = _to_bool(value)
    elif isinstance(current, int) and not isinstance(current, bool):
        value = int(value)
    elif isinstance(current, float):
        value = float(value)
    setattr(cfg, key, value)


def str2bool(value: Any) -> bool:
    return _to_bool(value)


def add_bool_arg(parser: argparse.ArgumentParser, name: str, default: bool, help: str) -> None:
    dest = name.replace("-", "_")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(f"--{name}", dest=dest, action="store_true", help=help)
    group.add_argument(f"--no-{name}", dest=dest, action="store_false", help=f"Disable {help}")
    parser.set_defaults(**{dest: default})


def load_yaml_file(path: str | os.PathLike[str]) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(text)
        return loaded or {}
    except ModuleNotFoundError:
        return _simple_yaml_load(text)


def dump_yaml(data: dict[str, Any]) -> str:
    try:
        import yaml  # type: ignore

        return yaml.safe_dump(data, sort_keys=False)
    except ModuleNotFoundError:
        lines: list[str] = []
        _dump_yaml_value(data, lines, 0)
        return "\n".join(lines) + "\n"


def _field_names() -> set[str]:
    return {item.name for item in dataclasses.fields(AblationConfig) if item.init}


def _simple_yaml_load(text: str) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, Any, Any | None, str | None]] = [(-1, root, None, None)]
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if stripped.startswith("- "):
            item = _parse_scalar(stripped[2:].strip())
            if isinstance(parent, dict):
                container_parent = stack[-1][2]
                container_key = stack[-1][3]
                if parent or not isinstance(container_parent, dict) or container_key is None:
                    continue
                replacement: list[Any] = []
                container_parent[container_key] = replacement
                stack[-1] = (stack[-1][0], replacement, container_parent, container_key)
                parent = replacement
            if isinstance(parent, list):
                parent.append(item)
            continue
        if ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        key = key.strip().strip("'\"")
        value = value.strip()
        if value == "":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child, parent, key))
        else:
            parent[key] = _parse_scalar(value)
    return root


def _dump_yaml_value(value: Any, lines: list[str], indent: int, key: str | None = None) -> None:
    prefix = " " * indent
    if isinstance(value, dict):
        if key is not None:
            lines.append(f"{prefix}{key}:")
            indent += 2
            prefix = " " * indent
        for child_key, child_value in value.items():
            _dump_yaml_value(child_value, lines, indent, str(child_key))
    elif isinstance(value, list):
        rendered = "[" + ", ".join(_format_scalar(v) for v in value) + "]"
        lines.append(f"{prefix}{key}: {rendered}")
    elif key is not None:
        lines.append(f"{prefix}{key}: {_format_scalar(value)}")


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    lower = value.lower()
    if lower in {"true", "false"}:
        return lower == "true"
    if lower in {"null", "~"}:
        return None
    if (value.startswith("'") and value.endswith("'")) or (
        value.startswith('"') and value.endswith('"')
    ):
        return value[1:-1]
    if value.startswith("[") and value.endswith("]"):
        try:
            return ast.literal_eval(value)
        except (SyntaxError, ValueError):
            inner = value[1:-1].strip()
            return [] if not inner else [_parse_scalar(part.strip()) for part in inner.split(",")]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _format_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return str(value)
    return repr(str(value))


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        if value.lower() in {"1", "true", "yes", "y", "on"}:
            return True
        if value.lower() in {"0", "false", "no", "n", "off"}:
            return False
    return bool(value)
