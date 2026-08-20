from __future__ import annotations

import argparse
import ast
import dataclasses
import hashlib
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
VALID_GRADIENT_TARGETS = {"current_state_chain_rule", "loss_state_direct", "next_state_direct"}
VALID_SAMPLER_PHASES = {"deterministic", "stochastic", "hybrid_d2s", "hybrid_s2d"}
VALID_GUIDANCE_SCHEDULES = {
    "constant",
    "delta",
    "bt",
    "cosine",
    "polynomial",
    "obs_decay",
}
VALID_CLIP_MODES = {"none", "global_norm", "per_component_norm", "per_sample_norm"}
VALID_PDE_REGIONS = {
    "full",
    "boundary_excluded",
    "coef_obs",
    "sol_obs",
    "active_obs_union",
    # Deprecated aliases retained for config compatibility. Both resolve to
    # the task-aware active observation union in sampling.masks.
    "observed",
    "union_obs",
}
VALID_SENSOR_MODES = {"random", "fixed", "grid", "sensor_column", "per_sample_random"}
VALID_TIME_GRIDS = {"uniform", "geometric", "cosine"}
VALID_STEP_METHODS = {"euler", "midpoint"}
VALID_STOCHASTIC_GUIDANCE_TIMES = {"t", "t_next"}
VALID_RESIDUAL_MODES = {
    "auto",
    "hermite_bridge",
    "near_endpoint_temporal",
    "endpoint_secant",
    "full_trajectory_fd",
    "full_time_space",
    "disabled",
}
VALID_MODEL_PROFILES = {"recommended", "light", "base", "heavy", "legacy_base"}
VALID_BOUNDARY_CONDITION_MODES = {"auto", "dirichlet_zero", "neumann_zero", "periodic", "mixed", "none", "legacy_ignore", "wall", "open"}
VALID_INITIAL_CONDITION_MODES = {"auto", "endpoint_initial", "observed_initial", "trajectory_initial", "none", "legacy_ignore"}
VALID_BOUNDARY_RESIDUAL_NORMALIZATION = {"mean", "sqrt_grid_over_mask", "mask_mean"}
VALID_NS_OPERATOR_MODES = {"generator_dealiased", "continuous_spectral"}
_TYPE_SUFFIX = "type"
DEPRECATED_LOSS_CONFIG_FIELDS = {f"loss_{_TYPE_SUFFIX}", f"obs_loss_{_TYPE_SUFFIX}", f"pde_loss_{_TYPE_SUFFIX}"}
DEPRECATED_LOSS_CONFIG_MESSAGE = (
    "loss reducers are no longer configurable; observation loss is masked MSE and PDE loss is component-wise MSE."
)
REMOVED_NEAR_ENDPOINT_CONFIG_FIELDS = {
    "num_near_endpoint_obs",
    "near_endpoint_sensor_mode",
    "near_endpoint_mask_seed",
    "near_endpoint_shared_mask",
}


_LOCAL_CHECKPOINTS = {
    "darcy": "outputs/pretrained/fm4darcy.pth",
    "poisson": "outputs/pretrained/fm4poisson.pth",
    "helmholtz": "outputs/pretrained/fm4helmholtz.pth",
    "nsnonbounded": "outputs/pretrained/fm4nsnonbounded.pth",
    "burger": "outputs/pretrained/fm4burgers.pth",
    "reaction_diffusion": "outputs/pretrained/fm4reaction_diffusion.pth",
    "shallow_water": "outputs/pretrained/fm4shallow_water.pth",
    "heat": "outputs/pretrained/fm4heat.pth",
    "wave": "outputs/pretrained/fm4wave.pth",
    "advection_diffusion": "outputs/pretrained/fm4advection_diffusion.pth",
    "steady_heat_conduction": "outputs/pretrained/fm4steady_heat_conduction.pth",
}


@dataclass
class AblationConfig:
    pde: str = "poisson"
    task: str = "forward"
    data_config_path: str = "configs/main/poisson.yaml"
    checkpoint_path: str = "outputs/pretrained/fm4poisson.pth"
    output_dir: str = "outputs/ablations"
    model_profile: str = "recommended"

    guidance_components: str = "obs_pde"
    loss_state: str = "endpoint"
    gradient_target: str = "current_state_chain_rule"
    sampler_phase: str = "stochastic"
    switch_ratio: float = 0.5
    guidance_schedule: str = "constant"
    clip_mode: str = "global_norm"
    clip_threshold: float = 1e10
    pde_residual_region: str = "full"

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
    step_method: str = "euler"

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
    stochastic_guidance_coeff: float = 0.1
    stochastic_guidance_time: str = "t"
    cfg_scale: float = 1.0
    obs_decay: float = 1.0
    obs_decay_start_ratio: float = 1.0
    polynomial_power: float = 2.0
    cosine_mode: str = "decay"
    time_grid_eta: float = 0.4
    pde_residual_status: str = "auto"
    residual_mode: str = "auto"
    enforce_boundary_conditions: bool = True
    enforce_initial_conditions: bool = True
    boundary_condition_mode: str = "auto"
    initial_condition_mode: str = "auto"
    bc_weight: float = 1.0
    ic_weight: float = 1.0
    endpoint_bc_weight: float = 1.0
    boundary_residual_normalization: str = "sqrt_grid_over_mask"
    ns_operator_mode: str = "generator_dealiased"
    allow_unknown_boundary_conditions: bool = False
    legacy_ignore_boundary: bool = False
    hermite_collocation_times: list[float] = field(default_factory=lambda: [0.25, 0.5, 0.75])
    hermite_num_collocation: int = 0
    hermite_include_integral_residual: bool = True
    hermite_integral_weight: float = 1.0
    data_path: str = ""
    offset: int = 0
    img_channels: int = 2
    img_resolution: int = 128
    coef_name: str = ""
    solution_name: str = ""
    loadby: str = ""
    ablation_name: str = ""
    allow_synthetic_data: bool = True
    empty_cache_each_step: bool = False
    k: int = 1
    extra: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        self.residual_mode = normalize_residual_mode(self.residual_mode)
        deprecated = DEPRECATED_LOSS_CONFIG_FIELDS.intersection(self.extra)
        if deprecated:
            fields = ", ".join(sorted(deprecated))
            raise ValueError(f"{fields}: {DEPRECATED_LOSS_CONFIG_MESSAGE}")
        removed_near = REMOVED_NEAR_ENDPOINT_CONFIG_FIELDS.intersection(self.extra)
        if removed_near:
            fields = ", ".join(sorted(removed_near))
            raise ValueError(
                f"{fields} have been removed. near_endpoint_temporal now reuses the main endpoint "
                "sensor mask; configure num_obs/sensor_mode or num_sensor_columns instead."
            )
        checks = [
            ("pde", self.pde, VALID_PDES),
            ("task", self.task, VALID_TASKS),
            ("guidance_components", self.guidance_components, VALID_GUIDANCE_COMPONENTS),
            ("loss_state", self.loss_state, VALID_LOSS_STATES),
            ("gradient_target", self.gradient_target, VALID_GRADIENT_TARGETS),
            ("sampler_phase", self.sampler_phase, VALID_SAMPLER_PHASES),
            ("guidance_schedule", self.guidance_schedule, VALID_GUIDANCE_SCHEDULES),
            ("clip_mode", self.clip_mode, VALID_CLIP_MODES),
            ("pde_residual_region", self.pde_residual_region, VALID_PDE_REGIONS),
            ("sensor_mode", self.sensor_mode, VALID_SENSOR_MODES),
            ("time_grid", self.time_grid, VALID_TIME_GRIDS),
            ("step_method", self.step_method, VALID_STEP_METHODS),
            ("stochastic_guidance_time", self.stochastic_guidance_time, VALID_STOCHASTIC_GUIDANCE_TIMES),
            ("residual_mode", self.residual_mode, VALID_RESIDUAL_MODES),
            ("model_profile", self.model_profile, VALID_MODEL_PROFILES),
            ("ns_operator_mode", self.ns_operator_mode, VALID_NS_OPERATOR_MODES),
            ("boundary_condition_mode", self.boundary_condition_mode, VALID_BOUNDARY_CONDITION_MODES),
            ("initial_condition_mode", self.initial_condition_mode, VALID_INITIAL_CONDITION_MODES),
            ("boundary_residual_normalization", self.boundary_residual_normalization, VALID_BOUNDARY_RESIDUAL_NORMALIZATION),
        ]
        for name, value, allowed in checks:
            if value not in allowed:
                raise ValueError(f"{name}={value!r} is invalid; expected one of {sorted(allowed)}")
        if not 0.0 <= self.switch_ratio <= 1.0:
            raise ValueError("switch_ratio must be in [0, 1]")
        if self.num_steps < 1:
            raise ValueError("num_steps must be positive")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.num_obs < 0:
            raise ValueError("num_obs must be non-negative")
        if self.sensor_mode == "sensor_column":
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
        if self.pde != "burger" and self.residual_mode in {"full_trajectory_fd", "full_time_space"}:
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
                "outputs the full predicted time-space field. Use auto, full_trajectory_fd, or full_time_space."
            )
        if self.initial_condition_mode in {"observed_initial", "trajectory_initial", "endpoint_initial"}:
            raise ValueError(
                f"initial_condition_mode={self.initial_condition_mode!r} would mix an observed/ground-truth field "
                "into PDE loss. Use observation guidance for measured initial values; sampling PDE loss is "
                "computed from model outputs only."
            )
        if self.hermite_num_collocation < 0:
            raise ValueError("hermite_num_collocation must be non-negative")
        if self.hermite_integral_weight < 0:
            raise ValueError("hermite_integral_weight must be non-negative")
        if self.bc_weight < 0 or self.ic_weight < 0 or self.endpoint_bc_weight < 0:
            raise ValueError("bc_weight, ic_weight and endpoint_bc_weight must be non-negative")
        for value in self.hermite_collocation_times:
            if not 0.0 < float(value) < 1.0:
                raise ValueError("hermite_collocation_times values must lie inside (0, 1)")
        if self.clip_threshold <= 0 and self.clip_mode != "none":
            raise ValueError("clip_threshold must be positive when clipping is enabled")
        if self.cfg_scale < 0:
            raise ValueError("cfg_scale must be non-negative")
        if self.gradient_target == "next_state_direct" and self.loss_state != "x_next":
            raise ValueError(
                "gradient_target='next_state_direct' is only connected when loss_state='x_next'"
            )
        if self.pde_residual_region == "coef_obs" and self.task not in {"forward", "both"}:
            raise ValueError(f"pde_residual_region='coef_obs' is inactive for task={self.task!r}")
        if self.pde_residual_region == "sol_obs" and self.task not in {"inverse", "both"}:
            raise ValueError(f"pde_residual_region='sol_obs' is inactive for task={self.task!r}")
        if self.pde_residual_region in {"active_obs_union", "observed", "union_obs"} and self.task == "unconditional":
            raise ValueError(
                f"pde_residual_region={self.pde_residual_region!r} is undefined for task='unconditional'"
            )
        validate_task_guidance(self.task, self.guidance_components)

    def resolved_ablation_name(self) -> str:
        if self.ablation_name:
            return self.ablation_name
        phase = self.sampler_phase
        if phase.startswith("hybrid"):
            phase = f"{phase}_{self.switch_ratio:g}"
        return (
            f"{self.guidance_components}_{self.loss_state}_{phase}_"
            f"{self.guidance_schedule}_{self.clip_mode}{self.clip_threshold:g}_"
            f"{self.sensor_mode}{self._sensor_budget()}_noise{self.noise_level:g}_"
            f"{self.time_grid}{self.num_steps}_{self.step_method}_"
            f"{self._short_residual_fragment()}_{self._short_bc_ic_fragment()}"
        )

    def _short_residual_fragment(self) -> str:
        mode_map = {
            "auto": "res-auto",
            "hermite_bridge": "res-hermite",
            "endpoint_secant": "res-secant",
            "full_trajectory_fd": "res-fulltraj",
            "full_time_space": "res-fulltime",
            "disabled": "res-off",
        }
        if self.residual_mode == "near_endpoint_temporal":
            return f"res-near-aligned-{self.sensor_mode}{self._sensor_budget()}"
        base = mode_map.get(self.residual_mode, f"res-{self.residual_mode}")
        if self.residual_mode in {"auto", "hermite_bridge"}:
            k = self.hermite_num_collocation if self.hermite_num_collocation > 0 else len(self.hermite_collocation_times)
            base = f"{base}-K{k}"
            if self.hermite_num_collocation <= 0 and self.hermite_collocation_times != [0.25, 0.5, 0.75]:
                digest = hashlib.sha1(",".join(f"{float(v):.8g}" for v in self.hermite_collocation_times).encode("utf-8")).hexdigest()[:6]
                base = f"{base}h{digest}"
            if self.hermite_include_integral_residual:
                base = f"{base}-int1-w{self.hermite_integral_weight:g}"
            else:
                base = f"{base}-int0"
        return base

    def _sensor_budget(self) -> int:
        if self.sensor_mode == "sensor_column":
            return int(self.num_sensor_columns or 0)
        return int(self.num_obs)

    def _short_bc_ic_fragment(self) -> str:
        return (
            f"bc-{self.boundary_condition_mode}-bcw{self.bc_weight:g}_"
            f"ic-{self.initial_condition_mode}-icw{self.ic_weight:g}_"
            f"epw{self.endpoint_bc_weight:g}_legacybc{int(bool(self.legacy_ignore_boundary))}"
        )

    def asdict(self) -> dict[str, Any]:
        data = dataclasses.asdict(self)
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
    aliases = {
        "": "auto",
        "default": "auto",
    }
    normalized = aliases.get(str(mode), str(mode))
    if normalized not in VALID_RESIDUAL_MODES:
        raise ValueError(f"residual_mode={mode!r} is invalid; expected one of {sorted(VALID_RESIDUAL_MODES)}")
    return normalized


def load_config(path: str | os.PathLike[str], overrides: dict[str, Any] | None = None) -> AblationConfig:
    raw = load_yaml_file(path)
    if "data" in raw or "generate" in raw or "model" in raw:
        raise ValueError(
            "This config uses the old data/generate/model schema. "
            "Please convert it to flat AblationConfig format."
        )
    else:
        cfg = AblationConfig(**{k: v for k, v in raw.items() if k in _field_names()})
        cfg.extra.update({k: v for k, v in raw.items() if k not in _field_names()})
    for key, value in (overrides or {}).items():
        set_config_value(cfg, key, value)
    cfg.validate()
    return cfg

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
        cfg.extra[key] = value
        return
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
    return {field.name for field in dataclasses.fields(AblationConfig)}


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
