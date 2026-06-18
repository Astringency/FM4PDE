from __future__ import annotations

import argparse
import ast
import dataclasses
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


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
VALID_LOSS_STATES = {"xt", "x_next", "endpoint", "denoised_endpoint"}
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
VALID_CLIP_MODES = {"none", "global_norm", "per_component_norm"}
VALID_PDE_REGIONS = {"full", "observed", "boundary_excluded", "union_obs"}
VALID_SENSOR_MODES = {"random", "fixed", "grid", "sensor_column", "per_sample_random"}
VALID_TIME_GRIDS = {"uniform", "geometric", "cosine"}
VALID_STEP_METHODS = {"euler", "midpoint"}
VALID_LOSS_TYPES = {"l1", "l2", "mse"}
VALID_STOCHASTIC_GUIDANCE_TIMES = {"t", "t_next"}
VALID_RESIDUAL_MODES = {
    "auto",
    "hermite_bridge",
    "near_endpoint_temporal",
    "endpoint_secant",
    "legacy_endpoint_secant",
    "disabled",
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
    data_config_path: str = "configs/poisson.yaml"
    checkpoint_path: str = "outputs/pretrained/fm4poisson.pth"
    output_dir: str = "outputs/ablations"

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
    dry_run: bool = False

    zeta_obs_a: float = 1.0
    zeta_obs_u: float = 1.0
    zeta_pde: float = 1.0
    stochastic_guidance_coeff: float = 0.1
    stochastic_guidance_time: str = "t"
    obs_decay: float = 1.0
    obs_decay_start_ratio: float = 1.0
    polynomial_power: float = 2.0
    cosine_mode: str = "decay"
    time_grid_eta: float = 0.4
    loss_type: str = "l2"
    pde_residual_status: str = "auto"
    residual_mode: str = "auto"
    hermite_collocation_times: list[float] = field(default_factory=lambda: [0.25, 0.5, 0.75])
    hermite_num_collocation: int = 0
    hermite_include_integral_residual: bool = True
    hermite_integral_weight: float = 1.0
    num_near_endpoint_obs: int = 0
    near_endpoint_sensor_mode: str = "random"
    near_endpoint_mask_seed: int = 0
    near_endpoint_shared_mask: bool = True
    data_path: str = ""
    offset: int = 0
    img_channels: int = 2
    img_resolution: int = 128
    coef_name: str = ""
    solution_name: str = ""
    loadby: str = ""
    ablation_name: str = ""
    legacy_mode: str = "sparse"
    allow_synthetic_data: bool = True
    empty_cache_each_step: bool = False
    legacy_pickle: bool = True
    legacy_minmax: bool = False
    k: int = 1
    extra: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        self.residual_mode = normalize_residual_mode(self.residual_mode)
        if self.loss_state == "denoised_endpoint":
            import warnings

            warnings.warn(
                "loss_state='denoised_endpoint' is deprecated and maps to 'endpoint'; "
                "there is no separate denoising step in the FM4PDE ablation sampler.",
                DeprecationWarning,
                stacklevel=2,
            )
        if self.sensor_mode == "time_varying":
            import warnings

            warnings.warn(
                "sensor_mode='time_varying' is deprecated for BCHW endpoint data; "
                "use 'per_sample_random' for independently sampled endpoint masks.",
                DeprecationWarning,
                stacklevel=2,
            )
            self.sensor_mode = "per_sample_random"
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
            ("loss_type", self.loss_type, VALID_LOSS_TYPES),
            ("stochastic_guidance_time", self.stochastic_guidance_time, VALID_STOCHASTIC_GUIDANCE_TIMES),
            ("residual_mode", self.residual_mode, VALID_RESIDUAL_MODES),
            ("near_endpoint_sensor_mode", self.near_endpoint_sensor_mode, VALID_SENSOR_MODES),
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
        if self.num_near_endpoint_obs < 0:
            raise ValueError("num_near_endpoint_obs must be non-negative")
        if self.residual_mode == "near_endpoint_temporal" and self.num_near_endpoint_obs <= 0:
            raise ValueError("residual_mode='near_endpoint_temporal' requires num_near_endpoint_obs > 0")
        if self.hermite_num_collocation < 0:
            raise ValueError("hermite_num_collocation must be non-negative")
        if self.hermite_integral_weight < 0:
            raise ValueError("hermite_integral_weight must be non-negative")
        for value in self.hermite_collocation_times:
            if not 0.0 < float(value) < 1.0:
                raise ValueError("hermite_collocation_times values must lie inside (0, 1)")
        if self.clip_threshold <= 0 and self.clip_mode != "none":
            raise ValueError("clip_threshold must be positive when clipping is enabled")
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
            f"{self.sensor_mode}{self.num_obs}_noise{self.noise_level:g}_"
            f"{self.time_grid}{self.num_steps}_{self.step_method}"
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
        "two_time_level": "endpoint_secant",
        "legacy": "legacy_endpoint_secant",
        "legacy_two_time_level": "legacy_endpoint_secant",
        "coarse_endpoint": "endpoint_secant",
    }
    normalized = aliases.get(str(mode), str(mode))
    if normalized not in VALID_RESIDUAL_MODES:
        raise ValueError(f"residual_mode={mode!r} is invalid; expected one of {sorted(VALID_RESIDUAL_MODES)}")
    return normalized


def load_config(path: str | os.PathLike[str], overrides: dict[str, Any] | None = None) -> AblationConfig:
    raw = load_yaml_file(path)
    if "data" in raw or "generate" in raw or "model" in raw:
        cfg = config_from_legacy(raw, str(path))
    else:
        cfg = AblationConfig(**{k: v for k, v in raw.items() if k in _field_names()})
        cfg.extra.update({k: v for k, v in raw.items() if k not in _field_names()})
    for key, value in (overrides or {}).items():
        set_config_value(cfg, key, value)
    cfg.validate()
    return cfg


def config_from_legacy(raw: dict[str, Any], data_config_path: str) -> AblationConfig:
    data = raw.get("data", {})
    generate = raw.get("generate", {})
    model = raw.get("model", {})
    output = raw.get("output", {})

    pde = str(data.get("name", "poisson")).strip("'\"")
    checkpoint = str(model.get("pre-trained", _LOCAL_CHECKPOINTS.get(pde, ""))).strip("'\"")
    if checkpoint and not Path(checkpoint).exists():
        local = _LOCAL_CHECKPOINTS.get(pde, checkpoint)
        if Path(local).exists():
            checkpoint = local

    return AblationConfig(
        pde=pde,
        task="both",
        data_config_path=data_config_path,
        checkpoint_path=checkpoint,
        output_dir=_safe_output_dir(str(output.get("savepath", "outputs/ablations")).strip("'\"")),
        guidance_components="obs_pde" if _to_bool(generate.get("guide", True)) else "noguide",
        sampler_phase=str(generate.get("samplemode", "stochastic")).strip("'\""),
        switch_ratio=float(generate.get("stochastic_t", 1.0)),
        guidance_schedule="obs_decay",
        clip_mode="global_norm",
        clip_threshold=1e10,
        num_obs=int(data.get("obs_size", 500)),
        sensor_mode="sensor_column" if _to_bool(generate.get("sensor", False)) else "random",
        shared_mask=False,
        mask_seed=int(generate.get("seed", 0)),
        noise_seed=int(generate.get("seed", 0)),
        time_grid="geometric" if str(generate.get("samplemode", "")) == "deterministic" else "uniform",
        num_steps=int(generate.get("num_steps", 100)),
        step_method=str(generate.get("method", "euler")).strip("'\""),
        batch_size=int(generate.get("batch_size", 1)),
        sample_seed=int(generate.get("seed", 0)),
        device=str(generate.get("device", "cpu")).strip("'\""),
        save_intermediate=_to_bool(generate.get("process", False)),
        save_plots=_to_bool(output.get("plot", False)),
        zeta_obs_a=float(generate.get("zeta_obs_a", 1.0)),
        zeta_obs_u=float(generate.get("zeta_obs_u", 1.0)),
        zeta_pde=float(generate.get("zeta_pde", 1.0)),
        obs_decay=float(generate.get("obsguide_decay", 1.0)),
        obs_decay_start_ratio=float(generate.get("obsguide_t", 1.0)),
        data_path=str(data.get("datapath", "")).strip("'\""),
        offset=int(data.get("offset", 0)),
        img_channels=int(data.get("img_channels", 2)),
        img_resolution=int(data.get("img_resolution", 128)),
        coef_name=str(data.get("coef", "")).strip("'\""),
        solution_name=str(data.get("solution", "")).strip("'\""),
        loadby=str(data.get("loadby", "")).strip("'\""),
        extra={"legacy": raw},
    )


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
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        key = key.strip().strip("'\"")
        value = value.strip()
        if value == "":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
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


def _safe_output_dir(path: str) -> str:
    if not path:
        return "outputs/ablations"
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        return str(candidate)
    probe = candidate
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    if probe.exists() and os.access(probe, os.W_OK):
        return str(candidate)
    return "outputs/ablations"
