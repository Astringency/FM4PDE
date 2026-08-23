from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from sampling.config import AblationConfig, load_config
from sampling.data import load_ground_truth
from sampling.runner import run_from_config_path, run_single_ablation
from sampling.sweep import expand_grid


MAIN_CONFIG_PATHS = sorted(Path("configs/main").glob("*/*.yaml"))
FORMAL_GRID = "configs/ablations/all_internal_ablation_grid.yaml"


def _formal_configs():
    jobs = expand_grid(FORMAL_GRID, selected_groups={"sampler_phase"})
    return [load_config(path, overrides=overrides) for path, overrides in jobs[::8]]


def test_formal_ablation_configs_disable_synthetic_fallback():
    for cfg in _formal_configs():
        assert cfg.allow_synthetic_data is False


def test_formal_data_paths_use_pde_named_directories():
    for cfg in _formal_configs():
        name = cfg.pde
        path = Path(cfg.data_path)
        path_text = path.as_posix()
        assert "/pair_h5/" not in path_text
        assert "/test1125/" not in path_text
        if name == "burger":
            assert path.parent.name in {"burger", "burgers"}
        else:
            assert path.parent.name == name


def test_main_task_data_paths_use_pde_named_directories():
    for config_path in MAIN_CONFIG_PATHS:
        name = config_path.stem
        cfg = load_config(config_path)
        path = Path(cfg.data_path)
        path_text = path.as_posix()
        assert "/pair_h5/" not in path_text
        assert "/test1125/" not in path_text
        expected_dir = "heat" if name == "heat_fixed" else name
        if name == "burger":
            assert path.parent.name in {"burger", "burgers"}
        else:
            assert path.parent.name == expected_dir


def test_formal_config_files_exist_and_load():
    for config_path in MAIN_CONFIG_PATHS:
        assert Path(config_path).is_file()
        load_config(config_path)
    assert list(_formal_configs())


def test_load_ground_truth_missing_formal_data_raises(tmp_path):
    cfg = AblationConfig(
        pde="heat",
        data_path=str(tmp_path / "missing_heat.h5"),
        loadby="pair_h5",
        allow_synthetic_data=False,
    )

    with pytest.raises(FileNotFoundError, match="Data path does not exist"):
        load_ground_truth(cfg)


def test_runner_missing_formal_data_raises_without_synthetic(tmp_path):
    cfg = AblationConfig(
        pde="heat",
        task="both",
        data_path=str(tmp_path / "missing_heat.h5"),
        loadby="pair_h5",
        output_dir=str(tmp_path / "runs"),
        dry_run=True,
        allow_synthetic_data=False,
        num_steps=1,
    )

    with pytest.raises(FileNotFoundError, match="Data path does not exist"):
        run_single_ablation(cfg)


def test_smoke_dry_run_can_use_synthetic_data(tmp_path):
    missing_data_path = tmp_path / "missing_poisson_smoke.mat"
    result = run_from_config_path(
        "configs/ablations/smoke.yaml",
        overrides={
            "dry_run": True,
            "output_dir": str(tmp_path),
            "num_steps": 1,
            "device": "cpu",
            "data_path": str(missing_data_path),
        },
    )

    assert result["status"] == "ok"
    assert result["synthetic_data"] is True
    assert Path(result["run_dir"]).exists()
