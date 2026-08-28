from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from data.DataGen.time_dependent import gen_nbns


ROOT = Path(__file__).resolve().parents[1]
GEN_SCRIPT = ROOT / "data" / "DataGen" / "gen_pde.sh"


@pytest.mark.parametrize(
    ("dataset_type", "alpha", "tau", "seed_offset"),
    [
        ("train", 2.5, 7.0, 0),
        ("id", 2.5, 7.0, 10_000_000),
        ("smooth", 3.0, 6.5, 20_000_000),
        ("rough", 1.5, 5.0, 30_000_000),
    ],
)
def test_nsnonbounded_generation_profiles(dataset_type, alpha, tau, seed_offset):
    profile = gen_nbns.GENERATION_PROFILES[dataset_type]

    assert profile == {"alpha": alpha, "tau": tau, "seed_offset": seed_offset}


@pytest.mark.parametrize(
    ("alias", "expected"),
    [
        ("train", "train"),
        ("test", "id"),
        ("easytest", "smooth"),
        ("hardtest", "rough"),
    ],
)
def test_nsnonbounded_generation_type_aliases(alias, expected):
    assert gen_nbns.canonical_dataset_type(alias) == expected


def test_nsnonbounded_rough_output_name(tmp_path):
    path = gen_nbns._output_path(
        tmp_path,
        dataset_type="rough",
        sample_count=1000,
        resolution=128,
        record_steps=10,
        file_index=0,
    )

    assert path.name == "nsnonbounded_test_1000-128-128-10_rough.mat"


@pytest.mark.parametrize(
    ("dataset_type", "expected_seed"),
    [
        ("id", "10000000"),
        ("smooth", "20000000"),
        ("rough", "30000000"),
    ],
)
def test_unified_generation_script_resolves_profiles(dataset_type, expected_seed, tmp_path):
    env = {
        **os.environ,
        "PDE": "poisson,nsnonbounded,burgers,heat,reaction_diffusion,shallow_water",
        "TYPE": dataset_type,
        "TEST_SAMPLES": "2",
        "RESOLUTION": "16",
        "OUT_ROOT": str(tmp_path),
        "DEVICE": "cpu",
        "DRY_RUN": "true",
    }
    result = subprocess.run(
        ["bash", str(GEN_SCRIPT)],
        cwd=ROOT,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )

    assert f"Types: {dataset_type}" in result.stdout
    assert f"type={dataset_type} seed={expected_seed}" in result.stdout
    assert f"generate_poisson\\(\\'{dataset_type}\\'" in result.stdout
    assert f"--dataset-type {dataset_type}" in result.stdout
    assert f"gen_burgers1\\(\\'{dataset_type}\\'" in result.stdout


def test_unified_generation_script_defaults_to_five_train_shards_and_three_tests(tmp_path):
    env = {
        **os.environ,
        "PDE": "poisson",
        "OUT_ROOT": str(tmp_path),
        "DRY_RUN": "true",
    }
    result = subprocess.run(
        ["bash", str(GEN_SCRIPT)],
        cwd=ROOT,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "Types: train id smooth rough" in result.stdout
    assert "Train: 5 shards x 10000 samples" in result.stdout
    assert result.stdout.count("generate_poisson\\(\\'train\\'") == 5
    for dataset_type in ("id", "smooth", "rough"):
        assert result.stdout.count(f"generate_poisson\\(\\'{dataset_type}\\'") == 1


def test_unified_generation_script_accepts_key_value_arguments(tmp_path):
    result = subprocess.run(
        [
            "bash",
            str(GEN_SCRIPT),
            "PDE=heat",
            "TYPE=smooth",
            "TEST_SAMPLES=3",
            "RESOLUTION=16",
            f"OUT_ROOT={tmp_path}",
            "DRY_RUN=true",
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "PDEs: heat" in result.stdout
    assert "Types: smooth" in result.stdout
    assert "--split test --n-test 3" in result.stdout
    assert "--dataset-type smooth" in result.stdout


def test_matlab_profile_table_contains_requested_static_distributions():
    profile_source = (ROOT / "data" / "DataGen" / "static" / "get_generation_profile.m").read_text(
        encoding="utf-8"
    )

    for alpha, tau in [(2.0, 3.0), (3.0, 4.0), (1.5, 5.0)]:
        assert f"alpha = {alpha:.1f};" in profile_source
        assert f"tau = {tau:.1f};" in profile_source
