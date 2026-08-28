from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from data.DataGen.time_dependent import gen_nbns


ROOT = Path(__file__).resolve().parents[1]
GEN_SCRIPT = ROOT / "data" / "DataGen" / "static" / "gen_pde.sh"


@pytest.mark.parametrize(
    ("dataset_type", "alpha", "tau", "seed_offset"),
    [
        ("train", 2.5, 7.0, 0),
        ("easytest", 3.0, 6.5, 10_000_000),
        ("hardtest", 1.5, 5.0, 20_000_000),
    ],
)
def test_nsnonbounded_generation_profiles(dataset_type, alpha, tau, seed_offset):
    profile = gen_nbns.GENERATION_PROFILES[dataset_type]

    assert profile == {"alpha": alpha, "tau": tau, "seed_offset": seed_offset}


@pytest.mark.parametrize(
    ("alias", "expected"),
    [
        ("train", "train"),
        ("test", "easytest"),
        ("smooth", "easytest"),
        ("rough", "hardtest"),
    ],
)
def test_nsnonbounded_generation_type_aliases(alias, expected):
    assert gen_nbns.canonical_dataset_type(alias) == expected


def test_nsnonbounded_hardtest_output_name(tmp_path):
    path = gen_nbns._output_path(
        tmp_path,
        dataset_type="hardtest",
        sample_count=1000,
        resolution=128,
        record_steps=10,
        file_index=0,
    )

    assert path.name == "nsnonbounded_hardtest_1000-128-128-10.mat"


@pytest.mark.parametrize(
    ("dataset_type", "expected_static", "expected_temporal", "expected_seed"),
    [
        ("train", "alpha=2.0 tau=3.0", "alpha/gamma=2.5 tau=7.0", "0"),
        ("easytest", "alpha=3.0 tau=4.0", "alpha/gamma=3.0 tau=6.5", "10000000"),
        ("hardtest", "alpha=1.5 tau=5.0", "alpha/gamma=1.5 tau=5.0", "20000000"),
    ],
)
def test_unified_generation_script_resolves_profiles(
    dataset_type, expected_static, expected_temporal, expected_seed, tmp_path
):
    result = subprocess.run(
        [
            "bash",
            str(GEN_SCRIPT),
            dataset_type,
            "--pdes",
            "poisson,nsnonbounded,burgers",
            "--samples",
            "2",
            "--resolution",
            "16",
            "--out-root",
            str(tmp_path),
            "--device",
            "cpu",
            "--dry-run",
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    assert f"dataset type:      {dataset_type}" in result.stdout
    assert expected_static in result.stdout
    assert expected_temporal in result.stdout
    assert f"seed:              {expected_seed}" in result.stdout
    assert f"generate_poisson\\(\\'{dataset_type}\\'" in result.stdout
    assert f"--dataset-type {dataset_type}" in result.stdout
    assert f"gen_burgers1\\(\\'{dataset_type}\\'" in result.stdout


def test_matlab_profile_table_contains_requested_static_distributions():
    profile_source = (ROOT / "data" / "DataGen" / "static" / "get_generation_profile.m").read_text(
        encoding="utf-8"
    )

    for alpha, tau in [(2.0, 3.0), (3.0, 4.0), (1.5, 5.0)]:
        assert f"alpha = {alpha:.1f};" in profile_source
        assert f"tau = {tau:.1f};" in profile_source
