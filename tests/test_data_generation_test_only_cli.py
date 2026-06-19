from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import argparse
import pytest

from data.DataGen.python.common import PairH5Config, files_with_splits
from data.DataGen.time_dependent import gen_nbns, gen_swe


ROOT = Path(__file__).resolve().parents[1]


def test_pair_h5_split_test_plans_only_test_file(tmp_path):
    config = PairH5Config(
        pde="heat",
        out_root=tmp_path,
        n_train=8,
        n_test=4,
        split="test",
        train_shards=2,
        samples_per_shard=4,
        resolution=16,
    )

    planned = files_with_splits(config)

    assert [(split, path.name) for _, _, _, split, path in planned] == [
        ("test", "heat_test_4-16-16.h5")
    ]


def test_generate_pair_h5s_split_test_dry_run_has_no_train_file(tmp_path):
    cmd = [
        sys.executable,
        str(ROOT / "data" / "DataGen" / "python" / "generate_pair_h5s.py"),
        "--pde",
        "heat",
        "--out-root",
        str(tmp_path),
        "--split",
        "test",
        "--n-test",
        "4",
        "--dry-run",
    ]
    result = subprocess.run(cmd, cwd=ROOT, check=True, text=True, capture_output=True)

    assert "heat_test_4-128-128.h5" in result.stdout
    assert "heat_10000-128-128_1.h5" not in result.stdout


def test_nsnonbounded_test_output_path_is_parameterized(tmp_path):
    path = gen_nbns._output_path(
        tmp_path,
        split="test",
        sample_count=10000,
        resolution=128,
        record_steps=10,
        file_index=0,
    )

    assert path.name == "nsnonbounded_test_10000-128-128-10.mat"


def test_nsnonbounded_test_split_rejects_multiple_test_files(tmp_path):
    args = argparse.Namespace(
        out_dir=tmp_path,
        split="test",
        total_samples=10000,
        samples_per_file=5000,
        resolution=128,
        record_steps=10,
        T=1.0,
        dt=1e-4,
        viscosity=1e-3,
        device="cpu",
        seed_offset=10000000,
        overwrite=False,
    )

    with pytest.raises(ValueError, match="test split writes one file"):
        gen_nbns.generate_dataset(args)


def test_shallow_water_test_output_path_is_parameterized(tmp_path):
    path = gen_swe.output_path(
        tmp_path,
        split="test",
        total_samples=10000,
        resolution=128,
        tsteps=10,
    )

    assert path.name == "shallow_water_test_10000-128-128-10.h5"


def test_time_dependent_generators_expose_split_help():
    for script in ["gen_nbns.py", "gen_swe.py"]:
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "data" / "DataGen" / "time_dependent" / script),
                "--help",
            ],
            cwd=ROOT,
            check=True,
            text=True,
            capture_output=True,
        )
        assert "--split" in result.stdout
        assert "--total-samples" in result.stdout
