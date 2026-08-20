import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
TIME_DEPENDENT_DIR = ROOT / "data" / "DataGen" / "time_dependent"
if str(TIME_DEPENDENT_DIR) not in sys.path:
    sys.path.insert(0, str(TIME_DEPENDENT_DIR))

from gen_rd import generate_dataset, resolve_seed_offset  # noqa: E402


def test_default_seed_offsets_are_disjoint_between_splits():
    assert resolve_seed_offset("train", None) == 0
    assert resolve_seed_offset("test", None) == 10_000_000
    assert resolve_seed_offset("test", 123) == 123


def _args(tmp_path, init_mode):
    return argparse.Namespace(
        save_path=tmp_path / init_mode,
        total_samples=2,
        samples_per_file=None,
        resolution=32,
        T=1.0,
        n_save_steps=10,
        init_mode=init_mode,
        init_mean=0.0,
        init_std=1.0,
        grf_length_scale=0.15,
        grf_spectral_power=2.0,
        no_grf_normalize=False,
        Du=2e-3,
        Dv=4e-3,
        k=3e-3,
        seed_offset=0,
        split="train",
        overwrite=True,
    )


def _neighbor_msd(field):
    dx = np.diff(field, axis=1)
    dy = np.diff(field, axis=0)
    return float(np.mean(dx**2) + np.mean(dy**2))


def _inspect(path, init_mode):
    smoothness = []
    with h5py.File(path, "r") as h5:
        assert h5.attrs["pde_name"] == "reaction_diffusion"
        assert h5.attrs["init_mode"] == init_mode
        assert h5.attrs["tdim"] == 11
        assert h5.attrs["n_save_steps"] == 10
        sample_keys = sorted(key for key in h5.keys() if key.isdigit())
        assert len(sample_keys) == 2
        for key in sample_keys:
            group = h5[key]
            data = group["data"][:]
            times = group["grid/t"][:]
            assert data.shape == (11, 32, 32, 2)
            assert len(times) == 11
            assert np.isclose(times[0], 0.0)
            assert np.isclose(times[-1], 1.0)
            assert group.attrs["init_mode"] == init_mode
            assert np.isfinite(data[0]).all()
            smoothness.append(_neighbor_msd(data[0, :, :, 0]))
            smoothness.append(_neighbor_msd(data[0, :, :, 1]))
    return float(np.mean(smoothness))


def test_reaction_diffusion_iid_and_grf_generation(tmp_path):
    iid_path = generate_dataset(_args(tmp_path, "iid"))[0]
    grf_path = generate_dataset(_args(tmp_path, "grf"))[0]

    assert "steps10" in iid_path.name
    assert "steps10" in grf_path.name
    iid_msd = _inspect(iid_path, "iid")
    grf_msd = _inspect(grf_path, "grf")
    assert grf_msd < iid_msd

    torch = pytest.importorskip("torch")
    from data.load import PDEloader

    data, labels = PDEloader("reaction_diffusion").load_data(str(tmp_path / "grf"), size=1)
    assert tuple(data.shape) == (2, 4, 32, 32)
    assert data.dtype == torch.float32
    assert torch.equal(labels, torch.full((2,), 5, dtype=torch.long))


def test_reaction_diffusion_training_and_sampling_use_endpoint_frames(tmp_path):
    torch = pytest.importorskip("torch")
    from data.load import PDEloader
    from sampling.config import AblationConfig
    from sampling.data import load_ground_truth

    path = tmp_path / "reaction_diffusion_test_grf_1-4-4-T1-steps10.h5"
    arr = np.zeros((11, 4, 4, 2), dtype=np.float32)
    arr[0, :, :, 0] = 1.0
    arr[0, :, :, 1] = 2.0
    arr[5, :, :, 0] = 50.0
    arr[5, :, :, 1] = 60.0
    arr[-1, :, :, 0] = 3.0
    arr[-1, :, :, 1] = 4.0
    with h5py.File(path, "w") as h5:
        h5.attrs["pde_name"] = "reaction_diffusion"
        h5.attrs["T"] = 1.0
        h5.attrs["Du"] = 2e-3
        h5.attrs["Dv"] = 4e-3
        h5.attrs["k"] = 3e-3
        h5.create_dataset("sample_seed", data=np.array([10], dtype=np.int64))
        group = h5.create_group("000000")
        group.create_dataset("data", data=arr)
        group.create_dataset("grid/t", data=np.linspace(0.0, 1.0, 11, dtype=np.float32))
        group.attrs["seed"] = 10
        group.attrs["T"] = 1.0
        group.attrs["Du"] = 2e-3
        group.attrs["Dv"] = 4e-3
        group.attrs["k"] = 3e-3

    train_data, _ = PDEloader("reaction_diffusion").load_data(str(path), size=1)
    assert tuple(train_data.shape) == (1, 4, 4, 4)
    assert torch.allclose(train_data[0, 0], torch.ones(4, 4) * 1.0)
    assert torch.allclose(train_data[0, 1], torch.ones(4, 4) * 2.0)
    assert torch.allclose(train_data[0, 2], torch.ones(4, 4) * 3.0)
    assert torch.allclose(train_data[0, 3], torch.ones(4, 4) * 4.0)

    cfg = AblationConfig(
        pde="reaction_diffusion",
        data_path=str(path),
        loadby="rd",
        img_channels=4,
        img_resolution=4,
        batch_size=1,
        allow_synthetic_data=False,
    )
    gt = load_ground_truth(cfg)
    assert torch.allclose(gt.coef[0, 0], torch.ones(4, 4) * 1.0)
    assert torch.allclose(gt.coef[0, 1], torch.ones(4, 4) * 2.0)
    assert torch.allclose(gt.sol[0, 0], torch.ones(4, 4) * 3.0)
    assert torch.allclose(gt.sol[0, 1], torch.ones(4, 4) * 4.0)
    assert torch.allclose(gt.pde_params["T"], torch.tensor([1.0]))
    assert torch.allclose(gt.pde_params["D_u"], torch.tensor([2e-3]))
    assert torch.allclose(gt.pde_params["D_v"], torch.tensor([4e-3]))
    assert torch.allclose(gt.pde_params["k"], torch.tensor([3e-3]))
