import h5py
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from data.load import PDEloader


def _write_rd(path, value, *, init_mode="grf", seed=0):
    arr = np.zeros((3, 4, 4, 2), dtype=np.float32)
    arr[0, :, :, 0] = value
    arr[0, :, :, 1] = value + 1
    arr[-1, :, :, 0] = value + 2
    arr[-1, :, :, 1] = value + 3
    with h5py.File(path, "w") as h5:
        h5.attrs["pde_name"] = "reaction_diffusion"
        h5.attrs["T"] = 1.5
        h5.attrs["Du"] = 2e-3
        h5.attrs["Dv"] = 4e-3
        h5.attrs["k"] = 3e-3
        h5.attrs["tdim"] = 3
        h5.attrs["n_save_steps"] = 2
        h5.attrs["init_mode"] = init_mode
        h5.attrs["x_range"] = (-1.0, 1.0)
        h5.attrs["y_range"] = (-2.0, 2.0)
        h5.create_dataset("sample_seed", data=np.array([seed], dtype=np.int64))
        group = h5.create_group("000000")
        group.create_dataset("data", data=arr)
        group.attrs["seed"] = seed


def test_reaction_diffusion_loader_defaults_to_new_format_when_legacy_coexists(tmp_path):
    pde_dir = tmp_path / "reaction_diffusion"
    pde_dir.mkdir()
    _write_rd(pde_dir / "reaction_diffusion-128-128-10_0.h5", 10.0, init_mode="iid", seed=11)
    _write_rd(pde_dir / "reaction_diffusion_grf_1-4-4-T1.5-steps2.h5", 20.0, init_mode="grf", seed=22)

    data, labels, metadata = PDEloader("reaction_diffusion").load_data(str(tmp_path), size=1, return_metadata=True)

    assert tuple(data.shape) == (1, 4, 4, 4)
    assert torch.allclose(data[0, 0], torch.full((4, 4), 20.0))
    assert torch.equal(labels, torch.full((1,), 5, dtype=torch.long))
    assert metadata["selected_file_format"] == "new_gen_rd"
    assert metadata["extra_metadata"]["candidate_file_formats"] == ["legacy_rd", "new_gen_rd"]
    assert torch.allclose(metadata["pde_params"]["T"], torch.tensor([1.5]))
    assert torch.allclose(metadata["pde_params"]["D_u"], torch.tensor([2e-3]))
    assert metadata["pde_param_summary"]["T"]["mean"] == pytest.approx(1.5)
    assert metadata["extra_metadata"]["init_mode"] == ["grf"]
    assert metadata["extra_metadata"]["sample_seed"] == [22]
    assert torch.allclose(metadata["pde_params"]["x_left"], torch.tensor([-1.0]))
    assert torch.allclose(metadata["pde_params"]["y_top"], torch.tensor([2.0]))


def test_reaction_diffusion_loader_reads_legacy_only_when_requested(tmp_path):
    pde_dir = tmp_path / "reaction_diffusion"
    pde_dir.mkdir()
    _write_rd(pde_dir / "reaction_diffusion-128-128-10_0.h5", 10.0, init_mode="iid", seed=11)
    _write_rd(pde_dir / "reaction_diffusion_grf_1-4-4-T1.5-steps2.h5", 20.0, init_mode="grf", seed=22)

    data, _labels, metadata = PDEloader("reaction_diffusion").load_data(
        str(pde_dir),
        size=1,
        legacy_rd_files=True,
        return_metadata=True,
    )

    assert torch.allclose(data[0, 0], torch.full((4, 4), 10.0))
    assert metadata["selected_file_format"] == "legacy_rd"
    assert metadata["extra_metadata"]["init_mode"] == ["iid"]


def test_reaction_diffusion_loader_ignores_test_files_when_searching_directory(tmp_path):
    pde_dir = tmp_path / "reaction_diffusion"
    pde_dir.mkdir()
    _write_rd(pde_dir / "reaction_diffusion_test_grf_1-4-4-T1.5-steps2.h5", 30.0, seed=33)

    with pytest.raises(FileNotFoundError, match="No reaction_diffusion training HDF5 files"):
        PDEloader("reaction_diffusion").load_data(str(tmp_path), size=1)
