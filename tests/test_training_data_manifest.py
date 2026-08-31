from __future__ import annotations

import argparse

import pytest

np = pytest.importorskip("numpy")
h5py = pytest.importorskip("h5py")
torch = pytest.importorskip("torch")

from data.load import PDEloader
from data.training_manifest import load_training_file_manifest


def _write_heat_pair(path, alpha_values):
    alpha = np.asarray(alpha_values, dtype=np.float32)
    n_samples = int(alpha.shape[0])
    values = alpha.reshape(n_samples, 1, 1, 1)
    with h5py.File(path, "w") as file:
        file.attrs["split"] = "train"
        file.attrs["T"] = 1.0
        file.create_dataset("input_data", data=np.broadcast_to(values, (n_samples, 1, 4, 4)))
        file.create_dataset("output_data", data=np.broadcast_to(values + 10.0, (n_samples, 1, 4, 4)))
        file.create_dataset("alpha", data=alpha)


def test_manifest_resolves_relative_files_against_data_path(tmp_path):
    data_root = tmp_path / "data"
    pde_dir = data_root / "heat"
    pde_dir.mkdir(parents=True)
    first = pde_dir / "heat_custom_a.h5"
    second = pde_dir / "heat_custom_b.h5"
    _write_heat_pair(first, [1.0, 2.0])
    _write_heat_pair(second, [3.0, 4.0])
    manifest = tmp_path / "training.yaml"
    manifest.write_text(
        "train_files:\n"
        "  heat:\n"
        "    - heat/heat_custom_b.h5\n"
        "    - heat/heat_custom_a.h5\n",
        encoding="utf-8",
    )

    resolved = load_training_file_manifest(
        manifest,
        data_root=data_root,
        pde_names=["heat"],
    )

    assert resolved == {"heat": [second.resolve(), first.resolve()]}


def test_single_pde_manifest_accepts_direct_file_list(tmp_path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    training_file = data_root / "custom.h5"
    _write_heat_pair(training_file, [1.0, 2.0])
    manifest = tmp_path / "training.yaml"
    manifest.write_text("train_files:\n  - custom.h5\n", encoding="utf-8")

    resolved = load_training_file_manifest(manifest, data_root=data_root, pde_names=["heat"])

    assert resolved == {"heat": [training_file.resolve()]}


def test_joint_manifest_requires_every_requested_pde(tmp_path):
    manifest = tmp_path / "training.yaml"
    manifest.write_text("train_files:\n  heat:\n    - heat.h5\n", encoding="utf-8")

    with pytest.raises(ValueError, match="no 'train_files' entry.*poisson"):
        load_training_file_manifest(
            manifest,
            data_root=tmp_path,
            pde_names=["heat", "poisson"],
        )


def test_manifest_rejects_missing_and_duplicate_files(tmp_path):
    missing_manifest = tmp_path / "missing.yaml"
    missing_manifest.write_text("train_files:\n  heat:\n    - missing.h5\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_training_file_manifest(missing_manifest, data_root=tmp_path, pde_names=["heat"])

    training_file = tmp_path / "heat.h5"
    _write_heat_pair(training_file, [1.0, 2.0])
    duplicate_manifest = tmp_path / "duplicate.yaml"
    duplicate_manifest.write_text(
        "train_files:\n  heat:\n    - heat.h5\n    - heat.h5\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate files"):
        load_training_file_manifest(duplicate_manifest, data_root=tmp_path, pde_names=["heat"])


def test_explicit_files_are_loaded_in_order_with_global_sample_cap(tmp_path):
    first = tmp_path / "heat_custom_a.h5"
    second = tmp_path / "heat_custom_b.h5"
    _write_heat_pair(first, [1.0, 2.0])
    _write_heat_pair(second, [3.0, 4.0])

    loader = PDEloader("heat")
    data, _labels = loader.load_data_files([second, first], max_samples=3)
    metadata = loader.metadata()

    assert data[:, 0, 0, 0].tolist() == pytest.approx([3.0, 4.0, 1.0])
    assert loader.pde_params["alpha"].tolist() == pytest.approx([3.0, 4.0, 1.0])
    assert metadata["extra_metadata"]["data_selection"] == "explicit_manifest"
    assert metadata["extra_metadata"]["configured_files"] == [str(second.resolve()), str(first.resolve())]
    assert metadata["selected_files"] == [str(second.resolve()), str(first.resolve())]
    assert metadata["num_loaded_samples"] == 3


def test_training_split_uses_only_manifest_files_and_records_selection(tmp_path):
    import train

    first = tmp_path / "heat_custom_a.h5"
    second = tmp_path / "heat_custom_b.h5"
    _write_heat_pair(first, [1.0, 2.0])
    _write_heat_pair(second, [3.0, 4.0])

    train_data, _train_labels, train_meta, val_data, _val_labels, val_meta, split = (
        train._load_training_and_validation_data(
            ["heat"],
            str(tmp_path),
            data_size=99,
            seed=3,
            train_files_by_pde={"heat": [second, first]},
        )
    )

    loaded_values = torch.cat([train_data[:, 0, 0, 0], val_data[:, 0, 0, 0]]).sort().values
    assert torch.equal(loaded_values, torch.tensor([1.0, 2.0, 3.0, 4.0]))
    assert split["per_pde"]["heat"]["original_loaded_samples"] == 4
    for metadata in (train_meta["heat"], val_meta["heat"]):
        extra = metadata["extra_metadata"]
        assert extra["configured_files"] == [str(second.resolve()), str(first.resolve())]
        assert extra["selected_files"] == [str(second.resolve()), str(first.resolve())]


def test_training_metadata_and_parser_record_optional_manifest():
    from train import _build_data_metadata
    from train_arg_parser import get_args_parser

    parser = get_args_parser()
    assert parser.parse_args([]).train_data_config is None
    parsed = parser.parse_args(["--train_data_config", "training.yaml"])
    assert parsed.train_data_config == "training.yaml"

    args = argparse.Namespace(
        dataset="heat",
        data_path="/data",
        data_size=5,
        train_data_config="training.yaml",
        max_train_samples=None,
        normalization_eps=1e-6,
        model_profile="recommended",
    )
    metadata = _build_data_metadata(
        args=args,
        pde_names=["heat"],
        data=torch.zeros(2, 2, 4, 4),
        label=torch.zeros(2, dtype=torch.long),
        loader_metadata={},
    )

    assert metadata["train_data_config"] == "training.yaml"
    assert metadata["training_data_selection"] == "explicit_manifest"
