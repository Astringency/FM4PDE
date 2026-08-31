from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

try:
    import yaml
except ModuleNotFoundError:  # Keep default training usable in minimal environments.
    yaml = None


TRAIN_FILES_KEY = "train_files"


def load_training_file_manifest(
    config_path: str | Path,
    *,
    data_root: str | Path,
    pde_names: Sequence[str],
) -> dict[str, list[Path]]:
    """Load explicit training files from YAML and resolve them against ``data_root``.

    The preferred schema supports one manifest shared by multiple PDEs::

        train_files:
          heat:
            - heat/heat_10000-128-128_0.h5

    For a single-PDE run, ``train_files`` may also be a string or a list of
    strings. Relative entries are resolved against ``data_root``; absolute
    entries are used as-is.
    """

    requested_pdes = [str(name) for name in pde_names]
    if not requested_pdes:
        raise ValueError("pde_names must contain at least one PDE")
    if yaml is None:
        raise ModuleNotFoundError(
            "--train_data_config requires PyYAML. Install the project environment "
            "from environment.yml or install the 'pyyaml' package."
        )

    manifest_path = Path(config_path).expanduser()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Training data config does not exist: {manifest_path}")

    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            document = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in training data config {manifest_path}: {exc}") from exc

    if not isinstance(document, dict):
        raise ValueError(
            f"Training data config {manifest_path} must contain a YAML mapping with "
            f"a {TRAIN_FILES_KEY!r} key"
        )
    if TRAIN_FILES_KEY not in document:
        raise ValueError(
            f"Training data config {manifest_path} is missing required key {TRAIN_FILES_KEY!r}"
        )

    raw_train_files = document[TRAIN_FILES_KEY]
    if isinstance(raw_train_files, dict):
        files_by_pde = raw_train_files
    elif len(requested_pdes) == 1:
        files_by_pde = {requested_pdes[0]: raw_train_files}
    else:
        raise ValueError(
            f"{TRAIN_FILES_KEY!r} must map each PDE to its files when joint training is requested; "
            f"requested PDEs={requested_pdes}"
        )

    missing_pdes = [pde_name for pde_name in requested_pdes if pde_name not in files_by_pde]
    if missing_pdes:
        raise ValueError(
            f"Training data config {manifest_path} has no {TRAIN_FILES_KEY!r} entry "
            f"for requested PDE(s) {missing_pdes}"
        )

    root = Path(data_root).expanduser()
    resolved: dict[str, list[Path]] = {}
    for pde_name in requested_pdes:
        entries = _normalize_file_entries(files_by_pde[pde_name], pde_name, manifest_path)
        paths = [_resolve_training_file(entry, root, manifest_path) for entry in entries]
        normalized_paths = [str(path) for path in paths]
        if len(set(normalized_paths)) != len(normalized_paths):
            raise ValueError(
                f"Training data config {manifest_path} contains duplicate files for PDE {pde_name!r}"
            )
        resolved[pde_name] = paths
    return resolved


def _normalize_file_entries(value: Any, pde_name: str, manifest_path: Path) -> list[str]:
    if isinstance(value, str):
        entries = [value]
    elif isinstance(value, list):
        entries = value
    else:
        raise ValueError(
            f"{TRAIN_FILES_KEY}.{pde_name} in {manifest_path} must be a file name "
            "or a list of file names"
        )
    if not entries:
        raise ValueError(f"{TRAIN_FILES_KEY}.{pde_name} in {manifest_path} must not be empty")
    if not all(isinstance(entry, str) and entry.strip() for entry in entries):
        raise ValueError(
            f"Every entry in {TRAIN_FILES_KEY}.{pde_name} in {manifest_path} "
            "must be a non-empty string"
        )
    return [entry.strip() for entry in entries]


def _resolve_training_file(entry: str, data_root: Path, manifest_path: Path) -> Path:
    path = Path(entry).expanduser()
    if not path.is_absolute():
        if data_root.exists() and not data_root.is_dir():
            raise ValueError(
                f"--data_path must be a directory when {manifest_path} contains relative file names; "
                f"got {data_root}"
            )
        path = data_root / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Training file listed in {manifest_path} does not exist or is not a file: {path}"
        )
    return path
