import h5py
import scipy.io
from tqdm import tqdm
import numpy as np
from pathlib import Path
import warnings

import torch
from torch.utils.data import Dataset

from data.specs import get_pde_spec


DEFAULT_TRAIN_SHARDS = 5

_SAMPLE_ALIGNED_EXTRA_KEYS = {
    "sample_id",
    "sample_seed",
    "init_mode",
    "boundary_condition",
}
_EXPLICIT_EXTRA_SKIP_KEYS = {
    "configured_files",
    "data_selection",
    "file_paths",
    "num_loaded_samples",
    "selected_file_format",
    "selected_files",
}


def _merge_explicit_extra_metadata(chunks):
    entries = [
        (chunk_loader.extra_metadata or {}, int(data.shape[0]), str(file_path))
        for chunk_loader, data, _labels, file_path in chunks
    ]
    keys = sorted(
        set().union(*(set(metadata) for metadata, _count, _path in entries))
        - _EXPLICIT_EXTRA_SKIP_KEYS
    )
    merged = {}
    per_file = {}
    for key in keys:
        values = [metadata.get(key) for metadata, _count, _path in entries]
        present = [key in metadata for metadata, _count, _path in entries]
        if key in _SAMPLE_ALIGNED_EXTRA_KEYS and all(present):
            aligned = [
                _sample_aligned_list(value, count)
                for value, (_metadata, count, _path) in zip(values, entries)
            ]
            if all(value is not None for value in aligned):
                merged[key] = [item for value in aligned for item in value]
                continue
        plain_values = [_metadata_to_plain(value) for value in values]
        if all(present) and all(value == plain_values[0] for value in plain_values[1:]):
            merged[key] = plain_values[0]
            continue
        per_file[key] = [
            {"file_path": path, "value": plain_value if is_present else None}
            for plain_value, is_present, (_metadata, _count, path) in zip(
                plain_values,
                present,
                entries,
            )
        ]
    if per_file:
        merged["per_file_metadata"] = per_file
    return merged


def _sample_aligned_list(value, count):
    if isinstance(value, torch.Tensor):
        if value.ndim > 0 and int(value.shape[0]) == int(count):
            return value.detach().cpu().tolist()
        return None
    if isinstance(value, np.ndarray):
        if value.ndim > 0 and int(value.shape[0]) == int(count):
            return value.tolist()
        return None
    if isinstance(value, (list, tuple)) and len(value) == int(count):
        return [_metadata_to_plain(item) for item in value]
    return None


def _metadata_to_plain(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _metadata_to_plain(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_metadata_to_plain(child) for child in value]
    return value


class TensorDataset(Dataset):
    def __init__(self, data, labels, scalar_conditioning=None):
        self.data = data
        self.labels = labels.to(torch.long)
        self.scalar_conditioning = None
        if scalar_conditioning is not None:
            scalar_conditioning = torch.as_tensor(scalar_conditioning, dtype=torch.float32)
            if scalar_conditioning.ndim != 2:
                raise ValueError(
                    "scalar_conditioning must have shape [N,K], "
                    f"got {tuple(scalar_conditioning.shape)}"
                )
            if int(scalar_conditioning.shape[0]) != int(data.shape[0]):
                raise ValueError(
                    "scalar_conditioning sample count must match data; "
                    f"got {int(scalar_conditioning.shape[0])} vs {int(data.shape[0])}"
                )
            self.scalar_conditioning = scalar_conditioning
        self.max_size = data.shape[0]
        self.num_channels = self.data.shape[1]
        self.resolution = self.data.shape[2]
        self.label_dim = 1
        self.scalar_conditioning_dim = (
            0 if self.scalar_conditioning is None else int(self.scalar_conditioning.shape[1])
        )

    def __getitem__(self, index):
        if self.scalar_conditioning is not None:
            return self.data[index], self.labels[index], self.scalar_conditioning[index]
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)


class PDEloader:
    def __init__(self, pde: str):
        self.pde = pde.lower()
        self.spec = get_pde_spec(self.pde)
        # Sample-aligned scalar PDE parameters cached by pair HDF5 and generated HDF5 loads.
        self.pde_params = {}
        self.pde_param_sources = {}
        self.pde_param_slices = []
        self.extra_metadata = {}
        self.load_func = {
                "darcy": self._darcy_load,
                "poisson": self._poisson_load,
                "helmholtz": self._helmholtz_load,
                "nsnonbounded": self._nsnonbounded_load,
                "burger": self._burger_load,
                "reaction_diffusion": self._reaction_diffusion_load,
                "shallow_water": self._shallow_water_load,
                "heat": self._heat_load,
                "wave": self._wave_load,
                "advection_diffusion": self._advection_diffusion_load,
                "steady_heat_conduction": self._steady_heat_conduction_load,
                }

        if self.pde not in self.load_func:
            raise ValueError(f"Unsupported PDE {pde!r}; expected one of {sorted(self.load_func)}")

    def load_data(self, *args, return_metadata=False, **kwargs):
        data, label = self.load_func[self.pde](*args, **kwargs)
        if return_metadata:
            return data, label, self.metadata()
        return data, label

    def load_data_files(self, file_paths, *, max_samples=None, return_metadata=False, **kwargs):
        """Load an explicit ordered list of files for this PDE.

        Each file goes through the existing single-file loader, so file-format
        validation remains identical to the default directory-discovery path.
        ``max_samples`` is applied across the complete list, not per file.
        """

        configured_paths = [Path(path).expanduser().resolve() for path in file_paths]
        if not configured_paths:
            raise ValueError(f"Explicit training file list for {self.pde!r} must not be empty")
        if max_samples is not None and int(max_samples) < 1:
            raise ValueError("max_samples must be positive when loading explicit files")

        chunks = []
        loaded_paths = []
        loaded_samples = 0
        for file_path in configured_paths:
            if not file_path.is_file():
                raise FileNotFoundError(f"Explicit training data file does not exist: {file_path}")
            remaining = None if max_samples is None else int(max_samples) - loaded_samples
            if remaining is not None and remaining <= 0:
                break

            chunk_loader = PDEloader(self.pde)
            data, labels = chunk_loader.load_data(
                file_path,
                size=1,
                max_samples=remaining,
                **kwargs,
            )
            if int(data.shape[0]) == 0:
                continue
            chunks.append((chunk_loader, data, labels, file_path))
            loaded_paths.append(file_path)
            loaded_samples += int(data.shape[0])

        if not chunks:
            raise ValueError(f"No samples were loaded from explicit files for PDE {self.pde!r}")

        self._merge_explicit_file_metadata(
            chunks,
            configured_paths=configured_paths,
            loaded_paths=loaded_paths,
            loaded_samples=loaded_samples,
        )
        data = torch.cat([chunk[1] for chunk in chunks], dim=0).to(torch.float32)
        labels = torch.cat([chunk[2] for chunk in chunks], dim=0).to(torch.long)
        if return_metadata:
            return data, labels, self.metadata()
        return data, labels

    def _merge_explicit_file_metadata(
        self,
        chunks,
        *,
        configured_paths,
        loaded_paths,
        loaded_samples,
    ):
        expected_param_names = set(chunks[0][0].pde_params)
        for chunk_loader, _data, _labels, file_path in chunks[1:]:
            param_names = set(chunk_loader.pde_params)
            if param_names != expected_param_names:
                raise ValueError(
                    "Explicit training files must provide the same sample-level PDE parameters; "
                    f"expected {sorted(expected_param_names)}, got {sorted(param_names)} in {file_path}"
                )

        self.pde_params = {
            name: torch.cat(
                [
                    torch.as_tensor(chunk_loader.pde_params[name]).reshape(-1)
                    for chunk_loader, *_rest in chunks
                ],
                dim=0,
            ).to(torch.float32)
            for name in sorted(expected_param_names)
        }
        self.pde_param_sources = {}
        for name in sorted(expected_param_names):
            sources = [
                chunk_loader.pde_param_sources.get(name, "unknown")
                for chunk_loader, *_rest in chunks
            ]
            self.pde_param_sources[name] = (
                sources[0] if all(source == sources[0] for source in sources) else "mixed"
            )

        self.pde_param_slices = []
        sample_offset = 0
        for chunk_loader, data, _labels, file_path in chunks:
            chunk_count = int(data.shape[0])
            if chunk_loader.pde_param_slices:
                for item in chunk_loader.pde_param_slices:
                    adjusted = dict(item)
                    adjusted["start"] = int(item.get("start", 0)) + sample_offset
                    adjusted["stop"] = int(item.get("stop", chunk_count)) + sample_offset
                    self.pde_param_slices.append(adjusted)
            else:
                self.pde_param_slices.append(
                    {
                        "file_path": str(file_path),
                        "start": sample_offset,
                        "stop": sample_offset + chunk_count,
                        "params": tuple(sorted(expected_param_names)),
                    }
                )
            sample_offset += chunk_count

        self.extra_metadata = _merge_explicit_extra_metadata(chunks)
        formats = [chunk[0].extra_metadata.get("selected_file_format") for chunk in chunks]
        nonempty_formats = [value for value in formats if value is not None]
        if nonempty_formats:
            self.extra_metadata["selected_file_format"] = (
                nonempty_formats[0]
                if all(value == nonempty_formats[0] for value in nonempty_formats)
                else "mixed"
            )
        self.extra_metadata.update(
            {
                "data_selection": "explicit_manifest",
                "configured_files": [str(path) for path in configured_paths],
                "selected_files": [str(path) for path in loaded_paths],
                "file_paths": [str(path) for path in loaded_paths],
                "num_loaded_samples": int(loaded_samples),
            }
        )

    def metadata(self):
        pde_param_summary = {}
        if self.pde_params:
            from data.metadata import summarize_pde_params

            pde_param_summary = summarize_pde_params({self.pde: self.pde_params}).get(self.pde, {})
        return {
            "pde": self.pde,
            "pde_params": self.pde_params,
            "pde_params_keys": sorted(self.pde_params),
            "pde_param_sources": dict(self.pde_param_sources),
            "pde_param_slices": list(self.pde_param_slices),
            "channel_names": self._channel_names_for_loaded_data(),
            "coef_channel_names": list(self.spec.coef_channel_names),
            "sol_channel_names": list(self.spec.sol_channel_names),
            "scalar_params_loaded": bool(self.pde_params),
            "pde_param_summary": pde_param_summary,
            "pde_data_spec": self.spec.to_metadata(),
            "scalar_param_names": list(self.spec.scalar_param_names),
            "optional_scalar_param_names": sorted(self.spec.optional_scalar_param_names),
            "residual_family": self.spec.residual_family,
            "loadby": self.spec.default_loadby,
            "extra_metadata": dict(self.extra_metadata),
            "selected_file_format": self.extra_metadata.get("selected_file_format"),
            "selected_files": list(self.extra_metadata.get("selected_files", [])),
            "num_loaded_samples": self.extra_metadata.get("num_loaded_samples"),
        }

    def _pde_dir(self, data_path):
        path = Path(data_path).expanduser()
        if path.is_file():
            return path.parent
        candidates = [path / self.pde]
        if self.pde == "burger":
            candidates.append(path / "burgers")
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    def _resolve_data_file(self, data_path, file_name):
        path = Path(data_path).expanduser()
        if path.is_file():
            return path
        return self._pde_dir(data_path) / file_name

    @staticmethod
    def _shard_indices(data_path, size, *, start):
        """Return one index for an explicit file, or all requested directory shards."""
        if int(size) < 1:
            raise ValueError("size must be positive")
        count = 1 if Path(data_path).expanduser().is_file() else int(size)
        return range(int(start), int(start) + count)

    def _finalize(self, data):
        data = torch.as_tensor(data, dtype=torch.float32).contiguous()
        self._assert_bchw(data, context=self.pde)
        label = torch.full((int(data.shape[0]),), int(self.spec.label_id), dtype=torch.long)
        return data, label

    @staticmethod
    def _assert_bchw(data, context="data"):
        if data.ndim != 4:
            raise ValueError(f"{context} data must be [N,C,H,W], got shape={tuple(data.shape)}")
        if data.shape[1] < 1:
            raise ValueError(f"{context} data must have at least one channel, got shape={tuple(data.shape)}")
        if data.shape[2] != data.shape[3]:
            raise ValueError(f"{context} data must have square spatial grid, got shape={tuple(data.shape)}")
        if data.dtype != torch.float32:
            raise TypeError(f"{context} data must be float32, got {data.dtype}")

    def _darcy_load(self, data_path, size=DEFAULT_TRAIN_SHARDS, max_samples=None):
        dataset = []
        remaining = max_samples
        for i in tqdm(self._shard_indices(data_path, size, start=1)):
            file_path = self._resolve_data_file(data_path, f"{self.pde}_10000-128-128_{i}.mat")
            with h5py.File(file_path, 'r') as file:
                total = file["thresh_a_data"].shape[-1]
                take = total if remaining is None else min(int(remaining), total)
                if take <= 0:
                    break
                a = file['thresh_a_data'][:, :, :take] # type: ignore
                u = file['thresh_p_data'][:, :, :take] # type: ignore
            dataset.append(np.stack([a, u], axis=0).transpose(3, 0, 1, 2)) # type: ignore
            if remaining is not None:
                remaining -= take
                if remaining <= 0:
                    break

        return self._finalize(np.concatenate(dataset, axis=0))

    def _poisson_load(self, data_path, size=DEFAULT_TRAIN_SHARDS, max_samples=None):
        dataset = []
        remaining = max_samples
        for i in tqdm(self._shard_indices(data_path, size, start=1)):
            file_path = self._resolve_data_file(data_path, f"{self.pde}_10000-128-128_{i}.mat")
            loaded = scipy.io.loadmat(file_path, variable_names=["f_data", "phi_data"])
            take = loaded["f_data"].shape[0] if remaining is None else min(int(remaining), loaded["f_data"].shape[0])
            if take <= 0:
                break
            dataset.append(np.stack([loaded["f_data"][:take], loaded["phi_data"][:take]], axis=1))
            if remaining is not None:
                remaining -= take
                if remaining <= 0:
                    break

        return self._finalize(np.concatenate(dataset, axis=0))
    
    def _helmholtz_load(self, data_path, size=DEFAULT_TRAIN_SHARDS, max_samples=None):
        dataset = []
        remaining = max_samples
        for i in tqdm(self._shard_indices(data_path, size, start=1)):
            file_path = self._resolve_data_file(data_path, f"{self.pde}_10000-128-128_{i}.mat")
            loaded = scipy.io.loadmat(file_path, variable_names=["f_data", "psi_data"])
            take = loaded["f_data"].shape[0] if remaining is None else min(int(remaining), loaded["f_data"].shape[0])
            if take <= 0:
                break
            dataset.append(np.stack([loaded["f_data"][:take], loaded["psi_data"][:take]], axis=1))
            if remaining is not None:
                remaining -= take
                if remaining <= 0:
                    break

        return self._finalize(np.concatenate(dataset, axis=0))

    def _nsnonbounded_load(self, data_path, size=DEFAULT_TRAIN_SHARDS, max_samples=None):
        dataset = []
        remaining = max_samples
        for i in tqdm(self._shard_indices(data_path, size, start=1)):
            file_path = self._resolve_data_file(data_path, f"{self.pde}_10000-128-128-10_{i}_new.mat")
            with h5py.File(file_path, 'r') as file:
                total = file["w0"].shape[0]
                take = total if remaining is None else min(int(remaining), total)
                if take <= 0:
                    break
                w0 = file['w0'][:take] # type: ignore
                wt = file['w'][:take, :, :, :] # type: ignore

            dataset.append(self._nsnonbounded_pair(w0, wt))
            if remaining is not None:
                remaining -= take
                if remaining <= 0:
                    break
        
        return self._finalize(np.concatenate(dataset, axis=0))
    
    def _burger_load(self, data_path, size=DEFAULT_TRAIN_SHARDS, max_samples=None):
        dataset = []
        remaining = max_samples
        for i in tqdm(self._shard_indices(data_path, size, start=1)):
            file_path = self._resolve_data_file(data_path, f"{self.pde}_10000-128-128_{i}.mat")
            output = scipy.io.loadmat(file_path, variable_names=["output"])["output"]
            take = output.shape[0] if remaining is None else min(int(remaining), output.shape[0])
            if take <= 0:
                break
            dataset.append(np.expand_dims(output[:take], axis=1))
            if remaining is not None:
                remaining -= take
                if remaining <= 0:
                    break
        
        return self._finalize(np.concatenate(dataset, axis=0))

    def _reaction_diffusion_load(
        self,
        data_path,
        size=DEFAULT_TRAIN_SHARDS,
        max_samples=None,
        rd_init_mode_filter=None,
    ):
        dataset = []
        sample_count = 0
        self.pde_params = {}
        self.pde_param_sources = {}
        self.pde_param_slices = []
        self.extra_metadata = {}
        (
            file_paths,
            detected_init_modes,
            mixed_init_modes,
        ) = self._reaction_diffusion_paths(
            data_path,
            size=size,
            rd_init_mode_filter=rd_init_mode_filter,
        )
        if not file_paths:
            raise FileNotFoundError(
                f"No reaction_diffusion training HDF5 files found under {data_path}. "
                "Expected files named reaction_diffusion_*.h5."
            )
        if mixed_init_modes and rd_init_mode_filter is None:
            warnings.warn(
                "Multiple reaction-diffusion init modes were detected in the training directory; "
                f"detected_init_modes={detected_init_modes}. Pass rd_init_mode_filter='grf' or 'iid' "
                "to avoid mixing GRF and IID training files.",
                RuntimeWarning,
                stacklevel=2,
            )

        param_chunks = {}
        param_sources = {}
        init_modes = []
        boundary_conditions = []
        sample_seeds = []
        sample_start = 0
        for file_path in file_paths:
            with h5py.File(file_path, "r") as f:
                sample_keys = sorted(
                    key
                    for key in f.keys()
                    if isinstance(f[key], h5py.Group) and "data" in f[key]
                )
                for local_idx, k in enumerate(tqdm(sample_keys)):
                    if max_samples is not None and sample_count >= max_samples:
                        break
                    arr = f[k]['data'] # type: ignore
                    u0 = np.expand_dims(arr[0, :, :, 0], axis=0)
                    v0 = np.expand_dims(arr[0, :, :, 1], axis=0)
                    u = np.expand_dims(arr[-1, :, :, 0], axis=0)
                    v = np.expand_dims(arr[-1, :, :, 1], axis=0)
                    dataset.append(np.stack([u0, v0, u, v], axis=1))
                    params, sources, extra = self._reaction_diffusion_sample_metadata(f, f[k], local_idx)
                    for name, value in params.items():
                        param_chunks.setdefault(name, []).append(value)
                        if name not in param_sources:
                            param_sources[name] = sources.get(name, "unknown")
                        elif param_sources[name] != sources.get(name, "unknown"):
                            param_sources[name] = "mixed"
                    if "init_mode" in extra:
                        init_modes.append(extra["init_mode"])
                    if "boundary_condition" in extra:
                        boundary_conditions.append(extra["boundary_condition"])
                    if "sample_seed" in extra:
                        sample_seeds.append(extra["sample_seed"])
                    sample_count += 1
                self.pde_param_slices.append(
                    {
                        "file_path": str(file_path),
                        "start": sample_start,
                        "stop": sample_count,
                        "params": tuple(sorted(param_chunks)),
                    }
                )
                sample_start = sample_count
            if max_samples is not None and sample_count >= max_samples:
                break

        if not dataset:
            raise FileNotFoundError(
                f"Reaction-diffusion files were found but no samples were loaded from {data_path}; "
                "check that each HDF5 file contains sample groups with a data dataset."
            )

        data, label = self._finalize(np.concatenate(dataset, axis=0))
        for name, values in param_chunks.items():
            if len(values) != len(data):
                raise ValueError(f"Reaction-diffusion parameter {name!r} was present for only part of the loaded samples")
            self.pde_params[name] = torch.tensor(values, dtype=torch.float32)
        self.pde_param_sources = param_sources
        self.extra_metadata = {
            "selected_file_format": "reaction_diffusion_h5",
            "selected_files": [str(path) for path in file_paths],
            "file_paths": [str(path) for path in file_paths],
            "num_loaded_samples": int(len(data)),
            "rd_init_mode_filter": rd_init_mode_filter,
            "detected_init_modes": detected_init_modes,
            "mixed_init_modes": bool(mixed_init_modes),
        }
        if init_modes:
            self.extra_metadata["init_mode"] = init_modes
        if boundary_conditions:
            self.extra_metadata["boundary_condition"] = boundary_conditions
            self.extra_metadata["boundary_condition_kind"] = "neumann"
        if sample_seeds:
            self.extra_metadata["sample_seed"] = sample_seeds
        return data, label

    def _reaction_diffusion_paths(
        self,
        data_path,
        size=DEFAULT_TRAIN_SHARDS,
        rd_init_mode_filter=None,
    ):
        if rd_init_mode_filter not in {None, "grf", "iid"}:
            raise ValueError("rd_init_mode_filter must be None, 'grf', or 'iid'")
        path = Path(data_path).expanduser()
        if path.is_file():
            if not path.name.startswith("reaction_diffusion_"):
                raise FileNotFoundError(
                    f"Reaction-diffusion files must use the reaction_diffusion_*.h5 naming scheme: {path}"
                )
            init_mode = self._reaction_diffusion_init_mode_from_name(path)
            detected = [init_mode] if init_mode else []
            if rd_init_mode_filter is not None and init_mode != rd_init_mode_filter:
                raise FileNotFoundError(
                    f"{path} has reaction-diffusion init_mode={init_mode!r}, "
                    f"which does not match rd_init_mode_filter={rd_init_mode_filter!r}"
                )
            return [path], detected, False

        pde_dirs = []
        for candidate in (path, self._pde_dir(data_path)):
            if candidate not in pde_dirs:
                pde_dirs.append(candidate)

        found_new_paths = []
        detected_init_modes: set[str] = set()
        mixed_init_modes = False
        for pde_dir in pde_dirs:
            all_new_paths = sorted(
                file_path
                for file_path in pde_dir.glob("reaction_diffusion_*.h5")
                if not file_path.name.startswith("reaction_diffusion_test_")
            )
            detected_init_modes = {
                mode
                for file_path in all_new_paths
                if (mode := self._reaction_diffusion_init_mode_from_name(file_path)) is not None
            }
            mixed_init_modes = len(detected_init_modes) > 1
            if rd_init_mode_filter is None:
                new_paths = all_new_paths
            else:
                new_paths = [
                    file_path
                    for file_path in all_new_paths
                    if self._reaction_diffusion_init_mode_from_name(file_path) == rd_init_mode_filter
                ]
            if new_paths:
                if size is not None:
                    new_paths = new_paths[: int(size)]
                found_new_paths = new_paths
                break

        if found_new_paths:
            return found_new_paths, sorted(detected_init_modes), mixed_init_modes
        return [], sorted(detected_init_modes), mixed_init_modes

    @staticmethod
    def _reaction_diffusion_init_mode_from_name(path):
        name = Path(path).name
        prefix = "reaction_diffusion_"
        if not name.startswith(prefix) or name.startswith("reaction_diffusion_test_"):
            return None
        mode = name[len(prefix) :].split("_", 1)[0]
        return mode if mode in {"grf", "iid"} else None

    def _reaction_diffusion_sample_metadata(self, file, group, sample_index):
        params = {}
        sources = {}
        extra = {}
        specs = {
            "T": ("T", "total_time"),
            "D_u": ("D_u", "Du"),
            "D_v": ("D_v", "Dv"),
            "k": ("k",),
            "init_mean": ("init_mean",),
            "init_std": ("init_std",),
            "n_save_steps": ("n_save_steps",),
            "tdim": ("tdim",),
            "x_left": ("x_left",),
            "x_right": ("x_right",),
            "y_bottom": ("y_bottom",),
            "y_top": ("y_top",),
            "dx": ("dx",),
            "dy": ("dy",),
        }
        for canonical, aliases in specs.items():
            value, source = self._rd_attr_value(file, group, aliases)
            if value is not None:
                params[canonical] = float(np.asarray(value, dtype=np.float32).reshape(-1)[0])
                sources[canonical] = source
        for range_name, left_name, right_name in (
            ("x_range", "x_left", "x_right"),
            ("y_range", "y_bottom", "y_top"),
        ):
            value, source = self._rd_attr_value(file, group, (range_name,))
            if value is None:
                continue
            values = np.asarray(value, dtype=np.float32).reshape(-1)
            if values.size >= 2:
                if left_name not in params:
                    params[left_name] = float(values[0])
                    sources[left_name] = source
                if right_name not in params:
                    params[right_name] = float(values[1])
                    sources[right_name] = source
        init_mode, _source = self._rd_attr_value(file, group, ("init_mode",))
        if init_mode is not None:
            extra["init_mode"] = self._decode_attr(init_mode)
        bc_value, bc_source = self._rd_attr_value(file, group, ("boundary_condition_kind", "boundary_condition", "bc"))
        extra["boundary_condition"] = self._decode_attr(bc_value) if bc_value is not None else "homogeneous_neumann"
        extra["boundary_condition_source"] = bc_source or "generator_default:homogeneous_neumann"
        seed_value, seed_source = self._rd_attr_value(file, group, ("sample_seed", "seed"))
        if seed_value is None and "sample_seed" in file:
            seed_data = np.asarray(file["sample_seed"][:])
            if sample_index < seed_data.shape[0]:
                seed_value = seed_data[sample_index]
                seed_source = "dataset:sample_seed"
        if seed_value is not None:
            seed_int = int(np.asarray(seed_value).reshape(-1)[0])
            params["sample_seed"] = float(seed_int)
            sources["sample_seed"] = seed_source or "unknown"
            extra["sample_seed"] = seed_int
        return params, sources, extra

    @staticmethod
    def _rd_attr_value(file, group, names):
        for name in names:
            if name in group.attrs:
                return group.attrs[name], f"group_attr:{name}"
        for name in names:
            if name in file.attrs:
                return file.attrs[name], f"root_attr:{name}"
        return None, ""

    @staticmethod
    def _decode_attr(value):
        if isinstance(value, bytes):
            return value.decode("utf-8")
        arr = np.asarray(value)
        if arr.ndim == 0:
            scalar = arr.item()
            return scalar.decode("utf-8") if isinstance(scalar, bytes) else str(scalar)
        return [item.decode("utf-8") if isinstance(item, bytes) else str(item) for item in arr.reshape(-1)]

    def _shallow_water_load(self, data_path, size=DEFAULT_TRAIN_SHARDS, max_samples=None):
        dataset = []
        remaining = None if max_samples is None else int(max_samples)
        self.pde_params = {}
        self.pde_param_sources = {}
        self.pde_param_slices = []
        self.extra_metadata = {"boundary_condition": "open/extrapolation", "boundary_condition_kind": "open"}
        param_chunks = {}
        param_sources = {}
        selected_files = []
        sample_start = 0
        sample_count = 0

        for i in self._shard_indices(data_path, size, start=0):
            file_path = self._resolve_data_file(data_path, f"2d_swe_128_128_10_{i}.h5")
            file_param_names = set()

            with h5py.File(file_path, "r") as f:
                selected_files.append(str(file_path))
                sample_keys = [k for k in f.keys() if isinstance(f[k], h5py.Group)]
                for k in sample_keys:
                    if remaining is not None and remaining <= 0:
                        break
                    group = f[k]
                    h0 = np.expand_dims(group['data']['h'][0, :, :, 0], axis=0)  # type: ignore
                    h = np.expand_dims(group['data']['h'][-1, :, :, 0], axis=0)  # type: ignore
                    hu0 = np.expand_dims(group['data']['hu'][0, :, :, 0], axis=0)  # type: ignore
                    hu = np.expand_dims(group['data']['hu'][-1, :, :, 0], axis=0)  # type: ignore
                    hv0 = np.expand_dims(group['data']['hv'][0, :, :, 0], axis=0)  # type: ignore
                    hv = np.expand_dims(group['data']['hv'][-1, :, :, 0], axis=0)  # type: ignore
                    dataset.append(np.stack([h0, hu0, hv0, h, hu, hv], axis=1))
                    params, sources, _extra = self._shallow_water_sample_metadata(f, group)
                    for name, value in params.items():
                        param_chunks.setdefault(name, []).append(float(value))
                        file_param_names.add(name)
                        if name not in param_sources:
                            param_sources[name] = sources.get(name, "unknown")
                        elif param_sources[name] != sources.get(name, "unknown"):
                            param_sources[name] = "mixed"
                    sample_count += 1
                    if remaining is not None:
                        remaining -= 1

            file_sample_count = sample_count - sample_start
            if file_sample_count > 0:
                sample_stop = sample_start + file_sample_count
                self.pde_param_slices.append(
                    {
                        "file_path": str(file_path),
                        "start": sample_start,
                        "stop": sample_stop,
                        "params": tuple(sorted(file_param_names)),
                    }
                )
                sample_start = sample_stop
            if remaining is not None and remaining <= 0:
                break

        if not dataset:
            raise ValueError(
                f"No shallow_water samples were loaded from data_path={data_path!r} with size={size}"
            )

        data, label = self._finalize(np.concatenate(dataset, axis=0))
        for name, values in param_chunks.items():
            tensor = np.asarray(values, dtype=np.float32)
            if tensor.shape[0] != len(data):
                raise ValueError(f"Scalar parameter {name!r} was present for only part of the loaded samples")
            self.pde_params[name] = torch.tensor(tensor, dtype=torch.float32)
        self.pde_param_sources = param_sources
        self.extra_metadata.update(
            {
                "selected_file_format": "swe",
                "selected_files": selected_files,
                "file_paths": selected_files,
                "num_loaded_samples": int(len(data)),
                "residual_family": self.spec.residual_family,
                "channel_names": list(self.spec.channel_names),
            }
        )
        return data, label

    @staticmethod
    def _as_scalar_float(value):
        values = np.asarray(value, dtype=np.float32).reshape(-1)
        if values.size != 1:
            raise ValueError(f"Expected scalar value for SWE metadata, got shape {values.shape}")
        return float(values[0])

    def _shallow_water_sample_metadata(self, file, group):
        params = {}
        sources = {}
        extra = {}

        t_value, t_source = self._rd_attr_value(file, group, ("T", "total_time"))
        if t_value is not None:
            t_float = self._as_scalar_float(t_value)
            params["T"] = t_float
            params["total_time"] = t_float
            sources["T"] = t_source
            sources["total_time"] = t_source

        g_value, g_source = self._rd_attr_value(file, group, ("grav", "g"))
        if g_value is not None:
            params["g"] = self._as_scalar_float(g_value)
            sources["g"] = g_source

        dt_value = None
        dt_source = ""
        if "grid" in group and "t" in group["grid"]:
            t_grid = np.asarray(group["grid"]["t"], dtype=np.float32).reshape(-1)
            if t_grid.size >= 2:
                dt_value = float(t_grid[1] - t_grid[0])
                dt_source = "group_dataset:grid/t"
        if dt_value is None and "tsteps" in file.attrs and "T" in params:
            tsteps = int(np.asarray(file.attrs["tsteps"]).reshape(-1)[0])
            if tsteps > 0:
                dt_value = params["T"] / float(tsteps)
                dt_source = "root_attr:tsteps"
        if dt_value is None and "n_time" in file.attrs and "T" in params:
            n_time = int(np.asarray(file.attrs["n_time"]).reshape(-1)[0])
            if n_time > 1:
                dt_value = params["T"] / float(n_time - 1)
                dt_source = "root_attr:n_time"

        if dt_value is not None:
            params["dt"] = dt_value
            sources["dt"] = dt_source

        sample_seed = group.attrs.get("seed", None)
        if sample_seed is not None:
            extra["sample_seed"] = int(self._as_scalar_float(sample_seed))

        return params, sources, extra

    def _heat_load(self, data_path, size=DEFAULT_TRAIN_SHARDS, split="train", max_samples=None):
        return self._pair_h5_load(data_path, size=size, split=split, max_samples=max_samples)

    def _wave_load(self, data_path, size=DEFAULT_TRAIN_SHARDS, split="train", max_samples=None):
        return self._pair_h5_load(data_path, size=size, split=split, max_samples=max_samples)

    def _advection_diffusion_load(self, data_path, size=DEFAULT_TRAIN_SHARDS, split="train", max_samples=None):
        return self._pair_h5_load(data_path, size=size, split=split, max_samples=max_samples)

    def _steady_heat_conduction_load(self, data_path, size=DEFAULT_TRAIN_SHARDS, split="train", max_samples=None):
        return self._pair_h5_load(data_path, size=size, split=split, max_samples=max_samples)

    @staticmethod
    def _nsnonbounded_pair(w0, wt):
        w0_nhw = PDEloader._as_nhw(np.asarray(w0, dtype=np.float32), "w0")
        wt_final = PDEloader._select_final_time(np.asarray(wt, dtype=np.float32), "w")
        wt_nhw = PDEloader._as_nhw(wt_final, "wT")
        if w0_nhw.shape != wt_nhw.shape:
            raise ValueError(f"NS w0/wT shape mismatch after conversion: {w0_nhw.shape} vs {wt_nhw.shape}")
        return np.stack([w0_nhw, wt_nhw], axis=1).astype(np.float32, copy=False)

    @staticmethod
    def _as_nhw(arr, name):
        if arr.ndim == 2:
            return arr[None, :, :]
        if arr.ndim != 3:
            raise ValueError(f"{name} must be convertible to [N,H,W], got {arr.shape}")
        if arr.shape[-2] == arr.shape[-1]:
            return arr
        if arr.shape[0] == arr.shape[1]:
            return np.transpose(arr, (2, 0, 1))
        if arr.shape[0] == arr.shape[2]:
            return np.transpose(arr, (1, 0, 2))
        raise ValueError(f"Cannot infer [N,H,W] layout for {name} with shape {arr.shape}")

    @staticmethod
    def _select_final_time(arr, name):
        if arr.ndim == 3:
            return arr
        if arr.ndim != 4:
            raise ValueError(f"{name} must be [N,H,W,T] or similar, got {arr.shape}")
        if arr.shape[1] == arr.shape[2]:
            return arr[:, :, :, -1]
        if arr.shape[2] == arr.shape[3]:
            if arr.shape[1] <= 64:
                return arr[:, -1, :, :]
            if arr.shape[0] <= 64:
                return arr[-1, :, :, :]
        if arr.shape[0] == arr.shape[1]:
            if arr.shape[2] <= 64:
                return arr[:, :, -1, :]
            if arr.shape[3] <= 64:
                return arr[:, :, :, -1]
        raise ValueError(f"Cannot infer final-time slice for {name} with shape {arr.shape}")

    def _pair_h5_load(self, data_path, size=DEFAULT_TRAIN_SHARDS, split="train", materialize_params=False, max_samples=None):
        """Load endpoint-pair HDF5 data as model channels plus scalar PDE metadata.

        Spatially constant PDE parameters are not Flow Matching input channels by default.
        They are cached in ``self.pde_params`` for PDE residual evaluation. Set
        ``materialize_params=True`` only for checkpoints that explicitly trained
        with scalar constants expanded into constant fields.
        """
        file_paths = self._pair_h5_paths(data_path, size=size, split=split)
        dataset = []
        param_chunks = {}
        param_sources = {}
        self.pde_params = {}
        self.pde_param_sources = {}
        self.pde_param_slices = []
        sample_ids = []
        sample_seeds = []
        seen_sample_ids = set()
        seen_sample_seeds = set()
        sample_start = 0
        for file_path in tqdm(file_paths):
            with h5py.File(file_path, "r") as file:
                self._validate_pair_h5_split(file, file_path, split)
                if "input_data" not in file or "output_data" not in file:
                    raise KeyError(f"{file_path} must contain data or input_data/output_data")
                remaining = None if max_samples is None else max_samples - sample_start
                if remaining is not None and remaining <= 0:
                    break
                arr = self._pair_h5_materialize(file, materialize_params=materialize_params, max_samples=remaining)
                params, sources = self._pair_h5_scalar_params(file, arr.shape[0])
                file_ids, file_seeds = self._pair_h5_sample_identities(file, file_path, arr.shape[0])
                if file_seeds is not None:
                    duplicate_seeds = sorted(set(file_seeds).intersection(seen_sample_seeds))
                    if duplicate_seeds:
                        raise ValueError(f"Duplicate pair_h5 sample_seed values detected: {duplicate_seeds[:5]}")
                    seen_sample_seeds.update(file_seeds)
                    sample_seeds.extend(file_seeds)
                duplicates = sorted(set(file_ids).intersection(seen_sample_ids))
                if duplicates:
                    raise ValueError(f"Duplicate pair_h5 sample_id values detected: {duplicates[:5]}")
                seen_sample_ids.update(file_ids)
                sample_ids.extend(file_ids)
            arr = np.asarray(arr, dtype=np.float32)
            if arr.ndim != 4:
                raise ValueError(f"{file_path} data must be [N,C,H,W], got {arr.shape}")
            if arr.shape[2] != arr.shape[3]:
                raise ValueError(f"{file_path} data must have square spatial grid, got {arr.shape[2:]}")
            if arr.shape[1] < 1:
                raise ValueError(f"{file_path} data must have at least one channel, got {arr.shape}")
            dataset.append(arr)
            sample_stop = sample_start + arr.shape[0]
            for name, values in params.items():
                param_chunks.setdefault(name, []).append(values)
                if name not in param_sources:
                    param_sources[name] = sources.get(name, "unknown")
                elif param_sources[name] != sources.get(name, "unknown"):
                    param_sources[name] = "mixed"
            self.pde_param_slices.append(
                {"file_path": str(file_path), "start": sample_start, "stop": sample_stop, "params": tuple(params)}
            )
            sample_start = sample_stop

        data, label = self._finalize(np.concatenate(dataset, axis=0))
        for name, chunks in param_chunks.items():
            values = np.concatenate(chunks, axis=0)
            if values.shape[0] != len(data):
                raise ValueError(f"Scalar parameter {name!r} was present for only part of the loaded samples")
            self.pde_params[name] = torch.tensor(values, dtype=torch.float32)
        self.pde_param_sources = param_sources
        self.extra_metadata.update(
            {
                "selected_file_format": "pair_h5",
                "selected_files": [str(path) for path in file_paths],
                "file_paths": [str(path) for path in file_paths],
                "num_loaded_samples": int(len(data)),
                "split": split,
                "sample_id": sample_ids,
                "residual_family": self.spec.residual_family,
                "channel_names": list(self.spec.channel_names),
            }
        )
        if sample_seeds:
            self.extra_metadata["sample_seed"] = sample_seeds
        return data, label

    @staticmethod
    def _validate_pair_h5_split(file, file_path, requested_split):
        stored = file.attrs.get("split")
        if stored is None:
            raise ValueError(f"{file_path} is missing required root HDF5 attr split={requested_split!r}")
        if isinstance(stored, bytes):
            stored = stored.decode("utf-8")
        if str(stored) != str(requested_split):
            raise ValueError(
                f"{file_path} root split={stored!r} does not match requested split={requested_split!r}"
            )

    @staticmethod
    def _pair_h5_sample_identities(file, file_path, n_samples):
        sample_ids = None
        for name in ("sample_id", "sample_ids"):
            if name in file:
                dataset = file[name]
                values = np.asarray(dataset[()] if dataset.shape == () else dataset[:n_samples]).reshape(-1)
                sample_ids = [str(value.decode("utf-8") if isinstance(value, bytes) else value) for value in values]
                break
        sample_seeds = None
        for name in ("sample_seed", "sample_seeds", "seed"):
            if name in file:
                dataset = file[name]
                values = np.asarray(dataset[()] if dataset.shape == () else dataset[:n_samples]).reshape(-1)
                sample_seeds = [int(value) for value in values]
                break
        if sample_ids is None:
            if sample_seeds is not None:
                sample_ids = [f"seed:{seed}" for seed in sample_seeds]
            else:
                resolved = str(Path(file_path).resolve())
                sample_ids = [f"{resolved}#{index}" for index in range(int(n_samples))]
        if len(sample_ids) != int(n_samples):
            raise ValueError(f"sample_id count must be {n_samples}, got {len(sample_ids)} in {file_path}")
        if sample_seeds is not None and len(sample_seeds) != int(n_samples):
            raise ValueError(f"sample_seed count must be {n_samples}, got {len(sample_seeds)} in {file_path}")
        if sample_seeds is not None and len(set(sample_seeds)) != len(sample_seeds):
            raise ValueError(f"Duplicate sample_seed values within {file_path}")
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError(f"Duplicate sample_id values within {file_path}")
        return sample_ids, sample_seeds

    def _pair_h5_materialize(self, file, materialize_params=False, max_samples=None):
        n_take = file["input_data"].shape[0] if max_samples is None else min(int(max_samples), file["input_data"].shape[0])
        input_data = np.asarray(file["input_data"][:n_take], dtype=np.float32)
        output_data = np.asarray(file["output_data"][:n_take], dtype=np.float32)
        if input_data.ndim != 4 or output_data.ndim != 4:
            raise ValueError(f"pair_h5 input/output must be [N,C,H,W], got {input_data.shape}, {output_data.shape}")
        if materialize_params:
            warnings.warn(
                "pair_h5 materialize_params=True is ignored. "
                "Scalar PDE parameters are returned as sample-level metadata, not FM channels.",
                RuntimeWarning,
                stacklevel=2,
            )
        return np.concatenate([input_data, output_data], axis=1)

    def _pair_h5_scalar_params(self, file, n_samples):
        params = {}
        sources = {}
        optional = self.spec.optional_scalar_param_names
        for name in self.spec.scalar_param_names:
            storage_name = self._resolve_param_storage_name(file, name)
            if storage_name is not None:
                params[name] = self._read_scalar_dataset_or_attr(file, storage_name, n_samples)
                sources[name] = "dataset" if storage_name in file else f"attrs:{storage_name}"
            elif name not in optional:
                raise KeyError(f"Missing scalar dataset or attr {name!r}")
        return params, sources

    @staticmethod
    def _expand_scalar_to_field(values, h, w):
        values = np.asarray(values, dtype=np.float32).reshape(-1, 1, 1, 1)
        return np.broadcast_to(values, (values.shape[0], 1, h, w)).copy()

    @staticmethod
    def _has_scalar_dataset_or_attr(file, name):
        return name in file or name in file.attrs

    def _resolve_param_storage_name(self, file, name):
        for candidate in self.spec.param_aliases.get(name, (name,)):
            if candidate in file or candidate in file.attrs:
                return candidate
        return None

    @staticmethod
    def _read_scalar_dataset_or_attr(file, name, n_samples, default=None):
        if name in file:
            dataset = file[name]
            values = np.asarray(dataset[()] if dataset.shape == () else dataset[:], dtype=np.float32)
        elif name in file.attrs:
            values = np.asarray(file.attrs[name], dtype=np.float32)
        elif default is not None:
            values = np.asarray(default, dtype=np.float32)
        else:
            raise KeyError(f"Missing scalar dataset or attr {name!r}")
        if values.ndim == 0:
            values = np.full((n_samples,), float(values), dtype=np.float32)
        elif values.shape[0] == 1 and n_samples > 1:
            values = np.repeat(values, n_samples, axis=0)
        elif values.shape[0] > n_samples:
            values = values[:n_samples]
        if values.shape[0] != n_samples:
            raise ValueError(f"{name} must have shape [{n_samples}], got {values.shape}")
        return values

    def _pair_h5_paths(self, data_path, size=5, split="train"):
        path = Path(data_path)
        if path.is_file():
            name = path.name
            if split == "train" and ("_test_" in name or "_val_" in name):
                raise ValueError(f"Training input must be a train shard, got {name!r}")
            if split == "test" and "_test_" not in name:
                raise ValueError(f"Test input filename must contain '_test_', got {name!r}")
            if split == "val" and "_val_" not in name:
                raise ValueError(f"Validation input filename must contain '_val_', got {name!r}")
            return [path]

        pde_dir = path / self.pde
        if not pde_dir.exists():
            pde_dir = path
        if not pde_dir.exists():
            raise FileNotFoundError(f"Pair HDF5 data directory does not exist: {pde_dir}")

        if split == "test":
            file_paths = sorted(pde_dir.glob(f"{self.pde}_test_*-*-*.h5"))
        elif split == "val":
            file_paths = sorted(pde_dir.glob(f"{self.pde}_val_*-*-*.h5"))
        elif split == "train":
            file_paths = sorted(
                (
                    p
                    for p in pde_dir.glob(f"{self.pde}_*-*-*_[0-9]*.h5")
                    if "_test_" not in p.name and "_val_" not in p.name
                ),
                key=self._pair_h5_sort_key,
            )
        else:
            raise ValueError(f"Unsupported split={split!r}; expected 'train', 'val', or 'test'")

        if size is not None and split == "train":
            file_paths = file_paths[:size]
        if not file_paths:
            raise FileNotFoundError(f"No {split} HDF5 files found for {self.pde} under {pde_dir}")
        return file_paths

    @staticmethod
    def _pair_h5_sort_key(path):
        stem = path.stem
        shard = stem.rsplit("_", 1)[-1]
        return int(shard) if shard.isdigit() else stem

    def _channel_names_for_loaded_data(self):
        return list(self.spec.channel_names)
