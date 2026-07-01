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

    def _legacy_path(self, data_path, file_name):
        path = Path(data_path).expanduser()
        if path.is_file():
            return path
        return self._pde_dir(data_path) / file_name

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
        for i in tqdm(range(1, size + 1)):
            file_path = self._legacy_path(data_path, f"{self.pde}_10000-128-128_{i}.mat")
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
        for i in tqdm(range(1, size + 1)):
            file_path = self._legacy_path(data_path, f"{self.pde}_10000-128-128_{i}.mat")
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
        for i in tqdm(range(1, size + 1)):
            file_path = self._legacy_path(data_path, f"{self.pde}_10000-128-128_{i}.mat")
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
        for i in tqdm(range(1, size + 1)):
            file_path = self._legacy_path(data_path, f"{self.pde}_10000-128-128-10_{i}_new.mat")
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
        for i in tqdm(range(1, size + 1)):
            file_path = self._legacy_path(data_path, f"{self.pde}_10000-128-128_{i}.mat")
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
        legacy_rd_files=False,
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
            selected_format,
            candidate_formats,
            detected_init_modes,
            mixed_init_modes,
        ) = self._reaction_diffusion_paths(
            data_path,
            size=size,
            legacy_rd_files=legacy_rd_files,
            rd_init_mode_filter=rd_init_mode_filter,
        )
        if not file_paths:
            raise FileNotFoundError(
                f"No reaction_diffusion training HDF5 files found under {data_path}. "
                "Expected new files named reaction_diffusion_*.h5; old "
                "reaction_diffusion-128-128-* files are ignored unless legacy_rd_files=True."
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
            "selected_file_format": selected_format,
            "candidate_file_formats": sorted(candidate_formats),
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
        legacy_rd_files=False,
        rd_init_mode_filter=None,
    ):
        if rd_init_mode_filter not in {None, "grf", "iid"}:
            raise ValueError("rd_init_mode_filter must be None, 'grf', or 'iid'")
        path = Path(data_path).expanduser()
        if path.is_file():
            selected_format = self._reaction_diffusion_file_format(path)
            if selected_format == "legacy_rd" and not legacy_rd_files:
                raise FileNotFoundError(
                    f"{path} matches the legacy reaction-diffusion file naming scheme; "
                    "pass legacy_rd_files=True to read legacy RD files explicitly."
                )
            init_mode = self._reaction_diffusion_init_mode_from_name(path)
            detected = [init_mode] if init_mode else []
            if selected_format == "new_gen_rd" and rd_init_mode_filter is not None and init_mode != rd_init_mode_filter:
                raise FileNotFoundError(
                    f"{path} has reaction-diffusion init_mode={init_mode!r}, "
                    f"which does not match rd_init_mode_filter={rd_init_mode_filter!r}"
                )
            return [path], selected_format, {selected_format}, detected, False

        pde_dirs = []
        for candidate in (path, self._pde_dir(data_path)):
            if candidate not in pde_dirs:
                pde_dirs.append(candidate)

        candidate_formats = set()
        legacy_paths = []
        for pde_dir in pde_dirs:
            legacy_paths = self._legacy_reaction_diffusion_paths(pde_dir, size)
            if legacy_paths:
                candidate_formats.add("legacy_rd")
                break

        found_new_paths = []
        detected_init_modes: set[str] = set()
        mixed_init_modes = False
        for pde_dir in pde_dirs:
            all_new_paths = sorted(
                file_path
                for file_path in pde_dir.glob("reaction_diffusion_*.h5")
                if not file_path.name.startswith("reaction_diffusion_test_")
            )
            if all_new_paths:
                candidate_formats.add("new_gen_rd")
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

        if legacy_rd_files and legacy_paths:
            return legacy_paths, "legacy_rd", candidate_formats, sorted(detected_init_modes), mixed_init_modes
        if found_new_paths:
            return found_new_paths, "new_gen_rd", candidate_formats, sorted(detected_init_modes), mixed_init_modes
        return [], "", candidate_formats, sorted(detected_init_modes), mixed_init_modes

    @staticmethod
    def _reaction_diffusion_file_format(path):
        name = path.name
        if name.startswith(("reaction_diffusion-128-128-10_", "reaction_diffusion-128-128-100_")):
            return "legacy_rd"
        return "new_gen_rd"

    @staticmethod
    def _reaction_diffusion_init_mode_from_name(path):
        name = Path(path).name
        prefix = "reaction_diffusion_"
        if not name.startswith(prefix) or name.startswith("reaction_diffusion_test_"):
            return None
        mode = name[len(prefix) :].split("_", 1)[0]
        return mode if mode in {"grf", "iid"} else None

    @staticmethod
    def _legacy_reaction_diffusion_paths(pde_dir, size):
        paths = []
        for i in range(int(size or 0)):
            for file_name in (
                f"reaction_diffusion-128-128-10_{i}.h5",
                f"reaction_diffusion-128-128-100_{i}.h5",
            ):
                file_path = pde_dir / file_name
                if file_path.exists():
                    paths.append(file_path)
                    break
        if paths:
            return paths
        globbed = sorted(pde_dir.glob("reaction_diffusion-128-128-*.h5"))
        if size is not None:
            globbed = globbed[: int(size)]
        return globbed

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
        sample_count = 0
        self.extra_metadata = {"boundary_condition": "open/extrapolation", "boundary_condition_kind": "open"}
        for i in range(size):
            file_path = self._legacy_path(data_path, f"2d_swe_128_128_10_{i}.h5")

            with h5py.File(file_path, "r") as f:
                for k in list(f.keys()):
                    if max_samples is not None and sample_count >= max_samples:
                        break
                    h0 = np.expand_dims(f[k]['data']['h'][0, :, :, 0], axis = 0) # type: ignore
                    h = np.expand_dims(f[k]['data']['h'][-1, :, :, 0], axis = 0) # type: ignore
                    hu0 = np.expand_dims(f[k]['data']['hu'][0, :, :, 0], axis = 0) # type: ignore
                    hu = np.expand_dims(f[k]['data']['hu'][-1, :, :, 0], axis = 0) # type: ignore
                    hv0 = np.expand_dims(f[k]['data']['hv'][0, :, :, 0], axis = 0) # type: ignore
                    hv = np.expand_dims(f[k]['data']['hv'][-1, :, :, 0], axis = 0) # type: ignore
                    dataset.append(np.stack([h0, hu0, hv0, h, hu, hv], axis=1))
                    sample_count += 1
            if max_samples is not None and sample_count >= max_samples:
                break

        return self._finalize(np.concatenate(dataset, axis=0))

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
        sample_start = 0
        for file_path in tqdm(file_paths):
            with h5py.File(file_path, "r") as file:
                if "input_data" not in file or "output_data" not in file:
                    raise KeyError(f"{file_path} must contain data or input_data/output_data")
                remaining = None if max_samples is None else max_samples - sample_start
                if remaining is not None and remaining <= 0:
                    break
                arr = self._pair_h5_materialize(file, materialize_params=materialize_params, max_samples=remaining)
                params, sources = self._pair_h5_scalar_params(file, arr.shape[0])
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
                "residual_family": self.spec.residual_family,
                "channel_names": list(self.spec.channel_names),
            }
        )
        return data, label

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
                (p for p in pde_dir.glob(f"{self.pde}_*-*-*_[0-9]*.h5") if "_test_" not in p.name),
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
