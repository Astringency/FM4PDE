import h5py
import scipy.io
from tqdm import tqdm
import numpy as np
from pathlib import Path
import warnings

import torch
from torch.utils.data import Dataset


DEFAULT_TRAIN_SHARDS = 5


class TensorDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels.to(torch.long)
        self.max_size = data.shape[0]
        self.num_channels = self.data.shape[1]
        self.resolution = self.data.shape[2]
        self.label_dim = 1

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)


class PDEloader:
    FUTURE_H5_SCALAR_PARAMS = {
        "heat": ("alpha", "T", "total_time", "dt"),
        "wave": ("c", "T", "total_time", "dt"),
        "advection_diffusion": ("b_x", "b_y", "kappa", "T", "total_time", "dt"),
        "steady_heat_conduction": ("u_D",),
    }
    OPTIONAL_FUTURE_H5_SCALAR_PARAMS = {
        "heat": {"alpha", "T", "total_time", "dt"},
        "wave": {"c", "T", "total_time", "dt"},
        "advection_diffusion": {"T", "total_time", "dt"},
    }
    FUTURE_H5_PARAM_ALIASES = {
        "alpha": ("alpha", "fixed_alpha"),
        "c": ("c", "fixed_c"),
    }

    def __init__(self, pde: str):
        self.pde = pde.lower()
        # Sample-aligned scalar PDE parameters cached by future HDF5 loads.
        self.pde_params = {}
        self.pde_param_sources = {}
        self.pde_param_slices = []
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
        return {
            "pde": self.pde,
            "pde_params": self.pde_params,
            "pde_params_keys": sorted(self.pde_params),
            "pde_param_sources": dict(self.pde_param_sources),
            "pde_param_slices": list(self.pde_param_slices),
            "channel_names": self._channel_names_for_loaded_data(),
            "scalar_params_loaded": bool(self.pde_params),
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

    def _finalize(self, data, label_value):
        data = torch.as_tensor(data, dtype=torch.float32).contiguous()
        self._assert_bchw(data, context=self.pde)
        label = torch.full((int(data.shape[0]),), int(label_value), dtype=torch.long)
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

        return self._finalize(np.concatenate(dataset, axis=0), 0)

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

        return self._finalize(np.concatenate(dataset, axis=0), 1)
    
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

        return self._finalize(np.concatenate(dataset, axis=0), 2)

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
        
        return self._finalize(np.concatenate(dataset, axis=0), 3)
    
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
        
        return self._finalize(np.concatenate(dataset, axis=0), 4)

    def _reaction_diffusion_load(self, data_path, size=DEFAULT_TRAIN_SHARDS, max_samples=None):
        dataset = []
        sample_count = 0
        path = Path(data_path).expanduser()
        if path.is_file():
            file_paths = [path]
        else:
            pde_dirs = []
            for candidate in (path, self._pde_dir(data_path)):
                if candidate not in pde_dirs:
                    pde_dirs.append(candidate)
            file_paths = []
            for pde_dir in pde_dirs:
                for i in range(size):
                    for file_name in (
                        f"reaction_diffusion-128-128-10_{i}.h5",
                        f"reaction_diffusion-128-128-100_{i}.h5",
                    ):
                        file_path = pde_dir / file_name
                        if file_path.exists():
                            file_paths.append(file_path)
                            break
                if file_paths:
                    break
            if not file_paths:
                for pde_dir in pde_dirs:
                    file_paths = sorted(
                        file_path
                        for file_path in pde_dir.glob("reaction_diffusion_*.h5")
                        if not file_path.name.startswith("reaction_diffusion_test_")
                    )[:size]
                    if file_paths:
                        break
        if not file_paths:
            raise FileNotFoundError(f"No reaction_diffusion HDF5 files found under {data_path}")

        for file_path in file_paths:
            with h5py.File(file_path, "r") as f:
                sample_keys = sorted(
                    key
                    for key in f.keys()
                    if isinstance(f[key], h5py.Group) and "data" in f[key]
                )
                for k in tqdm(sample_keys):
                    if max_samples is not None and sample_count >= max_samples:
                        break
                    arr = f[k]['data'] # type: ignore
                    u0 = np.expand_dims(arr[0, :, :, 0], axis=0)
                    v0 = np.expand_dims(arr[0, :, :, 1], axis=0)
                    u = np.expand_dims(arr[-1, :, :, 0], axis=0)
                    v = np.expand_dims(arr[-1, :, :, 1], axis=0)
                    dataset.append(np.stack([u0, v0, u, v], axis=1))
                    sample_count += 1
            if max_samples is not None and sample_count >= max_samples:
                break
        
        return self._finalize(np.concatenate(dataset, axis=0), 5)

    def _shallow_water_load(self, data_path, size=DEFAULT_TRAIN_SHARDS, max_samples=None):
        dataset = []
        sample_count = 0
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

        return self._finalize(np.concatenate(dataset, axis=0), 6)

    def _heat_load(self, data_path, size=DEFAULT_TRAIN_SHARDS, split="train", max_samples=None):
        return self._future_h5_load(data_path, size=size, split=split, label_value=7, max_samples=max_samples)

    def _wave_load(self, data_path, size=DEFAULT_TRAIN_SHARDS, split="train", max_samples=None):
        return self._future_h5_load(data_path, size=size, split=split, label_value=8, max_samples=max_samples)

    def _advection_diffusion_load(self, data_path, size=DEFAULT_TRAIN_SHARDS, split="train", max_samples=None):
        return self._future_h5_load(data_path, size=size, split=split, label_value=9, max_samples=max_samples)

    def _steady_heat_conduction_load(self, data_path, size=DEFAULT_TRAIN_SHARDS, split="train", max_samples=None):
        return self._future_h5_load(data_path, size=size, split=split, label_value=10, max_samples=max_samples)

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

    def _future_h5_load(self, data_path, size=DEFAULT_TRAIN_SHARDS, split="train", label_value=0, materialize_params=False, max_samples=None):
        """Load future HDF5 data as model channels plus scalar PDE metadata.

        Spatially constant PDE parameters are not Flow Matching input channels by default.
        They are cached in ``self.pde_params`` for PDE residual evaluation. Set
        ``materialize_params=True`` only for legacy checkpoints that explicitly trained
        with scalar constants expanded into constant fields.
        """
        file_paths = self._future_h5_paths(data_path, size=size, split=split)
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
                arr = self._future_h5_materialize(file, materialize_params=materialize_params, max_samples=remaining)
                params, sources = self._future_h5_scalar_params(file, arr.shape[0])
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

        data, label = self._finalize(np.concatenate(dataset, axis=0), label_value)
        for name, chunks in param_chunks.items():
            values = np.concatenate(chunks, axis=0)
            if values.shape[0] != len(data):
                raise ValueError(f"Scalar parameter {name!r} was present for only part of the loaded samples")
            self.pde_params[name] = torch.tensor(values, dtype=torch.float32)
        self.pde_param_sources = param_sources
        return data, label

    def _future_h5_materialize(self, file, materialize_params=False, max_samples=None):
        n_take = file["input_data"].shape[0] if max_samples is None else min(int(max_samples), file["input_data"].shape[0])
        input_data = np.asarray(file["input_data"][:n_take], dtype=np.float32)
        output_data = np.asarray(file["output_data"][:n_take], dtype=np.float32)
        if input_data.ndim != 4 or output_data.ndim != 4:
            raise ValueError(f"future_h5 input/output must be [N,C,H,W], got {input_data.shape}, {output_data.shape}")
        if materialize_params:
            warnings.warn(
                "future_h5 materialize_params=True is deprecated and ignored. "
                "Scalar PDE parameters are returned as sample-level metadata, not FM channels.",
                DeprecationWarning,
                stacklevel=2,
            )
        return np.concatenate([input_data, output_data], axis=1)

    def _future_h5_scalar_params(self, file, n_samples):
        params = {}
        sources = {}
        optional = self.OPTIONAL_FUTURE_H5_SCALAR_PARAMS.get(self.pde, set())
        for name in self.FUTURE_H5_SCALAR_PARAMS.get(self.pde, ()):
            storage_name = self._resolve_param_storage_name(file, name)
            if storage_name is not None:
                params[name] = self._read_scalar_dataset_or_attr(file, storage_name, n_samples)
                sources[name] = "dataset" if storage_name in file else f"attrs:{storage_name}"
            elif name not in optional:
                raise KeyError(f"Missing scalar dataset or attr {name!r}")
        if self.pde == "steady_heat_conduction":
            for name in ("residual_norm", "picard_iters", "converged", "n_sources", "source_x", "source_y", "source_amp", "source_sigma"):
                storage_name = self._resolve_param_storage_name(file, name)
                if storage_name is not None:
                    params[name] = self._read_scalar_dataset_or_attr(file, storage_name, n_samples)
                    sources[name] = "dataset" if storage_name in file else f"attrs:{storage_name}"
        return params, sources

    @staticmethod
    def _expand_scalar_to_field(values, h, w):
        values = np.asarray(values, dtype=np.float32).reshape(-1, 1, 1, 1)
        return np.broadcast_to(values, (values.shape[0], 1, h, w)).copy()

    @staticmethod
    def _has_scalar_dataset_or_attr(file, name):
        return name in file or name in file.attrs

    def _resolve_param_storage_name(self, file, name):
        for candidate in self.FUTURE_H5_PARAM_ALIASES.get(name, (name,)):
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

    def _future_h5_paths(self, data_path, size=5, split="train"):
        path = Path(data_path)
        if path.is_file():
            return [path]

        pde_dir = path / self.pde
        if not pde_dir.exists():
            pde_dir = path
        if not pde_dir.exists():
            raise FileNotFoundError(f"Future PDE data directory does not exist: {pde_dir}")

        if split == "test":
            file_paths = sorted(pde_dir.glob(f"{self.pde}_test_*-*-*.h5"))
        elif split == "val":
            file_paths = sorted(pde_dir.glob(f"{self.pde}_val_*-*-*.h5"))
        elif split == "train":
            file_paths = sorted(
                (p for p in pde_dir.glob(f"{self.pde}_*-*-*_[0-9]*.h5") if "_test_" not in p.name),
                key=self._future_h5_sort_key,
            )
        else:
            raise ValueError(f"Unsupported split={split!r}; expected 'train', 'val', or 'test'")

        if size is not None and split == "train":
            file_paths = file_paths[:size]
        if not file_paths:
            raise FileNotFoundError(f"No {split} HDF5 files found for {self.pde} under {pde_dir}")
        return file_paths

    @staticmethod
    def _future_h5_sort_key(path):
        stem = path.stem
        shard = stem.rsplit("_", 1)[-1]
        return int(shard) if shard.isdigit() else stem

    def _channel_names_for_loaded_data(self):
        names = {
            "heat": ["u0", "uT"],
            "wave": ["u0", "v0", "uT", "vT"],
            "advection_diffusion": ["u0", "uT"],
            "steady_heat_conduction": ["f", "u"],
        }
        return names.get(self.pde)
