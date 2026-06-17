import h5py
import scipy.io
from tqdm import tqdm
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import Dataset



class TensorDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
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
        "heat": ("alpha",),
        "wave": ("c",),
        "advection_diffusion": ("b_x", "b_y", "kappa"),
        "steady_heat_conduction": ("u_D",),
    }
    OPTIONAL_FUTURE_H5_SCALAR_PARAMS = {"heat", "wave"}

    def __init__(self, pde: str):
        self.pde = pde.lower()
        # Sample-aligned scalar PDE parameters cached by future HDF5 loads.
        self.pde_params = {}
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

        self.load_data = self.load_func[self.pde]

    def _darcy_load(self, data_path, size = 5):
        dataset = {}
        for i in tqdm(range(1, size + 1)):
            file_path = f'{data_path}{self.pde}/{self.pde}_10000-128-128_{i}.mat'
            with h5py.File(file_path, 'r') as file:
                a = file['thresh_a_data'][:] # type: ignore
                u = file['thresh_p_data'][:] # type: ignore
            dataset[i] = np.stack([a, u], axis=0).transpose(3, 0, 1, 2) # type: ignore

        data = torch.tensor(np.concatenate(list(dataset.values()), axis=0)).to(torch.float32)
        label = torch.zeros(len(data), dtype=torch.float32)
        return data, label

    def _poisson_load(self, data_path, size = 5):
        dataset = {}
        for i in tqdm(range(1, size + 1)):
            file_path = f'{data_path}{self.pde}/{self.pde}_10000-128-128_{i}.mat'
            f = scipy.io.loadmat(file_path)['f_data']
            phi = scipy.io.loadmat(file_path)['phi_data']
            dataset[i] = np.stack([f, phi], axis=1)

        data = torch.tensor(np.concatenate(list(dataset.values()), axis=0)).to(torch.float32)
        label = torch.zeros(len(data), dtype=torch.float32) + 1
        return data, label
    
    def _helmholtz_load(self, data_path, size = 5):
        dataset = {}
        for i in tqdm(range(1, size + 1)):
            file_path = f'{data_path}{self.pde}/{self.pde}_10000-128-128_{i}.mat'
            f = scipy.io.loadmat(file_path)['f_data']
            psi = scipy.io.loadmat(file_path)['psi_data']
            
            dataset[i] = np.stack([f, psi], axis=1)

        data = torch.tensor(np.concatenate(list(dataset.values()), axis=0)).to(torch.float32)
        label = torch.zeros(len(data), dtype=torch.float32) + 2
        return data, label

    def _nsnonbounded_load(self, data_path, size = 5):
        dataset = {}
        for i in tqdm(range(1, size + 1)):
            file_path = f'{data_path}{self.pde}/{self.pde}_10000-128-128-10_{i}_new.mat'
            with h5py.File(file_path, 'r') as file:
                w0 = file['w0'][:] # type: ignore
                wt = file['w'][:, :, :, :] # type: ignore

            w0 = np.expand_dims(w0, axis=-1) # type: ignore
            w = np.concatenate([w0, wt], axis=-1) # type: ignore

            dataset[i] = np.stack([w], axis=-1)
        
        data = torch.tensor(np.concatenate(list(dataset.values()), axis=0)).to(torch.float32)
        label = torch.zeros(len(data), dtype=torch.float32) + 3
        return data, label
    
    def _burger_load(self, data_path, size = 5):
        dataset = {}
        for i in tqdm(range(1, size + 1)):
            file_path = f'{data_path}{self.pde}/{self.pde}_10000-128-128_{i}.mat'
            output = scipy.io.loadmat(file_path)['output']
            dataset[i] = np.expand_dims(output, axis = 1)
        
        data = torch.tensor(np.concatenate(list(dataset.values()), axis=0)).to(torch.float32)
        label = torch.zeros(len(data), dtype=torch.float32) + 4
        return data, label

    def _reaction_diffusion_load(self, data_path, size = 5):
        dataset = {}
        for i in range(size):
            file_path = f"{data_path}{self.pde}/reaction_diffusion-128-128-100_{i}.h5"
            with h5py.File(file_path, "r") as f:
                for k in tqdm(list(f.keys())):
                    u0 = np.expand_dims(f[k]['data'][50, :, :, 0], axis = 0) # type: ignore
                    v0 = np.expand_dims(f[k]['data'][50, :, :, 1], axis = 0) # type: ignore
                    u = np.expand_dims(f[k]['data'][-1, :, :, 0], axis = 0) # type: ignore
                    v = np.expand_dims(f[k]['data'][-1, :, :, 1], axis = 0) # type: ignore
                    dataset[k] = np.stack([u0, v0, u, v], axis = 1)
        
        data = torch.tensor(np.concatenate(list(dataset.values()), axis=0)).to(torch.float32)
        label = torch.zeros(len(data), dtype=torch.float32) + 5
        return data, label

    def _shallow_water_load(self, data_path):
        dataset = {} 
        for i in range(5):
            file_path = f"{data_path}{self.pde}/2d_swe_128_128_10_{i}.h5"

            with h5py.File(file_path, "r") as f:
                for k in list(f.keys()):
                    h0 = np.expand_dims(f[k]['data']['h'][0, :, :, 0], axis = 0) # type: ignore
                    h = np.expand_dims(f[k]['data']['h'][-1, :, :, 0], axis = 0) # type: ignore
                    hu0 = np.expand_dims(f[k]['data']['hu'][0, :, :, 0], axis = 0) # type: ignore
                    hu = np.expand_dims(f[k]['data']['hu'][-1, :, :, 0], axis = 0) # type: ignore
                    hv0 = np.expand_dims(f[k]['data']['hv'][0, :, :, 0], axis = 0) # type: ignore
                    hv = np.expand_dims(f[k]['data']['hv'][-1, :, :, 0], axis = 0) # type: ignore
                    dataset[k] = np.stack([h0, hu0, hv0, h, hu, hv], axis = 1)

        data = torch.tensor(np.concatenate(list(dataset.values()), axis=0)).to(torch.float32)
        label = torch.zeros(len(data), dtype=torch.float32) + 6
        return data, label

    def _heat_load(self, data_path, size=5, split="train"):
        return self._future_h5_load(data_path, size=size, split=split, label_value=7)

    def _wave_load(self, data_path, size=5, split="train"):
        return self._future_h5_load(data_path, size=size, split=split, label_value=8)

    def _advection_diffusion_load(self, data_path, size=5, split="train"):
        return self._future_h5_load(data_path, size=size, split=split, label_value=9)

    def _steady_heat_conduction_load(self, data_path, size=5, split="train"):
        return self._future_h5_load(data_path, size=size, split=split, label_value=10)

    def _future_h5_load(self, data_path, size=5, split="train", label_value=0, materialize_params=False):
        file_paths = self._future_h5_paths(data_path, size=size, split=split)
        dataset = []
        param_chunks = {}
        self.pde_params = {}
        self.pde_param_slices = []
        sample_start = 0
        for file_path in tqdm(file_paths):
            with h5py.File(file_path, "r") as file:
                if "input_data" not in file or "output_data" not in file:
                    raise KeyError(f"{file_path} must contain data or input_data/output_data")
                arr = self._future_h5_materialize(file, materialize_params=materialize_params)
                params = self._future_h5_scalar_params(file, arr.shape[0])
            arr = np.asarray(arr, dtype=np.float32)
            if arr.ndim != 4:
                raise ValueError(f"{file_path} data must be [N,C,H,W], got {arr.shape}")
            if arr.shape[2] != arr.shape[3]:
                raise ValueError(f"{file_path} data must have square spatial grid, got {arr.shape[2:]}")
            if arr.shape[1] % 2 != 0:
                raise ValueError(f"{file_path} must have an even channel count for FM4PDE pair splitting")
            dataset.append(arr)
            sample_stop = sample_start + arr.shape[0]
            for name, values in params.items():
                param_chunks.setdefault(name, []).append(values)
            self.pde_param_slices.append(
                {"file_path": str(file_path), "start": sample_start, "stop": sample_stop, "params": tuple(params)}
            )
            sample_start = sample_stop

        data = torch.tensor(np.concatenate(dataset, axis=0)).to(torch.float32)
        label = torch.zeros(len(data), dtype=torch.float32) + label_value
        for name, chunks in param_chunks.items():
            values = np.concatenate(chunks, axis=0)
            if values.shape[0] != len(data):
                raise ValueError(f"Scalar parameter {name!r} was present for only part of the loaded samples")
            self.pde_params[name] = torch.tensor(values, dtype=torch.float32)
        return data, label

    def _future_h5_materialize(self, file, materialize_params=True):
        input_data = np.asarray(file["input_data"][:], dtype=np.float32)
        output_data = np.asarray(file["output_data"][:], dtype=np.float32)
        if input_data.ndim != 4 or output_data.ndim != 4:
            raise ValueError(f"future_h5 input/output must be [N,C,H,W], got {input_data.shape}, {output_data.shape}")
        n_samples, _, h, w = input_data.shape
        if not materialize_params:
            return np.concatenate([input_data, output_data], axis=1)

        if self.pde == "heat":
            if self._has_scalar_dataset_or_attr(file, "alpha"):
                alpha = self._read_scalar_dataset_or_attr(file, "alpha", n_samples)
                alpha_field = self._expand_scalar_to_field(alpha, h, w)
                return np.concatenate([input_data, alpha_field, output_data, alpha_field], axis=1)
            return np.concatenate([input_data, output_data], axis=1)

        if self.pde == "wave":
            if self._has_scalar_dataset_or_attr(file, "c"):
                c = self._read_scalar_dataset_or_attr(file, "c", n_samples)
                c_field = self._expand_scalar_to_field(c, h, w)
                return np.concatenate([input_data, c_field, output_data, c_field], axis=1)
            return np.concatenate([input_data, output_data], axis=1)

        if self.pde == "advection_diffusion":
            bx = self._read_scalar_dataset_or_attr(file, "b_x", n_samples)
            by = self._read_scalar_dataset_or_attr(file, "b_y", n_samples)
            kappa = self._read_scalar_dataset_or_attr(file, "kappa", n_samples)
            bx_field = self._expand_scalar_to_field(bx, h, w)
            by_field = self._expand_scalar_to_field(by, h, w)
            kappa_field = self._expand_scalar_to_field(kappa, h, w)
            return np.concatenate(
                [input_data, bx_field, by_field, kappa_field, output_data, bx_field, by_field, kappa_field],
                axis=1,
            )

        if self.pde == "steady_heat_conduction":
            u_d = self._read_scalar_dataset_or_attr(file, "u_D", n_samples)
            u_d_field = self._expand_scalar_to_field(u_d, h, w)
            return np.concatenate([input_data, u_d_field, output_data, u_d_field], axis=1)

        return np.concatenate([input_data, output_data], axis=1)

    def _future_h5_scalar_params(self, file, n_samples):
        params = {}
        for name in self.FUTURE_H5_SCALAR_PARAMS.get(self.pde, ()):
            if self._has_scalar_dataset_or_attr(file, name):
                params[name] = self._read_scalar_dataset_or_attr(file, name, n_samples)
            elif self.pde not in self.OPTIONAL_FUTURE_H5_SCALAR_PARAMS:
                raise KeyError(f"Missing scalar dataset or attr {name!r}")
        return params

    @staticmethod
    def _expand_scalar_to_field(values, h, w):
        values = np.asarray(values, dtype=np.float32).reshape(-1, 1, 1, 1)
        return np.broadcast_to(values, (values.shape[0], 1, h, w)).copy()

    @staticmethod
    def _has_scalar_dataset_or_attr(file, name):
        return name in file or name in file.attrs

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
        values = values.reshape(-1)
        if values.shape[0] == 1:
            values = np.full((n_samples,), float(values[0]), dtype=np.float32)
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
            raise ValueError(f"Unsupported split={split!r}; expected 'train' or 'test'")

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
