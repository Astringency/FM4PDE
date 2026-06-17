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
    def __init__(self, pde: str):
        self.pde = pde.lower()
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

    def _future_h5_load(self, data_path, size=5, split="train", label_value=0):
        file_paths = self._future_h5_paths(data_path, size=size, split=split)
        dataset = []
        for file_path in tqdm(file_paths):
            with h5py.File(file_path, "r") as file:
                if "data" in file:
                    arr = file["data"][:]
                elif "input_data" in file and "output_data" in file:
                    arr = np.concatenate([file["input_data"][:], file["output_data"][:]], axis=1)
                else:
                    raise KeyError(f"{file_path} must contain data or input_data/output_data")
            arr = np.asarray(arr, dtype=np.float32)
            if arr.ndim != 4:
                raise ValueError(f"{file_path} data must be [N,C,H,W], got {arr.shape}")
            if arr.shape[2] != arr.shape[3]:
                raise ValueError(f"{file_path} data must have square spatial grid, got {arr.shape[2:]}")
            if arr.shape[1] % 2 != 0:
                raise ValueError(f"{file_path} must have an even channel count for FM4PDE pair splitting")
            dataset.append(arr)

        data = torch.tensor(np.concatenate(dataset, axis=0)).to(torch.float32)
        label = torch.zeros(len(data), dtype=torch.float32) + label_value
        return data, label

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
