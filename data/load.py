import h5py
import scipy.io
from tqdm import tqdm
import numpy as np

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
                "shallow_water": self._shallow_water_load
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

    def _reaction_diffusion_load(self, data_path):
        dataset = {}
        file_path = f"{data_path}/reaction_diffusion/2D_diff-react_NA_NA.h5"
        with h5py.File(file_path, "r") as f:
            for k in tqdm(list(f.keys())):
                u0 = np.expand_dims(f[k]['data'][0, :, :, 0], axis = 0) # type: ignore
                v0 = np.expand_dims(f[k]['data'][0, :, :, 1], axis = 0) # type: ignore
                u = np.expand_dims(f[k]['data'][-1, :, :, 0], axis = 0) # type: ignore
                v = np.expand_dims(f[k]['data'][-1, :, :, 1], axis = 0) # type: ignore
                dataset[k] = np.stack([u0, v0, u, v], axis = 1)
        
        data = torch.tensor(np.concatenate(list(dataset.values()), axis=0)).to(torch.float32)
        label = torch.zeros(len(data), dtype=torch.float32) + 5
        return data, label

    def _shallow_water_load(self, data_path, size = 5):
        dataset = {} 
        for i in range(size):
            file_path = f"{data_path}/shallow_water/2d_swe_128_128_10_{i}.h5"

            with h5py.File(file_path, "r") as f:
                for k in list(f.keys()):
                    u0 = np.expand_dims(f[k]['data']['h'][0, :, :, 0], axis = 0) # type: ignore
                    u = np.expand_dims(f[k]['data']['h'][-1, :, :, 0], axis = 0) # type: ignore
                    dataset[k] = np.stack([u0, u], axis = 1)
        
        data = torch.tensor(np.concatenate(list(dataset.values()), axis=0)).to(torch.float32)
        label = torch.zeros(len(data), dtype=torch.float32) + 6
        return data, label

    # def _shallow_water_load(self, data_path):
    #     dataset = {} 
    #     for i in range(5):
    #         file_path = f"{data_path}{self.pde}/2d_swe_128_128_10_{i}.h5"
    #
    #         with h5py.File(file_path, "r") as f:
    #             for k in list(f.keys()):
    #                 h0 = np.expand_dims(f[k]['data']['h'][0, :, :, 0], axis = 0) # type: ignore
    #                 h = np.expand_dims(f[k]['data']['h'][-1, :, :, 0], axis = 0) # type: ignore
    #                 hu0 = np.expand_dims(f[k]['data']['hu'][0, :, :, 0], axis = 0) # type: ignore
    #                 hu = np.expand_dims(f[k]['data']['hu'][-1, :, :, 0], axis = 0) # type: ignore
    #                 hv0 = np.expand_dims(f[k]['data']['hv'][0, :, :, 0], axis = 0) # type: ignore
    #                 hv = np.expand_dims(f[k]['data']['hv'][-1, :, :, 0], axis = 0) # type: ignore
    #                 dataset[k] = np.stack([h0, hu0, hv0, h, hu, hv], axis = 1)
    #
    #     data = torch.tensor(np.concatenate(list(dataset.values()), axis=0)).to(torch.float32)
    #     label = torch.zeros(len(data), dtype=torch.float32) + 6
    #     return data, label
