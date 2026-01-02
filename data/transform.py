import torch



class PDEtransform:
    """
    Min-Max Normalization to [0, 1].
    input: data.shape = [N, C, H, W].
    """
    def __init__(self, data, mode = "train"):
        self.data = data
        self.mode = mode

        if self.mode == "train":
            self.dim = int(data.shape[1] / 2)
            self.min = self.data.amin(dim=(0, 2, 3), keepdim=True)
            self.max = self.data.amax(dim=(0, 2, 3), keepdim=True)
        elif self.mode == "sample":
            self.dim = int(data.shape[0] / 2)
            self.min = self.data.amin(dim=(1, 2), keepdim=True)
            self.max = self.data.amax(dim=(1, 2), keepdim=True)

        self.transform = self._transform_func
        self.inverse_transform = self._inverse_transform_func
        self.transform_sample = self._transform_func_sample
        self.inverse_transform_sample = self._inverse_transform_func_sample

    def _transform_func(self):
        return (self.data - self.min) / (self.max - self.min + 1e-8)
        
    def _inverse_transform_func(self):
        return self.data * (self.max - self.min + 1e-8) + self.min

    def _transform_func_sample(self, a, u):
        return (a - self.min[:self.dim]) / (self.max[:self.dim] - self.min[:self.dim] + 1e-8), (u - self.min[self.dim:]) / (self.max[self.dim:] - self.min[self.dim:] + 1e-8)
        # return a, u
        
    def _inverse_transform_func_sample(self, a, u):
        return a * (self.max[:self.dim] - self.min[:self.dim] + 1e-8) + self.min[:self.dim], u * (self.max[self.dim:] - self.min[self.dim:] + 1e-8) + self.min[self.dim:]
        # return a, u

