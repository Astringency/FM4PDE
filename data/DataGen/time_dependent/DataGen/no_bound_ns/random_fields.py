import torch
import math

class GaussianRF(object):
    def __init__(self, dim, size, alpha=2.5, tau=3.0, sigma=None, boundary="periodic", device=None):
        self.dim = dim
        self.device = device

        if sigma is None:
            sigma = tau**(0.5*(2*alpha - self.dim))

        k_max = size // 2

        if dim == 1:
            k = torch.cat((
                torch.arange(0, k_max, device=device),
                torch.arange(-k_max, 0, device=device)
            ))
            eig = (4 * math.pi**2 * k**2 + tau**2)**(-alpha / 2.0)
            eig[0] = 0.0
            self.sqrt_eig = size * math.sqrt(2.0) * sigma * eig

        elif dim == 2:
            k = torch.cat((
                torch.arange(0, k_max, device=device),
                torch.arange(-k_max, 0, device=device)
            )).repeat(size, 1)
            k_x = k.transpose(0, 1)
            k_y = k
            eig = (4 * math.pi**2 * (k_x**2 + k_y**2) + tau**2)**(-alpha / 2.0)
            eig[0, 0] = 0.0
            self.sqrt_eig = (size**2) * math.sqrt(2.0) * sigma * eig

        elif dim == 3:
            k = torch.cat((
                torch.arange(0, k_max, device=device),
                torch.arange(-k_max, 0, device=device)
            )).repeat(size, size, 1)
            k_x = k.transpose(1, 2)
            k_y = k
            k_z = k.transpose(0, 2)
            eig = (4 * math.pi**2 * (k_x**2 + k_y**2 + k_z**2) + tau**2)**(-alpha / 2.0)
            eig[0, 0, 0] = 0.0
            self.sqrt_eig = (size**3) * math.sqrt(2.0) * sigma * eig

        self.size = (size,) * dim

    def sample(self, N):
        real = torch.randn(N, *self.size, device=self.device)
        imag = torch.randn(N, *self.size, device=self.device)
        coeff = (real + 1j * imag) * self.sqrt_eig

        u = torch.fft.ifftn(coeff, dim=tuple(range(1, self.dim + 1)), norm='backward')
        return u.real
