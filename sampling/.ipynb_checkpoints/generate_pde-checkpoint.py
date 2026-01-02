import numpy as np
import torch

def random_index(k, grid_size, seed=0, device=torch.device('cuda')):
    '''randomly select k indices from a [grid_size, grid_size] grid.'''
    np.random.seed(seed)
    indices = np.random.choice(grid_size**2, k, replace=False)
    indices_2d = np.unravel_index(indices, (grid_size, grid_size))
    indices_list = list(zip(indices_2d[0], indices_2d[1]))
    mask = torch.zeros((grid_size, grid_size), dtype=torch.float32).to(device)
    for i in indices_list:
        mask[i] = 1
    return mask

def random_index_and_cylinder(center, radius, k, grid_size, seed=0, device=torch.device('cuda')):
    '''randomly select k% indices from a [grid_size, grid_size] grid as well as the known boundary of the cylinder.'''
    np.random.seed(seed)
    mask = torch.zeros((grid_size, grid_size), dtype=torch.float32).to(device)
    for i in range(grid_size):
        for j in range(grid_size):
            if (i - center[0])**2 + (j - center[1])**2 <= radius**2:
                mask[i, j] = 1
            else:
                if np.random.rand() < k:
                    mask[i, j] = 1
    num_ones = mask.sum().item()
    return mask, num_ones

def cylinder_index(center, radius, grid_size, device=torch.device('cuda')):
    '''return the known boundary of the cylinder.'''
    mask = torch.zeros((grid_size, grid_size), dtype=torch.float32).to(device)
    for i in range(grid_size):
        for j in range(grid_size):
            if (i - center[0])**2 + (j - center[1])**2 <= radius**2:
                mask[i, j] = 1
    num_ones = mask.sum().item()
    return mask, num_ones

class getPDEloss:
    def __init__(self, pde: str):
        self.pdes = {
            "darcy": self.darcy,
            "poisson": self.poisson,
            "helmholtz": self.helmholtz,
            "nsbounded": self.nsbounded,
            "nsnonbounded": self.nsnonbounded,
            "burger": self.burger,
            "shallow_water": self.shallow_water,
            "reaction_diffusion":self.reaction_diffusion
        }
        if pde not in self.pdes:
            raise ValueError(f"Unknown PDE: {pde}")
        self.func = self.pdes[pde]

    def darcy(self, a, u, a_GT, u_GT, a_mask, u_mask, device=torch.device('cuda')):
        """Return the loss of the Darcy Flow equation and the observation loss."""
        deriv_x = torch.tensor([[-1, 0, 1]], dtype=torch.float64, device=device).view(1, 1, 1, 3) / 2
        deriv_y = torch.tensor([[-1], [0], [1]], dtype=torch.float64, device=device).view(1, 1, 3, 1) / 2
        grad_x_next_x = torch.nn.functional.conv2d(u, deriv_x, padding=(0, 1))
        grad_x_next_y = torch.nn.functional.conv2d(u, deriv_y, padding=(1, 0))
        grad_x_next_x = a * grad_x_next_x
        grad_x_next_y = a * grad_x_next_y
        result = torch.nn.functional.conv2d(grad_x_next_x, deriv_x, padding=(0, 1)) + torch.nn.functional.conv2d(grad_x_next_y, deriv_y, padding=(1, 0))
        pde_loss = result + 1
        pde_loss = pde_loss.squeeze()
        
        observation_loss_a = (a - a_GT).squeeze()
        observation_loss_a = observation_loss_a * a_mask  
        observation_loss_u = (u - u_GT).squeeze()
        observation_loss_u = observation_loss_u * u_mask
        
        return pde_loss, observation_loss_a, observation_loss_u

    def poisson(self, a, u, a_GT, u_GT, a_mask, u_mask, device=torch.device('cuda')):
        """
        Return the loss of the Poisson equation and the observation loss.
        """
        S = u.size(2)
        h = 1 / (S - 1)
        a = a.view(1, 1, S, S)
        u_padded = torch.nn.functional.pad(u, (1, 1, 1, 1), 'constant', 0)
        d2u = (u_padded[:, :, :-2, 1:-1] + u_padded[:, :, 2:, 1:-1] +
               u_padded[:, :, 1:-1, :-2] + u_padded[:, :, 1:-1, 2:] - 4 * u[:, :, :, :]) / h**2
        pde_loss = d2u - a
        pde_loss = pde_loss.squeeze()
        pde_loss[0, :] = 0
        pde_loss[-1, :] = 0
        pde_loss[:, 0] = 0
        pde_loss[:, -1] = 0
        
        a_GT = a_GT.view(1, 1, S, S)
        u_GT = u_GT.view(1, 1, S, S)
        observation_loss_a = (a - a_GT).squeeze()
        observation_loss_a = observation_loss_a * a_mask  
        observation_loss_u = (u - u_GT).squeeze()
        observation_loss_u = observation_loss_u * u_mask
        
        return pde_loss.to(device), observation_loss_a.to(device), observation_loss_u.to(device)

    def helmholtz(self, a, u, a_GT, u_GT, a_mask, u_mask, device=torch.device('cuda')):
        """
        Return the loss of the Helmholtz equation and the observation loss.
        """
        S = u.size(2)
        h = 1 / (S - 1)
        a = a.view(1, 1, S, S)
        u_padded = torch.nn.functional.pad(u, (1, 1, 1, 1), 'constant', 0)
        d2u = (u_padded[:, :, :-2, 1:-1] + u_padded[:, :, 2:, 1:-1] +
               u_padded[:, :, 1:-1, :-2] + u_padded[:, :, 1:-1, 2:] - 4 * u[:, :, :, :]) / h**2
        pde_loss = d2u + u - a
        pde_loss = pde_loss.squeeze()
        
        a_GT = a_GT.view(1, 1, S, S)
        u_GT = u_GT.view(1, 1, S, S)
        observation_loss_a = (a - a_GT).squeeze()
        observation_loss_a = observation_loss_a * a_mask  
        observation_loss_u = (u - u_GT).squeeze()
        observation_loss_u = observation_loss_u * u_mask
        
        return pde_loss.to(device), observation_loss_a.to(device), observation_loss_u.to(device)
    
    def nsbounded(self, a, u, a_GT, u_GT, a_mask, u_mask, device=torch.device('cuda')):
        """
        Return the loss of the bounded NS equation and the observation loss.
        """
        deriv_x = torch.tensor([[1, 0, -1]], dtype=torch.float64, device=device).view(1, 1, 1, 3) / 2
        deriv_y = torch.tensor([[1], [0], [-1]], dtype=torch.float64, device=device).view(1, 1, 3, 1) / 2
        
        grad_x_next_x = torch.nn.functional.conv2d(u, deriv_x, padding=(0, 1))
        grad_x_next_y = torch.nn.functional.conv2d(u, deriv_y, padding=(1, 0))
        pde_loss = grad_x_next_x + grad_x_next_y
        pde_loss = pde_loss.squeeze()
        pde_loss[0, :] = 0
        pde_loss[-1, :] = 0
        pde_loss[:, 0] = 0
        pde_loss[:, -1] = 0
        
        a_GT = a_GT.view(1, 1, 128, 128)
        u_GT = u_GT.view(1, 1, 128, 128)
        observation_loss_a = (a - a_GT).squeeze()
        observation_loss_a = observation_loss_a * a_mask
        observation_loss_u = (u - u_GT).squeeze()
        observation_loss_u = observation_loss_u * u_mask
        
        return pde_loss, observation_loss_a, observation_loss_u

    def nsnonbounded(self, a, u, a_GT, u_GT, a_mask, u_mask, device=torch.device('cuda')):
        """Return the loss of the non-bounded NS equation and the observation loss."""
        deriv_x = torch.tensor([[1, 0, -1]], dtype=torch.float64, device=device).view(1, 1, 1, 3) / 2
        deriv_y = torch.tensor([[1], [0], [-1]], dtype=torch.float64, device=device).view(1, 1, 3, 1) / 2
        grad_x_next_x = torch.nn.functional.conv2d(u, deriv_x, padding=(0, 1))
        grad_x_next_y = torch.nn.functional.conv2d(u, deriv_y, padding=(1, 0))
        pde_loss = grad_x_next_x + grad_x_next_y
        pde_loss = pde_loss.squeeze()
        pde_loss[0, :] = 0
        pde_loss[-1, :] = 0
        pde_loss[:, 0] = 0
        pde_loss[:, -1] = 0
        
        a_GT = a_GT.view(1, 1, 128, 128)
        u_GT = u_GT.view(1, 1, 128, 128)
        observation_loss_a = (a - a_GT).squeeze()
        observation_loss_a = observation_loss_a * a_mask  
        observation_loss_u = (u - u_GT).squeeze()
        observation_loss_u = observation_loss_u * u_mask
        
        return pde_loss, observation_loss_a, observation_loss_u
    
    def burger(self, a, u, a_GT, u_GT, a_mask, u_mask, device=torch.device('cuda')):
        """Return the loss of the Burgers' equation and the observation loss."""
        u = u.view(1, 1, 128, 128)
        u_GT = u_GT.view(1, 1, 128, 128)

        deriv_t = torch.tensor([[-1], [0], [1]], dtype=torch.float64, device=device).view(1, 1, 3, 1) / 2 
        deriv_x = torch.tensor([[-1, 0, 1]], dtype=torch.float64, device=device).view(1, 1, 1, 3) / 2 

        u_t = torch.nn.functional.conv2d(u, deriv_t, padding=(1, 0)) 
        u_x = torch.nn.functional.conv2d(u, deriv_x, padding=(0, 1)) 
        u_xx = torch.nn.functional.conv2d(u_x, deriv_x, padding=(0, 1))

        pde_loss = u_t + u * u_x - 0.01 * u_xx
        pde_loss = pde_loss.squeeze()
        observation_loss = u - u_GT
        observation_loss = observation_loss.squeeze()
        observation_loss = observation_loss * u_mask
        return pde_loss, observation_loss, observation_loss

    def reaction_diffusion(self, a, u, a_GT, u_GT, a_mask, u_mask, device=torch.device('cuda')):
        a_GT = a_GT.view(1, 2, 128, 128)
        u_GT = u_GT.view(1, 2, 128, 128)

        # obs loss
        observation_loss_a = (a - a_GT).squeeze()
        observation_loss_a = observation_loss_a * a_mask  
        observation_loss_u = (u - u_GT).squeeze()
        observation_loss_u = observation_loss_u * u_mask

        # pde loss
        T = 5
        k = 5e-3
        D_u = 1e-3
        D_v = 5e-3
        S = u.size(2)
        h = 1 / (S - 1)
        u1_t = (u[:, :1, :, :] - a[:, :1, :, :]) / T
        u2_t = (u[:, 1:, :, :] - a[:, 1:, :, :]) / T

        
        pde_loss = torch.zeros((128, 128), dtype=torch.float64)
        
        return pde_loss, observation_loss_a, observation_loss_u
        
    def shallow_water(self, a, u, a_GT, u_GT, a_mask, u_mask, device=torch.device('cuda')):
        a_GT = a_GT.view(1, 1, 128, 128)
        u_GT = u_GT.view(1, 1, 128, 128)
        observation_loss_a = (a - a_GT).squeeze()
        observation_loss_a = observation_loss_a * a_mask  
        observation_loss_u = (u - u_GT).squeeze()
        observation_loss_u = observation_loss_u * u_mask

        pde_loss = torch.zeros((128, 128), dtype=torch.float64)
        
        return pde_loss, observation_loss_a, observation_loss_u
        


