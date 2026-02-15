import numpy as np
import torch
import sys

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

def random_sensor(k, grid_size, seed=0, device=torch.device('cuda')):
    """Return a index list with k sensors randomly placed in a grid of size [grid_size, grid_size]."""
    torch.manual_seed(seed)
    index = torch.zeros(grid_size, grid_size, dtype=torch.float64, device=device)
    known_index = torch.randperm(grid_size, device=device)[:k]
    for i in known_index:
        index[:, i]=1
    return index

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
            "nsnonbounded": self.nsnonbounded,
            "burger": self.burger,
            "shallow_water": self.shallow_water,
            "reaction_diffusion":self.reaction_diffusion
        }
        if pde not in self.pdes:
            raise ValueError(f"Unknown PDE: {pde}")
        self.func = self.pdes[pde]

    def darcy(self, a, u, a_GT, u_GT, a_mask, u_mask, perturb_rate=None, device=torch.device('cuda')):
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

    def poisson(self, a, u, a_GT, u_GT, a_mask, u_mask, perturb_rate=None, device=torch.device('cuda')):
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

    def helmholtz(self, a, u, a_GT, u_GT, a_mask, u_mask, perturb_rate=None, device=torch.device('cuda'), k=1):
        """
        Return the loss of the Helmholtz equation and the observation loss.
        """
        S = u.size(2)
        h = 1 / (S - 1)
        a = a.view(1, 1, S, S)
        u_padded = torch.nn.functional.pad(u, (1, 1, 1, 1), 'constant', 0)
        d2u = (u_padded[:, :, :-2, 1:-1] + u_padded[:, :, 2:, 1:-1] +
               u_padded[:, :, 1:-1, :-2] + u_padded[:, :, 1:-1, 2:] - 4 * u[:, :, :, :]) / h**2
        pde_loss = d2u + (k ** 2) * u - a
        pde_loss = pde_loss.squeeze()
        
        a_GT = a_GT.view(1, 1, S, S)
        u_GT = u_GT.view(1, 1, S, S)
        observation_loss_a = (a - a_GT).squeeze()
        observation_loss_a = observation_loss_a * a_mask  
        observation_loss_u = (u - u_GT).squeeze()
        observation_loss_u = observation_loss_u * u_mask
        
        return pde_loss.to(device), observation_loss_a.to(device), observation_loss_u.to(device)
    
    def nsnonbounded(self, a, u, a_GT, u_GT, a_mask, u_mask, perturb_rate=None, device=torch.device('cuda')):
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
    
    def burger(self, a, u, a_GT, u_GT, a_mask, u_mask, perturb_rate=None, device=torch.device('cuda')):
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

    def reaction_diffusion(self, a, u, a_GT, u_GT, a_mask, u_mask, perturb_rate=None, device=torch.device('cuda')):
        """Return the loss of the Reaction Diffusion equation and the observation loss."""
        a_GT = a_GT.view(1, 2, 128, 128)
        u_GT = u_GT.view(1, 2, 128, 128)
        a = a.view(1, 2, 128, 128)
        u = u.view(1, 2, 128, 128)

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

        # Extract components: [batch, channel, H, W]
        # a: initial state (t=0), u: state at time T
        a_u = a[:, 0:1, :, :]   # activator at t=0
        a_v = a[:, 1:2, :, :]   # inhibitor at t=0
        u_u = u[:, 0:1, :, :]   # activator at t=T
        u_v = u[:, 1:2, :, :]   # inhibitor at t=T

        
        # Time derivatives (forward difference)
        u_t = (u_u - a_u) / T   # ∂u/∂t
        v_t = (u_v - a_v) / T   # ∂v/∂t

        # Laplacian with periodic boundary conditions
        def laplacian(field):
            # field shape: [batch, 1, H, W]
            # Roll operations for periodic boundaries
            top = torch.roll(field, shifts=1, dims=2)
            bottom = torch.roll(field, shifts=-1, dims=2)
            left = torch.roll(field, shifts=1, dims=3)
            right = torch.roll(field, shifts=-1, dims=3)
            
            # Central difference approximation
            return (top + bottom + left + right - 4 * field) / (h ** 2)

        # Compute Laplacians for both fields at time T
        lap_u = laplacian(u_u)  # ∇²u
        lap_v = laplacian(u_v)  # ∇²v

        # Reaction terms (Fitzhugh-Nagumo)
        R_u = u_u - u_u**3 - k - u_v  # R_u(u,v)
        R_v = u_u - u_v               # R_v(u,v)

        # PDE residuals
        res_u = u_t - (D_u * lap_u + R_u)  # ∂u/∂t - (D_u∇²u + R_u)
        res_v = v_t - (D_v * lap_v + R_v)  # ∂v/∂t - (D_v∇²v + R_v)

        # Combined PDE loss (mean squared error)
        pde_loss = (res_u**2 + res_v**2).mean()
        
        return pde_loss, observation_loss_a, observation_loss_u
        
    def shallow_water(self, a, u, a_GT, u_GT, a_mask, u_mask, perturb_rate=None, device=torch.device('cuda')):
        """Return the loss of the Shallow Water equation and the observation loss."""
        # Reshape GT to (1, 3, 128, 128)
        a_GT = a_GT.reshape(1, 3, 128, 128)
        u_GT = u_GT.reshape(1, 3, 128, 128)

        # Extract ground truth: h0, hu0, hv0 and h, hu, hv
        h0 = a[:, 0:1, :, :]     # initial water depth
        hu0 = a[:, 1:2, :, :]    # initial x-momentum
        hv0 = a[:, 2:3, :, :]    # initial y-momentum

        h = u[:, 0:1, :, :]      # final water depth
        hu = u[:, 1:2, :, :]     # final x-momentum
        hv = u[:, 2:3, :, :]     # final y-momentum

        h_mid  = 0.5 * (h0 + h)
        hu_mid = 0.5 * (hu0 + hu)
        hv_mid = 0.5 * (hv0 + hv)

        # Observation losses for h (depth)
        observation_loss_a = (a - a_GT).squeeze()  # a is predicted h0
        observation_loss_a = observation_loss_a * a_mask  # apply mask

        observation_loss_u = (u - u_GT).squeeze()   # u is predicted h
        observation_loss_u = observation_loss_u * u_mask  # apply mask

        # Assume time step dt is known (you can pass it as parameter or set default)
        dt = 1.0  # adjust as needed

        # Use finite difference to approximate spatial derivatives
        # We'll use central difference with padding for boundaries
        def central_diff_x(f):
            # f: [1, 1, H, W]
            pad_f = torch.nn.functional.pad(f, (1, 1, 0, 0), mode='replicate')  # left/right pad
            dx = (pad_f[:, :, :, 2:] - pad_f[:, :, :, :-2]) / 2.0
            return dx

        def central_diff_y(f):
            pad_f = torch.nn.functional.pad(f, (0, 0, 1, 1), mode='replicate')  # top/bottom pad
            dy = (pad_f[:, :, 2:, :] - pad_f[:, :, :-2, :]) / 2.0
            return dy

        # For simplicity, we compute PDE residual at each point using the full field
        # But only accumulate loss at sparse observation points (a_mask and u_mask)

        # --- Mass Conservation: ∂t h + ∂x(hu) + ∂y(hv) = 0 ---
        dh_dt = (u - a) / dt
        dhu_dx = central_diff_x(hu_mid)  # use initial state for spatial derivative? Or average?
        dhv_dy = central_diff_y(hv_mid)
        mass_residual = dh_dt + dhu_dx + dhv_dy  # shape: [1, 1, 128, 128]

        # --- Momentum x: ∂t(hu) + ∂x( (hu)^2/h + 0.5*h^2 ) + ∂y( hu*hv/h ) = 0 ---
        dhu_dt = (hu - hu0) / dt
        # Avoid division by zero
        flux_x = (hu0 ** 2) / h_mid + 0.5 * h_mid ** 2
        flux_y = (hu0 * hv0) / h_mid
        dflux_x_dx = central_diff_x(flux_x)
        dflux_y_dy = central_diff_y(flux_y)
        mom_x_residual = dhu_dt + dflux_x_dx + dflux_y_dy

        # --- Momentum y: ∂t(hv) + ∂x( hu*hv/h ) + ∂y( (hv)^2/h + 0.5*h^2 ) = 0 ---
        dhv_dt = (hv - hv0) / dt
        flux_x_y = (hu0 * hv0) / h_mid
        flux_y_y = (hv0 ** 2) / h_mid + 0.5 * h_mid ** 2
        dflux_x_y_dx = central_diff_x(flux_x_y)
        dflux_y_y_dy = central_diff_y(flux_y_y)
        mom_y_residual = dhv_dt + dflux_x_y_dx + dflux_y_y_dy

        # Combine residuals (L2 norm per pixel)
        pde_loss_per_pixel = (
            mass_residual ** 2 +
            mom_x_residual ** 2 +
            mom_y_residual ** 2
        ).squeeze()  # shape: [128, 128]

        # Only apply loss where there are sparse observations (at initial or final time)
        # You can choose to apply at both times, or just one — here we use union
        combined_mask = (a_mask + u_mask).clamp(0, 1)  # boolean OR
        pde_loss = pde_loss_per_pixel * combined_mask

        pde_loss = pde_loss.mean()

        return pde_loss, observation_loss_a, observation_loss_u

    def shallow_water_sparse(self, a, u, a_GT, u_GT, a_mask, u_mask, perturb_rate=None, device=torch.device('cuda')):
        # Reshape GT to (1, 3, 128, 128)
        a_GT = a_GT.reshape(1, 3, 128, 128)
        u_GT = u_GT.reshape(1, 3, 128, 128)

        # Extract ground truth: h0, hu0, hv0 and h, hu, hv
        h0_GT = a_GT[:, 0:1, :, :]     # initial water depth
        hu0_GT = a_GT[:, 1:2, :, :]    # initial x-momentum
        hv0_GT = a_GT[:, 2:3, :, :]    # initial y-momentum

        h_GT = u_GT[:, 0:1, :, :]      # final water depth
        hu_GT = u_GT[:, 1:2, :, :]     # final x-momentum
        hv_GT = u_GT[:, 2:3, :, :]     # final y-momentum

        h_mid  = 0.5 * (a + u)
        hu_mid = 0.5 * (hu0_GT + hu_GT)
        hv_mid = 0.5 * (hv0_GT + hv_GT)

        # Observation losses for h (depth)
        observation_loss_a = (a - h0_GT).squeeze()  # a is predicted h0
        observation_loss_a = observation_loss_a * a_mask  # apply mask

        observation_loss_u = (u - h_GT).squeeze()   # u is predicted h
        observation_loss_u = observation_loss_u * u_mask  # apply mask

        # ========================
        # Compute PDE Loss
        # ========================

        # Assume time step dt is known (you can pass it as parameter or set default)
        dt = 1.0  # adjust as needed

        # Use finite difference to approximate spatial derivatives
        # We'll use central difference with padding for boundaries
        def central_diff_x(f):
            # f: [1, 1, H, W]
            pad_f = torch.nn.functional.pad(f, (1, 1, 0, 0), mode='replicate')  # left/right pad
            dx = (pad_f[:, :, :, 2:] - pad_f[:, :, :, :-2]) / 2.0
            return dx

        def central_diff_y(f):
            pad_f = torch.nn.functional.pad(f, (0, 0, 1, 1), mode='replicate')  # top/bottom pad
            dy = (pad_f[:, :, 2:, :] - pad_f[:, :, :-2, :]) / 2.0
            return dy

        # For simplicity, we compute PDE residual at each point using the full field
        # But only accumulate loss at sparse observation points (a_mask and u_mask)

        # --- Mass Conservation: ∂t h + ∂x(hu) + ∂y(hv) = 0 ---
        dh_dt = (u - a) / dt
        dhu_dx = central_diff_x(hu_mid)  # use initial state for spatial derivative? Or average?
        dhv_dy = central_diff_y(hv_mid)
        mass_residual = dh_dt + dhu_dx + dhv_dy  # shape: [1, 1, 128, 128]

        # --- Momentum x: ∂t(hu) + ∂x( (hu)^2/h + 0.5*h^2 ) + ∂y( hu*hv/h ) = 0 ---
        dhu_dt = (hu_GT - hu0_GT) / dt
        # Avoid division by zero
        flux_x = (hu0_GT ** 2) / h_mid + 0.5 * h_mid ** 2
        flux_y = (hu0_GT * hv0_GT) / h_mid
        dflux_x_dx = central_diff_x(flux_x)
        dflux_y_dy = central_diff_y(flux_y)
        mom_x_residual = dhu_dt + dflux_x_dx + dflux_y_dy

        # --- Momentum y: ∂t(hv) + ∂x( hu*hv/h ) + ∂y( (hv)^2/h + 0.5*h^2 ) = 0 ---
        dhv_dt = (hv_GT - hv0_GT) / dt
        flux_x_y = (hu0_GT * hv0_GT) / h_mid
        flux_y_y = (hv0_GT ** 2) / h_mid + 0.5 * h_mid ** 2
        dflux_x_y_dx = central_diff_x(flux_x_y)
        dflux_y_y_dy = central_diff_y(flux_y_y)
        mom_y_residual = dhv_dt + dflux_x_y_dx + dflux_y_y_dy

        # Combine residuals (L2 norm per pixel)
        pde_loss_per_pixel = (
            mass_residual ** 2 +
            mom_x_residual ** 2 +
            mom_y_residual ** 2
        ).squeeze()  # shape: [128, 128]

        # Only apply loss where there are sparse observations (at initial or final time)
        # You can choose to apply at both times, or just one — here we use union
        combined_mask = (a_mask + u_mask).clamp(0, 1)  # boolean OR
        pde_loss = pde_loss_per_pixel * combined_mask

        pde_loss = pde_loss.mean()

        return pde_loss, observation_loss_a, observation_loss_u
