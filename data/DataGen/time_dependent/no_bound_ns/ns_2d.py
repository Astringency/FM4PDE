import torch
import math
import torch
import math
from tqdm import tqdm

def navier_stokes_2d(w0, f, visc, T, delta_t=1e-4, record_steps=1):
    """
    Simulate 2D Navier-Stokes in vorticity form.
    
    Args:
        w0: Tensor [batch, N, N] initial vorticity field
        f: Tensor [batch, N, N] forcing term
        visc: float viscosity
        T: float total simulation time
        delta_t: float time step
        record_steps: int number of saved time steps
    
    Returns:
        sol: [batch, N, N, record_steps, 3] (ω, u, v)
        sol_t: [record_steps] time points
    """
    # Grid size - must be power of 2
    N = w0.size(-1)
    batch = w0.size(0)

    # Frequency and steps
    k_max = math.floor(N / 2.0)
    steps = math.ceil(T / delta_t)
    record_time = math.floor(steps / record_steps)

    # FFT of initial fields
    w_h = torch.fft.fft2(w0, norm='forward')
    f_h = torch.fft.fft2(f, norm='forward')
    if len(f_h.shape) < len(w_h.shape):
        f_h = f_h.unsqueeze(0)

    # Wavenumbers
    k_y = torch.cat((
        torch.arange(0, k_max, device=w0.device),
        torch.arange(-k_max, 0, device=w0.device)
    ), 0).repeat(N, 1)
    k_x = k_y.t()

    # Laplacian operator (avoid div by zero)
    lap = 4 * math.pi ** 2 * (k_x ** 2 + k_y ** 2)
    lap[0, 0] = 1.0  # avoid division by zero

    # Dealiasing mask
    dealias = ((torch.abs(k_y) <= (2. / 3.) * k_max) &
               (torch.abs(k_x) <= (2. / 3.) * k_max)).float().unsqueeze(0)

    # Output tensor: [batch, N, N, record_steps]
    sol_vx0 = torch.zeros(batch, N, N, device=w0.device)
    sol_vy0 = torch.zeros(batch, N, N, device=w0.device)
    sol_w = torch.zeros(batch, N, N, record_steps, device=w0.device)
    sol_vx = torch.zeros(batch, N, N, record_steps, device=w0.device)
    sol_vy = torch.zeros(batch, N, N, record_steps, device=w0.device)
    sol_t = torch.zeros(record_steps, device=w0.device)

    # Time loop
    t = 0.0
    c = 0

    for j in tqdm(range(steps)):
        # Poisson solver: psi_h = w_h / lap
        psi_h = w_h / lap

        # Velocity field
        q_h = 1j * 2 * math.pi * k_y * psi_h   # u = ∂ψ/∂y
        v_h = -1j * 2 * math.pi * k_x * psi_h  # v = -∂ψ/∂x
        q = torch.fft.ifft2(q_h, norm='forward')
        v = torch.fft.ifft2(v_h, norm='forward')

        # Derivatives of vorticity
        w_x_h = 1j * 2 * math.pi * k_x * w_h
        w_y_h = 1j * 2 * math.pi * k_y * w_h
        w_x = torch.fft.ifft2(w_x_h, norm='forward')
        w_y = torch.fft.ifft2(w_y_h, norm='forward')

        # Nonlinear term F = u*∂w/∂x + v*∂w/∂y
        F = q * w_x + v * w_y
        F_h = torch.fft.fft2(F, norm='forward')

        # Dealias
        F_h = F_h * dealias
        f_h = f_h * dealias

        # Crank-Nicolson update
        A = 1.0 + 0.5 * delta_t * visc * lap
        B = 1.0 - 0.5 * delta_t * visc * lap
        w_h = (B * w_h - delta_t * F_h + delta_t * f_h) / A

        # Update time
        t += delta_t

        # Record data
        if j == 0:
            sol_vx0 = q.real
            sol_vy0 = v.real
        elif (j + 1) % record_time == 0:
            w = torch.fft.ifft2(w_h, norm='forward').real
            sol_w[..., c] = w
            sol_vx[..., c] = q.real
            sol_vy[..., c] = v.real
            sol_t[c] = t
            c += 1
        else:
            continue

    return sol_vx0, sol_vy0, sol_w, sol_vx, sol_vy, sol_t
