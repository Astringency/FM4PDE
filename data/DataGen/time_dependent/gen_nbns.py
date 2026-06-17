from tqdm import tqdm
from no_bound_ns.random_fields import GaussianRF
from no_bound_ns.ns_2d import navier_stokes_2d
import h5py
import torch
import math


def main(batch=5, N_each_batch=10000, resolution=128, device='cuda', if_test=False):
    device = torch.device(device)

    for i in range(batch):
        print(f">>> Generate 2D Non-bounded Navier Stokes {i+1} <<<")
        # Resolution
        s = resolution

        # Number of solutions to generate
        N = N_each_batch

        # Set up 2d GRF with covariance parameters
        if if_test:
            GRF = GaussianRF(2, s, alpha=3, tau=6.5, device=device)
        else:
            GRF = GaussianRF(2, s, alpha=2.5, tau=7, device=device)

        # Forcing function: 0.1*(sin(2pi(x+y)) + cos(2pi(x+y)))
        t = torch.linspace(0, 1, s+1, device=device)
        t = t[0:-1]

        X, Y = torch.meshgrid(t, t, indexing='ij')
        f = 0.1*(torch.sin(2*math.pi*(X + Y)) + torch.cos(2*math.pi*(X + Y)))

        # Number of snapshots from solution
        record_steps = 10

        # Solve equations in batches (order of magnitude speed-up)
        w0 = GRF.sample(N)
        sol_vx0, sol_vy0, sol_w, sol_vx, sol_vy, sol_t = navier_stokes_2d(w0, f, 1e-3, 1.0, 1e-4, record_steps)
        a = w0.real
        
        if if_test:
            filename = f'/large_storage/zhangxf/PDEdata/test1125/nsnonbounded_{N}-{s}-{s}-{record_steps}_{i+1}.mat'
        else:
            filename = f'/large_storage/zhangxf/PDEdata/nsnonbounded/nsnonbounded_{N}-{s}-{s}-{record_steps}_{i+1}_new.mat'
        
        with h5py.File(filename, 'w') as f:
            f.create_dataset('w0', data=a.cpu().numpy())
            f.create_dataset('w', data=sol_w.cpu().numpy())
            f.create_dataset('vx0', data=sol_vx0.cpu().numpy())
            f.create_dataset('vy0', data=sol_vy0.cpu().numpy())
            f.create_dataset('vx', data=sol_vx.cpu().numpy())
            f.create_dataset('vy', data=sol_vy.cpu().numpy())
            f.create_dataset('t', data=sol_t.cpu().numpy())

    print(f"Done. Generate {batch * N_each_batch} Non-bounded navier-Stokes Equations.")

if __name__ == "__main__":
    batch = 5
    N_each = 10000
    batch_test = 1
    N_each_test = 1000
    resolution = 128
    device = 'cuda:0'

    # main(batch, N_each, resolution, device)
    main(batch_test, N_each_test, resolution, device, True)
