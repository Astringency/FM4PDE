import sys
import torch
from flow_matching.solver import Solver, ODESolver

class pdeSampler:
    def __init__(self, net, mode: str = "deterministic", useNODE: bool = False):
        self.node = useNODE
        self.modes = {
            "deterministic": self.deterministic,
            "deterministic_new": self.deterministic_new,
            "stochastic": self.stochastic
        }

        if mode not in self.modes.keys():
            raise ValueError(f"Unknown method: {mode}")

        self.net = net
        self.sampler = self.modes[mode]

    def deterministic(self, x_cur, t_grid, step_size, method="euler", device="cuda"):
        if self.node == True:
            solver = ODESolver(velocity_model = self.net)
            x_N = solver.sample(time_grid=t_grid, x_init=x_cur, method=method, step_size=step_size, enable_grad = True, return_intermediates=False)
        else:
            if isinstance(t_grid, (float, int)):
                t = t_grid
            else:
                t = t_grid[0]
            if isinstance(t, torch.Tensor):
                t = t.detach().clone().to(device)
            else:
                t = torch.tensor(t, device=device)
            
            if method == "euler":
                x_N = x_cur + self.net(x_cur, t) * step_size
            elif method == "midpoint":
                t_mid = t + step_size * 0.5
                x_mid = x_cur + self.net(x_cur, t) * step_size * 0.5
                x_N = x_cur + self.net(x_mid, t_mid) * step_size
            else:
                print(">>> Set 'useNODE = True' to apply more solvers. <<<")
                x_N = None
                sys.exit()

        x_1 = x_N

        return x_1, x_N

    def deterministic_new(self, x_init, x_cur, t_grid, step_size, method="euler"):
        t = t_grid[0]
        if method == "euler":
            x_1 = x_cur + self.net(x_cur, t) * (1 - t)
            x_N = (1 - t - step_size) * x_init + (t + step_size) * x_1
        elif method == "midpoint":
            t_mid = t + (1 - t) * 0.5
            x_mid = x_cur + self.net(x_cur, t) * (1 - t) * 0.5
            x_1 = x_cur + self.net(x_mid, t_mid) * (1 - t)
            x_N = (1 - t - step_size) * x_init + (t + step_size) * x_1
        else:
            print(">>> Use euler or midpoint to apply stochastic sampler. <<<")
            x_N = None
            sys.exit()

        return x_1, x_N

    def stochastic(self, x, t, step_size, method="euler", device='cuda'):
        if method == "euler":
            x_0 = torch.randn_like(x, device=device)
            x_1 = x + self.net(x, t) * (1 - t)
            x_N = (1 - t - step_size) * x_0 + (t + step_size) * x_1
        elif method == "midpoint":
            x_0 = torch.randn_like(x, device=device)
            t_mid = t + (1 - t) * 0.5
            x_mid = x + self.net(x, t) * (1 - t) * 0.5
            x_1 = x + self.net(x_mid, t_mid) * (1 - t)
            x_N = (1 - t - step_size) * x_0 + (t + step_size) * x_1
        else:
            print(">>> Use euler or midpoint to apply stochastic sampler. <<<")
            x_N = None
            sys.exit()

        return x_1, x_N

