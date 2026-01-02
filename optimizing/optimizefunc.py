import torch

class getPDEoptfunc:
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

    def darcy(self, device=torch.device('cuda')):
        pass

    def poisson(self, device=torch.device('cuda')):
        pass

    def helmholtz(self, device=torch.device('cuda')):
        k = torch.empty(1, device=device).uniform_(0.1, 1.0)
        return k

    def nsnonbounded(self, device=torch.device('cuda')):
        pass
    
    def burger(self, device=torch.device('cuda')):
        pass
        
    def shallow_water(self, device=torch.device('cuda')):
        lower = -5.0
        upper = 5.0
        hu0 = torch.nn.Parameter(torch.empty(1, 1, 128, 128, device=device).uniform_(lower, upper))
        hv0 = torch.nn.Parameter(torch.empty(1, 1, 128, 128, device=device).uniform_(lower, upper))
        hu  = torch.nn.Parameter(torch.empty(1, 1, 128, 128, device=device).uniform_(lower, upper))
        hv  = torch.nn.Parameter(torch.empty(1, 1, 128, 128, device=device).uniform_(lower, upper))
        return hu0, hv0, hu, hv

    def reaction_diffusion(self, device=torch.device('cuda')):
        k = torch.nn.Parameter(torch.empty(1, device=device).uniform_(0.1, 1.0))
        D_u = torch.nn.Parameter(torch.empty(1, device=device).uniform_(0.1, 1.0))
        D_v = torch.nn.Parameter(torch.empty(1, device=device).uniform_(0.1, 1.0))
        return k, D_u, D_v

