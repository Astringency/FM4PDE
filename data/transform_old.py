import numpy as np



class PDEtransform:
    def __init__(self, pde: str):
        self.pde = pde.lower()
        self.transform_func = {
                "darcy": self._darcy_transform,
                "poisson": self._poisson_transform,
                "helmholtz": self._helmholtz_transform,
                "nsbounded": self._nsbounded_transform,
                "nsnonbounded": self._nsnonbounded_transform,
                "burger": self._burger_transform,
                "shallow_water": self._shallow_water_transform,
                "reaction_diffusion": self._reaction_diffusion_transform
                }

        self.inverse_transform_func = {
                "darcy": self._darcy_inverse_transform,
                "poisson": self._poisson_inverse_transform,
                "helmholtz": self._helmholtz_inverse_transform,
                "nsbounded": self._nsbounded_inverse_transform,
                "nsnonbounded": self._nsnonbounded_inverse_transform,
                "burger": self._burger_inverse_transform,
                "shallow_water": self._shallow_water_inverse_transform,
                "reaction_diffusion": self._reaction_diffusion_inverse_transform
                }

        self.transform = self.transform_func[self.pde]
        self.inverse_transform = self.inverse_transform_func[self.pde]

    def _darcy_transform(self, a, u):
        a_transform = a * 0.2 - 1.5
        u_transform = u * 115 - 0.9
        return a_transform, u_transform

    def _darcy_inverse_transform(self, a, u):
        a_transform = (a + 1.5) / 0.2
        u_transform = (u + 0.9) / 115
        return a_transform, u_transform

    def _poisson_transform(self, f, phi):
        f_transform = f / 2.5
        phi_transform = phi * 36.5
        return f_transform, phi_transform
    
    def _poisson_inverse_transform(self, f, phi):
        f_transform = f * 2.5
        phi_transform = phi / 36.5
        return f_transform, phi_transform
    
    def _helmholtz_transform(self, f, phi):
        f_transform = f / 2.15
        phi_transform = phi / 0.028
        # f_transform = (f + 2.0640 ) / (2.1616 + 2.0640)
        # phi_transform = (phi + 0.3319) / (0.3469 + 0.3319)
        return f_transform, phi_transform
    
    def _helmholtz_inverse_transform(self, f, phi):
        f_transform = f * 2.15
        phi_transform = phi * 0.028
        # f_transform = f * (2.1616 + 2.0640) - 2.0640
        # phi_transform = phi * (0.3469 + 0.3319) - 0.3319
        return f_transform, phi_transform

    def _nsbounded_transform(self, f, phi):
        f_transform = f / 10
        phi_transform = phi / 10
        return f_transform, phi_transform
    
    def _nsbounded_inverse_transform(self, f, phi):
        f_transform = f * 10
        phi_transform = phi * 10
        return f_transform, phi_transform

    def _nsnonbounded_transform(self, f, phi):
        f_transform = f / 1.6
        phi_transform = phi / 1.6
        return f_transform, phi_transform
    
    def _nsnonbounded_inverse_transform(self, f, phi):
        f_transform = f * 1.6
        phi_transform = phi * 1.6
        return f_transform, phi_transform

    def _burger_transform(self, a, u):
        a_transform = a / 1.415
        u_transform = u / 1.415
        return a_transform, u_transform

    def _burger_inverse_transform(self, a, u):
        a_transform = a * 1.415
        u_transform = u * 1.415
        return a_transform, u_transform

    def _shallow_water_transform(self, h0, h):
        return h0, h

    def _shallow_water_inverse_transform(self, h0, h):
        return h0, h

    def _reaction_diffusion_transform(self, uv0, uv):
        return uv0, uv

    def _reaction_diffusion_inverse_transform(self, uv0, uv):
        return uv0, uv
