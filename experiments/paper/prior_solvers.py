"""Reference operators used for the appendix's unguided generated fields."""
from types import SimpleNamespace

import numpy as np


def solve(pde, a, u, parameters, folder, args):
    def p(name):
        return float(parameters[name][0])

    if pde in {'poisson', 'helmholtz', 'darcy'}:
        from data.DataGen.static_solvers import solve_static
        result, details = solve_static(pde, a[0])
        return result[None], details
    if pde in {'heat', 'advection_diffusion', 'wave'}:
        from data.DataGen.python.common import periodic_wavenumbers
        kx, ky, ksq = periodic_wavenumbers(a.shape[-1])
        ahat = np.fft.fft2(a)
        if pde == 'heat':
            result = np.fft.ifft2(ahat*np.exp(-p('alpha')*ksq*p('T'))).real
        elif pde == 'advection_diffusion':
            multiplier = np.exp(-(p('kappa')*ksq+1j*(p('b_x')*kx+p('b_y')*ky))*p('T'))
            result = np.fft.ifft2(ahat*multiplier).real
        else:
            ck = p('c')*np.sqrt(ksq)
            cosine, sine = np.cos(ck*p('T')), np.sin(ck*p('T'))
            factor = np.full_like(ck, p('T'))
            np.divide(sine, ck, out=factor, where=ck > 0)
            result = np.fft.ifft2(np.stack([ahat[0]*cosine+ahat[1]*factor,
                                          -ahat[0]*ck*sine+ahat[1]*cosine])).real
        return result, dict(solver='periodic Fourier reference')
    if pde == 'reaction_diffusion':
        from data.DataGen.time_dependent.pdebench.data_gen.src.sim_diff_react import Simulator
        sim = Simulator(Du=p('D_u'), Dv=p('D_v'), k=p('k'), t=p('T'), tdim=int(p('tdim')),
                        xdim=a.shape[-1], ydim=a.shape[-2], x_left=p('x_left'), x_right=p('x_right'),
                        y_bottom=p('y_bottom'), y_top=p('y_top'))
        initial = iter(a.copy())
        sim._sample_initial_field = lambda rng: next(initial).reshape(-1)
        return sim.generate_sample()[-1].transpose(2, 0, 1), dict(solver='generator finite-volume reference')
    if pde == 'shallow_water':
        from data.DataGen.time_dependent.pdebench.data_gen.src.sim_radial_dam_break import RadialDamBreak2D
        if np.min(a[0]) <= 0:
            raise ValueError('Generated shallow-water depth is nonpositive; no clipping is applied')
        sim = RadialDamBreak2D(a.shape[-2], a.shape[-1], grav=p('g'))
        sim.claw_state.q[:] = a
        sim.run(T=p('T'), tsteps=10)
        return sim.claw_state.q.copy(), dict(minimum_generated_depth=float(np.min(a[0])), saved_intervals=10)
    if pde == 'steady_heat_conduction':
        from data.DataGen.python.generate_steady_heat_conduction import solve_nonlinear_heat
        boundary = float(np.mean(u[0, 0]))
        answer = solve_nonlinear_heat(a[0], boundary, SimpleNamespace(extra={}))
        if not answer['converged']:
            raise RuntimeError('Steady-heat reference iteration did not converge')
        result = answer.pop('u')[None]
        return result, dict(answer, boundary_temperature=boundary,
                            boundary_source='mean of generated Dirichlet trace u[0,:]')
    if pde == 'nsnonbounded':
        import torch
        from data.DataGen.time_dependent.no_bound_ns.ns_2d import navier_stokes_2d
        if abs(p('nu')-.001) >= 1e-9 or abs(p('solver_dt')-1e-4) >= 1e-10:
            raise ValueError('Unexpected Navier–Stokes reference parameters')
        grid = torch.arange(a.shape[-1], dtype=torch.float32)/a.shape[-1]
        xx, yy = torch.meshgrid(grid, grid, indexing='ij')
        forcing = .1*(torch.sin(2*torch.pi*(xx+yy))+torch.cos(2*torch.pi*(xx+yy)))
        answer = navier_stokes_2d(torch.from_numpy(a.astype(np.float32)), forcing,
                                 .001, p('T'), delta_t=1e-4, record_steps=10)
        return answer[2][..., -1].numpy().astype(np.float64), dict(nu=.001, solver_dt=.0001, saved_intervals=10)
    if pde == 'burger':
        if not args.chebfun:
            raise ValueError('Set CHEBFUN_ROOT for the Burgers reference solver')
        from experiments.paper.consistency_helpers import solve as solve_burgers
        result = solve_burgers(u[:, 0], folder/'reference_solver', args)
        return result[0].numpy(), dict(solver='Chebfun SPIN', viscosity=.01, T=1.)
    raise ValueError(pde)
