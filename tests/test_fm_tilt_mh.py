import math
import torch
from experiments.fm_tilt_mh import log_acceptance, proposal, transition


def test_ratio_matches_full_gaussian_density_for_wrong_drift():
    # Deliberately wrong, nonlinear drift: MH still corrects the actual target.
    rng = torch.Generator().manual_seed(591)
    z, y = [torch.randn(7, 9, generator=rng, dtype=torch.float64) for _ in range(2)]
    force = lambda x: torch.sin(x) + .3 * x.square()
    energy = lambda x: (.2 * (x - 2).square()).sum(1)
    beta = torch.linspace(.01, .9, 7, dtype=torch.float64)
    mean = lambda x: proposal(x, force(x), beta, torch.zeros_like(x))
    direct = energy(z) - energy(y) - .5 * (y.square() - z.square()).sum(1)
    direct += ((y - mean(z)).square().sum(1) - (z - mean(y)).square().sum(1)) / (2 * beta.square())
    actual = log_acceptance(z, y, force(z), force(y), energy(z), energy(y), beta)
    torch.testing.assert_close(actual, direct, rtol=2e-10, atol=2e-10)


def test_zero_drift_is_exact_pcn_acceptance():
    z, y = torch.randn(4, 11).double(), torch.randn(4, 11).double()
    ez, ey = z.square().sum(1), y.square().sum(1)
    actual = log_acceptance(z, y, z * 0, y * 0, ez, ey, torch.full((4,), .2))
    torch.testing.assert_close(actual, ez - ey, rtol=0, atol=0)


def test_wrong_surrogate_samples_full_nonlinear_generator_tilt():
    import numpy as np
    from scipy.integrate import quad
    torch.set_num_threads(2)
    generator = lambda z: z + .15 * z**3
    target = lambda z: (generator(z), 2 * (generator(z)[:, 0] - 1.2).square())
    # Surrogate omits nonlinear term and uses a wrong observation scale.
    force = lambda z: 2 * (z - 1.2)
    rng = torch.Generator().manual_seed(451)
    z = torch.randn(512, 1, generator=rng, dtype=torch.float64)
    r, e = target(z)
    g = force(z)
    beta = torch.full((len(z),), .6, dtype=torch.float64)
    samples = []
    for i in range(600):
        z, r, e, g, accept, prob = transition(z, r, e, g, beta, target=target, force=force, rng=rng)
        if i >= 200:
            samples.append(r.clone())
    density = lambda x: math.exp(-.5*x*x - 2*(x+.15*x**3-1.2)**2)
    norm = quad(density, -8, 8, epsabs=1e-12)[0]
    exact_mean = quad(lambda x:(x+.15*x**3)*density(x),-8,8,epsabs=1e-12)[0]/norm
    exact_second = quad(lambda x:(x+.15*x**3)**2*density(x),-8,8,epsabs=1e-12)[0]/norm
    r = torch.cat(samples).numpy()
    assert abs(r.mean()-exact_mean) < .012
    assert abs((r*r).mean()-exact_second) < .02
