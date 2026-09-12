import torch
from experiments.fm_tilt_geometry import cosine_basis, GaussianReference, fit_reference
from experiments.fm_tilt_mh import log_acceptance, proposal


def test_reference_change_of_variables_matches_original_target_density():
    rng = torch.Generator().manual_seed(408)
    basis = cosine_basis(n=4, width=2, dtype=torch.float64)
    torch.testing.assert_close(basis @ basis.T, torch.eye(8, dtype=torch.float64))
    root = torch.diag(torch.linspace(.02, .7, 8, dtype=torch.float64))
    mean = torch.randn(1, 2, 4, 4, generator=rng, dtype=torch.float64)
    ref = GaussianReference(basis, root, mean)
    w, v = [torch.randn(4, 2, 4, 4, generator=rng, dtype=torch.float64) for _ in range(2)]
    energy = lambda z: (torch.sin(z) - .3).square().flatten(1).sum(1)
    # Deliberately incorrect proposal drift: the full MH correction must hold.
    force = lambda x: .3 * x.square()
    beta = torch.full((4,), .6, dtype=torch.float64)
    z, y = ref.decode(w), ref.decode(v)
    uw, uv = ref.residual_energy(w, z, energy(z)), ref.residual_energy(v, y, energy(y))
    actual = log_acceptance(w, v, force(w), force(v), uw, uv, beta)
    qmean = lambda x: proposal(x, force(x), beta, torch.zeros_like(x))
    expected = energy(z) - energy(y) + .5*(z.square()-y.square()).flatten(1).sum(1)
    expected += ((v-qmean(w)).square()-(w-qmean(v)).square()).flatten(1).sum(1)/(2*beta.square())
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
    w.requires_grad_(True)
    z = ref.decode(w)
    g, = torch.autograd.grad(energy(z).sum(), z, retain_graph=True)
    expected, = torch.autograd.grad(ref.residual_energy(w, z, energy(z)).sum(), w)
    torch.testing.assert_close(ref.residual_force(w, z, g), expected)


def test_gauss_newton_reference_agrees_with_dense_linear_posterior():
    from types import SimpleNamespace
    rng = torch.Generator().manual_seed(419)
    basis = cosine_basis(n=4, width=4, dtype=torch.float64)
    j = torch.randn(32, 2, 4, 4, generator=rng, dtype=torch.float64)/5
    center = torch.randn(1, 2, 4, 4, generator=rng, dtype=torch.float64)
    intercept = torch.randn(1, 2, 4, 4, generator=rng, dtype=torch.float64)
    endpoint = intercept + ((center.flatten(1)@basis.T)@j.flatten(1)).reshape_as(center)
    observation = torch.randn(1, 2, 4, 4, generator=rng, dtype=torch.float64)
    precision = torch.linspace(.1, 20, 32, dtype=torch.float64).reshape(1, 2, 4, 4)
    adapter = SimpleNamespace(singles=[None]*4,
        potential=lambda x: (.5*precision*(x-observation).square()).flatten(1).sum(1))
    ref, report = fit_reference(basis, j, center, endpoint, adapter)
    a = j.flatten(1).T @ basis
    h = torch.eye(32, dtype=torch.float64)+a.T @ torch.diag(precision.flatten()) @ a
    expected = torch.linalg.solve(h, a.T @ (precision*(observation-intercept)).flatten())
    torch.testing.assert_close(ref.mean.flatten(), expected, atol=1e-10, rtol=1e-10)
    l = ref.linear(torch.eye(32, dtype=torch.float64).reshape(32, 2, 4, 4)).flatten(1)
    torch.testing.assert_close(l@l.T, torch.linalg.inv(h), atol=1e-10, rtol=1e-10)
