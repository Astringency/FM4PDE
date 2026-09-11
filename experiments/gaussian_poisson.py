"""Full-grid Gaussian Poisson reference with the generator's DCT/DST operators.

The prescribed GRF prior is a different prior from a trained FM checkpoint.
No learned-prior accuracy claim follows from this oracle control alone.
"""
from __future__ import annotations
import numpy as np
from scipy.fft import dctn, idctn, dstn, idstn


class PoissonPrior:
    def __init__(self, n=128, alpha=2., tau=3., mean=None, scale=None):
        self.n = n
        k = np.arange(n)
        k2 = k[:, None] ** 2 + k[None, :] ** 2
        self.amplitude = n * tau ** (alpha - 1) * (np.pi ** 2 * k2 + tau ** 2) ** (-alpha / 2)
        self.amplitude[0, 0] = 0
        interior = np.arange(1, n - 1)
        eigen = -4 * (n - 1) ** 2 * np.sin(np.pi * interior / (2 * (n - 1))) ** 2
        self.laplacian_eigen = eigen[:, None] + eigen[None, :]
        self.mean = np.asarray([0., 0.] if mean is None else mean).reshape(1, 2, 1, 1)
        self.scale = np.asarray([1., 1.] if scale is None else scale).reshape(1, 2, 1, 1)
        self.mu = np.broadcast_to(-self.mean / self.scale, (1, 2, n, n)).copy()
        self.approx_solution_gain = -1 / (np.pi ** 2 * (k2 + 2))

    def solve(self, source):
        out = np.zeros_like(source)
        v = dstn(source[..., 1:-1, 1:-1], type=1, norm="ortho", axes=(-2, -1))
        out[..., 1:-1, 1:-1] = idstn(v / self.laplacian_eigen, type=1, norm="ortho", axes=(-2, -1))
        return out

    def A(self, z):
        source = idctn(self.amplitude * z, type=2, norm="ortho", axes=(-2, -1))
        return np.stack((source, self.solve(source)), axis=1) / self.scale

    def AT(self, r):
        physical_dual = r / self.scale
        source_dual = physical_dual[:, 0] + self.solve(physical_dual[:, 1])
        return self.amplitude * dctn(source_dual, type=2, norm="ortho", axes=(-2, -1))

    def decode(self, r):
        return r * self.scale + self.mean


def cg_batch(matvec, rhs, *, diagonal, initial=None, rtol=1e-9, maxiter=3000):
    """Independent CG recurrence per RHS, with verified true residuals."""
    x = np.zeros_like(rhs) if initial is None else initial.copy()
    axes = tuple(range(1, rhs.ndim))
    dot = lambda a, b: np.sum(a * b, axis=axes)
    broadcast = lambda a: a.reshape((-1,) + (1,) * len(axes))
    tol2 = np.maximum(rtol ** 2 * dot(rhs, rhs), 1e-26)
    r = rhs - matvec(x)
    z = r / diagonal
    direction = z.copy()
    rz = dot(r, z)
    active = dot(r, r) > tol2
    for it in range(maxiter + 1):
        if not np.any(active):
            true = rhs - matvec(x)
            relative = np.sqrt(dot(true, true) / np.maximum(dot(rhs, rhs), 1e-300))
            if np.all(dot(true, true) <= tol2 * 1.01):
                return x, dict(iterations=it, max_relative_residual=float(relative.max()))
            r = true
            z = r / diagonal
            direction = z.copy()
            rz = dot(r, z)
            active = dot(r, r) > tol2
        Ad = matvec(direction)
        curvature = dot(direction, Ad)
        if not np.all(curvature[active] > 0):
            raise RuntimeError("CG lost positive curvature; inspect adjoints")
        step = np.zeros_like(rz)
        step[active] = rz[active] / curvature[active]
        x += broadcast(step) * direction
        r -= broadcast(step) * Ad
        z = r / diagonal
        next_rz = dot(r, z)
        beta = np.zeros_like(rz)
        beta[active] = next_rz[active] / rz[active]
        direction = z + broadcast(beta) * direction
        rz = next_rz
        active &= dot(r, r) > tol2
        direction *= broadcast(active)
    raise RuntimeError(f"CG did not converge in {maxiter} iterations")


class GaussianTilt:
    def __init__(self, prior, masks, physical_observation, weights, rtol=1e-9):
        self.prior = prior
        self.masks = np.asarray(masks, dtype=np.float64)
        self.weights = np.asarray(weights, dtype=np.float64)
        count = self.masks.sum(axis=(-2, -1), keepdims=True)
        self.factor = np.sqrt(2 * self.weights.reshape(1, 2, 1, 1) / count) * self.masks
        self.y = self.factor * (physical_observation - prior.mean)
        self.obs_rhs = prior.AT(self.BT(self.y - self.B(prior.mu)))
        self.rtol = rtol

    def B(self, r):
        return self.factor * self.prior.scale * r

    def BT(self, obs):
        return self.factor * self.prior.scale * obs

    def diagonal(self, t, tilted):
        p = self.prior
        sa, su = p.scale.reshape(-1)
        gain2 = p.approx_solution_gain ** 2
        diag = (1 - t) ** 2 + t ** 2 * p.amplitude ** 2 * (1 / sa ** 2 + gain2 / su ** 2)
        if tilted:
            diag += (1 - t) ** 2 * p.amplitude ** 2 * 2 / p.n ** 2 * (self.weights[0] + self.weights[1] * gain2)
        return diag

    def precision(self, z, t, tilted):
        p = self.prior
        az = p.A(z)
        out = (1 - t) ** 2 * z + t ** 2 * p.AT(az)
        if tilted:
            out += (1 - t) ** 2 * p.AT(self.BT(self.B(az)))
        return out

    def moment(self, x, t, *, tilted=True, initial=None):
        p = self.prior
        rhs = t * p.AT(x - t * p.mu)
        if tilted:
            rhs += (1 - t) ** 2 * self.obs_rhs
        z, report = cg_batch(lambda z: self.precision(z, t, tilted), rhs,
            diagonal=self.diagonal(t, tilted), initial=initial, rtol=self.rtol)
        return p.mu + p.A(z), z, report

    def endpoint_draw(self, rng):
        # The random RHS has covariance Q; Q^{-1} RHS has covariance Q^{-1}.
        p = self.prior
        noise = rng.standard_normal(self.obs_rhs.shape) + p.AT(self.BT(rng.standard_normal(self.y.shape)))
        z, report = cg_batch(lambda z: self.precision(z, 0., True), self.obs_rhs + noise,
                            diagonal=self.diagonal(0., True), rtol=self.rtol)
        return p.mu + p.A(z), report

    def plugin_gradient(self, endpoint, t):
        # The endpoint Jacobian is t A Q_0^{-1} A^T, which is symmetric.
        p = self.prior
        rhs = p.AT(self.BT(self.B(endpoint) - self.y))
        z, report = cg_batch(lambda z: self.precision(z, t, False), rhs,
                            diagonal=self.diagonal(t, False), rtol=self.rtol)
        return t * p.A(z), report


def self_test():
    rng = np.random.default_rng(42)
    n = 8
    p = PoissonPrior(n=n, mean=[.13, -.08], scale=[.8, .003])
    z, r = rng.standard_normal((2, n, n)), rng.standard_normal((2, 2, n, n))
    adjoint_error = abs(np.sum(p.A(z) * r) - np.sum(z * p.AT(r)))
    assert adjoint_error < 1e-11
    mask = np.zeros((1, 2, n, n))
    mask[..., 1:4, 2:5] = 1
    truth = p.decode(p.mu + p.A(z))
    guide = GaussianTilt(p, mask, truth, [300., 1e6], rtol=1e-11)
    basis = np.eye(n * n).reshape(n * n, n, n)
    matrix = p.A(basis).reshape(n * n, -1).T
    H = guide.B(p.A(basis)).reshape(n * n, -1).T
    max_error = 0.
    for t in [0., .03, .5, .99]:
        x = rng.standard_normal(r.shape)
        mean, latent, report = guide.moment(x, t)
        q = (1 - t) ** 2 * np.eye(n * n) + t ** 2 * matrix.T @ matrix + (1 - t) ** 2 * H.T @ H
        rhs = t * (x - t * p.mu).reshape(2, -1) @ matrix + (1 - t) ** 2 * guide.obs_rhs.reshape(2, -1)
        expected = np.linalg.solve(q, rhs.T).T.reshape(2, n, n)
        error = np.max(abs(mean - (p.mu + p.A(expected))))
        max_error = max(max_error, float(error))
    assert max_error < 1e-8, max_error
    x = rng.standard_normal(r.shape)
    endpoint, _, _ = guide.moment(x, .4, tilted=False)
    grad, _ = guide.plugin_gradient(endpoint, .4)
    direction = rng.standard_normal(x.shape)
    eps = 1e-5
    plus, _, _ = guide.moment(x + eps * direction, .4, tilted=False)
    minus, _, _ = guide.moment(x - eps * direction, .4, tilted=False)
    finite_diff = ((guide.B(plus) - guide.y) ** 2).sum() / (2 * eps) / 2
    finite_diff -= ((guide.B(minus) - guide.y) ** 2).sum() / (2 * eps) / 2
    gradient_error = abs(finite_diff - np.sum(grad * direction)) / max(1., abs(finite_diff))
    assert gradient_error < 1e-6
    return dict(adjoint_error=adjoint_error, dense_mean_max_error=max_error,
                plugin_gradient_relative_error=gradient_error, passed=True)


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), indent=2))
