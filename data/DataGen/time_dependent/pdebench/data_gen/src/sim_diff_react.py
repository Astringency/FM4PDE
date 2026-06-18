from __future__ import annotations

import logging

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import diags


class Simulator:
    """
    Finite-volume simulator for the 2D reaction-diffusion system

        u_t = u - u^3 - k - v + Du * Delta u
        v_t = u - v + Dv * Delta v

    on a cell-centered rectangular grid with homogeneous Neumann boundaries.

    ``init_mode`` controls the initial fields for u and v. ``"iid"`` keeps the
    original per-grid-point standard normal white-noise behavior, while
    ``"grf"`` samples spatially correlated Gaussian random fields. The two
    channels are sampled from independent random streams derived from ``seed``.

    ``t`` is the final simulation time. ``tdim`` is the number of saved time
    nodes including the initial state, so 10 saved intervals on [0, 1] use
    ``t=1.0`` and ``tdim=11``.
    """

    def __init__(
        self,
        Du: float = 1e-3,
        Dv: float = 5e-3,
        k: float = 5e-3,
        t: float = 1.0,
        tdim: int = 11,
        n_save_steps: int | None = None,
        x_left: float = -1.0,
        x_right: float = 1.0,
        xdim: int = 50,
        y_bottom: float = -1.0,
        y_top: float = 1.0,
        ydim: int = 50,
        n: int = 1,  # noqa: ARG002
        seed: int = 0,
        init_mode: str = "grf",
        init_mean: float = 0.0,
        init_std: float = 1.0,
        grf_length_scale: float = 0.15,
        grf_spectral_power: float = 2.0,
        grf_normalize: bool = True,
    ):
        """
        Constructor method initializing the parameters for the diffusion
        reaction problem.
        :param Du: The diffusion coefficient of u
        :param Dv: The diffusion coefficient of v
        :param k: The reaction parameter
        :param t: Stop time of the simulation
        :param tdim: Number of saved frames, including the initial state
        :param n_save_steps: Optional number of saved intervals; if provided,
            it must satisfy tdim == n_save_steps + 1
        :param x_left: Left end of the 2D simulation field
        :param x_right: Right end of the 2D simulation field
        :param xdim: Number of spatial steps between x_left and x_right
        :param y_bottom: bottom end of the 2D simulation field
        :param y_top: top end of the 2D simulation field
        :param ydim: Number of spatial steps between y_bottom and y_top
        :param n: Number of batches
        :param seed: Base seed for reproducible u/v initial fields
        :param init_mode: "iid"/"standard_normal" or
            "grf"/"gaussian_random_field"
        :param init_mean: Target initial-field mean
        :param init_std: Target initial-field standard deviation
        :param grf_length_scale: Physical GRF correlation length scale
        :param grf_spectral_power: Spectral decay power for GRF filtering
        :param grf_normalize: Whether to normalize each GRF sample before
            applying init_mean/init_std
        """

        # Set class parameters
        self.Du = Du
        self.Dv = Dv
        self.k = k

        self.T = t
        self.X0 = x_left
        self.X1 = x_right
        self.Y0 = y_bottom
        self.Y1 = y_top

        self.Nx = xdim
        self.Ny = ydim
        self.Nt = tdim
        if self.Nt < 2:
            raise ValueError("tdim must be at least 2 because it includes t=0 and t=T")
        if n_save_steps is not None and self.Nt != int(n_save_steps) + 1:
            raise ValueError(
                f"tdim={self.Nt} is inconsistent with n_save_steps={n_save_steps}; "
                "tdim must be n_save_steps + 1"
            )

        self.init_mode = self._canonical_init_mode(init_mode)
        self.init_mean = float(init_mean)
        self.init_std = float(init_std)
        self.grf_length_scale = float(grf_length_scale)
        self.grf_spectral_power = float(grf_spectral_power)
        self.grf_normalize = bool(grf_normalize)

        # Calculate grid size and generate grid
        self.dx = (self.X1 - self.X0) / (self.Nx)
        self.dy = (self.Y1 - self.Y0) / (self.Ny)

        self.x = np.linspace(self.X0 + self.dx / 2, self.X1 - self.dx / 2, self.Nx)
        self.y = np.linspace(self.Y0 + self.dy / 2, self.Y1 - self.dy / 2, self.Ny)

        # Time steps to store the simulation results
        self.t = np.linspace(0, self.T, self.Nt)

        # Initialize the logger
        self.log = logging.getLogger(__name__)

        self.seed = seed

    @staticmethod
    def _canonical_init_mode(init_mode: str) -> str:
        mode = str(init_mode).lower()
        aliases = {
            "iid": "iid",
            "standard_normal": "iid",
            "white_noise": "iid",
            "grf": "grf",
            "gaussian_random_field": "grf",
        }
        if mode not in aliases:
            raise ValueError(
                f"Unknown init_mode={init_mode!r}; expected one of "
                "'iid', 'standard_normal', 'grf', or 'gaussian_random_field'"
            )
        return aliases[mode]

    def _sample_initial_field(self, rng: np.random.Generator) -> np.ndarray:
        if self.init_mode == "iid":
            field = rng.standard_normal((self.Ny, self.Nx))
            return (self.init_mean + self.init_std * field).reshape(self.Nx * self.Ny)

        noise = rng.standard_normal((self.Ny, self.Nx))
        freq_x = 2.0 * np.pi * np.fft.fftfreq(self.Nx, d=self.dx)
        freq_y = 2.0 * np.pi * np.fft.fftfreq(self.Ny, d=self.dy)
        kx, ky = np.meshgrid(freq_x, freq_y)
        ksq = kx**2 + ky**2

        ell = max(abs(self.grf_length_scale), np.finfo(float).eps)
        alpha = max(float(self.grf_spectral_power), np.finfo(float).eps)
        amplitude = (ksq + ell**-2) ** (-0.5 * alpha)
        amplitude[0, 0] = 0.0

        field = np.fft.ifft2(np.fft.fft2(noise) * amplitude).real
        if not np.isfinite(field).all():
            raise RuntimeError(
                "Non-finite values encountered while sampling reaction-diffusion "
                f"GRF initial field: seed={self.seed}, init_mode={self.init_mode}, "
                f"length_scale={self.grf_length_scale}, spectral_power={self.grf_spectral_power}"
            )

        if self.grf_normalize:
            field = field - float(np.mean(field))
            field_std = float(np.std(field))
            if field_std > 1e-12:
                field = field / field_std

        field = self.init_mean + self.init_std * field
        return field.reshape(self.Nx * self.Ny)

    def generate_sample(self):
        """
        Single sample generation using the parameters of this simulator.
        :return: The generated sample as numpy array(t, y, x, num_features)
        """

        seed_sequence = np.random.SeedSequence(self.seed)
        rng_u, rng_v = [
            np.random.default_rng(child_seed)
            for child_seed in seed_sequence.spawn(2)
        ]

        u0 = self._sample_initial_field(rng_u)
        v0 = self._sample_initial_field(rng_v)
        u0 = np.concatenate((u0, v0))

        # # Normalize u0
        # u0 = 2 * (u0 - u0.min()) / (u0.max() - u0.min()) - 1

        # Generate arrays as diagonal inputs to the Laplacian matrix
        main_diag = (
            -2 * np.ones(self.Nx) / self.dx**2 - 2 * np.ones(self.Nx) / self.dy**2
        )
        main_diag[0] = -1 / self.dx**2 - 2 / self.dy**2
        main_diag[-1] = -1 / self.dx**2 - 2 / self.dy**2
        main_diag = np.tile(main_diag, self.Ny)
        main_diag[: self.Nx] = -2 / self.dx**2 - 1 / self.dy**2
        main_diag[self.Nx * (self.Ny - 1) :] = -2 / self.dx**2 - 1 / self.dy**2
        main_diag[0] = -1 / self.dx**2 - 1 / self.dy**2
        main_diag[self.Nx - 1] = -1 / self.dx**2 - 1 / self.dy**2
        main_diag[self.Nx * (self.Ny - 1)] = -1 / self.dx**2 - 1 / self.dy**2
        main_diag[-1] = -1 / self.dx**2 - 1 / self.dy**2

        left_diag = np.ones(self.Nx)
        left_diag[0] = 0
        left_diag = np.tile(left_diag, self.Ny)
        left_diag = left_diag[1:] / self.dx**2

        right_diag = np.ones(self.Nx)
        right_diag[-1] = 0
        right_diag = np.tile(right_diag, self.Ny)
        right_diag = right_diag[:-1] / self.dx**2

        bottom_diag = np.ones(self.Nx * (self.Ny - 1)) / self.dy**2

        top_diag = np.ones(self.Nx * (self.Ny - 1)) / self.dy**2

        # Generate the sparse Laplacian matrix
        diagonals = [main_diag, left_diag, right_diag, bottom_diag, top_diag]
        offsets = [0, -1, 1, -self.Nx, self.Nx]
        self.lap = diags(diagonals, offsets)

        # Solve the diffusion reaction problem
        prob = solve_ivp(self.rc_ode, (0, self.T), u0, t_eval=self.t)
        if not prob.success:
            raise RuntimeError(
                "Reaction-diffusion solve_ivp failed: "
                f"seed={self.seed}, T={self.T}, Nt={self.Nt}, "
                f"init_mode={self.init_mode}, message={prob.message}"
            )
        ode_data = prob.y

        sample_u = np.transpose(ode_data[: self.Nx * self.Ny]).reshape(
            -1, self.Ny, self.Nx
        )
        sample_v = np.transpose(ode_data[self.Nx * self.Ny :]).reshape(
            -1, self.Ny, self.Nx
        )

        return np.stack((sample_u, sample_v), axis=-1)

    def rc_ode(self, t, y):  # noqa: ARG002
        """
        Solves a given equation for a particular time step.
        :param t: The current time step
        :param y: The equation values to solve
        :return: A finite volume solution
        """

        # Separate y into u and v
        u = y[: self.Nx * self.Ny]
        v = y[self.Nx * self.Ny :]

        # Calculate reaction function for each unknown
        react_u = u - u**3 - self.k - v
        react_v = u - v

        # Calculate time derivative for each unknown
        u_t = react_u + self.Du * (self.lap @ u)
        v_t = react_v + self.Dv * (self.lap @ v)

        # Stack the time derivative into a single array y_t
        return np.concatenate((u_t, v_t))
