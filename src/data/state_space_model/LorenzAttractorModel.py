"""Lorenz attractor based nonlinear state-space model."""

from __future__ import annotations

import math
from typing import Callable, Optional, Tuple

import numpy as np

from utils.tools import dB_to_lin


class LorenzAttractorModel:
    """Approximate Lorenz dynamics via truncated Taylor expansion."""

    def __init__(
        self,
        d: int,
        J: int,
        delta: float,
        delta_d: float,
        A_fn: Optional[Callable[[np.ndarray], np.ndarray]],
        h_fn: Optional[Callable[[np.ndarray], np.ndarray]],
        decimate: bool = False,
        mu_e: Optional[np.ndarray] = None,
        mu_w: Optional[np.ndarray] = None,
        use_Taylor: bool = True,
    ) -> None:
        self.n_states = d
        self.J = J
        self.delta = delta
        self.delta_d = delta_d
        self.A_fn = A_fn if A_fn is not None else (lambda x: np.eye(d))
        self._h_fn = h_fn if h_fn is not None else (lambda x: x)
        self.n_obs = self._h_fn(np.random.randn(d, 1)).shape[0]
        self.decimate = decimate
        self.mu_e = mu_e if mu_e is not None else np.zeros(d)
        self.mu_w = mu_w if mu_w is not None else np.zeros(self.n_obs)
        self.use_Taylor = use_Taylor

        self.Q = None
        self.R = None
        self.q2 = None
        self.r2 = None

    def h_fn(self, x: np.ndarray) -> np.ndarray:
        """Observation mapping wrapper."""

        return self._h_fn(x)

    def f_linearize(self, x: np.ndarray) -> np.ndarray:
        """One-step prediction using truncated matrix exponential."""

        if not self.use_Taylor:
            return self.A_fn(x) @ x

        self.F = np.eye(self.n_states)
        for j in range(1, self.J + 1):
            self.F += np.linalg.matrix_power(self.A_fn(x) * self.delta, j) / math.factorial(j)
        return self.F @ x

    def init_noise_covs(self) -> None:
        """Initialise diagonal covariances based on ``q2`` and ``r2``."""

        self.Q = self.q2 * np.eye(self.n_states)
        self.R = self.r2 * np.eye(self.n_obs)

    def generate_single_sequence(
        self,
        T: int,
        inverse_r2_dB: float,
        nu_dB: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a single Lorenz trajectory and its observations."""

        x = np.zeros((T + 1, self.n_states))
        y = np.zeros((T, self.n_obs))

        r2 = 1.0 / dB_to_lin(inverse_r2_dB)
        q2 = dB_to_lin(nu_dB - inverse_r2_dB)
        self.r2 = r2
        self.q2 = q2
        self.init_noise_covs()

        e = np.random.multivariate_normal(self.mu_e, self.Q, size=(T + 1,))
        v = np.random.multivariate_normal(self.mu_w, self.R, size=(T,))

        for t in range(T):
            x[t + 1] = self.f_linearize(x[t]) + e[t]
            y[t] = self.h_fn(x[t]) + v[t]

        if self.decimate:
            K = max(int(round(self.delta_d / self.delta)), 1)
            x_lorenz_d = x[0:T:K, :]
            y_lorenz_d = self.h_fn(x_lorenz_d) + np.random.multivariate_normal(
                self.mu_w, self.R, size=(len(x_lorenz_d),)
            )
        else:
            x_lorenz_d = x
            y_lorenz_d = y

        return x_lorenz_d, y_lorenz_d

