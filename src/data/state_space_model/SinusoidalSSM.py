"""Sinusoidal nonlinear state-space model definition."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from utils.tools import dB_to_lin


class SinusoidalSSM:
    """Simple sinusoidal state transition with affine observation mapping."""

    def __init__(
        self,
        n_states: int,
        alpha: float = 0.9,
        beta: float = 1.1,
        phi: float = 0.1 * np.pi,
        delta: float = 0.01,
        delta_d: Optional[float] = None,
        a: float = 1.0,
        b: float = 1.0,
        c: float = 0.0,
        decimate: bool = False,
        mu_e: Optional[np.ndarray] = None,
        mu_w: Optional[np.ndarray] = None,
        use_Taylor: bool = False,
    ) -> None:
        self.n_states = n_states
        self.delta = delta
        self.alpha = alpha
        self.beta = beta
        self.phi = phi
        self.a = a
        self.b = b
        self.c = c
        self.delta_d = delta_d if delta_d is not None else delta
        self.n_obs = self.h_fn(np.random.randn(self.n_states, 1)).shape[0]
        self.decimate = decimate
        self.mu_e = mu_e if mu_e is not None else np.zeros(self.n_states)
        self.mu_w = mu_w if mu_w is not None else np.zeros(self.n_obs)
        self.use_Taylor = use_Taylor

    def init_noise_covs(self) -> None:
        """Initialise diagonal noise covariance matrices."""

        self.Q = self.q * np.eye(self.n_states)
        self.R = self.r * np.eye(self.n_obs)

    def h_fn(self, x: np.ndarray) -> np.ndarray:
        """Observation mapping: y = a * (b * x + c)."""

        return self.a * (self.b * x + self.c)

    def f_fn(self, x: np.ndarray) -> np.ndarray:
        """State transition: x_next = alpha * sin(beta * x + phi) + delta."""

        return self.alpha * np.sin(self.beta * x + self.phi) + self.delta

    def generate_single_sequence(
        self,
        T: int,
        inverse_r2_dB: float,
        nu_dB: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a sinusoidal SSM trajectory."""

        x = np.zeros((T + 1, self.n_states))
        y = np.zeros((T, self.n_obs))

        r2 = 1.0 / dB_to_lin(inverse_r2_dB)
        q2 = dB_to_lin(nu_dB - inverse_r2_dB)
        self.r = r2
        self.q = q2
        self.init_noise_covs()

        e = np.random.multivariate_normal(self.mu_e, q2 * np.eye(self.n_states), size=(T + 1,))
        v = np.random.multivariate_normal(self.mu_w, r2 * np.eye(self.n_obs), size=(T,))

        for t in range(T):
            x[t + 1] = self.f_fn(x[t]) + e[t]
            y[t] = self.h_fn(x[t]) + v[t]

        if self.decimate:
            K = max(int(self.delta_d // self.delta), 1)
            x_d = x[0:T:K, :]
            y_d = self.h_fn(x_d) + np.random.multivariate_normal(self.mu_e, self.R, size=(len(x_d),))
        else:
            x_d = x
            y_d = y

        return x_d, y_d
