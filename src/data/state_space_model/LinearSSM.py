"""Linear state-space model utilities for dataset generation.

This module implements a configurable linear SSM used by the data generation
pipeline. It was extracted from ``generate_data.py`` to decouple model
definitions from the dataset tooling layer.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

from utils.tools import dB_to_lin, generate_normal


class LinearSSM:
    """Linear state-space model with optional control input.

    The dynamics follow

    x_{k+1} = F x_k + G u_k + e_k
    y_k     = H x_k + w_k

    where both ``e_k`` and ``w_k`` are Gaussian noises.
    """

    def __init__(
        self,
        n_states: int,
        n_obs: int,
        F: Optional[np.ndarray] = None,
        G: Optional[np.ndarray] = None,
        H: Optional[np.ndarray] = None,
        mu_e=0.0,
        mu_w=0.0,
        q2: float = 1.0,
        r2: float = 1.0,
        Q: Optional[np.ndarray] = None,
        R: Optional[np.ndarray] = None,
    ) -> None:
        self.n_states = n_states
        self.n_obs = n_obs

        if F is None and H is None:
            self.F = self.construct_F()
            self.H = self.construct_H()
        else:
            self.F = F
            self.H = H

        self.G = G
        self.mu_e = mu_e
        self.mu_w = mu_w
        self.q2 = q2
        self.r2 = r2
        self.Q = Q
        self.R = R

        if self.Q is None and self.R is None:
            self.init_noise_covs()

    def construct_F(self) -> np.ndarray:
        """Construct a default transition matrix when none is provided."""

        m = self.n_states
        F_sys = np.eye(m) + np.concatenate(
            (
                np.zeros((m, 1)),
                np.concatenate((np.ones((1, m - 1)), np.zeros((m - 1, m - 1))), axis=0),
            ),
            axis=1,
        )
        return F_sys

    def construct_H(self) -> np.ndarray:
        """Construct a default observation matrix when none is provided."""

        H_sys = np.rot90(np.eye(self.n_states, self.n_states)) + np.concatenate(
            (
                np.concatenate(
                    (np.ones((1, self.n_states - 1)), np.zeros((self.n_states - 1, self.n_states - 1))),
                    axis=0,
                ),
                np.zeros((self.n_states, 1)),
            ),
            axis=1,
        )
        return H_sys[: self.n_obs, : self.n_states]

    def init_noise_covs(self) -> None:
        """Initialise process and measurement covariances."""

        self.Q = self.q2 * np.eye(self.n_states)
        self.R = self.r2 * np.eye(self.n_obs)

    def generate_driving_noise(self, k: int, a: float = 1.2, add_noise: bool = False) -> np.ndarray:
        """Generate optional driving noise signal for control input."""

        if not add_noise:
            u_k = np.cos(a * (k + 1))
        else:
            u_k = np.cos(a * (k + 1) + np.random.normal(loc=0.0, scale=math.pi, size=(1, 1)))
        return np.asarray(u_k).reshape(-1, 1)

    def generate_single_sequence(
        self,
        T: int,
        inverse_r2_dB: float = 0.0,
        nu_dB: float = 0.0,
        drive_noise: bool = False,
        add_noise_flag: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate one trajectory of length ``T``."""

        x_arr = np.zeros((T + 1, self.n_states))
        y_arr = np.zeros((T, self.n_obs))

        r2 = 1.0 / dB_to_lin(inverse_r2_dB)
        q2 = dB_to_lin(nu_dB - inverse_r2_dB)
        self.r2 = r2
        self.q2 = q2
        self.init_noise_covs()

        e_k_arr = generate_normal(N=T, mean=self.mu_e, Sigma2=self.Q)
        w_k_arr = generate_normal(N=T, mean=self.mu_w, Sigma2=self.R)

        for k in range(T):
            if drive_noise:
                u_k = self.generate_driving_noise(k, a=1.2, add_noise=add_noise_flag)
            else:
                u_k = np.zeros((self.G.shape[1], 1)) if self.G is not None else np.zeros((1, 1))

            e_k = e_k_arr[k]
            w_k = w_k_arr[k]

            control_term = self.G @ u_k if self.G is not None else 0.0
            x_next = self.F @ x_arr[k].reshape((-1, 1)) + control_term + e_k.reshape((-1, 1))
            x_arr[k + 1] = x_next.reshape((-1,))
            y_arr[k] = self.H @ x_arr[k] + w_k

        return x_arr, y_arr

