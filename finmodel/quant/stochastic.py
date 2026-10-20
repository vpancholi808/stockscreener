"""Stochastic-calculus foundations: the simulation engine everything else uses.

Implements the core Itô processes as path simulators plus a generic
Euler-Maruyama integrator:

  - Brownian motion (arithmetic, Bachelier-style)
  - Geometric Brownian motion (Black-Scholes underlying)
  - Ornstein-Uhlenbeck (mean-reverting; basis for Vasicek rates, pairs spreads)
  - Merton jump-diffusion (GBM + compound-Poisson jumps)
  - Heston (stochastic variance via a CIR process, full-truncation Euler)

All simulators return arrays shaped (n_paths, n_steps + 1) including the initial
value. A fixed seed gives reproducible draws for testing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


def _rng(seed: Optional[int]) -> np.random.Generator:
    return np.random.default_rng(seed)


def brownian_motion(T: float, n_steps: int, n_paths: int = 1, mu: float = 0.0,
                    sigma: float = 1.0, x0: float = 0.0,
                    seed: Optional[int] = None) -> np.ndarray:
    """Arithmetic BM: dX = mu dt + sigma dW."""
    dt = T / n_steps
    g = _rng(seed)
    dW = g.normal(0.0, np.sqrt(dt), (n_paths, n_steps))
    incr = mu * dt + sigma * dW
    paths = np.empty((n_paths, n_steps + 1))
    paths[:, 0] = x0
    paths[:, 1:] = x0 + np.cumsum(incr, axis=1)
    return paths


def gbm(S0: float, mu: float, sigma: float, T: float, n_steps: int,
        n_paths: int = 1, seed: Optional[int] = None) -> np.ndarray:
    """Geometric BM: dS = mu S dt + sigma S dW (exact log-Euler)."""
    dt = T / n_steps
    g = _rng(seed)
    dW = g.normal(0.0, np.sqrt(dt), (n_paths, n_steps))
    log_incr = (mu - 0.5 * sigma ** 2) * dt + sigma * dW
    paths = np.empty((n_paths, n_steps + 1))
    paths[:, 0] = S0
    paths[:, 1:] = S0 * np.exp(np.cumsum(log_incr, axis=1))
    return paths


def ornstein_uhlenbeck(x0: float, kappa: float, theta: float, sigma: float,
                       T: float, n_steps: int, n_paths: int = 1,
                       seed: Optional[int] = None) -> np.ndarray:
    """Mean-reverting OU: dX = kappa(theta - X) dt + sigma dW (exact scheme)."""
    dt = T / n_steps
    g = _rng(seed)
    paths = np.empty((n_paths, n_steps + 1))
    paths[:, 0] = x0
    exp_k = np.exp(-kappa * dt)
    std = sigma * np.sqrt((1 - np.exp(-2 * kappa * dt)) / (2 * kappa)) if kappa > 0 \
        else sigma * np.sqrt(dt)
    for t in range(n_steps):
        z = g.normal(0.0, 1.0, n_paths)
        paths[:, t + 1] = theta + (paths[:, t] - theta) * exp_k + std * z
    return paths


def jump_diffusion(S0: float, mu: float, sigma: float, T: float, n_steps: int,
                   lam: float, jump_mean: float, jump_std: float,
                   n_paths: int = 1, seed: Optional[int] = None) -> np.ndarray:
    """Merton jump-diffusion: GBM plus compound-Poisson log-normal jumps."""
    dt = T / n_steps
    g = _rng(seed)
    paths = np.empty((n_paths, n_steps + 1))
    paths[:, 0] = S0
    for t in range(n_steps):
        dW = g.normal(0.0, np.sqrt(dt), n_paths)
        n_jumps = g.poisson(lam * dt, n_paths)
        jump = g.normal(jump_mean, jump_std, n_paths) * n_jumps
        drift = (mu - 0.5 * sigma ** 2) * dt
        paths[:, t + 1] = paths[:, t] * np.exp(drift + sigma * dW + jump)
    return paths


def heston_paths(S0: float, v0: float, mu: float, kappa: float, theta: float,
                 xi: float, rho: float, T: float, n_steps: int, n_paths: int = 1,
                 seed: Optional[int] = None) -> tuple[np.ndarray, np.ndarray]:
    """Heston: dS = mu S dt + sqrt(v) S dW1 ; dv = kappa(theta - v) dt + xi sqrt(v) dW2,
    corr(dW1,dW2)=rho. Full-truncation Euler keeps variance non-negative."""
    dt = T / n_steps
    g = _rng(seed)
    S = np.empty((n_paths, n_steps + 1)); S[:, 0] = S0
    v = np.empty((n_paths, n_steps + 1)); v[:, 0] = v0
    for t in range(n_steps):
        z1 = g.normal(0.0, 1.0, n_paths)
        z2 = rho * z1 + np.sqrt(1 - rho ** 2) * g.normal(0.0, 1.0, n_paths)
        vt = np.maximum(v[:, t], 0.0)
        sq = np.sqrt(vt * dt)
        S[:, t + 1] = S[:, t] * np.exp((mu - 0.5 * vt) * dt + sq * z1)
        v[:, t + 1] = v[:, t] + kappa * (theta - vt) * dt + xi * np.sqrt(vt) * np.sqrt(dt) * z2
    return S, v


def euler_maruyama(drift: Callable, diffusion: Callable, x0: float, T: float,
                   n_steps: int, n_paths: int = 1,
                   seed: Optional[int] = None) -> np.ndarray:
    """Generic SDE integrator: dX = drift(x,t) dt + diffusion(x,t) dW."""
    dt = T / n_steps
    g = _rng(seed)
    paths = np.empty((n_paths, n_steps + 1))
    paths[:, 0] = x0
    for t in range(n_steps):
        x, tau = paths[:, t], t * dt
        dW = g.normal(0.0, np.sqrt(dt), n_paths)
        paths[:, t + 1] = x + drift(x, tau) * dt + diffusion(x, tau) * dW
    return paths


def heston_mc_price(S0: float, K: float, r: float, v0: float, kappa: float,
                    theta: float, xi: float, rho: float, T: float,
                    option: str = "call", n_steps: int = 100,
                    n_paths: int = 100_000, seed: Optional[int] = None) -> float:
    """European option price under Heston by Monte Carlo (risk-neutral, mu=r)."""
    S, _ = heston_paths(S0, v0, r, kappa, theta, xi, rho, T, n_steps, n_paths, seed)
    ST = S[:, -1]
    payoff = np.maximum(ST - K, 0.0) if option == "call" else np.maximum(K - ST, 0.0)
    return float(np.exp(-r * T) * payoff.mean())
