"""Short-rate interest-rate models: Vasicek and CIR.

Both are affine term-structure models, so a zero-coupon bond has a closed form
P(t,T) = A(T-t) * exp(-B(T-t) * r). We expose the analytic bond price (and the
implied continuously-compounded yield) plus a path simulator, so the analytic
price can be checked against the Monte-Carlo average of exp(-∫ r ds).

  Vasicek: dr = kappa(theta - r) dt + sigma dW        (Gaussian; can go negative)
  CIR:     dr = kappa(theta - r) dt + sigma sqrt(r) dW (non-negative if Feller holds)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_trapz = getattr(np, "trapezoid", None) or np.trapz  # numpy 2.x renamed trapz

from .stochastic import ornstein_uhlenbeck


@dataclass
class Vasicek:
    kappa: float
    theta: float
    sigma: float
    r0: float

    def zcb_price(self, T: float) -> float:
        k, th, s = self.kappa, self.theta, self.sigma
        B = (1 - np.exp(-k * T)) / k
        A = np.exp((th - s ** 2 / (2 * k ** 2)) * (B - T) - s ** 2 * B ** 2 / (4 * k))
        return float(A * np.exp(-B * self.r0))

    def yield_to(self, T: float) -> float:
        return float(-np.log(self.zcb_price(T)) / T)

    def simulate(self, T: float, n_steps: int, n_paths: int = 1, seed=None):
        return ornstein_uhlenbeck(self.r0, self.kappa, self.theta, self.sigma,
                                  T, n_steps, n_paths, seed)

    def zcb_price_mc(self, T: float, n_steps: int = 200, n_paths: int = 50_000,
                     seed=None) -> float:
        paths = self.simulate(T, n_steps, n_paths, seed)
        dt = T / n_steps
        integral = _trapz(paths, dx=dt, axis=1)
        return float(np.mean(np.exp(-integral)))


@dataclass
class CIR:
    kappa: float
    theta: float
    sigma: float
    r0: float

    @property
    def feller_satisfied(self) -> bool:
        """2*kappa*theta >= sigma^2 keeps the rate strictly positive."""
        return 2 * self.kappa * self.theta >= self.sigma ** 2

    def zcb_price(self, T: float) -> float:
        k, th, s = self.kappa, self.theta, self.sigma
        h = np.sqrt(k ** 2 + 2 * s ** 2)
        e = np.exp(h * T)
        denom = 2 * h + (k + h) * (e - 1)
        A = (2 * h * np.exp((k + h) * T / 2) / denom) ** (2 * k * th / s ** 2)
        B = 2 * (e - 1) / denom
        return float(A * np.exp(-B * self.r0))

    def yield_to(self, T: float) -> float:
        return float(-np.log(self.zcb_price(T)) / T)

    def simulate(self, T: float, n_steps: int, n_paths: int = 1, seed=None):
        dt = T / n_steps
        g = np.random.default_rng(seed)
        paths = np.empty((n_paths, n_steps + 1))
        paths[:, 0] = self.r0
        for t in range(n_steps):
            r = np.maximum(paths[:, t], 0.0)
            dW = g.normal(0.0, np.sqrt(dt), n_paths)
            paths[:, t + 1] = np.maximum(
                r + self.kappa * (self.theta - r) * dt + self.sigma * np.sqrt(r) * dW,
                0.0)
        return paths

    def zcb_price_mc(self, T: float, n_steps: int = 200, n_paths: int = 50_000,
                     seed=None) -> float:
        paths = self.simulate(T, n_steps, n_paths, seed)
        dt = T / n_steps
        integral = _trapz(paths, dx=dt, axis=1)
        return float(np.mean(np.exp(-integral)))


def yield_curve(model, maturities) -> "list[tuple[float, float]]":
    """Convenience: (maturity, yield) pairs for plotting a term structure."""
    return [(float(T), model.yield_to(float(T))) for T in maturities]
