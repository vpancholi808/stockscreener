"""Markowitz mean-variance optimization.

Provides the global minimum-variance and maximum-Sharpe (tangency) portfolios,
an efficient frontier, and target-return portfolios. Two solver paths:

  - analytic (unconstrained, weights sum to 1, shorting allowed) via the
    closed-form two-fund theorem — exact and used to validate the numeric path;
  - numeric (long-only or custom bounds) via scipy SLSQP.

Covariance defaults to Ledoit-Wolf shrinkage toward a scaled identity, which is
far better conditioned than the raw sample covariance when assets >~ observations.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .universe import Universe


def ledoit_wolf_cov(returns: pd.DataFrame, annualize: int = 1) -> pd.DataFrame:
    """Ledoit-Wolf shrinkage of the sample covariance toward mu*I."""
    X = returns.to_numpy()
    n, p = X.shape
    Xc = X - X.mean(axis=0)
    sample = (Xc.T @ Xc) / n
    mu = np.trace(sample) / p
    target = mu * np.eye(p)
    # Shrinkage intensity (Ledoit-Wolf 2004).
    d2 = np.sum((sample - target) ** 2)
    b2 = 0.0
    for i in range(n):
        xi = Xc[i][:, None]
        b2 += np.sum((xi @ xi.T - sample) ** 2)
    b2 = b2 / n ** 2
    b2 = min(b2, d2)
    shrink = b2 / d2 if d2 > 0 else 0.0
    cov = shrink * target + (1 - shrink) * sample
    return pd.DataFrame(cov * annualize, index=returns.columns, columns=returns.columns)


@dataclass
class Portfolio:
    weights: pd.Series
    expected_return: float
    volatility: float
    sharpe: float

    def __str__(self) -> str:
        w = ", ".join(f"{k}={v:.2%}" for k, v in self.weights.items())
        return (f"E[r]={self.expected_return:.2%}  vol={self.volatility:.2%}  "
                f"Sharpe={self.sharpe:.2f}\n  weights: {w}")


def _stats(w, mu, cov, rf):
    r = float(w @ mu)
    v = float(np.sqrt(w @ cov @ w))
    return r, v, (r - rf) / v if v else float("nan")


class MarkowitzOptimizer:
    def __init__(self, mu: pd.Series, cov: pd.DataFrame, rf: float = 0.0):
        common = mu.index.intersection(cov.index)
        self.assets = list(common)
        self.mu = mu.loc[common].to_numpy()
        self.cov = cov.loc[common, common].to_numpy()
        self.rf = rf
        self._inv = np.linalg.pinv(self.cov)

    @classmethod
    def from_universe(cls, u: Universe, rf: float = 0.0,
                      shrinkage: bool = True) -> "MarkowitzOptimizer":
        mu = u.mean_returns(annualized=True)
        cov = (ledoit_wolf_cov(u.returns, u.periods_per_year) if shrinkage
               else u.cov_matrix(annualized=True))
        return cls(mu, cov, rf)

    def _wrap(self, w) -> Portfolio:
        r, v, s = _stats(w, self.mu, self.cov, self.rf)
        return Portfolio(pd.Series(w, index=self.assets), r, v, s)

    # ---- analytic (unconstrained) ----------------------------------------
    def min_variance(self) -> Portfolio:
        ones = np.ones(len(self.assets))
        w = self._inv @ ones
        return self._wrap(w / (ones @ w))

    def max_sharpe(self) -> Portfolio:
        excess = self.mu - self.rf
        w = self._inv @ excess
        return self._wrap(w / w.sum())

    def target_return(self, target: float) -> Portfolio:
        # Two-fund frontier: solve for weights with min variance at given return.
        ones = np.ones(len(self.assets))
        A = ones @ self._inv @ ones
        B = ones @ self._inv @ self.mu
        C = self.mu @ self._inv @ self.mu
        D = A * C - B ** 2
        lam = (C - B * target) / D
        gam = (A * target - B) / D
        w = lam * (self._inv @ ones) + gam * (self._inv @ self.mu)
        return self._wrap(w)

    # ---- numeric (constrained, e.g. long-only) ---------------------------
    def optimize(self, objective: str = "max_sharpe", long_only: bool = True,
                 bounds: Optional[tuple] = None,
                 target_return: Optional[float] = None) -> Portfolio:
        n = len(self.assets)
        w0 = np.ones(n) / n
        bnds = bounds or ((0.0, 1.0) if long_only else (-1.0, 1.0),) * n
        if len(bnds) != n:
            bnds = (bnds[0],) * n
        cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
        if target_return is not None:
            cons.append({"type": "eq",
                         "fun": lambda w: float(w @ self.mu) - target_return})

        if objective == "min_variance":
            fun = lambda w: float(w @ self.cov @ w)
        elif objective == "max_sharpe":
            def fun(w):
                r, v, s = _stats(w, self.mu, self.cov, self.rf)
                return -s
        else:
            raise ValueError("objective must be 'max_sharpe' or 'min_variance'")

        res = minimize(fun, w0, method="SLSQP", bounds=bnds, constraints=cons,
                       options={"maxiter": 500, "ftol": 1e-10})
        return self._wrap(res.x)

    def efficient_frontier(self, points: int = 25,
                           long_only: bool = False) -> pd.DataFrame:
        lo, hi = self.mu.min(), self.mu.max()
        targets = np.linspace(lo, hi, points)
        rows = []
        for t in targets:
            try:
                p = (self.optimize("min_variance", long_only=long_only,
                                   target_return=float(t)) if long_only
                     else self.target_return(float(t)))
                rows.append({"target_return": t, "return": p.expected_return,
                             "volatility": p.volatility, "sharpe": p.sharpe})
            except Exception:
                continue
        return pd.DataFrame(rows)
