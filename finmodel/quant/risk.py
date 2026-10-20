"""Portfolio downside risk: Value-at-Risk, Conditional VaR, and stress testing.

VaR is reported as a positive loss fraction at a given confidence (e.g. 0.95).
Three estimators:
  - historical: empirical quantile of realized returns (no distribution assumed);
  - parametric (Gaussian): mean - z*sigma;
  - Cornish-Fisher: parametric adjusted for skew and excess kurtosis (fat tails).
CVaR (expected shortfall) is the mean loss beyond the VaR threshold.

Stress testing applies named shock scenarios (or replays a historical window) to
a portfolio and reports the implied P&L.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats


def portfolio_returns(asset_returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    w = weights.reindex(asset_returns.columns).fillna(0.0)
    return asset_returns @ w


@dataclass
class RiskReport:
    confidence: float
    horizon_days: int
    var_historical: float
    var_parametric: float
    var_cornish_fisher: float
    cvar_historical: float
    volatility_annual: float
    max_drawdown: float
    skew: float
    excess_kurtosis: float

    def __str__(self) -> str:
        c = int(self.confidence * 100)
        return (f"{c}% VaR ({self.horizon_days}d): hist={self.var_historical:.2%}  "
                f"param={self.var_parametric:.2%}  CF={self.var_cornish_fisher:.2%}\n"
                f"{c}% CVaR: {self.cvar_historical:.2%} | "
                f"ann vol {self.volatility_annual:.2%} | "
                f"max DD {self.max_drawdown:.2%} | "
                f"skew {self.skew:.2f} kurt(excess) {self.excess_kurtosis:.2f}")


def _max_drawdown(returns: pd.Series) -> float:
    curve = (1 + returns).cumprod()
    peak = curve.cummax()
    return float((curve / peak - 1).min())


def risk_report(returns: pd.Series, confidence: float = 0.95,
                horizon_days: int = 1, periods_per_year: int = 252) -> RiskReport:
    r = returns.dropna().to_numpy()
    alpha = 1 - confidence
    z = stats.norm.ppf(alpha)
    mu, sigma = r.mean(), r.std(ddof=1)
    s = stats.skew(r)
    k = stats.kurtosis(r)  # excess kurtosis (Fisher)

    scale = np.sqrt(horizon_days)
    var_hist = -np.quantile(r, alpha) * scale
    var_param = -(mu + z * sigma) * scale
    # Cornish-Fisher adjusted quantile.
    z_cf = (z + (z**2 - 1) * s / 6 + (z**3 - 3*z) * k / 24
            - (2*z**3 - 5*z) * s**2 / 36)
    var_cf = -(mu + z_cf * sigma) * scale
    tail = r[r <= np.quantile(r, alpha)]
    cvar = -tail.mean() * scale if tail.size else float("nan")

    return RiskReport(
        confidence=confidence, horizon_days=horizon_days,
        var_historical=var_hist, var_parametric=var_param,
        var_cornish_fisher=var_cf, cvar_historical=cvar,
        volatility_annual=sigma * np.sqrt(periods_per_year),
        max_drawdown=_max_drawdown(returns), skew=float(s), excess_kurtosis=float(k))


def stress_test(weights: pd.Series, scenarios: dict[str, dict[str, float]],
                portfolio_value: float = 1.0) -> pd.DataFrame:
    """Apply named per-asset shock scenarios. scenarios = {name: {ticker: ret}}."""
    rows = []
    for name, shocks in scenarios.items():
        shock_vec = pd.Series(shocks).reindex(weights.index).fillna(0.0)
        pnl = float((weights * shock_vec).sum())
        rows.append({"scenario": name, "portfolio_return": pnl,
                     "pnl": pnl * portfolio_value})
    return pd.DataFrame(rows).set_index("scenario")


def historical_scenario(asset_returns: pd.DataFrame, weights: pd.Series,
                        start: str, end: str) -> dict:
    """Replay a historical window (e.g. a crash) on the current weights."""
    window = asset_returns.loc[start:end]
    port = portfolio_returns(window, weights)
    cumulative = float((1 + port).prod() - 1)
    return {"start": start, "end": end, "cumulative_return": cumulative,
            "worst_day": float(port.min()), "days": len(port)}
