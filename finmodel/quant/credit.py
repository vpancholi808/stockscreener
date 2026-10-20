"""Merton structural credit-risk model.

Treats a firm's equity as a European call on its asset value V, struck at the
face value of debt D (the option to walk away). Observed equity value E and
equity volatility sigma_E imply the unobservable asset value and asset
volatility via the simultaneous equations:

    E = V N(d1) - D e^{-rT} N(d2)
    sigma_E * E = N(d1) * sigma_V * V

From the solved (V, sigma_V) we read off distance-to-default, default
probability, and the credit spread. Ties directly to the 3-statement balance
sheet (D = total debt) and the CAPM layer (sigma_E from return volatility).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import fsolve
from scipy.stats import norm


@dataclass
class MertonCredit:
    asset_value: float
    asset_volatility: float
    equity_value: float
    equity_volatility: float
    debt_face: float
    r: float
    T: float

    @property
    def distance_to_default(self) -> float:
        d2 = (np.log(self.asset_value / self.debt_face)
              + (self.r - 0.5 * self.asset_volatility ** 2) * self.T) \
            / (self.asset_volatility * np.sqrt(self.T))
        return float(d2)

    @property
    def default_probability(self) -> float:
        """Risk-neutral PD = N(-DD)."""
        return float(norm.cdf(-self.distance_to_default))

    @property
    def risky_debt_value(self) -> float:
        d1 = (np.log(self.asset_value / self.debt_face)
              + (self.r + 0.5 * self.asset_volatility ** 2) * self.T) \
            / (self.asset_volatility * np.sqrt(self.T))
        d2 = d1 - self.asset_volatility * np.sqrt(self.T)
        return float(self.asset_value * norm.cdf(-d1)
                     + self.debt_face * np.exp(-self.r * self.T) * norm.cdf(d2))

    @property
    def credit_spread(self) -> float:
        """Yield premium of risky debt over the risk-free rate (continuous)."""
        y = -np.log(self.risky_debt_value / self.debt_face) / self.T
        return float(y - self.r)

    def __str__(self) -> str:
        return (f"Asset value {self.asset_value:,.2f}, asset vol "
                f"{self.asset_volatility:.2%}\n"
                f"Distance-to-default {self.distance_to_default:.2f}  |  "
                f"PD {self.default_probability:.2%}  |  "
                f"credit spread {self.credit_spread*1e4:.0f} bps")


def solve_merton(equity_value: float, equity_vol: float, debt_face: float,
                 r: float, T: float = 1.0) -> MertonCredit:
    """Recover (asset value, asset vol) from observed equity value and vol."""
    def equations(x):
        V, sV = x
        if V <= 0 or sV <= 0:
            return [1e6, 1e6]
        d1 = (np.log(V / debt_face) + (r + 0.5 * sV ** 2) * T) / (sV * np.sqrt(T))
        d2 = d1 - sV * np.sqrt(T)
        eq1 = V * norm.cdf(d1) - debt_face * np.exp(-r * T) * norm.cdf(d2) - equity_value
        eq2 = norm.cdf(d1) * sV * V - equity_vol * equity_value
        return [eq1, eq2]

    V0 = equity_value + debt_face * np.exp(-r * T)   # sensible starting guess
    sV0 = equity_vol * equity_value / V0
    V, sV = fsolve(equations, [V0, max(sV0, 1e-3)], full_output=False)
    return MertonCredit(asset_value=float(V), asset_volatility=float(sV),
                        equity_value=equity_value, equity_volatility=equity_vol,
                        debt_face=debt_face, r=r, T=T)


def merton_from_model(three_statement, equity_value: float, equity_vol: float,
                      r: float = 0.042, T: float = 1.0) -> MertonCredit:
    """Build a Merton credit assessment using the balance sheet's total debt."""
    bs = three_statement.financials.balance_sheet
    y = three_statement.financials.latest_year()

    def g(item):
        return bs.at[item, y] if item in bs.index and y in bs.columns else float("nan")

    debt = float(np.nansum([g("long_term_debt"), g("short_term_debt")]))
    if debt <= 0:
        raise ValueError("No reported debt; Merton model needs a positive debt face value.")
    return solve_merton(equity_value, equity_vol, debt, r, T)
