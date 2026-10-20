"""CAPM / beta layer.

Computes beta by regressing a stock's excess returns on the market's excess
returns (OLS), then cost of equity via CAPM and a WACC that pulls capital
structure and cost of debt from the 3-statement model. WACC is the discount
rate the DCF layer consumes next.

Beta methodology (defaults are the common cost-of-equity convention):
  - 5 years of monthly total returns
  - excess returns over the per-period risk-free rate
  - raw beta = OLS slope; adjusted beta = Blume: 0.67*raw + 0.33 (mean-reversion
    toward 1.0, as reported by Bloomberg) — use adjusted for forward-looking ke
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .data.prices import (FramePriceFeed, PriceFeed, YFinancePriceFeed,
                          default_window, periods_per_year, to_returns)


@dataclass
class CapmConfig:
    risk_free_rate: float = 0.042       # annualized; SET to current 10Y Treasury
    equity_risk_premium: float = 0.05   # annualized market premium over rf
    market_ticker: str = "^GSPC"        # S&P 500; override e.g. "ACWI" for global
    frequency: str = "M"                # "M" monthly, "W" weekly, "D" daily
    lookback_years: int = 5
    min_observations: int = 24


@dataclass
class BetaResult:
    raw_beta: float
    adjusted_beta: float
    alpha_annualized: float
    r_squared: float
    correlation: float
    std_error: float
    observations: int
    frequency: str

    def __str__(self) -> str:
        return (f"beta(raw)={self.raw_beta:.3f}  beta(adj)={self.adjusted_beta:.3f}  "
                f"R²={self.r_squared:.2f}  α(ann)={self.alpha_annualized:+.2%}  "
                f"n={self.observations} {self.frequency}")


def compute_beta(stock_returns: pd.Series, market_returns: pd.Series,
                 rf_per_period: float = 0.0, frequency: str = "M") -> BetaResult:
    """OLS of stock excess return on market excess return."""
    df = pd.concat([stock_returns.rename("s"), market_returns.rename("m")],
                   axis=1, join="inner").dropna()
    n = len(df)
    if n < 3:
        raise ValueError(f"Only {n} aligned return observations; need more data.")
    x = df["m"].to_numpy() - rf_per_period
    y = df["s"].to_numpy() - rf_per_period

    # OLS with intercept via least squares.
    X = np.column_stack([np.ones(n), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    intercept, slope = float(coef[0]), float(coef[1])

    resid = y - X @ coef
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot else float("nan")
    sxx = float(((x - x.mean()) ** 2).sum())
    se = float(np.sqrt((ss_res / (n - 2)) / sxx)) if n > 2 and sxx else float("nan")
    corr = float(np.sign(slope) * np.sqrt(r2)) if r2 == r2 and r2 >= 0 else float("nan")

    ppy = periods_per_year(frequency)
    alpha_ann = (1 + intercept) ** ppy - 1
    adjusted = 0.67 * slope + 0.33  # Blume adjustment toward 1.0

    return BetaResult(raw_beta=slope, adjusted_beta=adjusted,
                      alpha_annualized=alpha_ann, r_squared=r2, correlation=corr,
                      std_error=se, observations=n, frequency=frequency)


def cost_of_equity(beta: float, cfg: CapmConfig) -> float:
    """CAPM: ke = rf + beta * equity risk premium."""
    return cfg.risk_free_rate + beta * cfg.equity_risk_premium


@dataclass
class WaccResult:
    wacc: float
    cost_of_equity: float
    pretax_cost_of_debt: float
    after_tax_cost_of_debt: float
    equity_value: float
    debt_value: float
    equity_weight: float
    debt_weight: float
    tax_rate: float
    equity_basis: str  # "market cap" or "book equity"

    def __str__(self) -> str:
        return (f"WACC={self.wacc:.2%} | ke={self.cost_of_equity:.2%} "
                f"(w={self.equity_weight:.0%}, {self.equity_basis}) | "
                f"kd(after-tax)={self.after_tax_cost_of_debt:.2%} "
                f"(w={self.debt_weight:.0%})")


def compute_wacc(cost_of_equity_: float, pretax_cost_of_debt: float,
                 tax_rate: float, equity_value: float, debt_value: float,
                 equity_basis: str = "market cap") -> WaccResult:
    debt_value = max(0.0, debt_value)
    total = equity_value + debt_value
    we, wd = (equity_value / total, debt_value / total) if total else (1.0, 0.0)
    kd_at = pretax_cost_of_debt * (1 - tax_rate)
    wacc = we * cost_of_equity_ + wd * kd_at
    return WaccResult(wacc, cost_of_equity_, pretax_cost_of_debt, kd_at,
                      equity_value, debt_value, we, wd, tax_rate, equity_basis)


@dataclass
class CapmModel:
    ticker: str
    beta: BetaResult
    cost_of_equity: float
    config: CapmConfig
    last_price: Optional[float] = None
    wacc: Optional[WaccResult] = None

    @classmethod
    def from_ticker(cls, ticker: str, config: Optional[CapmConfig] = None,
                    feed: Optional[PriceFeed] = None,
                    three_statement=None) -> "CapmModel":
        cfg = config or CapmConfig()
        feed = feed or YFinancePriceFeed()
        start, end = default_window(cfg.lookback_years)

        stock_px = feed.history(ticker, start, end)
        market_px = feed.history(cfg.market_ticker, start, end)
        s_ret = to_returns(stock_px, cfg.frequency)
        m_ret = to_returns(market_px, cfg.frequency)

        ppy = periods_per_year(cfg.frequency)
        rf_pp = (1 + cfg.risk_free_rate) ** (1 / ppy) - 1
        beta = compute_beta(s_ret, m_ret, rf_pp, cfg.frequency)
        if beta.observations < cfg.min_observations:
            print(f"WARNING: only {beta.observations} observations "
                  f"(< {cfg.min_observations}); beta may be unstable.")

        ke = cost_of_equity(beta.adjusted_beta, cfg)
        last_price = float(stock_px.iloc[-1])

        wacc = None
        if three_statement is not None:
            wacc = _wacc_from_model(ke, three_statement, cfg, last_price)

        return cls(ticker=ticker, beta=beta, cost_of_equity=ke, config=cfg,
                   last_price=last_price, wacc=wacc)


def _wacc_from_model(ke: float, m, cfg: CapmConfig,
                     last_price: float) -> WaccResult:
    """Build WACC using capital structure + cost of debt from the 3-stmt model."""
    bs = m.financials.balance_sheet
    is_ = m.financials.income_statement
    y = m.financials.latest_year()

    def g(df, item):
        return df.at[item, y] if item in df.index and y in df.columns else float("nan")

    debt = float(np.nansum([g(bs, "long_term_debt"), g(bs, "short_term_debt")]))
    interest = abs(g(is_, "interest_expense"))
    pretax_kd = (interest / debt) if debt and pd.notna(interest) else 0.05
    pretax_kd = min(max(pretax_kd, 0.0), 0.25)  # sanity-clamp the proxy

    tax_rate = getattr(m.projection.assumptions, "tax_rate", 0.21) or 0.21

    # Equity value: market cap if shares available, else book equity.
    shares = g(is_, "shares_diluted")
    if pd.notna(shares) and shares > 0:
        equity_value, basis = last_price * float(shares), "market cap"
    else:
        equity_value, basis = float(g(bs, "total_equity")), "book equity"

    return compute_wacc(ke, pretax_kd, tax_rate, equity_value, debt, basis)
