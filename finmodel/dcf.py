"""Discounted cash flow valuation.

Discounts **unlevered free cash flow to the firm** at WACC, adds a terminal
value, and bridges enterprise value to an intrinsic value per share that's
directly comparable to the market price.

Why unlevered FCFF (not the projection's levered FCF line): WACC already
reflects the tax benefit of debt in its after-tax cost of debt. Discounting
levered cash flow (which is after interest) at WACC would count the financing
benefit twice. So:

    FCFF = EBIT*(1 - tax) + D&A - CapEx - increase in net working capital

Terminal value: Gordon growth perpetuity (default) or an EV/EBITDA exit
multiple. Mid-year convention is on by default (cash flows arrive on average
mid-period), which is the more accurate and common practitioner choice.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class DcfConfig:
    terminal_growth: float = 0.025        # perpetual growth; must be < WACC
    terminal_method: str = "gordon"       # "gordon" or "exit_multiple"
    exit_ev_ebitda: float = 10.0          # used if terminal_method == "exit_multiple"
    mid_year: bool = True                 # mid-year discounting convention
    tax_rate: Optional[float] = None      # override; else from projection assumptions


@dataclass
class DcfResult:
    value_per_share: float
    enterprise_value: float
    equity_value: float
    net_debt: float
    shares: float
    current_price: Optional[float]
    upside: Optional[float]
    pv_explicit: float
    pv_terminal: float
    terminal_value: float
    terminal_pct: float
    wacc: float
    terminal_growth: float
    table: pd.DataFrame
    equity_basis: str

    def summary(self) -> str:
        lines = [f"DCF intrinsic value: {self.value_per_share:,.2f} / share"]
        if self.current_price is not None and self.upside is not None:
            tag = "UNDERVALUED" if self.upside > 0 else "OVERVALUED"
            lines.append(f"Market price: {self.current_price:,.2f}  -> "
                         f"{self.upside:+.1%} ({tag})")
        lines += [
            f"Enterprise value: {self.enterprise_value/1e9:,.2f}B "
            f"(explicit {self.pv_explicit/1e9:,.2f}B + terminal "
            f"{self.pv_terminal/1e9:,.2f}B)",
            f"Terminal share of EV: {self.terminal_pct:.0%}"
            + ("  (high — sensitive to terminal assumptions)"
               if self.terminal_pct > 0.75 else ""),
            f"- Net debt: {self.net_debt/1e9:,.2f}B  = Equity "
            f"{self.equity_value/1e9:,.2f}B / {self.shares/1e9:,.2f}B sh "
            f"({self.equity_basis})",
            f"WACC {self.wacc:.2%} | terminal g {self.terminal_growth:.2%}",
        ]
        return "\n".join(lines)


def _fcff_from_projection(proj, tax_rate: float) -> pd.Series:
    """Unlevered FCFF per projected year from the linked projection."""
    is_, cf = proj.income_statement, proj.cash_flow
    ebit = is_.loc["operating_income"]
    da = is_.loc["depreciation_amortization"]
    capex = cf.loc["capex"].abs()
    # change_in_nwc on the cash-flow statement is already signed as a cash
    # impact (negative when NWC grows), so adding it subtracts an NWC increase.
    chg_nwc = cf.loc["change_in_nwc"]
    nopat = ebit * (1 - tax_rate)
    return (nopat + da - capex + chg_nwc).astype("float64")


def run_dcf(three_statement, wacc: float, config: Optional[DcfConfig] = None,
            current_price: Optional[float] = None,
            net_debt: Optional[float] = None,
            shares: Optional[float] = None) -> DcfResult:
    cfg = config or DcfConfig()
    proj = three_statement.projection
    tax_rate = cfg.tax_rate if cfg.tax_rate is not None else \
        (proj.assumptions.tax_rate or 0.21)

    if cfg.terminal_method == "gordon" and cfg.terminal_growth >= wacc:
        raise ValueError(f"terminal growth ({cfg.terminal_growth:.2%}) must be "
                         f"below WACC ({wacc:.2%}) for a Gordon perpetuity.")

    fcff = _fcff_from_projection(proj, tax_rate)
    years = list(fcff.index)
    n = len(years)
    periods = np.arange(1, n + 1, dtype="float64")
    if cfg.mid_year:
        periods = periods - 0.5
    discount = 1.0 / (1.0 + wacc) ** periods
    pv = fcff.to_numpy() * discount
    pv_explicit = float(pv.sum())

    # Terminal value at end of explicit horizon.
    if cfg.terminal_method == "exit_multiple":
        terminal_ebitda = float(proj.income_statement.loc["ebitda"].iloc[-1])
        terminal_value = cfg.exit_ev_ebitda * terminal_ebitda
    else:
        last_fcff = float(fcff.iloc[-1])
        terminal_value = last_fcff * (1 + cfg.terminal_growth) / \
            (wacc - cfg.terminal_growth)
    # Discount TV back over the full explicit horizon (mid-year aware).
    tv_period = n - 0.5 if cfg.mid_year else n
    pv_terminal = terminal_value / (1.0 + wacc) ** tv_period

    enterprise_value = pv_explicit + pv_terminal

    # EV -> equity bridge using the latest actual balance sheet.
    bs = three_statement.financials.balance_sheet
    is_hist = three_statement.financials.income_statement
    y = three_statement.financials.latest_year()

    def g(df, item):
        return df.at[item, y] if item in df.index and y in df.columns else float("nan")

    if net_debt is None:
        debt = float(np.nansum([g(bs, "long_term_debt"), g(bs, "short_term_debt")]))
        cash = float(np.nansum([g(bs, "cash"), g(bs, "short_term_investments")]))
        net_debt = debt - cash
    if shares is None:
        shares = g(is_hist, "shares_diluted")

    equity_value = enterprise_value - net_debt
    if pd.notna(shares) and shares > 0:
        value_per_share = equity_value / float(shares)
        equity_basis = "diluted shares"
    else:
        value_per_share = float("nan")
        equity_basis = "shares unavailable"

    upside = (value_per_share / current_price - 1.0) \
        if current_price and pd.notna(value_per_share) else None

    table = pd.DataFrame({
        "fcff": fcff.to_numpy(),
        "period": periods,
        "discount_factor": discount,
        "pv": pv,
    }, index=years)

    return DcfResult(
        value_per_share=value_per_share, enterprise_value=enterprise_value,
        equity_value=equity_value, net_debt=net_debt,
        shares=float(shares) if pd.notna(shares) else float("nan"),
        current_price=current_price, upside=upside,
        pv_explicit=pv_explicit, pv_terminal=pv_terminal,
        terminal_value=terminal_value,
        terminal_pct=pv_terminal / enterprise_value if enterprise_value else float("nan"),
        wacc=wacc, terminal_growth=cfg.terminal_growth, table=table,
        equity_basis=equity_basis)


def sensitivity(three_statement, base_wacc: float,
                config: Optional[DcfConfig] = None,
                wacc_range: float = 0.02, growth_range: float = 0.01,
                steps: int = 5, **kw) -> pd.DataFrame:
    """Grid of value-per-share across WACC (rows) and terminal growth (cols)."""
    cfg = config or DcfConfig()
    waccs = np.linspace(base_wacc - wacc_range, base_wacc + wacc_range, steps)
    growths = np.linspace(cfg.terminal_growth - growth_range,
                          cfg.terminal_growth + growth_range, steps)
    grid = pd.DataFrame(index=[f"{w:.2%}" for w in waccs],
                        columns=[f"{gr:.2%}" for gr in growths], dtype="float64")
    for w in waccs:
        for gr in growths:
            c = DcfConfig(terminal_growth=float(gr),
                          terminal_method=cfg.terminal_method,
                          exit_ev_ebitda=cfg.exit_ev_ebitda,
                          mid_year=cfg.mid_year, tax_rate=cfg.tax_rate)
            try:
                r = run_dcf(three_statement, float(w), c, **kw)
                grid.at[f"{w:.2%}", f"{gr:.2%}"] = round(r.value_per_share, 2)
            except ValueError:
                grid.at[f"{w:.2%}", f"{gr:.2%}"] = float("nan")
    grid.index.name = "WACC \\ g"
    return grid


@dataclass
class DcfModel:
    result: DcfResult
    sensitivity_table: pd.DataFrame

    @classmethod
    def from_models(cls, three_statement, capm=None, wacc: Optional[float] = None,
                    config: Optional[DcfConfig] = None,
                    current_price: Optional[float] = None) -> "DcfModel":
        if wacc is None:
            if capm is None or capm.wacc is None:
                raise ValueError("Provide wacc=... or a CapmModel with a WACC.")
            wacc = capm.wacc.wacc
        if current_price is None and capm is not None:
            current_price = capm.last_price
        result = run_dcf(three_statement, wacc, config, current_price=current_price)
        sens = sensitivity(three_statement, wacc, config, current_price=current_price)
        return cls(result=result, sensitivity_table=sens)
