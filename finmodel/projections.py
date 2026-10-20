"""Driver-based 3-statement projection.

Projects the income statement, balance sheet, and cash flow forward in a linked
way: net income flows to retained earnings and operating cash flow; capex and
D&A roll PP&E forward; working-capital items scale with revenue; cash is the
balancing plug. Assumptions default to recent historical averages but every
driver is overridable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .statements import Financials


def _avg_ratio(num: pd.Series, den: pd.Series) -> float:
    """Mean of num/den across years where both are present and den != 0."""
    pairs = [(n, d) for n, d in zip(num, den)
             if pd.notna(n) and pd.notna(d) and d != 0]
    if not pairs:
        return float("nan")
    return float(np.mean([n / d for n, d in pairs]))


def _latest_ratio(num: pd.Series, den: pd.Series) -> float:
    """num/den for the most recent year where both are present and den != 0."""
    for n, d in zip(reversed(list(num)), reversed(list(den))):
        if pd.notna(n) and pd.notna(d) and d != 0:
            return float(n / d)
    return float("nan")


def _median_ratio(num: pd.Series, den: pd.Series) -> float:
    vals = [n / d for n, d in zip(num, den)
            if pd.notna(n) and pd.notna(d) and d != 0]
    return float(np.median(vals)) if vals else float("nan")


_BASIS = {"average": _avg_ratio, "latest": _latest_ratio, "median": _median_ratio}


@dataclass
class Assumptions:
    """Projection drivers. None -> infer from history.

    driver_basis controls HOW history is turned into drivers:
      "latest"  (default) - use the most recent year. Correct for companies
                 whose economics have structurally changed, and essential for
                 cyclicals: averaging a ratio across a loss year and a boom
                 year produces drivers that are internally incoherent.
      "median"  - middle year; a mid-cycle view, robust to one-off extremes.
      "average" - mean across all years (the old behavior; assumes stability).
    """
    years: int = 5
    driver_basis: str = "latest"
    revenue_growth: Optional[float] = None      # e.g. 0.08 for 8%/yr
    gross_margin: Optional[float] = None
    opex_pct_revenue: Optional[float] = None     # operating costs below gross
    da_pct_revenue: Optional[float] = None
    capex_pct_revenue: Optional[float] = None
    nwc_pct_revenue: Optional[float] = None       # net working capital intensity
    tax_rate: Optional[float] = None
    dividend_payout: Optional[float] = None       # fraction of net income
    debt_schedule: Optional[list[float]] = None   # total debt per year; None = flat

    def infer(self, fin: Financials) -> "Assumptions":
        is_, bs, cf = fin.income_statement, fin.balance_sheet, fin.cash_flow
        yrs = fin.years
        rev = is_.loc["revenue", yrs]

        if self.driver_basis not in _BASIS:
            raise ValueError(f"driver_basis must be one of {list(_BASIS)}")
        ratio = _BASIS[self.driver_basis]

        def filled(val, fallback):
            return fallback if val is None else val

        # Revenue growth: historical CAGR.
        if self.revenue_growth is None:
            clean = rev.dropna()
            if len(clean) >= 2 and clean.iloc[0] > 0:
                n = len(clean) - 1
                self.revenue_growth = (clean.iloc[-1] / clean.iloc[0]) ** (1 / n) - 1
            else:
                self.revenue_growth = 0.05

        gp = is_.loc["gross_profit", yrs]
        self.gross_margin = filled(self.gross_margin, ratio(gp, rev))
        oi = is_.loc["operating_income", yrs]
        # opex as % rev = gross margin - operating margin
        op_margin = ratio(oi, rev)
        if self.opex_pct_revenue is None and pd.notna(self.gross_margin) and pd.notna(op_margin):
            self.opex_pct_revenue = self.gross_margin - op_margin
        self.da_pct_revenue = filled(self.da_pct_revenue,
                                     ratio(cf.loc["depreciation_amortization", yrs], rev))
        self.capex_pct_revenue = filled(self.capex_pct_revenue,
                                        ratio(cf.loc["capex", yrs], rev))

        # Net working capital = current assets ex-cash - current liabilities ex-debt.
        ca = bs.loc["current_assets", yrs]
        cash = bs.loc["cash", yrs]
        cl = bs.loc["current_liabilities", yrs]
        nwc = (ca - cash) - cl
        self.nwc_pct_revenue = filled(self.nwc_pct_revenue, ratio(nwc, rev))

        # Effective tax rate.
        tax = is_.loc["income_tax", yrs]
        pre = is_.loc["pretax_income", yrs]
        rate = ratio(tax, pre)
        self.tax_rate = filled(self.tax_rate,
                               rate if pd.notna(rate) and 0 <= rate <= 0.6 else 0.21)

        # Dividend payout.
        div = cf.loc["dividends_paid", yrs].abs()
        ni = is_.loc["net_income", yrs]
        self.dividend_payout = filled(self.dividend_payout,
                                      max(0.0, min(1.0, ratio(div, ni)))
                                      if pd.notna(ratio(div, ni)) else 0.0)

        # --- coherence guards -------------------------------------------------
        # Drivers inferred from a violent cycle can be economically incoherent
        # (e.g. margins that imply permanent losses, or capex that compounds
        # faster than the business). Surface it rather than silently projecting.
        self.warnings = []
        implied_op_margin = ((self.gross_margin or 0) - (self.opex_pct_revenue or 0))
        if implied_op_margin <= 0:
            self.warnings.append(
                f"implied operating margin is {implied_op_margin:.1%} - projection "
                f"assumes permanent losses; set gross_margin/opex_pct_revenue explicitly")
        if (self.capex_pct_revenue or 0) > 0.25:
            self.warnings.append(
                f"capex is {self.capex_pct_revenue:.1%} of revenue; combined with "
                f"{(self.revenue_growth or 0):.1%} growth this compounds fast - check realism")
        if (self.revenue_growth or 0) > 0.30:
            self.warnings.append(
                f"revenue growth {self.revenue_growth:.1%} held flat for {self.years}y "
                f"is aggressive; consider a fade")
        for w in self.warnings:
            print(f"WARNING [assumptions]: {w}")

        return self


@dataclass
class Projection:
    income_statement: pd.DataFrame
    balance_sheet: pd.DataFrame
    cash_flow: pd.DataFrame
    assumptions: Assumptions
    balance_check: pd.Series  # assets - (liabilities + equity) per year; ~0 is good


def project(fin: Financials, assumptions: Optional[Assumptions] = None) -> Projection:
    a = (assumptions or Assumptions()).infer(fin)
    base = fin.latest_year()
    is_, bs, cf = fin.income_statement, fin.balance_sheet, fin.cash_flow
    fcols = list(range(base + 1, base + 1 + a.years))

    pis = pd.DataFrame(index=["revenue", "gross_profit", "operating_income",
                              "ebitda", "depreciation_amortization", "interest_expense",
                              "pretax_income", "income_tax", "net_income"], columns=fcols, dtype="float64")
    pbs = pd.DataFrame(index=["cash", "net_working_capital", "ppe_net",
                              "total_assets", "total_debt", "total_equity",
                              "total_liabilities_and_equity"], columns=fcols, dtype="float64")
    pcf = pd.DataFrame(index=["net_income", "depreciation_amortization", "change_in_nwc",
                              "operating_cash_flow", "capex", "investing_cash_flow",
                              "dividends", "debt_change", "financing_cash_flow",
                              "net_change_in_cash", "free_cash_flow"], columns=fcols, dtype="float64")

    # Starting points from the last actual year.
    prev_rev = is_.at["revenue", base]
    prev_ppe = bs.at["ppe_net", base]
    prev_cash = bs.at["cash", base]
    prev_equity = bs.at["total_equity", base]
    prev_debt = (np.nansum([bs.at["long_term_debt", base], bs.at["short_term_debt", base]]))
    prev_nwc = ((bs.at["current_assets", base] - bs.at["cash", base])
                - bs.at["current_liabilities", base])
    prev_interest = is_.at["interest_expense", base]
    # Other (non-modeled) assets held flat as a residual plug for balancing.
    other_assets = (bs.at["total_assets", base]
                    - np.nansum([prev_cash, prev_ppe, bs.at["current_assets", base] - prev_cash]))
    other_liab = (bs.at["total_liabilities", base] - prev_debt
                  - bs.at["current_liabilities", base]) if pd.notna(bs.at["total_liabilities", base]) else 0.0
    other_assets = 0.0 if pd.isna(other_assets) else other_assets
    other_liab = 0.0 if pd.isna(other_liab) else other_liab

    for i, y in enumerate(fcols):
        rev = prev_rev * (1 + a.revenue_growth)
        gp = rev * a.gross_margin
        da = rev * a.da_pct_revenue
        opex = rev * (a.opex_pct_revenue or 0.0)
        oi = gp - opex
        ebitda = oi + da
        debt = (a.debt_schedule[i] if a.debt_schedule and i < len(a.debt_schedule)
                else prev_debt)
        interest = prev_interest if pd.notna(prev_interest) else 0.0
        pretax = oi - interest
        tax = max(0.0, pretax) * a.tax_rate
        ni = pretax - tax

        capex = rev * a.capex_pct_revenue
        nwc = rev * a.nwc_pct_revenue
        d_nwc = nwc - prev_nwc
        ppe = prev_ppe + capex - da
        dividends = ni * a.dividend_payout
        equity = prev_equity + ni - dividends
        debt_change = debt - prev_debt

        ocf = ni + da - d_nwc
        icf = -capex
        fcf = ocf + icf
        fin_cf = -dividends + debt_change
        net_change_cash = ocf + icf + fin_cf
        cash = prev_cash + net_change_cash

        # Balance sheet, with cash as computed; other items hold flat.
        total_liab = debt + other_liab
        total_le = total_liab + equity
        total_assets = cash + ppe + nwc + other_assets  # nwc proxies net current items

        pis[y] = [rev, gp, oi, ebitda, da, interest, pretax, tax, ni]
        pbs[y] = [cash, nwc, ppe, total_assets, debt, equity, total_le]
        pcf[y] = [ni, da, -d_nwc, ocf, -capex, icf, -dividends, debt_change,
                  fin_cf, net_change_cash, fcf]

        prev_rev, prev_ppe, prev_cash = rev, ppe, cash
        prev_equity, prev_debt, prev_nwc = equity, debt, nwc

    balance_check = pbs.loc["total_assets"] - pbs.loc["total_liabilities_and_equity"]
    return Projection(pis, pbs, pcf, a, balance_check)
