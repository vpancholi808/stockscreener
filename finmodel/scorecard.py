"""Financial-health scorecard.

Computes the metrics an investor screens on and grades each against a tunable
threshold. Thresholds live in `ScorecardConfig` so you can dial them to your
strategy (e.g. demand Net Debt/EBITDA < 2.0 instead of 3.0). Every metric
returns its raw value plus a pass/fail so results feed cleanly into a screener
or the long/short model later.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from .statements import Financials


def _safe(a, b):
    a, b = float(a), float(b)
    return a / b if b not in (0, 0.0) and not pd.isna(b) else float("nan")


def _return_on(income, capital):
    """A return on a capital base, undefined when that base is not positive.

    A company that has bought back most of its own stock carries book equity
    near zero or below it. Net income divided by a negative denominator comes
    out as a large negative number that reads as catastrophic performance
    when the business is in fact highly profitable: on a real S&P 500 run
    McDonald's showed -478% ROE on $8.6B of earnings, Mettler-Toledo -3,677%
    and Colgate +3,948%, every one of them from an equity base near zero
    rather than from anything about the company.

    This is the rule valuation._d already applies to a P/E on negative
    earnings: a ratio on a non-positive denominator is undefined, not
    extreme. Returning it corrupts every average, benchmark and screen it
    reaches, and marks profitable companies as failures on the scorecard.
    """
    try:
        income, capital = float(income), float(capital)
    except (TypeError, ValueError):
        return float("nan")
    if pd.isna(capital) or capital <= 0:
        return float("nan")
    return income / capital


@dataclass
class ScorecardConfig:
    # Leverage / solvency
    max_debt_to_equity: float = 2.0
    max_net_debt_to_ebitda: float = 3.0
    min_interest_coverage: float = 3.0       # EBIT / interest
    # Liquidity
    min_current_ratio: float = 1.2
    min_quick_ratio: float = 1.0
    # Cash flow quality
    min_fcf_margin: float = 0.05             # FCF / revenue
    min_cash_conversion: float = 0.8         # operating cash flow / net income
    require_positive_fcf: bool = True
    # Profitability
    min_gross_margin: float = 0.30
    min_operating_margin: float = 0.10
    min_roe: float = 0.12
    min_roic: float = 0.08


@dataclass
class Metric:
    name: str
    value: float
    threshold: float
    passed: bool
    higher_is_better: bool


def _grade(name, value, threshold, higher_is_better) -> Metric:
    if pd.isna(value):
        passed = False
    else:
        passed = value >= threshold if higher_is_better else value <= threshold
    return Metric(name, value, threshold, bool(passed), higher_is_better)


def compute_scorecard(fin: Financials, cfg: ScorecardConfig | None = None,
                      year: int | None = None) -> dict:
    cfg = cfg or ScorecardConfig()
    y = year or fin.latest_year()
    is_, bs, cf = fin.income_statement, fin.balance_sheet, fin.cash_flow

    def g(df, item):
        return df.at[item, y] if item in df.index and y in df.columns else float("nan")

    revenue = g(is_, "revenue")
    ebit = g(is_, "operating_income")
    ebitda = g(is_, "ebitda")
    net_income = g(is_, "net_income")
    interest = abs(g(is_, "interest_expense")) if pd.notna(g(is_, "interest_expense")) else float("nan")
    gross_profit = g(is_, "gross_profit")

    cash = g(bs, "cash")
    debt = np.nansum([g(bs, "long_term_debt"), g(bs, "short_term_debt")])
    net_debt = debt - (cash if pd.notna(cash) else 0.0)
    equity = g(bs, "total_equity")
    total_assets = g(bs, "total_assets")
    current_assets = g(bs, "current_assets")
    current_liab = g(bs, "current_liabilities")
    inventory = g(bs, "inventory")

    ocf = g(cf, "operating_cash_flow")
    capex = abs(g(cf, "capex")) if pd.notna(g(cf, "capex")) else float("nan")
    fcf = ocf - capex if pd.notna(ocf) and pd.notna(capex) else float("nan")

    invested_capital = (debt if pd.notna(debt) else 0.0) + (equity if pd.notna(equity) else float("nan"))
    nopat = ebit * (1 - 0.21) if pd.notna(ebit) else float("nan")

    metrics = [
        _grade("debt_to_equity", _safe(debt, equity), cfg.max_debt_to_equity, False),
        _grade("net_debt_to_ebitda", _safe(net_debt, ebitda), cfg.max_net_debt_to_ebitda, False),
        _grade("interest_coverage", _safe(ebit, interest), cfg.min_interest_coverage, True),
        _grade("current_ratio", _safe(current_assets, current_liab), cfg.min_current_ratio, True),
        _grade("quick_ratio", _safe((current_assets - (inventory if pd.notna(inventory) else 0)), current_liab),
               cfg.min_quick_ratio, True),
        _grade("fcf_margin", _safe(fcf, revenue), cfg.min_fcf_margin, True),
        _grade("cash_conversion", _safe(ocf, net_income), cfg.min_cash_conversion, True),
        _grade("gross_margin", _safe(gross_profit, revenue), cfg.min_gross_margin, True),
        _grade("operating_margin", _safe(ebit, revenue), cfg.min_operating_margin, True),
        _grade("roe", _return_on(net_income, equity), cfg.min_roe, True),
        _grade("roic", _return_on(nopat, invested_capital), cfg.min_roic, True),
    ]
    if cfg.require_positive_fcf:
        metrics.append(_grade("positive_fcf", fcf, 0.0, True))

    passed = sum(m.passed for m in metrics)
    total = len(metrics)
    return {
        "ticker": fin.ticker,
        "fiscal_year": y,
        "metrics": {m.name: m for m in metrics},
        "raw": {
            "revenue": revenue, "ebitda": ebitda, "ebit": ebit,
            "net_income": net_income, "fcf": fcf, "operating_cash_flow": ocf,
            "total_debt": debt, "net_debt": net_debt, "cash": cash, "equity": equity,
        },
        "score": passed / total if total else float("nan"),
        "passed": passed,
        "total": total,
    }


def scorecard_to_frame(result: dict) -> pd.DataFrame:
    rows = []
    for m in result["metrics"].values():
        rows.append({
            "metric": m.name,
            "value": round(m.value, 4) if pd.notna(m.value) else None,
            "threshold": m.threshold,
            "rule": ("≥" if m.higher_is_better else "≤"),
            "pass": "PASS" if m.passed else "FAIL",
        })
    return pd.DataFrame(rows)
