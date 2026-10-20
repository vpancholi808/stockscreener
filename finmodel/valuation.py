"""Relative valuation — what the market charges for this business.

The scorecard answers "is this a good business?". These multiples answer the
separate question "what am I paying for it?". A company can pass every health
check and still be expensive; the two readings are independent and should stay
that way.

Every multiple needs a market price, so all of this is gated on market-cap
availability. Where it is missing the metric returns None rather than a
fabricated number.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import pandas as pd


def _d(a, b):
    """Divide, but return None for a meaningless result.

    Negative denominators matter here: a P/E on negative earnings is not a
    'cheap' low number, it is undefined. Returning it would quietly corrupt
    every sector median it lands in."""
    if a is None or b is None:
        return None
    try:
        if pd.isna(a) or pd.isna(b):
            return None
    except (TypeError, ValueError):
        pass
    a, b = float(a), float(b)
    if b <= 0:
        return None
    return a / b


def _get(df, item, year):
    if item in df.index and year in df.columns:
        v = df.at[item, year]
        return None if pd.isna(v) else float(v)
    return None


@dataclass
class Multiples:
    """Relative-valuation readings. None where undefined or unavailable."""
    market_cap: Optional[float] = None
    enterprise_value: Optional[float] = None
    pe: Optional[float] = None                  # price / earnings
    forward_pe: Optional[float] = None          # needs estimates; usually None
    pb: Optional[float] = None                  # price / book
    ps: Optional[float] = None                  # price / sales
    ev_ebitda: Optional[float] = None
    ev_sales: Optional[float] = None
    ev_ebit: Optional[float] = None
    fcf_yield: Optional[float] = None           # FCF / market cap
    earnings_yield: Optional[float] = None      # net income / market cap
    peg: Optional[float] = None                 # PE / (growth % )
    dividend_yield: Optional[float] = None

    def as_dict(self) -> dict:
        return asdict(self)

    def available(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


# Direction matters for every comparison downstream: for most multiples a
# LOWER reading is cheaper, but yields invert. Getting this wrong would rank
# expensive companies as attractive.
LOWER_IS_CHEAPER = {
    "pe", "forward_pe", "pb", "ps", "ev_ebitda", "ev_sales", "ev_ebit", "peg",
}
HIGHER_IS_CHEAPER = {"fcf_yield", "earnings_yield", "dividend_yield"}

MULTIPLE_LABELS = {
    "pe": "P/E", "forward_pe": "Forward P/E", "pb": "P/B", "ps": "P/S",
    "ev_ebitda": "EV/EBITDA", "ev_sales": "EV/Sales", "ev_ebit": "EV/EBIT",
    "fcf_yield": "FCF yield", "earnings_yield": "Earnings yield",
    "peg": "PEG", "dividend_yield": "Dividend yield",
}


def compute_multiples(model, market_cap: Optional[float],
                      dividends_paid: Optional[float] = None,
                      growth_override: Optional[float] = None) -> Multiples:
    """Build relative-valuation multiples for one company."""
    if market_cap is None or market_cap <= 0:
        return Multiples()

    fin = model.financials
    if not fin.years:
        return Multiples(market_cap=market_cap)
    y = fin.latest_year()
    is_, bs, cf = fin.income_statement, fin.balance_sheet, fin.cash_flow

    revenue = _get(is_, "revenue", y)
    net_income = _get(is_, "net_income", y)
    ebit = _get(is_, "operating_income", y)
    ebitda = _get(is_, "ebitda", y)
    equity = _get(bs, "total_equity", y)

    cash = sum(v for v in (_get(bs, "cash", y),
                           _get(bs, "short_term_investments", y)) if v is not None)
    debt = sum(v for v in (_get(bs, "long_term_debt", y),
                           _get(bs, "short_term_debt", y)) if v is not None)
    ev = market_cap + debt - cash

    ocf = _get(cf, "operating_cash_flow", y)
    capex = _get(cf, "capex", y)
    fcf = (ocf - abs(capex)) if ocf is not None and capex is not None else None

    divs = dividends_paid
    if divs is None:
        divs = _get(cf, "dividends_paid", y)
    divs = abs(divs) if divs is not None else None

    pe = _d(market_cap, net_income)

    # PEG is only meaningful on positive growth; a negative denominator flips
    # the sign and would make a shrinking company look like a bargain.
    growth = growth_override
    if growth is None:
        series = [_get(is_, "revenue", yr) for yr in fin.years]
        clean = [v for v in series if v is not None and v > 0]
        if len(clean) >= 2:
            growth = (clean[-1] / clean[0]) ** (1 / (len(clean) - 1)) - 1
    peg = None
    if pe is not None and growth is not None and growth > 0.01:
        peg = pe / (growth * 100)

    return Multiples(
        market_cap=market_cap,
        enterprise_value=ev,
        pe=pe,
        forward_pe=None,            # requires analyst estimates; not in filings
        pb=_d(market_cap, equity),
        ps=_d(market_cap, revenue),
        ev_ebitda=_d(ev, ebitda) if ev > 0 else None,
        ev_sales=_d(ev, revenue) if ev > 0 else None,
        ev_ebit=_d(ev, ebit) if ev > 0 else None,
        fcf_yield=(fcf / market_cap) if fcf is not None else None,
        earnings_yield=(net_income / market_cap) if net_income is not None else None,
        peg=peg,
        dividend_yield=(divs / market_cap) if divs is not None else None,
    )
