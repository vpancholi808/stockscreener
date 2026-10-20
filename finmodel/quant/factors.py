"""Financial-statement factor models.

Turns each company's fundamentals (from a ThreeStatementModel) plus its market
cap into factor exposures across five themes — value, quality, growth,
profitability, and FCF yield — then z-scores them cross-sectionally across the
universe so names are directly comparable and rankable. Sign conventions are
normalized so a HIGHER composite always means MORE attractive (cheaper, higher
quality, faster growing, more profitable). The composite feeds the long/short
portfolio layer.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# metric -> (group, direction)  direction +1 = higher is better, -1 = lower better
_METRICS = {
    "earnings_yield":       ("value", +1),
    "fcf_yield":            ("value", +1),
    "book_to_price":        ("value", +1),
    "ebitda_to_ev":         ("value", +1),
    "roe":                  ("quality", +1),
    "roic":                 ("quality", +1),
    "gross_profitability":  ("quality", +1),
    "net_debt_to_ebitda":   ("quality", -1),
    "revenue_cagr":         ("growth", +1),
    "earnings_cagr":        ("growth", +1),
    "gross_margin":         ("profitability", +1),
    "operating_margin":     ("profitability", +1),
    "net_margin":           ("profitability", +1),
}
GROUPS = ["value", "quality", "growth", "profitability"]


def _safe(a, b):
    a, b = float(a), float(b)
    return a / b if b not in (0, 0.0) and not pd.isna(b) else float("nan")


def _cagr(series: pd.Series) -> float:
    s = series.dropna()
    if len(s) < 2 or s.iloc[0] <= 0:
        return float("nan")
    return (s.iloc[-1] / s.iloc[0]) ** (1 / (len(s) - 1)) - 1


def company_factors(model, market_cap: float) -> dict:
    """Raw (un-scored) factor metrics for a single company."""
    fin = model.financials
    is_, bs, cf = fin.income_statement, fin.balance_sheet, fin.cash_flow
    y = fin.latest_year()

    def g(df, item):
        return df.at[item, y] if item in df.index and y in df.columns else float("nan")

    revenue = g(is_, "revenue")
    ni = g(is_, "net_income")
    gross_profit = g(is_, "gross_profit")
    ebit = g(is_, "operating_income")
    ebitda = g(is_, "ebitda")
    equity = g(bs, "total_equity")
    assets = g(bs, "total_assets")
    debt = float(np.nansum([g(bs, "long_term_debt"), g(bs, "short_term_debt")]))
    cash = float(np.nansum([g(bs, "cash"), g(bs, "short_term_investments")]))
    net_debt = debt - cash
    ocf = g(cf, "operating_cash_flow")
    capex = abs(g(cf, "capex")) if pd.notna(g(cf, "capex")) else float("nan")
    fcf = ocf - capex if pd.notna(ocf) and pd.notna(capex) else float("nan")
    ev = market_cap + net_debt
    nopat = ebit * (1 - 0.21) if pd.notna(ebit) else float("nan")

    return {
        "earnings_yield":      _safe(ni, market_cap),
        "fcf_yield":           _safe(fcf, market_cap),
        "book_to_price":       _safe(equity, market_cap),
        "ebitda_to_ev":        _safe(ebitda, ev),
        "roe":                 _safe(ni, equity),
        "roic":                _safe(nopat, debt + (equity if pd.notna(equity) else 0)),
        "gross_profitability": _safe(gross_profit, assets),
        "net_debt_to_ebitda":  _safe(net_debt, ebitda),
        "revenue_cagr":        _cagr(is_.loc["revenue", fin.years]),
        "earnings_cagr":       _cagr(is_.loc["net_income", fin.years]),
        "gross_margin":        _safe(gross_profit, revenue),
        "operating_margin":    _safe(ebit, revenue),
        "net_margin":          _safe(ni, revenue),
    }


def _zscore(col: pd.Series, winsor: float = 3.0) -> pd.Series:
    z = (col - col.mean()) / col.std(ddof=0)
    return z.clip(-winsor, winsor)


@dataclass
class FactorModel:
    raw: pd.DataFrame          # tickers x metrics (un-scored)
    zscores: pd.DataFrame      # direction-adjusted cross-sectional z-scores
    composites: pd.DataFrame   # per-group composite + overall, plus rank

    @classmethod
    def from_models(cls, models: dict, market_caps: dict,
                    group_weights: dict | None = None) -> "FactorModel":
        """models/market_caps keyed by ticker."""
        raw = pd.DataFrame({t: company_factors(m, market_caps[t])
                            for t, m in models.items()}).T
        raw = raw[list(_METRICS.keys())]

        z = pd.DataFrame(index=raw.index)
        for metric, (_, direction) in _METRICS.items():
            z[metric] = _zscore(raw[metric]) * direction

        comp = pd.DataFrame(index=raw.index)
        for grp in GROUPS:
            cols = [m for m, (g, _) in _METRICS.items() if g == grp]
            comp[grp] = z[cols].mean(axis=1, skipna=True)

        w = group_weights or {g: 1.0 for g in GROUPS}
        wsum = sum(w.get(g, 0) for g in GROUPS)
        comp["composite"] = sum(comp[g] * w.get(g, 0) for g in GROUPS) / wsum
        comp["rank"] = comp["composite"].rank(ascending=False).astype("Int64")
        comp = comp.sort_values("composite", ascending=False)
        return cls(raw=raw, zscores=z, composites=comp)

    def long_short(self, n: int = 1) -> dict:
        """Top-n (long) and bottom-n (short) by composite score."""
        ordered = self.composites.sort_values("composite", ascending=False)
        return {"long": list(ordered.index[:n]),
                "short": list(ordered.index[-n:][::-1])}
