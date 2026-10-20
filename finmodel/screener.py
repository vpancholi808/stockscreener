"""Financial screener.

Runs the analysis engine across a watchlist and collects the values that matter
into one ranked, filterable table: health score, growth, margins, leverage,
returns, cash flow, and (when market caps are supplied) valuation multiples.
Each row also carries multi-year trend series so the dashboard can show how a
company's numbers are moving, not just today's snapshot.

The screen itself is a set of thresholds (ScreenConfig); a company "passes" when
it meets every active criterion. Nothing here re-derives finance — it reuses the
validated ThreeStatementModel, scorecard, and factor composite.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import pandas as pd

from .model import ThreeStatementModel


@dataclass
class ScreenConfig:
    """Active screen thresholds. None = criterion not applied."""
    min_health_score: Optional[float] = 0.60
    min_revenue_growth: Optional[float] = 0.0
    min_fcf_margin: Optional[float] = 0.05
    max_net_debt_to_ebitda: Optional[float] = 3.0
    min_operating_margin: Optional[float] = 0.10
    min_roic: Optional[float] = 0.08

    def criteria(self) -> list[tuple[str, str, float]]:
        c = []
        if self.min_health_score is not None: c.append(("health_score", ">=", self.min_health_score))
        if self.min_revenue_growth is not None: c.append(("revenue_growth", ">=", self.min_revenue_growth))
        if self.min_fcf_margin is not None: c.append(("fcf_margin", ">=", self.min_fcf_margin))
        if self.max_net_debt_to_ebitda is not None: c.append(("net_debt_to_ebitda", "<=", self.max_net_debt_to_ebitda))
        if self.min_operating_margin is not None: c.append(("operating_margin", ">=", self.min_operating_margin))
        if self.min_roic is not None: c.append(("roic", ">=", self.min_roic))
        return c


def _safe(a, b):
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return float("nan")
    return a / b if b not in (0, 0.0) and not pd.isna(b) else float("nan")


def _cagr(series: pd.Series) -> float:
    s = series.dropna()
    if len(s) < 2 or s.iloc[0] <= 0:
        return float("nan")
    return (s.iloc[-1] / s.iloc[0]) ** (1 / (len(s) - 1)) - 1


def _row_from_model(model: ThreeStatementModel, market_cap: Optional[float]) -> dict:
    fin = model.financials
    is_, bs, cf = fin.income_statement, fin.balance_sheet, fin.cash_flow
    y = fin.latest_year()
    yrs = fin.years

    def g(df, item):
        return df.at[item, y] if item in df.index and y in df.columns else float("nan")

    revenue = g(is_, "revenue")
    ebit = g(is_, "operating_income")
    ebitda = g(is_, "ebitda")
    ni = g(is_, "net_income")
    gross_profit = g(is_, "gross_profit")
    equity = g(bs, "total_equity")
    debt = float(np.nansum([g(bs, "long_term_debt"), g(bs, "short_term_debt")]))
    cash = float(np.nansum([g(bs, "cash"), g(bs, "short_term_investments")]))
    net_debt = debt - cash
    ocf = g(cf, "operating_cash_flow")
    capex = abs(g(cf, "capex")) if pd.notna(g(cf, "capex")) else float("nan")
    fcf = ocf - capex if pd.notna(ocf) and pd.notna(capex) else float("nan")
    nopat = ebit * (1 - 0.21) if pd.notna(ebit) else float("nan")

    # Multi-year trend series (for sparklines / tracking).
    rev_series = [is_.at["revenue", yr] if yr in is_.columns else np.nan for yr in yrs]
    ni_series = [is_.at["net_income", yr] if yr in is_.columns else np.nan for yr in yrs]
    fcf_series = []
    for yr in yrs:
        o = cf.at["operating_cash_flow", yr] if yr in cf.columns else np.nan
        cx = cf.at["capex", yr] if yr in cf.columns else np.nan
        fcf_series.append((o - abs(cx)) if pd.notna(o) and pd.notna(cx) else np.nan)
    margin_series = [(_safe(is_.at["operating_income", yr] if yr in is_.columns else np.nan,
                            is_.at["revenue", yr] if yr in is_.columns else np.nan))
                     for yr in yrs]

    sc = model.scorecard
    row = {
        "ticker": model.financials.ticker,
        "name": model.financials.entity_name,
        "health_passed": sc["passed"], "health_total": sc["total"],
        "health_score": sc["score"],
        "revenue": revenue, "revenue_growth": _cagr(is_.loc["revenue", yrs]),
        "ebitda": ebitda, "ebitda_margin": _safe(ebitda, revenue),
        "fcf": fcf, "fcf_margin": _safe(fcf, revenue),
        "net_debt": net_debt, "net_debt_to_ebitda": _safe(net_debt, ebitda),
        "gross_margin": _safe(gross_profit, revenue),
        "operating_margin": _safe(ebit, revenue),
        "net_margin": _safe(ni, revenue),
        "roe": _safe(ni, equity),
        "roic": _safe(nopat, debt + (equity if pd.notna(equity) else 0)),
        "quality_grade": model.quality["grade"],
        "years": yrs,
        "trend_revenue": rev_series, "trend_net_income": ni_series,
        "trend_fcf": fcf_series, "trend_op_margin": margin_series,
    }
    if market_cap is not None and market_cap > 0:
        ev = market_cap + net_debt
        row.update({
            "market_cap": market_cap,
            "pe": _safe(market_cap, ni),
            "ev_ebitda": _safe(ev, ebitda),
            "fcf_yield": _safe(fcf, market_cap),
        })
    return row


def _passes(row: dict, cfg: ScreenConfig) -> bool:
    for metric, op, thr in cfg.criteria():
        val = row.get(metric, float("nan"))
        if pd.isna(val):
            return False
        if op == ">=" and not (val >= thr): return False
        if op == "<=" and not (val <= thr): return False
    return True


@dataclass
class Screener:
    rows: list[dict]
    config: ScreenConfig
    as_of: Optional[str] = None

    @classmethod
    def from_models(cls, models: dict, market_caps: Optional[dict] = None,
                    config: Optional[ScreenConfig] = None,
                    composites: Optional[pd.Series] = None,
                    as_of: Optional[str] = None) -> "Screener":
        cfg = config or ScreenConfig()
        caps = market_caps or {}
        rows = []
        for t, m in models.items():
            r = _row_from_model(m, caps.get(t))
            if composites is not None and t in composites.index:
                r["factor_composite"] = float(composites.loc[t])
            r["passes_screen"] = _passes(r, cfg)
            rows.append(r)
        rows.sort(key=lambda r: (r["passes_screen"], r["health_score"]), reverse=True)
        return cls(rows=rows, config=cfg, as_of=as_of)

    @classmethod
    def from_tickers(cls, tickers: list[str], market_caps: Optional[dict] = None,
                     config: Optional[ScreenConfig] = None,
                     as_of: Optional[str] = None, **model_kw) -> "Screener":
        models = {}
        for t in tickers:
            try:
                models[t] = ThreeStatementModel.from_ticker(t, as_of=as_of, **model_kw)
            except Exception as e:
                print(f"WARNING: skipping {t}: {e}")
        return cls.from_models(models, market_caps, config, as_of=as_of)

    # ---- views ------------------------------------------------------------
    _DISPLAY = ["ticker", "name", "health_passed", "health_total", "health_score",
                "revenue_growth", "operating_margin", "fcf_margin",
                "net_debt_to_ebitda", "roic", "roe", "passes_screen"]

    def table(self, display_only: bool = True) -> pd.DataFrame:
        df = pd.DataFrame(self.rows)
        cols = [c for c in self._DISPLAY if c in df.columns] if display_only else list(df.columns)
        # drop bulky trend columns from the flat table view
        cols = [c for c in cols if not c.startswith("trend_") and c != "years"]
        return df[cols].set_index("ticker")

    def passing(self) -> pd.DataFrame:
        df = self.table()
        return df[df["passes_screen"]]

    def summary(self) -> str:
        n = len(self.rows)
        p = sum(r["passes_screen"] for r in self.rows)
        return (f"Screened {n} companies — {p} pass the screen"
                + (f" (as of {self.as_of})" if self.as_of else ""))
