"""Forensic accounting scores.

Four decades-old composite diagnostics computed straight from the statements,
plus the economic-spread test. Each bundles many ratios into one number:

  Piotroski F-score  (0-9)   fundamental strength / improving vs deteriorating
  Altman Z-score             bankruptcy risk
  Beneish M-score            fingerprints of earnings manipulation
  Sloan accruals ratio       earnings quality (high accruals -> weaker future returns)
  ROIC - WACC spread         is the business creating or destroying value

All require two consecutive years; they return `available=False` with a reason
rather than a misleading number when inputs are missing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


def _g(df, item, year):
    if item in df.index and year in df.columns:
        v = df.at[item, year]
        return float(v) if pd.notna(v) else None
    return None


def _div(a, b):
    if a is None or b is None or b == 0:
        return None
    return a / b


@dataclass
class ScoreResult:
    name: str
    value: Optional[float]
    verdict: str
    detail: dict = field(default_factory=dict)
    available: bool = True
    reason: str = ""

    def __str__(self):
        if not self.available:
            return f"{self.name}: unavailable ({self.reason})"
        v = f"{self.value:.2f}" if self.value is not None else "n/a"
        return f"{self.name}: {v} — {self.verdict}"


# ---------------------------------------------------------------- Piotroski
def piotroski_f(fin, year: int | None = None) -> ScoreResult:
    """9 binary tests: profitability (4), leverage/liquidity (3), efficiency (2)."""
    y = year or fin.latest_year()
    yrs = fin.years
    if y not in yrs or yrs.index(y) == 0:
        return ScoreResult("Piotroski F-score", None, "", available=False,
                           reason="needs a prior year")
    p = yrs[yrs.index(y) - 1]
    is_, bs, cf = fin.income_statement, fin.balance_sheet, fin.cash_flow

    ni_t, ni_p = _g(is_, "net_income", y), _g(is_, "net_income", p)
    ta_t, ta_p = _g(bs, "total_assets", y), _g(bs, "total_assets", p)
    ocf_t = _g(cf, "operating_cash_flow", y)
    roa_t, roa_p = _div(ni_t, ta_t), _div(ni_p, ta_p)

    ltd_t, ltd_p = _g(bs, "long_term_debt", y), _g(bs, "long_term_debt", p)
    ca_t, ca_p = _g(bs, "current_assets", y), _g(bs, "current_assets", p)
    cl_t, cl_p = _g(bs, "current_liabilities", y), _g(bs, "current_liabilities", p)
    sh_t, sh_p = _g(is_, "shares_diluted", y), _g(is_, "shares_diluted", p)
    rev_t, rev_p = _g(is_, "revenue", y), _g(is_, "revenue", p)
    gp_t, gp_p = _g(is_, "gross_profit", y), _g(is_, "gross_profit", p)

    tests = {}
    tests["ROA positive"] = (roa_t is not None and roa_t > 0)
    tests["Operating CF positive"] = (ocf_t is not None and ocf_t > 0)
    tests["ROA improving"] = (roa_t is not None and roa_p is not None and roa_t > roa_p)
    tests["OCF > net income (quality)"] = (ocf_t is not None and ni_t is not None and ocf_t > ni_t)
    tests["Leverage decreased"] = (_div(ltd_t, ta_t) is not None and _div(ltd_p, ta_p) is not None
                                   and _div(ltd_t, ta_t) < _div(ltd_p, ta_p))
    tests["Current ratio improved"] = (_div(ca_t, cl_t) is not None and _div(ca_p, cl_p) is not None
                                       and _div(ca_t, cl_t) > _div(ca_p, cl_p))
    tests["No share dilution"] = (sh_t is not None and sh_p is not None and sh_t <= sh_p * 1.02)
    tests["Gross margin improved"] = (_div(gp_t, rev_t) is not None and _div(gp_p, rev_p) is not None
                                      and _div(gp_t, rev_t) > _div(gp_p, rev_p))
    tests["Asset turnover improved"] = (_div(rev_t, ta_t) is not None and _div(rev_p, ta_p) is not None
                                        and _div(rev_t, ta_t) > _div(rev_p, ta_p))

    score = sum(1 for v in tests.values() if v)
    verdict = ("strong — improving fundamentals" if score >= 8 else
               "solid" if score >= 6 else
               "mixed" if score >= 4 else "weak — deteriorating")
    return ScoreResult("Piotroski F-score", float(score), f"{score}/9 · {verdict}",
                       detail=tests)


# ------------------------------------------------------------------- Altman
def altman_z(fin, market_cap: Optional[float] = None,
             year: int | None = None) -> ScoreResult:
    """Z = 1.2·WC/TA + 1.4·RE/TA + 3.3·EBIT/TA + 0.6·MVE/TL + 1.0·Sales/TA"""
    y = year or fin.latest_year()
    is_, bs = fin.income_statement, fin.balance_sheet
    ta = _g(bs, "total_assets", y)
    ca, cl = _g(bs, "current_assets", y), _g(bs, "current_liabilities", y)
    re = _g(bs, "retained_earnings", y)
    ebit = _g(is_, "operating_income", y)
    tl = _g(bs, "total_liabilities", y)
    sales = _g(is_, "revenue", y)

    missing = [n for n, v in [("total assets", ta), ("current assets", ca),
                              ("current liabilities", cl), ("EBIT", ebit),
                              ("total liabilities", tl), ("revenue", sales)] if v is None]
    if missing:
        return ScoreResult("Altman Z-score", None, "", available=False,
                           reason=f"missing {', '.join(missing)}")
    if re is None:
        return ScoreResult("Altman Z-score", None, "", available=False,
                           reason="missing retained earnings")
    mve = market_cap if market_cap else _g(bs, "total_equity", y)
    if mve is None:
        return ScoreResult("Altman Z-score", None, "", available=False,
                           reason="no market cap or book equity")

    x1, x2, x3 = (ca - cl) / ta, re / ta, ebit / ta
    x4, x5 = mve / tl, sales / ta
    z = 1.2*x1 + 1.4*x2 + 3.3*x3 + 0.6*x4 + 1.0*x5
    verdict = ("safe zone" if z > 2.99 else
               "grey zone" if z >= 1.81 else "distress zone")
    return ScoreResult("Altman Z-score", float(z), verdict,
                       detail={"WC/TA": x1, "RE/TA": x2, "EBIT/TA": x3,
                               "MVE/TL": x4, "Sales/TA": x5,
                               "equity basis": "market" if market_cap else "book"})


# ------------------------------------------------------------------ Beneish
def beneish_m(fin, year: int | None = None) -> ScoreResult:
    """8-variable earnings-manipulation model. M > -1.78 flags scrutiny."""
    y = year or fin.latest_year()
    yrs = fin.years
    if y not in yrs or yrs.index(y) == 0:
        return ScoreResult("Beneish M-score", None, "", available=False,
                           reason="needs a prior year")
    p = yrs[yrs.index(y) - 1]
    is_, bs, cf = fin.income_statement, fin.balance_sheet, fin.cash_flow

    def gy(df, item):
        return _g(df, item, y), _g(df, item, p)

    (rec_t, rec_p) = gy(bs, "receivables")
    (rev_t, rev_p) = gy(is_, "revenue")
    (gp_t, gp_p) = gy(is_, "gross_profit")
    (ta_t, ta_p) = gy(bs, "total_assets")
    (ca_t, ca_p) = gy(bs, "current_assets")
    (ppe_t, ppe_p) = gy(bs, "ppe_net")
    (dep_t, dep_p) = gy(cf, "depreciation_amortization")
    (sga_t, sga_p) = gy(is_, "sga_expense")
    (ni_t, _) = gy(is_, "net_income")
    (ocf_t, _) = gy(cf, "operating_cash_flow")
    (ltd_t, ltd_p) = gy(bs, "long_term_debt")
    (cl_t, cl_p) = gy(bs, "current_liabilities")

    need = [rec_t, rec_p, rev_t, rev_p, gp_t, gp_p, ta_t, ta_p, ca_t, ca_p,
            ppe_t, ppe_p, dep_t, dep_p, ni_t, ocf_t]
    if any(v is None for v in need):
        return ScoreResult("Beneish M-score", None, "", available=False,
                           reason="missing receivables / PP&E / D&A history")

    dsri = _div(rec_t / rev_t, rec_p / rev_p)
    gmi = _div(gp_p / rev_p, gp_t / rev_t)
    aqi = _div(1 - (ca_t + ppe_t) / ta_t, 1 - (ca_p + ppe_p) / ta_p)
    sgi = rev_t / rev_p
    depi = _div(dep_p / (dep_p + ppe_p), dep_t / (dep_t + ppe_t))
    sgai = (_div(sga_t / rev_t, sga_p / rev_p)
            if sga_t is not None and sga_p is not None else 1.0)
    tata = (ni_t - ocf_t) / ta_t
    lvgi = (_div((ltd_t + cl_t) / ta_t, (ltd_p + cl_p) / ta_p)
            if None not in (ltd_t, ltd_p, cl_t, cl_p) else 1.0)

    parts = [dsri, gmi, aqi, sgi, depi, sgai, lvgi]
    if any(v is None or not np.isfinite(v) for v in parts):
        return ScoreResult("Beneish M-score", None, "", available=False,
                           reason="index term undefined (sign change in inputs)")

    m = (-4.84 + 0.920*dsri + 0.528*gmi + 0.404*aqi + 0.892*sgi
         + 0.115*depi - 0.172*sgai + 4.679*tata - 0.327*lvgi)
    verdict = ("flagged — warrants scrutiny" if m > -1.78 else "no flag")
    return ScoreResult("Beneish M-score", float(m), verdict,
                       detail={"DSRI": dsri, "GMI": gmi, "AQI": aqi, "SGI": sgi,
                               "DEPI": depi, "SGAI": sgai, "TATA": tata, "LVGI": lvgi})


# ------------------------------------------------------- accruals & spread
def sloan_accruals(fin, year: int | None = None) -> ScoreResult:
    y = year or fin.latest_year()
    is_, bs, cf = fin.income_statement, fin.balance_sheet, fin.cash_flow
    ni, ocf, ta = (_g(is_, "net_income", y), _g(cf, "operating_cash_flow", y),
                   _g(bs, "total_assets", y))
    if None in (ni, ocf, ta):
        return ScoreResult("Sloan accruals ratio", None, "", available=False,
                           reason="missing net income / OCF / assets")
    r = (ni - ocf) / ta
    verdict = ("high accruals — lower earnings quality" if r > 0.10 else
               "elevated" if r > 0.05 else "clean — earnings backed by cash")
    return ScoreResult("Sloan accruals ratio", float(r), verdict)


def economic_spread(roic: Optional[float], wacc: Optional[float]) -> ScoreResult:
    if roic is None or wacc is None:
        return ScoreResult("ROIC − WACC spread", None, "", available=False,
                           reason="needs both ROIC and WACC")
    s = roic - wacc
    verdict = ("creating value" if s > 0.02 else
               "roughly at cost of capital" if s > -0.02 else "destroying value")
    return ScoreResult("ROIC − WACC spread", float(s), verdict,
                       detail={"ROIC": roic, "WACC": wacc})


@dataclass
class ForensicReport:
    piotroski: ScoreResult
    altman: ScoreResult
    beneish: ScoreResult
    accruals: ScoreResult
    spread: ScoreResult

    def all_scores(self):
        return [self.piotroski, self.altman, self.beneish, self.accruals, self.spread]

    def __str__(self):
        return "\n".join(f"  {s}" for s in self.all_scores())


def forensic_report(model, market_cap: Optional[float] = None,
                    wacc: Optional[float] = None) -> ForensicReport:
    fin = model.financials
    roic = model.scorecard["metrics"]["roic"].value if "roic" in model.scorecard["metrics"] else None
    if roic is not None and pd.isna(roic):
        roic = None
    return ForensicReport(
        piotroski=piotroski_f(fin),
        altman=altman_z(fin, market_cap),
        beneish=beneish_m(fin),
        accruals=sloan_accruals(fin),
        spread=economic_spread(roic, wacc),
    )
