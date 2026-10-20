"""Data-quality checks on the extracted historical statements.

Missing fundamentals shouldn't fail silently (a missing debt tag reading as zero
debt would flatter every leverage metric). This flags coverage gaps, missing key
items, and whether the historical balance sheet articulates
(Assets ~ Liabilities + Equity) where all three are present.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

KEY_ITEMS = {
    "income_statement": ["revenue", "operating_income", "net_income"],
    "balance_sheet": ["total_assets", "total_equity", "cash"],
    "cash_flow": ["operating_cash_flow", "capex"],
}


def quality_report(fin) -> dict:
    y = fin.latest_year()
    frames = {"income_statement": fin.income_statement,
              "balance_sheet": fin.balance_sheet,
              "cash_flow": fin.cash_flow}

    coverage, missing_key = {}, []
    for name, df in frames.items():
        present = df[y].notna().sum() if y in df.columns else 0
        coverage[name] = round(float(present) / len(df.index), 3) if len(df.index) else 0.0
        for item in KEY_ITEMS[name]:
            if item not in df.index or pd.isna(df.at[item, y]):
                missing_key.append(f"{name}:{item}")

    # Historical balance-sheet articulation: A ~ L + E.
    bs = fin.balance_sheet
    articulation = []
    for yr in fin.years:
        A = bs.at["total_assets", yr] if "total_assets" in bs.index else np.nan
        L = bs.at["total_liabilities", yr] if "total_liabilities" in bs.index else np.nan
        E = bs.at["total_equity", yr] if "total_equity" in bs.index else np.nan
        if pd.notna(A) and pd.notna(L) and pd.notna(E) and A != 0:
            resid = (A - (L + E)) / A
            articulation.append({"year": yr, "residual_pct": round(float(resid), 4),
                                 "ok": abs(resid) < 0.01})

    debt_present = any(("long_term_debt" in bs.index and pd.notna(bs.at["long_term_debt", y]),
                        "short_term_debt" in bs.index and pd.notna(bs.at["short_term_debt", y]))) \
        if y in bs.columns else False

    flags = []
    if not debt_present:
        flags.append("No debt tags found — leverage metrics assume zero debt; verify manually.")
    if any(not a["ok"] for a in articulation):
        flags.append("Balance sheet does not articulate in some years (A != L+E).")
    if missing_key:
        flags.append(f"Missing key items: {', '.join(missing_key)}")

    grade = "clean" if not flags else ("warnings" if debt_present and not missing_key else "check")
    return {"ticker": fin.ticker, "fiscal_year": y, "coverage": coverage,
            "missing_key_items": missing_key, "articulation": articulation,
            "flags": flags, "grade": grade}


def quality_summary(report: dict) -> str:
    lines = [f"Data quality [{report['grade']}] — {report['ticker']} FY{report['fiscal_year']}"]
    cov = report["coverage"]
    lines.append("  coverage: " + ", ".join(f"{k.split('_')[0]} {v:.0%}" for k, v in cov.items()))
    for f in report["flags"]:
        lines.append(f"  ! {f}")
    if not report["flags"]:
        lines.append("  no issues detected")
    return "\n".join(lines)
