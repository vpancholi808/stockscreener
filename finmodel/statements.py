"""Build clean, annual historical statements from raw SEC companyfacts JSON.

Output is a `Financials` object holding three pandas DataFrames (income
statement, balance sheet, cash flow), each indexed by line item with one
column per fiscal year. Missing line items are filled with NaN rather than
dropped, so downstream code can decide how to handle gaps.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional

import pandas as pd

from .data.concepts import ALL_MAPS, INSTANT_ITEMS, SUBSET_MAX


def _parse(d: str) -> date:
    return date.fromisoformat(d)


def _facts_for_tag(facts: dict, tag: str) -> list[dict]:
    """Return the USD (or shares) fact entries for a us-gaap tag, or []."""
    node = facts.get("facts", {}).get("us-gaap", {}).get(tag)
    if not node:
        return []
    units = node.get("units", {})
    for key in ("USD", "shares", "USD/shares"):
        if key in units:
            return units[key]
    return []


def _annual_values_for_tag(facts: dict, tag: str, instant: bool,
                           as_of: str | None = None) -> dict[int, float]:
    """Map fiscal-year -> value for one tag, using only annual (10-K/FY) data.

    Point-in-time: if `as_of` (an ISO date) is given, only facts *filed* on or
    before that date are eligible, and among those the latest-filed value for
    each fiscal year wins (so a restatement is used only once it was public).
    With as_of=None the behavior is unchanged (period-length scored)."""
    out: dict[int, tuple[float, tuple]] = {}  # year -> (value, sort_key)
    for f in _facts_for_tag(facts, tag):
        form = f.get("form", "")
        if not form.startswith("10-K"):
            continue
        try:
            end = _parse(f["end"])
        except (KeyError, ValueError):
            continue
        filed = f.get("filed")
        if as_of is not None:
            if filed is None or filed > as_of:
                continue  # not yet public as of the requested date
        year = end.year
        if instant:
            score = 1 if f.get("fp") == "FY" else 0
        else:
            if "start" not in f:
                continue
            span = (end - _parse(f["start"])).days
            if not (350 <= span <= 380):
                continue
            score = span
        val = f.get("val")
        if val is None:
            continue
        # Point-in-time ranks by filing recency; default ranks by period fit.
        sort_key = (filed or "", score) if as_of is not None else (score,)
        prev = out.get(year)
        if prev is None or sort_key >= prev[1]:
            out[year] = (float(val), sort_key)
    return {y: v for y, (v, _) in out.items()}


def _merge_by_priority(facts: dict, tags: list[str], instant: bool,
                       as_of: str | None = None) -> dict[int, float]:
    """Fill each year using the first tag (in priority order) that has a value."""
    merged: dict[int, float] = {}
    for tag in tags:
        for year, val in _annual_values_for_tag(facts, tag, instant, as_of).items():
            merged.setdefault(year, val)
    return merged


def _merge_by_max(facts: dict, tags: list[str], instant: bool,
                  as_of: str | None = None) -> dict[int, float]:
    """Fill each year with the LARGEST candidate rather than the first.

    Only for line items in concepts.SUBSET_MAX, where the alternative tags
    are partial by accounting standard rather than merely different. A REIT
    that reports both `Revenues` and ASC 606 contract revenue is reporting a
    total and a fraction of it; first-wins ordering can pick the fraction,
    and the resulting margin is nonsense. See the note in concepts.py.
    """
    merged: dict[int, float] = {}
    for tag in tags:
        for year, val in _annual_values_for_tag(facts, tag, instant, as_of).items():
            if year not in merged or val > merged[year]:
                merged[year] = val
    return merged


@dataclass
class Financials:
    ticker: str
    entity_name: str
    income_statement: pd.DataFrame
    balance_sheet: pd.DataFrame
    cash_flow: pd.DataFrame
    years: list[int]

    def latest_year(self) -> int:
        return self.years[-1]

    def all_statements(self) -> pd.DataFrame:
        """Stacked view of all three statements for quick inspection."""
        frames = []
        for name, df in (("IS", self.income_statement),
                         ("BS", self.balance_sheet),
                         ("CF", self.cash_flow)):
            tagged = df.copy()
            tagged.index = [f"{name}:{i}" for i in tagged.index]
            frames.append(tagged)
        return pd.concat(frames)


def build_financials(facts: dict, ticker: str = "", n_years: int = 5,
                     as_of: str | None = None) -> Financials:
    """Construct annual historical statements from raw companyfacts JSON.

    Pass `as_of` (ISO date) for a point-in-time view that only uses data filed
    on or before that date — essential for look-ahead-free backtesting."""
    entity = facts.get("entityName", ticker)
    statements: dict[str, pd.DataFrame] = {}
    all_years: set[int] = set()

    for stmt_name, mapping in ALL_MAPS.items():
        rows: dict[str, dict[int, float]] = {}
        for item, tags in mapping.items():
            instant = item in INSTANT_ITEMS
            merge = _merge_by_max if item in SUBSET_MAX else _merge_by_priority
            rows[item] = merge(facts, tags, instant, as_of)
            all_years.update(rows[item].keys())
        statements[stmt_name] = rows

    years = sorted(all_years)[-n_years:] if all_years else []

    def to_df(rows: dict[str, dict[int, float]]) -> pd.DataFrame:
        df = pd.DataFrame(index=list(rows.keys()), columns=years, dtype="float64")
        for item, by_year in rows.items():
            for y in years:
                df.at[item, y] = by_year.get(y, float("nan"))
        return df

    return Financials(
        ticker=ticker or entity,
        entity_name=entity,
        income_statement=to_df(statements["income_statement"]),
        balance_sheet=to_df(statements["balance_sheet"]),
        cash_flow=to_df(statements["cash_flow"]),
        years=years,
    )


def derive_missing(fin: Financials) -> Financials:
    """Fill obviously-derivable gaps so the scorecard has what it needs.

    Examples: gross_profit = revenue - cogs; ebitda = operating_income + D&A;
    net debt = total debt - cash. These are stored back on the relevant frame.
    """
    is_ = fin.income_statement
    cf = fin.cash_flow
    for y in fin.years:
        rev, cogs = is_.at["revenue", y], is_.at["cogs", y]
        if pd.isna(is_.at["gross_profit", y]) and pd.notna(rev) and pd.notna(cogs):
            is_.at["gross_profit", y] = rev - cogs
        # EBITDA = operating income + D&A (D&A usually lives on the cash flow).
        oi = is_.at["operating_income", y]
        da = cf.at["depreciation_amortization", y]
        if "ebitda" not in is_.index:
            is_.loc["ebitda"] = float("nan")
        if pd.notna(oi) and pd.notna(da):
            is_.at["ebitda", y] = oi + da
        elif pd.notna(oi):
            is_.at["ebitda", y] = oi
    return fin
