"""Investment research summaries — evidence, not verdicts.

Produces a structured written assessment of a company: what the figures
support, what they flag, how the valuation sits against sector peers, and what
the data cannot tell you.

DELIBERATELY NO BUY / SELL / HOLD FIELD, and this is not an oversight.

  1. A site that tells the public whether to invest is giving investment
     advice, which is a regulated activity in most jurisdictions.
  2. Every figure here comes from annual filings and can be up to twelve months
     stale. A verdict built on that would be confidently wrong often enough to
     cause real losses.
  3. Financial health and price attractiveness are different questions, and a
     single label collapses them into a false one.

Each finding carries the number it rests on, so a reader can check the claim
rather than trust the label. Findings are generated from explicit rules — no
model, no randomness, same input gives the same output.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

import pandas as pd

from .sectors import Comparison, valuation_summary

Severity = str  # "strength" | "concern" | "watch" | "context"


@dataclass
class Finding:
    kind: Severity
    topic: str
    statement: str
    evidence: str
    metric: Optional[str] = None

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class ResearchNote:
    ticker: str
    name: str
    sector: str
    fiscal_year: Optional[int]
    headline: str
    summary: str
    strengths: list[Finding] = field(default_factory=list)
    concerns: list[Finding] = field(default_factory=list)
    watch_items: list[Finding] = field(default_factory=list)
    context: list[Finding] = field(default_factory=list)
    valuation_label: Optional[str] = None
    valuation_percentile: Optional[float] = None
    health_passed: Optional[int] = None
    health_total: Optional[int] = None
    caveats: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        d = asdict(self)
        for k in ("strengths", "concerns", "watch_items", "context"):
            d[k] = [f for f in d[k]]
        return d

    def as_text(self) -> str:
        lines = [f"{self.name} ({self.ticker}) — {self.sector}",
                 f"FY{self.fiscal_year} as filed", "",
                 self.headline, "", self.summary, ""]
        for title, items in (("What the figures support", self.strengths),
                             ("What the figures flag", self.concerns),
                             ("Worth watching", self.watch_items),
                             ("Context", self.context)):
            if items:
                lines.append(title.upper())
                for f in items:
                    lines.append(f"  - {f.statement} ({f.evidence})")
                lines.append("")
        if self.caveats:
            lines.append("LIMITATIONS")
            lines += [f"  - {c}" for c in self.caveats]
        lines += ["", "This is an evidence summary, not investment advice. "
                      "No buy or sell recommendation is made or implied."]
        return "\n".join(lines)


def _pct(v, digits=1):
    return "n/a" if v is None else f"{v * 100:.{digits}f}%"


def _x(v, digits=1):
    return "n/a" if v is None else f"{v:.{digits}f}x"


def _num(v, digits=2):
    return "n/a" if v is None else f"{v:.{digits}f}"


def _ok(v) -> bool:
    if v is None:
        return False
    try:
        return not pd.isna(v)
    except (TypeError, ValueError):
        return True


def generate_research(company: dict,
                      comparisons: Optional[dict[str, Comparison]] = None,
                      scorecard_metrics: Optional[dict] = None) -> ResearchNote:
    """Build a research note from a company row plus its sector comparisons.

    `company` is the flat row produced by the loader (margins, returns,
    leverage, forensic scores, multiples). `comparisons` comes from
    sectors.compare_to_sector."""
    comparisons = comparisons or {}
    s: list[Finding] = []
    c: list[Finding] = []
    w: list[Finding] = []
    ctx: list[Finding] = []

    ticker = company.get("ticker", "")
    name = company.get("name", ticker)
    sector = company.get("sector", "Unknown")
    passed = company.get("health_passed")
    total = company.get("health_total")

    def cmp_note(metric: str) -> str:
        comp = comparisons.get(metric)
        if comp is None or not comp.available:
            return ""
        med = _pct(comp.sector_median) if comp.is_percent else _x(comp.sector_median)
        return f"; sector median {med}, {comp.verdict()}"

    # ---------------- profitability & returns ----------------
    om = company.get("operating_margin")
    if _ok(om):
        if om >= 0.20:
            s.append(Finding("strength", "Profitability",
                             "Operating margin is strong in absolute terms",
                             f"operating margin {_pct(om)}{cmp_note('operating_margin')}",
                             "operating_margin"))
        elif om < 0:
            c.append(Finding("concern", "Profitability",
                             "The business is loss-making at the operating line",
                             f"operating margin {_pct(om)}", "operating_margin"))
        elif om < 0.05:
            c.append(Finding("concern", "Profitability",
                             "Operating margin leaves little room for error",
                             f"operating margin {_pct(om)}{cmp_note('operating_margin')}",
                             "operating_margin"))

    roic = company.get("roic")
    if _ok(roic):
        if roic >= 0.15:
            s.append(Finding("strength", "Returns",
                             "Returns on invested capital are high",
                             f"ROIC {_pct(roic)}{cmp_note('roic')}", "roic"))
        elif roic < 0.05:
            c.append(Finding("concern", "Returns",
                             "Returns on capital are low; growth may not create value",
                             f"ROIC {_pct(roic)}{cmp_note('roic')}", "roic"))

    # ROE far above ROIC usually means leverage is doing the work.
    roe = company.get("roe")
    if _ok(roe) and _ok(roic) and roic > 0 and roe > roic * 2.0 and roe > 0.15:
        w.append(Finding("watch", "Returns",
                         "Return on equity is flattered by leverage rather than "
                         "operating performance",
                         f"ROE {_pct(roe)} against ROIC {_pct(roic)}", "roe"))

    # ---------------- cash generation ----------------
    fcfm = company.get("fcf_margin")
    fcf = company.get("fcf")
    if _ok(fcf) and fcf < 0:
        c.append(Finding("concern", "Cash flow",
                         "Free cash flow is negative; operations plus capital "
                         "spending consumed cash",
                         f"free cash flow {fcf / 1e6:,.0f}M", "fcf"))
    elif _ok(fcfm):
        if fcfm >= 0.15:
            s.append(Finding("strength", "Cash flow",
                             "Converts a high share of revenue into free cash",
                             f"FCF margin {_pct(fcfm)}{cmp_note('fcf_margin')}",
                             "fcf_margin"))
        elif fcfm < 0.03:
            w.append(Finding("watch", "Cash flow",
                             "Thin free-cash-flow margin limits self-funding",
                             f"FCF margin {_pct(fcfm)}{cmp_note('fcf_margin')}",
                             "fcf_margin"))

    cc = company.get("cash_conversion")
    if _ok(cc):
        if cc < 0.8:
            c.append(Finding("concern", "Earnings quality",
                             "Reported profit is not fully backed by operating cash",
                             f"operating cash flow is {_x(cc, 2)} net income",
                             "cash_conversion"))
        elif cc >= 1.2:
            s.append(Finding("strength", "Earnings quality",
                             "Operating cash flow comfortably exceeds reported profit",
                             f"cash conversion {_x(cc, 2)}", "cash_conversion"))

    acc = company.get("accruals")
    if _ok(acc) and acc > 0.10:
        c.append(Finding("concern", "Earnings quality",
                         "High accruals — a persistent gap between profit and cash",
                         f"accruals ratio {_num(acc, 3)}", "accruals"))

    # ---------------- balance sheet ----------------
    nde = company.get("net_debt_to_ebitda")
    if _ok(nde):
        if nde > 4:
            c.append(Finding("concern", "Leverage",
                             "Leverage is high relative to earnings",
                             f"net debt {_x(nde)} EBITDA{cmp_note('net_debt_to_ebitda')}",
                             "net_debt_to_ebitda"))
        elif nde < 0:
            s.append(Finding("strength", "Balance sheet",
                             "Holds net cash rather than net debt",
                             f"net debt / EBITDA {_x(nde)}", "net_debt_to_ebitda"))

    ic = company.get("interest_coverage")
    if _ok(ic) and ic < 3:
        c.append(Finding("concern", "Leverage",
                         "Interest cover is thin; earnings barely service the debt",
                         f"EBIT covers interest {_x(ic)}", "interest_coverage"))

    cr = company.get("current_ratio")
    if _ok(cr) and cr < 1.0:
        c.append(Finding("concern", "Liquidity",
                         "Current liabilities exceed current assets",
                         f"current ratio {_x(cr, 2)}", "current_ratio"))

    # ---------------- growth ----------------
    g = company.get("revenue_growth")
    if _ok(g):
        if g >= 0.20:
            s.append(Finding("strength", "Growth",
                             "Revenue has compounded quickly over the reported years",
                             f"revenue CAGR {_pct(g)}{cmp_note('revenue_growth')}",
                             "revenue_growth"))
        elif g < 0:
            c.append(Finding("concern", "Growth",
                             "Revenue has contracted over the reported period",
                             f"revenue CAGR {_pct(g)}{cmp_note('revenue_growth')}",
                             "revenue_growth"))

    # ---------------- forensic ----------------
    pio = company.get("piotroski")
    if _ok(pio):
        if pio >= 8:
            s.append(Finding("strength", "Fundamental trend",
                             "Fundamentals improved on nearly every Piotroski test",
                             f"Piotroski F-score {int(pio)}/9", "piotroski"))
        elif pio <= 3:
            c.append(Finding("concern", "Fundamental trend",
                             "Fundamentals deteriorated across most Piotroski tests",
                             f"Piotroski F-score {int(pio)}/9", "piotroski"))

    alt = company.get("altman")
    if _ok(alt):
        if alt < 1.81:
            c.append(Finding("concern", "Solvency",
                             "Altman Z sits in the distress range",
                             f"Altman Z {_num(alt)} (below 1.81)", "altman"))
        elif alt < 2.99:
            w.append(Finding("watch", "Solvency",
                             "Altman Z sits in the grey zone rather than the safe range",
                             f"Altman Z {_num(alt)}", "altman"))

    ben = company.get("beneish")
    if _ok(ben) and ben > -1.78:
        w.append(Finding("watch", "Earnings quality",
                         "Beneish M-score is above the scrutiny threshold — a prompt "
                         "to read the filings closely, not evidence of wrongdoing",
                         f"Beneish M {_num(ben)} (above -1.78)", "beneish"))

    # ---------------- valuation ----------------
    val = valuation_summary(comparisons)
    if val["available"]:
        parts = []
        for m in ("pe", "ev_ebitda", "ps", "pb"):
            comp = comparisons.get(m)
            if comp and comp.available:
                parts.append(f"{comp.label} {_x(comp.value)} vs sector "
                             f"{_x(comp.sector_median)}")
        ctx.append(Finding("context", "Valuation",
                           f"Trades {val['label']}",
                           "; ".join(parts) if parts else "multiple comparison",
                           "valuation"))

    fy = company.get("fcf_yield")
    if _ok(fy) and fy >= 0.06:
        s.append(Finding("strength", "Valuation",
                         "Free-cash-flow yield is high relative to the market price",
                         f"FCF yield {_pct(fy)}{cmp_note('fcf_yield')}", "fcf_yield"))

    # ---------------- headline & summary ----------------
    health = (f"{passed}/{total}" if passed is not None and total else "n/a")
    quality_word = "strong" if (passed or 0) >= 10 else \
        "mixed" if (passed or 0) >= 7 else "weak"
    headline = (f"{name} passes {health} financial-health checks "
                f"({quality_word} on the balance sheet and cash-flow tests)")
    if val["available"]:
        headline += f", and trades {val['label']}."
    else:
        headline += "."

    bits = []
    if _ok(g):
        bits.append(f"revenue has compounded at {_pct(g)} across the reported years")
    if _ok(om):
        bits.append(f"operating margin is {_pct(om)}")
    if _ok(roic):
        bits.append(f"ROIC is {_pct(roic)}")
    first = "On the reported figures, " + ", ".join(bits) + "." if bits else ""
    second = (f" The scorecard flags {len(c)} concern"
              f"{'s' if len(c) != 1 else ''}"
              f" and {len(s)} area{'s' if len(s) != 1 else ''} of strength."
              if (c or s) else "")
    third = ""
    if val["available"]:
        third = (f" Against {sector} peers the shares screen as "
                 f"{val['label'].replace(' vs sector', '')}.")
    summary = (first + second + third).strip()

    caveats = [
        "Figures come from the most recent annual filing and can be up to "
        "twelve months out of date; for a company mid-cycle the gap to current "
        "trading can be large.",
        "Sector comparisons use the median of peers in the same SIC grouping; "
        "sectors with few constituents give noisy benchmarks.",
        "No analyst estimates are used, so forward multiples and growth "
        "forecasts are absent.",
        "This is a quantitative screen of filed statements. It does not assess "
        "management, competitive position, regulation, or anything not in the "
        "numbers.",
    ]

    return ResearchNote(
        ticker=ticker, name=name, sector=sector,
        fiscal_year=company.get("fiscal_year"),
        headline=headline, summary=summary,
        strengths=s, concerns=c, watch_items=w, context=ctx,
        valuation_label=val.get("label") if val["available"] else None,
        valuation_percentile=val.get("percentile"),
        health_passed=passed, health_total=total,
        caveats=caveats,
    )
