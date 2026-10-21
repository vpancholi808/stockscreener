"""Methodology: what is computed, from what, and what it cannot tell you."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from finmodel.web import ui

st.title("Methodology")
st.caption("Where every figure comes from, and what it cannot tell you.")

st.subheader("Source data")
st.markdown("""
Fundamentals come from **SEC EDGAR XBRL company facts** — the structured data
companies file with their 10-K. Nothing is estimated, surveyed or scraped from
a summary site. Each line item is resolved through a chain of XBRL tag
fallbacks, because filers legitimately use different tags for the same concept.

Prices, where enabled, come from a batched Yahoo Finance request: five years of
monthly total-return history in one pass, which yields last price, annualised
volatility, Blume-adjusted beta, one-year and total return, and maximum
drawdown together.

**Nothing is fetched while a page renders.** SEC rate-limits to roughly ten
requests a second and the bulk dataset is about 1.5 GB, so a request-time pull
would be slow and would get the deployment blocked. A separate worker
precomputes everything on a schedule; these pages only read.
""")

st.subheader("The 12-metric health scorecard")
st.markdown("""
Twelve pass/fail checks across solvency, liquidity, cash-flow quality and
profitability. Missing data counts as **n/a** and is excluded from the score,
never as a failure — a company that does not report a line item has not failed
a test on it.
""")
st.dataframe(
    pd.DataFrame([
        ("Debt / equity", "≤ 2.0", "Solvency"),
        ("Net debt / EBITDA", "≤ 3.0", "Solvency"),
        ("Interest coverage", "≥ 3.0×", "Solvency"),
        ("Current ratio", "≥ 1.2", "Liquidity"),
        ("Quick ratio", "≥ 1.0", "Liquidity"),
        ("Free cash flow", "positive", "Cash flow"),
        ("FCF margin", "≥ 5%", "Cash flow"),
        ("Cash conversion", "≥ 0.8", "Cash flow"),
        ("Gross margin", "≥ 30%", "Profitability"),
        ("Operating margin", "≥ 10%", "Profitability"),
        ("Return on equity", "≥ 12%", "Profitability"),
        ("Return on invested capital", "≥ 8%", "Profitability"),
    ], columns=["Metric", "Threshold", "Category"]),
    hide_index=True, width="stretch", height=460)

st.info("""
**These absolute thresholds are sector-biased, which is why they are not the
headline number.** Run them across a universe and software passes nearly
everything while grocery distribution passes almost nothing — not because
distributors are badly run, but because a 30% gross margin is structurally
impossible in that business. The scorecard is useful as a checklist; the
ranking you see first is peer-relative instead.
""", icon="💡")

st.subheader("Sector-relative ranking")
st.markdown("""
The headline **sector rank** is a two-step calculation:

1. Percentile-rank the company within its sector on each of six metrics —
   health score, revenue growth, operating margin, FCF margin, ROIC and ROE.
2. Average those six ranks into a composite, then **rank the composite again**
   within the sector.

The second step is not redundant. A mean of percentiles is not itself a
percentile: it compresses toward 0.5 and never reaches 1, so reporting it as
"Nth percentile" would understate every strong company and overstate every weak
one. Ranking the composite makes the number mean what the label says — this
company scores above N% of its sector peers.

Sector benchmarks use the **median, never the mean**, and a sector needs at
least five constituents with the metric present before any benchmark is
reported. Valuation multiples are heavily right-skewed; one company at 400×
P/E drags a mean far above anything a typical constituent trades at. The mean
is stored for reference, because the gap between the two is itself informative.
""")

st.subheader("Forensic accounting checks")
st.markdown("""
- **Piotroski F-score** (0–9) — nine binary tests of improving fundamentals.
- **Altman Z-score** — distress model; below ~1.8 is the classic distress zone.
- **Beneish M-score** — eight variables associated with earnings manipulation.
  A high score means the accounting *resembles* manipulated filings, not that
  manipulation occurred.
- **Sloan accruals** — the gap between accounting earnings and cash. Large
  positive accruals have historically preceded weak returns.
- **ROIC − WACC spread** — whether the business earns more than its capital
  costs.
""")

st.subheader("Valuation")
st.markdown("""
P/E, P/B, P/S, EV/EBITDA, EV/Sales, EV/EBIT, PEG, FCF yield, earnings yield and
dividend yield. Market cap is price × diluted shares *from the filings*, so the
numerator and denominator of every multiple sit on the same basis.

**A multiple on a non-positive denominator returns nothing, not a number.** A
P/E computed on negative earnings is undefined, not cheap, and letting it
through would corrupt every sector median it landed in.

**Forward P/E is always empty.** It requires analyst consensus estimates, which
the SEC does not publish and this project does not buy.
""")

st.subheader("Why there are no buy, sell or hold verdicts")
st.markdown("""
The research notes report evidence — margins, leverage, cash conversion,
valuation against peers — and stop. There is no verdict field, and this is
deliberate for three reasons:

1. A public site telling people whether to invest is giving investment advice,
   a regulated activity in most jurisdictions.
2. Every figure comes from annual filings and can be up to twelve months stale.
   A verdict built on that would be confidently wrong often enough to cause
   real losses.
3. Financial health and price attractiveness are different questions. One label
   collapses them into a false single answer.

The test suite enforces this: it asserts that no verdict field exists and that
no buy/sell language appears anywhere outside the disclaimer.
""")

st.subheader("Not every metric applies to every sector")
st.markdown("""
Some columns are empty for some companies, and that is the honest answer
rather than a gap in the data.

- **Gross margin** requires a cost-of-revenue line. Banks, insurers and REITs
  do not report one, because the concept does not apply to them — there is no
  "cost" of making a loan in the way there is a cost of making a phone. Those
  sectors show no gross margin and no gross-margin benchmark.
- **Altman Z-score** was estimated on manufacturers. Its inputs are often
  absent for financials, and where they exist the score is not comparable to
  the thresholds it is famous for. Treat a missing Z for a bank as correct.
- **Interest coverage** needs a separately reported interest expense. Many
  filers fold it into a net financing line.
- **Current ratio** assumes a classified balance sheet. Banks and insurers
  present unclassified ones, so there is no "current" split to divide.

A missing value is always excluded from the health score rather than counted
as a failure. A company is never marked down for not reporting something its
accounting framework does not contain.
""")

st.subheader("Known limitations")
st.markdown("""
- **Annual data only.** Every figure is as-of the last filed 10-K. For a
  company mid-cycle the gap to current trading can be very large. Quarterly/TTM
  parsing is the top-priority next addition and should be treated as a
  prerequisite for serious use, not a follow-up.
- **No forward estimates**, for the reason above.
- **Point-in-time is available but off by default.** The loader can filter to
  data filed on or before a given date (`as_of`), which is required for any
  backtest to avoid look-ahead bias. The live pages use the latest filing.
- **Missing debt tags read as zero debt.** Verify leverage on unfamiliar names.
- **US SEC filers only.** Foreign issuers filing Form 20-F have partial
  coverage.
- **No backtester yet**, and therefore no machine learning on these signals.
  Walk-forward validation with purging and embargo has to exist first;
  without it, any model fitted here would be measuring its own leakage.
""")

st.subheader("Price data licensing")
st.markdown(f"""
SEC filings are U.S. government work and in the public domain, so fundamentals
and every ratio derived from them can be republished freely.

Scraped price data is different: the provider's terms generally prohibit
redistribution. Price-derived fields are therefore withheld unless
`SERVE_MARKET_DATA=true`, which should only be set once you have moved to a
feed whose licence permits redistribution.

On this deployment, price-derived fields are
**{'served' if ui.SERVE_MARKET_DATA else 'withheld'}**.
""")

ui.footer()
