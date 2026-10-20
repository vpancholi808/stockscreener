"""Screener: filter, sort and rank the universe.

Reached through streamlit_app.py, which declares the navigation.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from finmodel.web import ui

# Columns shown in the table, in order, with how to render each one.
DISPLAY = [
    ("ticker", "Ticker", None),
    ("name", "Company", None),
    ("sector", "Sector", None),
    ("fiscal_year", "FY", None),
    ("sector_percentile", "Sector rank", "pct_bar"),
    ("health_score", "Health", "pct_bar"),
    ("revenue", "Revenue", "money"),
    ("revenue_growth", "Growth", "pct"),
    ("operating_margin", "Op margin", "pct"),
    ("fcf_margin", "FCF margin", "pct"),
    ("roic", "ROIC", "pct"),
    ("roe", "ROE", "pct"),
    ("net_debt_to_ebitda", "ND/EBITDA", "x"),
    ("interest_coverage", "Int cov", "x"),
    ("current_ratio", "Current", "x"),
    ("piotroski", "Piotroski", "int"),
    ("altman", "Altman Z", "num"),
    ("data_quality", "Data", None),
]

SORT_OPTIONS = {
    "Sector rank": "sector_percentile",
    "Health score": "health_score",
    "Revenue": "revenue",
    "Revenue growth": "revenue_growth",
    "Operating margin": "operating_margin",
    "FCF margin": "fcf_margin",
    "ROIC": "roic",
    "ROE": "roe",
    "Net debt / EBITDA": "net_debt_to_ebitda",
    "Interest coverage": "interest_coverage",
    "Piotroski F-score": "piotroski",
    "Altman Z-score": "altman",
    "Ticker": "ticker",
}


def column_config() -> dict:
    cfg = {}
    for key, label, kind in DISPLAY:
        if kind == "pct_bar":
            cfg[key] = st.column_config.ProgressColumn(
                label, min_value=0.0, max_value=1.0, format="percent")
        elif kind == "pct":
            cfg[key] = st.column_config.NumberColumn(label, format="percent")
        elif kind == "money":
            cfg[key] = st.column_config.NumberColumn(label, format="compact")
        elif kind == "x":
            cfg[key] = st.column_config.NumberColumn(label, format="%.1f×")
        elif kind == "num":
            cfg[key] = st.column_config.NumberColumn(label, format="%.2f")
        elif kind == "int":
            cfg[key] = st.column_config.NumberColumn(label, format="%d")
        else:
            cfg[key] = st.column_config.TextColumn(label)
    return cfg


# --------------------------------------------------------------------------
st.title("Equity health screener")
st.caption(
    "Financial health, forensic accounting checks and peer-relative ranking, "
    "computed from SEC filings. No estimates, no price targets, no verdicts."
)

if not ui.require_data():
    st.stop()

stats = ui.load_stats()
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Companies", f"{stats.get('companies') or 0:,}")
c2.metric("Pass screen", f"{stats.get('passing') or 0:,}")
c3.metric("Sectors", stats.get("sectors") or 0)
c4.metric("Latest FY", stats.get("latest_year") or "—")
c5.metric("Backend", ui.backend_name())

# --------------------------------------------------------------------------
# filters
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("Filters")
    search = st.text_input("Search", placeholder="Ticker or company name")
    sectors = ui.sector_names()
    sector = st.selectbox("Sector", ["All sectors"] + sectors)
    passing_only = st.toggle(
        "Only companies passing the screen",
        help="Passes the default scorecard thresholds — see Methodology.")

    st.subheader("Quality")
    min_health = st.slider("Min health score", 0.0, 1.0, 0.0, 0.05,
                           format="%.2f")
    min_roic = st.slider("Min ROIC %", -20, 60, -20, 1)
    min_op_margin = st.slider("Min operating margin %", -50, 60, -50, 1)
    min_fcf_margin = st.slider("Min FCF margin %", -50, 50, -50, 1)

    st.subheader("Growth and size")
    min_growth = st.slider("Min revenue growth %", -50, 100, -50, 1)
    min_revenue = st.number_input("Min revenue ($M)", 0, 1_000_000, 0, 100)

    st.subheader("Balance sheet")
    max_nd_ebitda = st.slider("Max net debt / EBITDA", -5.0, 15.0, 15.0, 0.5)
    min_int_cov = st.slider("Min interest coverage", 0.0, 30.0, 0.0, 0.5)
    min_current = st.slider("Min current ratio", 0.0, 5.0, 0.0, 0.1)

    st.subheader("Forensic")
    min_piotroski = st.slider("Min Piotroski F-score", 0, 9, 0, 1)
    min_altman = st.slider("Min Altman Z-score", -5.0, 10.0, -5.0, 0.5)

    st.divider()
    sort_label = st.selectbox("Sort by", list(SORT_OPTIONS), index=0)
    descending = st.toggle("Highest first", value=True)
    row_limit = st.select_slider("Rows", [50, 100, 250, 500, 1000], value=250)

    st.divider()
    if st.button("Refresh data", width="stretch",
                 help="Clear cached queries and re-read the database."):
        ui.clear_caches()
        st.rerun()

# Only send a filter when it is away from its default, so the SQL stays as
# cheap as possible and the cache key does not churn on untouched sliders.
filters: list[tuple[str, float]] = []
if min_health > 0:
    filters.append(("health_score__min", min_health))
if min_roic > -20:
    filters.append(("roic__min", min_roic / 100))
if min_op_margin > -50:
    filters.append(("operating_margin__min", min_op_margin / 100))
if min_fcf_margin > -50:
    filters.append(("fcf_margin__min", min_fcf_margin / 100))
if min_growth > -50:
    filters.append(("revenue_growth__min", min_growth / 100))
if min_revenue > 0:
    filters.append(("revenue__min", min_revenue * 1e6))
if max_nd_ebitda < 15.0:
    filters.append(("net_debt_to_ebitda__max", max_nd_ebitda))
if min_int_cov > 0:
    filters.append(("interest_coverage__min", min_int_cov))
if min_current > 0:
    filters.append(("current_ratio__min", min_current))
if min_piotroski > 0:
    filters.append(("piotroski__min", min_piotroski))
if min_altman > -5.0:
    filters.append(("altman__min", min_altman))

result = ui.load_screen(
    filters=tuple(filters),
    sector=None if sector == "All sectors" else sector,
    search=search,
    passing_only=passing_only,
    sort_by=SORT_OPTIONS[sort_label],
    desc=descending,
    limit=int(row_limit),
)

rows, total = result["rows"], result["total"]
st.divider()

if not rows:
    st.info("No companies match these filters. Widen them in the sidebar.")
    ui.footer()
    st.stop()

df = ui.strip_market(pd.DataFrame(rows))
shown = [(k, lab) for k, lab, _ in DISPLAY if k in df.columns]
table = df[[k for k, _ in shown]]

left, right = st.columns([3, 1])
with left:
    st.subheader(f"{total:,} match" + ("" if total == 1 else "es")
                 + (f" — showing {len(table):,}" if total > len(table) else ""))
with right:
    st.download_button(
        "Download CSV",
        df.to_csv(index=False).encode(),
        file_name="screen.csv",
        mime="text/csv",
        width="stretch",
    )

st.dataframe(
    table,
    column_config=column_config(),
    hide_index=True,
    width="stretch",
    height=min(640, 38 * len(table) + 40),
)

st.caption(
    "**Sector rank** is peer-relative: each company is ranked inside its own "
    "sector, because a 30% gross margin is unremarkable for software and "
    "exceptional for a distributor. Open any ticker on the **Company** page "
    "for the research note and the full peer comparison."
)

ui.footer()
