"""Company detail: statements, peer comparison and the research note."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from finmodel.web import ui

SEVERITY_ICON = {
    "strength": "✅", "positive": "✅",
    "concern": "⚠️", "negative": "⚠️",
    "watch": "👁️", "neutral": "•", "context": "•",
}


def finding_list(items: list[dict], empty: str) -> None:
    if not items:
        st.caption(empty)
        return
    for f in items:
        kind = str(f.get("kind", "")).lower().split(".")[-1]
        icon = SEVERITY_ICON.get(kind, "•")
        st.markdown(f"{icon} **{f.get('topic', '')}** — {f.get('statement', '')}")
        if f.get("evidence"):
            st.caption(f"　　{f['evidence']}")


# --------------------------------------------------------------------------
st.title("Company analysis")

if not ui.require_data():
    st.stop()

tickers = ui.load_tickers()
if not tickers:
    st.info("No companies in the database.")
    st.stop()

# A ticker in the URL makes a company page shareable: ?ticker=MU
qp = st.query_params
preset = (qp.get("ticker") or "").upper()
index = tickers.index(preset) if preset in tickers else 0

ticker = st.selectbox("Company", tickers, index=index)
if ticker != preset:
    st.query_params["ticker"] = ticker

company = ui.load_company(ticker)
if not company:
    st.error(f"No data for {ticker}.")
    st.stop()

name = company.get("name") or ticker
sector = company.get("sector") or "Unknown"
fy = company.get("fiscal_year")

st.header(f"{name} ({ticker})")
st.caption(f"{sector} · FY{fy} as filed · data quality "
           f"{company.get('data_quality') or '—'}")

# --- headline numbers -----------------------------------------------------
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Health", ui.health_label(company.get("health_passed"),
                                    company.get("health_total")),
          help="Checks passed out of the 12-metric scorecard.")
pct = company.get("sector_percentile")
c2.metric("Sector rank",
          f"{pct * 100:.0f}th" if pct is not None else "—",
          help=f"Percentile within {sector}, not the whole market.")
c3.metric("Revenue", ui.fmt_money(company.get("revenue")))
c4.metric("Revenue growth", ui.fmt_pct(company.get("revenue_growth")))
c5.metric("ROIC", ui.fmt_pct(company.get("roic")))

tab_fin, tab_peers, tab_research = st.tabs(
    ["Financials", "Versus sector", "Research note"])

# --------------------------------------------------------------------------
# financials
# --------------------------------------------------------------------------
with tab_fin:
    groups = {
        "Profitability": [
            ("gross_margin", "Gross margin", "pct"),
            ("operating_margin", "Operating margin", "pct"),
            ("net_margin", "Net margin", "pct"),
            ("roe", "Return on equity", "pct"),
            ("roic", "Return on invested capital", "pct"),
        ],
        "Cash": [
            ("fcf", "Free cash flow", "money"),
            ("fcf_margin", "FCF margin", "pct"),
            ("cash_conversion", "Cash conversion", "num"),
            ("ebitda", "EBITDA", "money"),
            ("accruals", "Sloan accruals", "num"),
        ],
        "Balance sheet": [
            ("net_debt", "Net debt", "money"),
            ("net_debt_to_ebitda", "Net debt / EBITDA", "x"),
            ("debt_to_equity", "Debt / equity", "x"),
            ("interest_coverage", "Interest coverage", "x"),
            ("current_ratio", "Current ratio", "x"),
        ],
        "Forensic": [
            ("piotroski", "Piotroski F-score", "int9"),
            ("altman", "Altman Z-score", "num"),
            ("beneish", "Beneish M-score", "num"),
        ],
    }
    fmt = {"pct": ui.fmt_pct, "money": ui.fmt_money, "x": ui.fmt_x,
           "num": ui.fmt_num}

    cols = st.columns(len(groups))
    for col, (title, items) in zip(cols, groups.items()):
        with col:
            st.subheader(title)
            for key, label, kind in items:
                v = company.get(key)
                if kind == "int9":
                    shown = "—" if v is None else f"{int(v)}/9"
                else:
                    shown = fmt[kind](v)
                st.markdown(f"**{shown}**  \n{label}")

    history = company.get("history") or []
    if history:
        st.divider()
        st.subheader("Five-year history")
        hist = pd.DataFrame(history).set_index("fiscal_year").sort_index()

        a, b = st.columns(2)
        with a:
            st.caption("Revenue and free cash flow")
            have = [c for c in ("revenue", "fcf") if c in hist.columns]
            if have:
                st.bar_chart(hist[have], height=260)
        with b:
            st.caption("Margins")
            m = pd.DataFrame(index=hist.index)
            if {"gross_profit", "revenue"} <= set(hist.columns):
                m["gross"] = hist["gross_profit"] / hist["revenue"]
            if {"operating_income", "revenue"} <= set(hist.columns):
                m["operating"] = hist["operating_income"] / hist["revenue"]
            if {"net_income", "revenue"} <= set(hist.columns):
                m["net"] = hist["net_income"] / hist["revenue"]
            if not m.empty:
                st.line_chart(m, height=260)

        with st.expander("History table"):
            st.dataframe(hist, width="stretch")
    else:
        st.caption("No filed history stored for this company.")

# --------------------------------------------------------------------------
# versus sector
# --------------------------------------------------------------------------
with tab_peers:
    research = ui.load_research(ticker)
    comparisons = (research or {}).get("comparisons") or []
    if not comparisons:
        st.info(
            f"No peer comparison available. {sector} needs at least five "
            "constituents with the metric present before a benchmark is "
            "reported — a median over three companies is noise."
        )
    else:
        st.subheader(f"Against the {sector} median")
        st.caption(
            "Comparisons use the **median**, not the mean. Valuation multiples "
            "are right-skewed, so one company at 400× P/E drags a sector mean "
            "far above anything a typical constituent trades at."
        )
        rows = []
        for c in comparisons:
            val, med = c.get("value"), c.get("sector_median")
            is_pct = c.get("is_percent")
            rows.append({
                "Metric": c.get("label") or c.get("metric"),
                "Company": ui.fmt_pct(val) if is_pct else ui.fmt_num(val),
                "Sector median": ui.fmt_pct(med) if is_pct else ui.fmt_num(med),
                "Gap": ui.fmt_pct(c["premium"]) if c.get("premium") is not None else "—",
                "Percentile": c.get("percentile"),
                "Standing": c.get("verdict"),
                "Peers": c.get("peers"),
            })
        st.dataframe(
            pd.DataFrame(rows),
            hide_index=True,
            width="stretch",
            column_config={
                "Percentile": st.column_config.ProgressColumn(
                    "Percentile", min_value=0.0, max_value=1.0,
                    format="percent",
                    help="Direction-adjusted: higher is always better, "
                         "including for metrics where a lower raw reading is "
                         "the good outcome."),
                "Peers": st.column_config.NumberColumn("Peers", format="%d"),
            },
            height=min(620, 38 * len(rows) + 40),
        )

# --------------------------------------------------------------------------
# research note
# --------------------------------------------------------------------------
with tab_research:
    research = ui.load_research(ticker)
    if not research:
        st.info("No research note available for this company.")
    else:
        note = research["note"]
        st.subheader(note.get("headline") or name)
        st.write(note.get("summary") or "")

        if note.get("valuation_label"):
            st.info(f"**Valuation context** — {note['valuation_label']}",
                    icon="🏷️")

        left, right = st.columns(2)
        with left:
            st.markdown("#### What the figures support")
            finding_list(note.get("strengths") or [],
                         "Nothing in the filed figures stands out favourably.")
        with right:
            st.markdown("#### What the figures flag")
            finding_list(note.get("concerns") or [],
                         "No flags raised by the checks that ran.")

        if note.get("watch_items"):
            st.markdown("#### Worth watching")
            finding_list(note["watch_items"], "")
        if note.get("context"):
            st.markdown("#### Context")
            finding_list(note["context"], "")

        if note.get("caveats"):
            with st.expander("Limitations of this note", expanded=True):
                for c in note["caveats"]:
                    st.markdown(f"- {c}")

        st.warning(ui.NO_RECOMMENDATION, icon="⚖️")
        st.download_button(
            "Download note as text",
            research["text"].encode(),
            file_name=f"{ticker}_research.txt",
            mime="text/plain",
        )

ui.footer()
