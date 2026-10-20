"""Portfolio construction and assessment.

Risk is enforced as *constraints*, not as a score: position caps, sector caps,
solvency floors and a minimum number of holdings. Where two caps conflict the
position cap wins and the result says so.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from finmodel.portfolio_builder import RISK_PROFILES, STRATEGIES
from finmodel.web import ui

ALLOCATIONS = {
    "score": ("Strategy-weighted",
              "Weight by strategy score, capped per position."),
    "equal": ("Equal weight",
              "Every holding the same size. Hard to beat when expected "
              "returns are uncertain."),
    "inverse_vol": ("Inverse volatility",
                    "Smaller weights to more volatile names. Needs price "
                    "history, so it falls back to equal weight without it."),
}


@st.cache_data(ttl=300, show_spinner=False)
def build(strategy: str, risk_profile: str, allocation: str,
          max_holdings: int, sector_limit: float,
          sectors: tuple[str, ...]) -> dict:
    from finmodel.portfolio_builder import build_portfolio
    rows = ui.load_universe()
    if sectors:
        wanted = {s.lower() for s in sectors}
        rows = [r for r in rows if str(r.get("sector", "")).lower() in wanted]
    try:
        result = build_portfolio(
            rows, strategy=strategy, risk_profile=risk_profile,
            allocation=allocation, max_holdings=max_holdings,
            sector_limit=sector_limit)
    except ValueError as e:
        return {"error": str(e)}
    out = result.as_dict()
    if not ui.SERVE_MARKET_DATA:
        for h in out["holdings"]:
            h.pop("pe", None)
        out["metrics"] = {
            k: v for k, v in out["metrics"].items()
            if not any(p in k for p in ui.PRICE_DERIVED_METRICS)}
    return out


@st.cache_data(ttl=300, show_spinner=False)
def assess(holdings: tuple[tuple[str, float], ...]) -> dict:
    """`holdings` arrives as a tuple of pairs so the cache key is hashable;
    assess_portfolio wants ticker -> weight."""
    from finmodel.portfolio_builder import assess_portfolio
    book: dict[str, float] = {}
    for ticker, weight in holdings:
        book[ticker] = book.get(ticker, 0.0) + weight
    result = assess_portfolio(book, ui.load_universe())
    if not ui.SERVE_MARKET_DATA and "holdings" in result:
        for h in result["holdings"]:
            h.pop("pe", None)
        result["metrics"] = {
            k: v for k, v in result["metrics"].items()
            if not any(p in k for p in ui.PRICE_DERIVED_METRICS)}
    return result


PCT_METRICS = ("margin", "roe", "roic", "growth", "yield", "weight")


def metric_rows(metrics: dict) -> pd.DataFrame:
    out = []
    for key, val in metrics.items():
        if isinstance(val, dict) or val is None:
            continue
        label = key.replace("_", " ").replace("weighted ", "").strip().capitalize()
        if any(p in key for p in PCT_METRICS):
            shown = ui.fmt_pct(val)
        elif isinstance(val, (int, float)):
            shown = ui.fmt_num(val)
        else:
            shown = str(val)
        out.append({"Measure": label, "Value": shown})
    return pd.DataFrame(out)


def show_result(out: dict) -> None:
    holdings = out.get("holdings") or []
    if not holdings:
        st.warning("No holdings survived the constraints. Loosen the risk "
                   "profile or widen the sector selection.")
        return

    metrics = out.get("metrics") or {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Holdings", len(holdings))
    eff = metrics.get("effective_holdings") or metrics.get("effective_n")
    c2.metric("Effective holdings",
              ui.fmt_num(eff, 1) if eff is not None else "—",
              help="1/HHI. A more honest read of diversification than a raw "
                   "count: ten positions where one is 60% behaves like three.")
    hhi = metrics.get("herfindahl") or metrics.get("hhi")
    c3.metric("Concentration (HHI)",
              ui.fmt_num(hhi, 3) if hhi is not None else "—")
    c4.metric("Universe screened",
              f"{out.get('eligible_size', 0):,} of {out.get('universe_size', 0):,}")

    for w in out.get("warnings") or []:
        st.warning(w, icon="⚠️")

    hdf = pd.DataFrame(holdings)
    order = [c for c in ("ticker", "name", "sector", "weight",
                         "strategy_score", "health_passed", "health_total",
                         "roic", "fcf_margin", "revenue_growth",
                         "net_debt_to_ebitda", "pe") if c in hdf.columns]

    st.subheader("Holdings")
    st.dataframe(
        hdf[order], hide_index=True, width="stretch",
        column_config={
            "ticker": st.column_config.TextColumn("Ticker"),
            "name": st.column_config.TextColumn("Company"),
            "sector": st.column_config.TextColumn("Sector"),
            "weight": st.column_config.ProgressColumn(
                "Weight", min_value=0.0,
                max_value=float(hdf["weight"].max()), format="percent"),
            "strategy_score": st.column_config.NumberColumn(
                "Score", format="%.2f"),
            "health_passed": st.column_config.NumberColumn("Passed", format="%d"),
            "health_total": st.column_config.NumberColumn("of", format="%d"),
            "roic": st.column_config.NumberColumn("ROIC", format="percent"),
            "fcf_margin": st.column_config.NumberColumn("FCF margin", format="percent"),
            "revenue_growth": st.column_config.NumberColumn("Growth", format="percent"),
            "net_debt_to_ebitda": st.column_config.NumberColumn("ND/EBITDA", format="%.1f×"),
            "pe": st.column_config.NumberColumn("P/E", format="%.1f"),
        },
        height=min(600, 38 * len(hdf) + 40),
    )

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Sector exposure")
        sw = out.get("sector_weights") or {}
        if sw:
            s = pd.Series(sw, name="weight").sort_values(ascending=False)
            st.bar_chart(s, height=max(220, 26 * len(s)), horizontal=True)
    with right:
        st.subheader("Weighted characteristics")
        mdf = metric_rows(metrics)
        if not mdf.empty:
            st.dataframe(mdf, hide_index=True, width="stretch")
        dropped = metrics.get("excluded_outliers")
        if dropped:
            st.caption(
                f"{dropped} reading(s) excluded as out of bounds. A company "
                "that has bought back most of its equity shows ROE in the "
                "hundreds of percent — a fact about its capital structure, "
                "not its profitability — and would destroy the average."
            )

    rejected = out.get("rejected") or {}
    if rejected:
        with st.expander("Why companies were excluded"):
            st.dataframe(
                pd.DataFrame(
                    [{"Reason": k.replace("_", " ").capitalize(), "Companies": v}
                     for k, v in sorted(rejected.items(), key=lambda x: -x[1])]),
                hide_index=True, width="stretch")

    st.download_button(
        "Download holdings CSV",
        hdf[order].to_csv(index=False).encode(),
        file_name="portfolio.csv", mime="text/csv")


# --------------------------------------------------------------------------
st.title("Portfolio")

if not ui.require_data():
    st.stop()

build_tab, assess_tab = st.tabs(["Build", "Assess what you hold"])

with build_tab:
    st.caption(
        "Screens the universe by strategy, then allocates under the risk "
        "profile's hard constraints."
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        strategy = st.selectbox(
            "Strategy", list(STRATEGIES),
            format_func=lambda k: STRATEGIES[k].name)
        st.caption(STRATEGIES[strategy].description)
    with c2:
        profile = st.selectbox(
            "Risk profile", list(RISK_PROFILES), index=1,
            format_func=lambda k: RISK_PROFILES[k].name)
        st.caption(RISK_PROFILES[profile].description)
    with c3:
        allocation = st.selectbox(
            "Allocation", list(ALLOCATIONS),
            format_func=lambda k: ALLOCATIONS[k][0])
        st.caption(ALLOCATIONS[allocation][1])

    risk = RISK_PROFILES[profile]
    c4, c5 = st.columns(2)
    with c4:
        max_holdings = st.slider(
            "Target holdings", risk.min_holdings, 60,
            min(risk.target_holdings, 60))
    with c5:
        # Integer percent, not a 0-1 fraction: a "%.0f%%" format applied to
        # 0.35 renders as "0%", which is what it used to show.
        sector_pct = st.slider(
            "Max weight in any one sector", 10, 100,
            int(round(risk.max_sector * 100)), 5, format="%d%%")
        sector_limit = sector_pct / 100

    all_sectors = ui.sector_names()
    chosen = st.multiselect(
        "Restrict to sectors (optional)", all_sectors,
        help="Leave empty to screen every sector.")

    with st.expander("Constraints this profile enforces"):
        st.markdown(
            f"- Max single position: **{risk.max_position:.0%}**\n"
            f"- Max single sector: **{risk.max_sector:.0%}**\n"
            f"- Minimum holdings: **{risk.min_holdings}**\n"
            f"- Minimum health score: **{risk.min_health_score:.0%}**\n"
            + (f"- Max net debt / EBITDA: **{risk.max_net_debt_to_ebitda:.1f}×**\n"
               if risk.max_net_debt_to_ebitda is not None else "")
            + (f"- Positive free cash flow required\n"
               if risk.require_positive_fcf else "")
        )
        st.caption(
            "The position and sector caps are satisfied jointly, by "
            "alternating until both hold. Applying them in sequence lets "
            "sector redistribution push a position back over its limit."
        )

    if st.button("Build portfolio", type="primary"):
        with st.spinner("Screening and allocating…"):
            out = build(strategy, profile, allocation, int(max_holdings),
                        float(sector_limit), tuple(chosen))
        if out.get("error"):
            st.error(out["error"])
        else:
            st.session_state["portfolio"] = out

    if st.session_state.get("portfolio"):
        st.divider()
        show_result(st.session_state["portfolio"])
        st.warning(
            "A screening result, not a recommendation. The allocation above "
            "reflects the criteria you selected, not advice about what you "
            "should own.", icon="⚖️")

with assess_tab:
    st.caption(
        "Paste what you hold and get concentration, sector dominance and "
        "balance-sheet exposure measured against the same database."
    )
    # Default to tickers that are actually loaded, so the example works on a
    # demo deployment instead of reporting four unmatched symbols.
    sample = ui.load_tickers()[:4] or ["AAPL", "MSFT", "NVDA", "MU"]
    default = "\n".join(f"{t}, {w}" for t, w in
                        zip(sample, (0.30, 0.25, 0.25, 0.20)))
    raw = st.text_area(
        "Holdings — one per line, `TICKER, weight`", value=default, height=180,
        help="Weights can be fractions or percentages; they are normalised. "
             "A repeated ticker has its weights added together.")

    if st.button("Assess", type="primary"):
        parsed: list[tuple[str, float]] = []
        bad: list[str] = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.replace("\t", ",").split(",")]
            if len(parts) < 2:
                bad.append(line)
                continue
            try:
                w = float(parts[1].rstrip("%"))
            except ValueError:
                bad.append(line)
                continue
            parsed.append((parts[0].upper(), w))
        if bad:
            st.warning("Could not read: " + "; ".join(bad))
        if not parsed:
            st.error("No readable holdings.")
        else:
            with st.spinner("Assessing…"):
                out = assess(tuple(parsed))
            if out.get("error"):
                st.error(out["error"])
                if out.get("unmatched"):
                    st.caption("Not in the database: "
                               + ", ".join(out["unmatched"]))
            else:
                if out.get("unmatched"):
                    st.warning(
                        "Not in the database, so excluded from every figure "
                        "below: " + ", ".join(out["unmatched"]))
                for f in out.get("flags") or []:
                    st.warning(f, icon="⚠️")
                show_result(out)

ui.footer()
