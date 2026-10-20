"""Shared data access, caching and formatting for the Streamlit app.

Streamlit re-executes the entire script top to bottom on every widget
interaction, so an uncached query runs again on every click. Everything here is
therefore cached in one of two ways:

  * @st.cache_resource for the database connection — one per process, reused
    across reruns and across sessions. Caching a connection as *data* would
    pickle it, which does not work and would open a new connection per rerun.
  * @st.cache_data for query results — keyed on the arguments, with a TTL so a
    worker run becomes visible without restarting the app.

The same two public-deployment constraints as the JSON API apply here, for the
same reasons, and are implemented the same way:

  * SERVE_MARKET_DATA (env, default false). SEC filings are US government work
    and public domain, so fundamentals and every ratio derived from them can be
    republished freely. Scraped price data generally cannot be redistributed
    under the provider's terms, so price-derived fields are withheld unless you
    have moved to a feed whose licence permits it.
  * Nothing here issues a buy/sell/hold verdict, and the disclaimer is rendered
    on every page rather than left to one of them to remember.
"""
from __future__ import annotations

import os
from typing import Any, Optional

import pandas as pd
import streamlit as st

from .store import Database

DISCLAIMER = (
    "Computed from public SEC filings for informational purposes only. "
    "Not investment advice. Figures are as last filed and may be out of date; "
    "verify against primary sources before acting."
)

NO_RECOMMENDATION = (
    "This reports what the filed figures show. It contains no buy, sell or "
    "hold recommendation, and none should be inferred."
)

SERVE_MARKET_DATA = os.environ.get("SERVE_MARKET_DATA", "false").lower() == "true"

# Fields computed from a scraped price. Withheld unless SERVE_MARKET_DATA.
PRICE_DERIVED_METRICS = {
    "pe", "forward_pe", "pb", "ps", "ev_ebitda", "ev_sales", "ev_ebit",
    "peg", "fcf_yield", "earnings_yield", "dividend_yield",
}
MARKET_FIELDS = {
    "price", "market_cap", "enterprise_value", "beta", "volatility",
    "return_1y", "return_total", "max_drawdown", "price_observations",
} | PRICE_DERIVED_METRICS

CACHE_TTL = int(os.environ.get("UI_CACHE_TTL", "300"))


# --------------------------------------------------------------------------
# connection
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_db() -> Database:
    """One connection per process, shared across reruns and sessions.

    The SQLite connection is opened with check_same_thread=False in store.py,
    which it has to be: Streamlit serves each session from a different thread.
    """
    db = Database()
    db.init_schema()          # idempotent; makes a fresh deployment work
    return db


def backend_name() -> str:
    return "PostgreSQL" if get_db().is_postgres else "SQLite"


# --------------------------------------------------------------------------
# reads
# --------------------------------------------------------------------------
@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def load_stats() -> dict:
    return get_db().stats()


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def load_sectors() -> list[dict]:
    return get_db().sectors()


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def sector_names() -> list[str]:
    return [r["sector"] for r in load_sectors() if r.get("sector")]


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def load_screen(filters: Optional[tuple] = None, sector: Optional[str] = None,
                search: str = "", passing_only: bool = False,
                sort_by: str = "sector_percentile", desc: bool = True,
                limit: int = 500, offset: int = 0) -> dict:
    """Filters arrive as a tuple of (key, value) pairs so they are hashable and
    the cache key is stable; a dict would not be."""
    return get_db().screen(
        filters=dict(filters or ()), sector=sector, search=search or None,
        passing_only=passing_only, sort_by=sort_by, desc=desc,
        limit=limit, offset=offset)


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def load_company(ticker: str) -> Optional[dict]:
    return get_db().company(ticker)


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def load_universe(limit: int = 100000) -> list[dict]:
    return get_db().universe_rows(limit=limit)


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def load_tickers() -> list[str]:
    rows = get_db().screen(sort_by="ticker", desc=False, limit=10000)["rows"]
    return [r["ticker"] for r in rows if r.get("ticker")]


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def load_benchmarks(sector: str) -> list[dict]:
    rows = get_db().benchmarks(sector)
    if SERVE_MARKET_DATA:
        return rows
    return [r for r in rows if not r.get("price_derived")]


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def load_research(ticker: str) -> Optional[dict]:
    """Research note plus peer comparisons, assembled exactly as the JSON API
    assembles it so the two surfaces cannot drift apart."""
    from ..research import generate_research
    from ..sectors import (BENCHMARK_METRICS, MetricStats, SectorBenchmark,
                           compare_to_sector)

    db = get_db()
    rows = db.universe_rows(limit=100000)
    match = [r for r in rows
             if str(r.get("ticker", "")).upper() == ticker.upper()]
    if not match:
        return None
    company = match[0]
    sector = company.get("sector") or "Unknown"

    stored = db.benchmarks(sector)
    bench = SectorBenchmark(
        sector=sector,
        n_companies=stored[0]["n_companies"] if stored else 0,
        metrics={r["metric"]: MetricStats(
            metric=r["metric"], count=r["count"] or 0, median=r["median"],
            mean=r["mean"], p25=r["p25"], p75=r["p75"], std=r["std"])
            for r in stored},
    )
    peers = db.peer_values(sector, list(BENCHMARK_METRICS))
    comparisons = compare_to_sector(company, bench, peer_values=peers)
    if not SERVE_MARKET_DATA:
        comparisons = {k: v for k, v in comparisons.items()
                       if k not in PRICE_DERIVED_METRICS}

    note = generate_research(company, comparisons)
    return {
        "note": note.as_dict(),
        "text": note.as_text(),
        "comparisons": [
            {"metric": c.metric, "label": c.label, "value": c.value,
             "sector_median": c.sector_median, "percentile": c.percentile,
             "direction": c.direction, "peers": c.peers,
             "is_percent": c.is_percent, "verdict": c.verdict(),
             "premium": c.premium}
            for c in comparisons.values() if c.available
        ],
    }


def clear_caches() -> None:
    load_stats.clear()
    load_sectors.clear()
    sector_names.clear()
    load_screen.clear()
    load_company.clear()
    load_universe.clear()
    load_tickers.clear()
    load_benchmarks.clear()
    load_research.clear()


# --------------------------------------------------------------------------
# licensing gate
# --------------------------------------------------------------------------
def strip_market(df: pd.DataFrame) -> pd.DataFrame:
    if SERVE_MARKET_DATA:
        return df
    drop = [c for c in df.columns if c in MARKET_FIELDS]
    return df.drop(columns=drop) if drop else df


# --------------------------------------------------------------------------
# formatting
# --------------------------------------------------------------------------
def fmt_money(v: Any, unit: str = "$") -> str:
    if v is None or (isinstance(v, float) and v != v):
        return "—"
    v = float(v)
    a = abs(v)
    for cut, suf in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if a >= cut:
            return f"{unit}{v / cut:,.2f}{suf}"
    return f"{unit}{v:,.0f}"


def fmt_pct(v: Any, digits: int = 1) -> str:
    if v is None or (isinstance(v, float) and v != v):
        return "—"
    return f"{float(v) * 100:.{digits}f}%"


def fmt_x(v: Any, digits: int = 1) -> str:
    if v is None or (isinstance(v, float) and v != v):
        return "—"
    return f"{float(v):.{digits}f}×"


def fmt_num(v: Any, digits: int = 2) -> str:
    if v is None or (isinstance(v, float) and v != v):
        return "—"
    return f"{float(v):,.{digits}f}"


def health_label(passed: Any, total: Any) -> str:
    if passed is None or total in (None, 0):
        return "—"
    return f"{int(passed)}/{int(total)}"


# --------------------------------------------------------------------------
# page furniture
# --------------------------------------------------------------------------
def footer() -> None:
    st.divider()
    st.caption(DISCLAIMER)
    if not SERVE_MARKET_DATA:
        st.caption(
            "Price-derived fields (market cap, P/E, EV/EBITDA, yields, beta) "
            "are hidden because `SERVE_MARKET_DATA` is false. SEC filing data "
            "is public domain; scraped price data generally is not "
            "redistributable. Switch to a licensed feed before enabling it."
        )


def require_data() -> bool:
    """A fresh deployment has an empty database. Say so plainly and offer the
    offline demo set, rather than rendering empty tables that look broken."""
    stats = load_stats()
    if (stats.get("companies") or 0) > 0:
        return True

    st.warning("No data loaded yet.", icon="📭")
    st.markdown(
        "The app reads from a database that a separate worker fills. Nothing "
        "is fetched on page load — SEC rate-limits to roughly 10 requests a "
        "second, so a request-time pull would be slow and would get the "
        "deployment blocked."
    )
    left, right = st.columns(2)
    with left:
        st.markdown(
            "**Demo data** — 141 companies across 10 sectors, five fiscal "
            "years, generated offline. No network, no SEC key. Good for "
            "seeing how the app works."
        )
        if st.button("Load demo data", type="primary", width="stretch"):
            with st.spinner("Seeding…"):
                from .seed import seed, seed_prices
                from .loader import compute_benchmarks
                db = get_db()
                seed(db)
                seed_prices(db)
                compute_benchmarks(db)
                db.compute_sector_percentiles()
            clear_caches()
            st.rerun()
    with right:
        st.markdown(
            "**Real filings** — run the worker from a terminal, then reload:\n"
            "```bash\n"
            'export EDGAR_USER_AGENT="Your Name you@email.com"\n'
            "python worker.py AAPL MSFT NVDA MU   # a few names\n"
            "python worker.py --sp500             # S&P 500\n"
            "```\n"
            "SEC requires a real name and email on every request. On a hosted "
            "deployment the worker needs a persistent database — see "
            "`STREAMLIT.md`."
        )
    footer()
    return False
