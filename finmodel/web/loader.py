"""Turn finmodel analysis output into database rows.

This is the bridge between the Python engine and the web app. It runs offline
(scheduled worker), never inside an HTTP request: SEC rate-limits to roughly
10 requests/second and a market-wide pass takes minutes plus a 1.5 GB download,
so fetching on request would be both slow and a good way to get blocked.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from ..forensic import altman_z, beneish_m, piotroski_f, sloan_accruals
from ..model import ThreeStatementModel
from ..screener import ScreenConfig, _passes, _row_from_model
from .store import Database


def _f(v):
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return float(v)


def model_to_rows(model: ThreeStatementModel, cik: int, ticker: str, name: str,
                  sector: str = "Unknown",
                  config: Optional[ScreenConfig] = None) -> tuple[dict, list[dict]]:
    """-> (company row, [history rows])"""
    cfg = config or ScreenConfig()
    fin = model.financials
    r = _row_from_model(model, None)
    passes = _passes(r, cfg)
    sc = model.scorecard
    m = sc["metrics"]

    def metric(key):
        return _f(m[key].value) if key in m else None

    pio = piotroski_f(fin)
    alt = altman_z(fin)
    ben = beneish_m(fin)
    acc = sloan_accruals(fin)

    company = {
        "cik": int(cik), "ticker": ticker, "name": name, "sector": sector,
        "fiscal_year": int(fin.latest_year()) if fin.years else None,
        "health_passed": int(r["health_passed"]),
        "health_total": int(r["health_total"]),
        "health_score": _f(r["health_score"]),
        "revenue": _f(r["revenue"]), "revenue_growth": _f(r["revenue_growth"]),
        "ebitda": _f(r["ebitda"]), "fcf": _f(r["fcf"]),
        "net_debt": _f(r["net_debt"]),
        "gross_margin": _f(r["gross_margin"]),
        "operating_margin": _f(r["operating_margin"]),
        "net_margin": _f(r["net_margin"]),
        "fcf_margin": _f(r["fcf_margin"]),
        "roe": _f(r["roe"]), "roic": _f(r["roic"]),
        "net_debt_to_ebitda": _f(r["net_debt_to_ebitda"]),
        "interest_coverage": metric("interest_coverage"),
        "current_ratio": metric("current_ratio"),
        "debt_to_equity": metric("debt_to_equity"),
        "cash_conversion": metric("cash_conversion"),
        "piotroski": int(pio.value) if pio.available and pio.value is not None else None,
        "altman": _f(alt.value) if alt.available else None,
        "beneish": _f(ben.value) if ben.available else None,
        "accruals": _f(acc.value) if acc.available else None,
        "shares_diluted": _f(
            fin.income_statement.at["shares_diluted", fin.latest_year()]
            if "shares_diluted" in fin.income_statement.index else None),
        "data_quality": model.quality["grade"],
        "passes_screen": bool(passes),
        "sector_percentile": None,          # filled by compute_sector_percentiles
    }

    is_, bs, cf = fin.income_statement, fin.balance_sheet, fin.cash_flow

    def g(df, item, year):
        if item in df.index and year in df.columns:
            return _f(df.at[item, year])
        return None

    history = []
    for y in fin.years:
        ocf = g(cf, "operating_cash_flow", y)
        capex = g(cf, "capex", y)
        debt_parts = [g(bs, "long_term_debt", y), g(bs, "short_term_debt", y)]
        debt = sum(p for p in debt_parts if p is not None) \
            if any(p is not None for p in debt_parts) else None
        history.append({
            "cik": int(cik), "fiscal_year": int(y),
            "revenue": g(is_, "revenue", y),
            "gross_profit": g(is_, "gross_profit", y),
            "operating_income": g(is_, "operating_income", y),
            "net_income": g(is_, "net_income", y),
            "ebitda": g(is_, "ebitda", y),
            "operating_cash_flow": ocf, "capex": capex,
            "fcf": (ocf - abs(capex)) if ocf is not None and capex is not None else None,
            "total_assets": g(bs, "total_assets", y),
            "total_equity": g(bs, "total_equity", y),
            "total_debt": debt,
        })
    return company, history


def load_universe(user_agent: str, db: Optional[Database] = None,
                  limit: Optional[int] = None, min_revenue: float = 50e6,
                  n_years: int = 5, bulk_dir: Optional[str] = None,
                  config: Optional[ScreenConfig] = None,
                  verbose: bool = True) -> dict:
    """Stream SEC's bulk dataset into the database. The scheduled worker job."""
    from ..data.bulk import BulkData, sector_for_sic

    db = db or Database()
    db.init_schema()
    bulk = BulkData(bulk_dir, user_agent)
    bulk.ensure()
    tmap = bulk.ticker_map()
    if verbose:
        print("  %d listed companies in SEC ticker map" % len(tmap))

    loaded = skipped = 0
    cbatch: list[dict] = []
    hbatch: list[dict] = []

    for cik, facts in bulk.iter_companies(limit=limit, only_ciks=set(tmap)):
        meta = tmap.get(cik, {})
        try:
            model = ThreeStatementModel.from_facts(
                facts, ticker=meta.get("ticker", str(cik)), n_years=n_years)
            fin = model.financials
            if not fin.years:
                skipped += 1
                continue
            y = fin.latest_year()
            rev = fin.income_statement.at["revenue", y] \
                if "revenue" in fin.income_statement.index else None
            if rev is None or pd.isna(rev) or rev < min_revenue:
                skipped += 1
                continue
            sic = facts.get("sic") or facts.get("entityInfo", {}).get("sic")
            c, h = model_to_rows(model, cik, meta.get("ticker", ""),
                                 meta.get("name", fin.entity_name),
                                 sector_for_sic(sic), config)
            cbatch.append(c)
            hbatch.extend(h)
            loaded += 1
        except Exception:
            skipped += 1          # one malformed filer must not kill the run
            continue

        if len(cbatch) >= 200:
            db.upsert_companies(cbatch)
            db.upsert_history(hbatch)
            cbatch, hbatch = [], []
            if verbose:
                print("\r  loaded %d | skipped %d" % (loaded, skipped),
                      end="", flush=True)

    db.upsert_companies(cbatch)
    db.upsert_history(hbatch)
    db.compute_sector_percentiles()
    db.record_run(loaded, skipped, "sec-bulk")
    if verbose:
        print("\r  loaded %d | skipped %d%s" % (loaded, skipped, " " * 20))
    return {"loaded": loaded, "skipped": skipped}


def load_tickers(tickers: list[str], user_agent: str,
                 db: Optional[Database] = None, n_years: int = 5,
                 config: Optional[ScreenConfig] = None,
                 sectors: Optional[dict] = None,
                 verbose: bool = True) -> dict:
    """Load a specific list of tickers through the per-company EDGAR API.

    Use for a curated set; use load_universe() for the whole market."""
    from ..data.edgar import EdgarClient

    db = db or Database()
    db.init_schema()
    client = EdgarClient(user_agent=user_agent)
    loaded = skipped = 0
    cbatch, hbatch = [], []
    for t in tickers:
        try:
            cik = client.ticker_to_cik(t)
            model = ThreeStatementModel.from_ticker(
                t, user_agent=user_agent, n_years=n_years)
            # Prefer the index's own GICS sector when supplied; it is a far
            # better grouping than SIC for peer comparison.
            sector = (sectors or {}).get(t.upper()) or "Unknown"
            c, h = model_to_rows(model, cik, t.upper(),
                                 model.financials.entity_name, sector, config)
            cbatch.append(c)
            hbatch.extend(h)
            loaded += 1
            if verbose:
                print("  loaded %s" % t)
        except Exception as e:
            skipped += 1
            if verbose:
                print("  skipped %s: %s" % (t, e))
    db.upsert_companies(cbatch)
    db.upsert_history(hbatch)
    db.compute_sector_percentiles()
    db.record_run(loaded, skipped, "sec-api")
    return {"loaded": loaded, "skipped": skipped}


# ---------------------------------------------------------------------------
# Valuation + sector benchmarks (run after companies are loaded)
# ---------------------------------------------------------------------------

PRICE_DERIVED = {"pe", "forward_pe", "pb", "ps", "ev_ebitda", "ev_sales",
                 "ev_ebit", "peg", "fcf_yield", "earnings_yield",
                 "dividend_yield"}


def compute_benchmarks(db: Optional[Database] = None,
                       verbose: bool = True) -> int:
    """Rebuild per-sector reference statistics from whatever is in the DB.

    Run this after every load: benchmarks are only as current as the companies
    behind them, and a stale median silently distorts every comparison."""
    from ..sectors import BENCHMARK_METRICS, build_benchmarks

    db = db or Database()
    rows = db.universe_rows(limit=100000)
    if not rows:
        if verbose:
            print("  no companies loaded; nothing to benchmark")
        return 0

    benchmarks = build_benchmarks(rows, BENCHMARK_METRICS)
    out = []
    for bench in benchmarks.values():
        for r in bench.to_rows():
            r["price_derived"] = 1 if r["metric"] in PRICE_DERIVED else 0
            out.append(r)
    n = db.upsert_benchmarks(out)
    if verbose:
        print("  benchmarked %d sectors (%d metric rows)" % (len(benchmarks), n))
    return n


def attach_valuation(db: Optional[Database] = None,
                     prices: Optional[dict] = None,
                     user_agent: Optional[str] = None,
                     batch_size: int = 200,
                     source: str = "yfinance",
                     history_period: str = "5y",
                     with_history: bool = True,
                     verbose: bool = True) -> int:
    """Fetch prices, derive market cap and every relative-valuation multiple.

    Market cap is price x diluted shares from the filings rather than a
    vendor's own field: it keeps the numerator and denominator of every
    multiple on the same basis, and avoids yfinance's slow per-ticker .info
    call, which would take hours across a full universe."""
    from ..valuation import compute_multiples
    from ..model import ThreeStatementModel

    db = db or Database()
    conn = db.connect()
    cur = conn.cursor()
    cur.execute("SELECT cik, ticker, revenue, ebitda, fcf, net_debt, net_margin,"
                " fiscal_year, shares_diluted FROM companies"
                " WHERE ticker IS NOT NULL AND ticker <> ''")
    companies = db._rows_to_dicts(cur)
    if not companies:
        return 0

    stats: dict = {}
    if prices is None:
        tickers = [c["ticker"] for c in companies]
        try:
            if with_history:
                # One pass over 5 years of monthly history gives the latest
                # price AND volatility, beta, drawdown and momentum — all from
                # the same request, instead of a second round trip per name.
                from ..data.market import batch_price_history, price_statistics
                if verbose:
                    print("  fetching %s price history for %d tickers ..."
                          % (history_period, len(tickers)))
                panel = batch_price_history(
                    tickers + ["^GSPC"], period=history_period,
                    interval="1mo", batch_size=min(batch_size, 150),
                    verbose=verbose)
                stats = price_statistics(panel)
                prices = {t: v["last_price"] for t, v in stats.items()}
            else:
                from ..data.market import batch_last_prices
                if verbose:
                    print("  fetching prices for %d tickers ..." % len(tickers))
                prices = batch_last_prices(tickers, batch_size=batch_size,
                                           verbose=verbose)
        except ImportError:
            print("  yfinance not installed - skipping valuation "
                  "(pip install yfinance)")
            return 0

    rows = []
    for c in companies:
        px = prices.get(c["ticker"])
        if px is None:
            continue
        shares = c.get("shares_diluted") or _shares_for(db, c["cik"])
        if not shares or shares <= 0:
            continue
        market_cap = float(px) * float(shares)
        model = _light_model(db, c)
        m = compute_multiples(model, market_cap) if model else None
        row = {"cik": c["cik"], "price": float(px), "shares_diluted": shares,
               "market_cap": market_cap, "source": source}
        st = stats.get(c["ticker"])
        if st:
            row.update({"beta": st.get("beta"),
                        "volatility": st.get("volatility"),
                        "return_1y": st.get("return_1y"),
                        "return_total": st.get("return_total"),
                        "max_drawdown": st.get("max_drawdown"),
                        "price_observations": st.get("observations")})
        if m:
            row.update({k: v for k, v in m.as_dict().items()
                        if k not in ("market_cap",)})
        rows.append(row)

    n = db.upsert_market(rows)
    if verbose:
        print("  valuation attached for %d companies" % n)
    return n


def _shares_for(db: Database, cik: int) -> Optional[float]:
    """Diluted share count from the company's own filings.

    Reads `companies`, not `market_data`: the share count is a filed fact, and
    sourcing it from the price table would be circular on a first run."""
    conn = db.connect()
    cur = conn.cursor()
    cur.execute("SELECT shares_diluted FROM companies WHERE cik = %s" % db.ph,
                (cik,))
    r = cur.fetchone()
    return float(r[0]) if r and r[0] else None


def _light_model(db: Database, company: dict):
    """A minimal stand-in exposing the attributes compute_multiples reads.

    Rebuilding the full ThreeStatementModel for every company just to compute
    multiples would mean re-parsing the filings; the stored history already
    carries every line item the multiples need."""
    import pandas as pd
    from types import SimpleNamespace

    conn = db.connect()
    cur = conn.cursor()
    cur.execute("SELECT * FROM financial_history WHERE cik = %s "
                "ORDER BY fiscal_year" % db.ph, (company["cik"],))
    hist = db._rows_to_dicts(cur)
    if not hist:
        return None
    years = [h["fiscal_year"] for h in hist]

    def frame(mapping):
        df = pd.DataFrame(index=list(mapping), columns=years, dtype="float64")
        for item, key in mapping.items():
            for h in hist:
                df.at[item, h["fiscal_year"]] = h.get(key)
        return df

    is_ = frame({"revenue": "revenue", "gross_profit": "gross_profit",
                 "operating_income": "operating_income",
                 "net_income": "net_income", "ebitda": "ebitda"})
    bs = frame({"total_equity": "total_equity", "total_assets": "total_assets",
                "long_term_debt": "total_debt"})
    bs.loc["short_term_debt"] = float("nan")
    bs.loc["cash"] = float("nan")
    bs.loc["short_term_investments"] = float("nan")
    cf = frame({"operating_cash_flow": "operating_cash_flow", "capex": "capex"})
    cf.loc["dividends_paid"] = float("nan")

    fin = SimpleNamespace(years=years, income_statement=is_, balance_sheet=bs,
                          cash_flow=cf, latest_year=lambda: years[-1])
    return SimpleNamespace(financials=fin)
