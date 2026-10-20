"""Market-wide screening — run the model across every filer, not a watchlist.

Design constraints that matter at this scale:

  * memory      companies are streamed one at a time out of the bulk zip
  * failures    thousands of filers have odd or missing tags; one bad company
                must never kill a multi-hour run, so every parse is isolated
  * resumable   results are checkpointed to SQLite as they complete, so a crash
                or a Ctrl-C costs you nothing
  * sectors     a 30% gross-margin rule is meaningless for a grocery
                distributor. Absolute thresholds are kept, but percentile ranks
                are computed WITHIN sector so names compete against peers.
"""
from __future__ import annotations

import sqlite3
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from .data.bulk import BulkData, sector_for_sic
from .forensic import piotroski_f, altman_z, beneish_m, sloan_accruals
from .model import ThreeStatementModel
from .screener import ScreenConfig, _row_from_model, _passes

SCHEMA = """
CREATE TABLE IF NOT EXISTS results (
  cik INTEGER PRIMARY KEY, ticker TEXT, name TEXT, sector TEXT,
  fiscal_year INTEGER, health_passed INTEGER, health_total INTEGER,
  health_score REAL, revenue REAL, revenue_growth REAL, ebitda REAL,
  gross_margin REAL, operating_margin REAL, net_margin REAL,
  fcf REAL, fcf_margin REAL, net_debt REAL, net_debt_to_ebitda REAL,
  roe REAL, roic REAL, piotroski REAL, altman REAL, beneish REAL,
  accruals REAL, shares_diluted REAL, market_cap REAL, pe REAL,
  ev_ebitda REAL, fcf_yield REAL,
  quality TEXT, passes_screen INTEGER
);
CREATE TABLE IF NOT EXISTS failures (cik INTEGER PRIMARY KEY, reason TEXT);
CREATE INDEX IF NOT EXISTS idx_sector ON results(sector);
CREATE INDEX IF NOT EXISTS idx_score ON results(health_score);
"""

_COLS = [c.strip().split()[0] for c in
         SCHEMA.split("CREATE TABLE IF NOT EXISTS results (")[1]
         .split(");")[0].split(",")]


@dataclass
class UniverseScreen:
    db_path: Path

    # ------------------------------------------------------------------ run
    @classmethod
    def run(cls, user_agent: Optional[str] = None,
            db_path: str = "universe.db", bulk_dir: Optional[str] = None,
            config: Optional[ScreenConfig] = None,
            limit: Optional[int] = None, min_revenue: float = 50e6,
            n_years: int = 4, resume: bool = True,
            fetch_prices: bool = True, verbose: bool = True) -> "UniverseScreen":
        cfg = config or ScreenConfig()
        bulk = BulkData(bulk_dir, user_agent)
        bulk.ensure()

        if verbose:
            print("\n  loading ticker map ...")
        tmap = bulk.ticker_map()
        if verbose:
            print("  %d listed companies in SEC ticker map" % len(tmap))

        db = Path(db_path)
        conn = sqlite3.connect(db)
        conn.executescript(SCHEMA)
        conn.commit()

        done = set()
        if resume:
            done = {r[0] for r in conn.execute("SELECT cik FROM results")}
            done |= {r[0] for r in conn.execute("SELECT cik FROM failures")}
            if done and verbose:
                print("  resuming — %d companies already processed" % len(done))

        # only companies that are actually listed (have a ticker)
        only = set(tmap.keys())
        processed, kept, failed, batch = 0, 0, 0, []

        if verbose:
            print("  streaming bulk filings ...\n")
        for cik, facts in bulk.iter_companies(limit=limit, only_ciks=only):
            if cik in done:
                continue
            processed += 1
            meta = tmap.get(cik, {})
            ticker = meta.get("ticker", "")
            try:
                row = cls._process(cik, facts, meta, n_years, cfg, min_revenue)
                if row is None:
                    conn.execute("INSERT OR REPLACE INTO failures VALUES (?,?)",
                                 (cik, "below size floor or no usable data"))
                    failed += 1
                else:
                    batch.append(row)
                    kept += 1
            except Exception as e:           # never let one filer kill the run
                conn.execute("INSERT OR REPLACE INTO failures VALUES (?,?)",
                             (cik, "%s: %s" % (type(e).__name__, str(e)[:180])))
                failed += 1

            if len(batch) >= 200:
                cls._flush(conn, batch)
                batch = []
            if verbose and processed % 250 == 0:
                print("\r  processed %d | kept %d | skipped %d"
                      % (processed, kept, failed), end="", flush=True)

        cls._flush(conn, batch)
        conn.commit()
        if verbose:
            print("\r  processed %d | kept %d | skipped %d%s"
                  % (processed, kept, failed, " " * 12))

        screen = cls(db)
        if fetch_prices:
            screen.attach_market_data(verbose=verbose)
        screen.rank_within_sectors()
        conn.close()
        return screen

    # -------------------------------------------------------------- helpers
    @staticmethod
    def _process(cik, facts, meta, n_years, cfg, min_revenue):
        model = ThreeStatementModel.from_facts(
            facts, ticker=meta.get("ticker", str(cik)), n_years=n_years)
        fin = model.financials
        if not fin.years:
            return None
        y = fin.latest_year()
        rev = fin.income_statement.at["revenue", y] \
            if "revenue" in fin.income_statement.index else None
        if rev is None or pd.isna(rev) or rev < min_revenue:
            return None

        r = _row_from_model(model, None)
        r["passes_screen"] = _passes(r, cfg)
        pio = piotroski_f(fin)
        alt = altman_z(fin)
        ben = beneish_m(fin)
        acc = sloan_accruals(fin)
        sic = facts.get("sic") or facts.get("entityInfo", {}).get("sic")

        def f(v):
            return None if v is None or (isinstance(v, float) and pd.isna(v)) else float(v)

        return (cik, meta.get("ticker", ""), meta.get("name", fin.entity_name),
                sector_for_sic(sic), int(y),
                int(r["health_passed"]), int(r["health_total"]), f(r["health_score"]),
                f(r["revenue"]), f(r["revenue_growth"]), f(r["ebitda"]),
                f(r["gross_margin"]), f(r["operating_margin"]), f(r["net_margin"]),
                f(r["fcf"]), f(r["fcf_margin"]), f(r["net_debt"]),
                f(r["net_debt_to_ebitda"]), f(r["roe"]), f(r["roic"]),
                f(pio.value), f(alt.value), f(ben.value), f(acc.value),
                f(fin.income_statement.at["shares_diluted", y]
                  if "shares_diluted" in fin.income_statement.index else None),
                None, None, None, None,
                model.quality["grade"], int(bool(r["passes_screen"])))

    @staticmethod
    def _flush(conn, batch):
        if batch:
            conn.executemany(
                "INSERT OR REPLACE INTO results VALUES (%s)"
                % ",".join("?" * len(batch[0])), batch)
            conn.commit()

    # --------------------------------------------------------- market data
    def attach_market_data(self, batch_size: int = 200, verbose: bool = True):
        """Batched prices -> market cap = price x diluted shares -> multiples.

        Never calls yfinance's per-ticker .info (one HTTP round trip each,
        hours at market scale). Share counts come from the filings."""
        from .data.market import batch_last_prices
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            "SELECT cik, ticker, shares_diluted, revenue, ebitda, fcf, net_debt, "
            "net_margin FROM results WHERE ticker IS NOT NULL AND ticker <> ''"
        ).fetchall()
        tickers = [r[1] for r in rows]
        if verbose:
            print("\n  fetching prices for %d tickers (batched) ..." % len(tickers))
        try:
            prices = batch_last_prices(tickers, batch_size=batch_size, verbose=verbose)
        except ImportError:
            print("  yfinance not installed - skipping market data "
                  "(pip install yfinance)")
            conn.close()
            return

        updates = []
        for cik, ticker, shares, revenue, ebitda, fcf, net_debt, net_margin in rows:
            px = prices.get(ticker)
            if px is None or not shares or shares <= 0:
                continue
            cap = px * float(shares)
            ni = (revenue * net_margin) if revenue and net_margin else None
            pe = (cap / ni) if ni and ni > 0 else None
            ev = cap + (net_debt or 0.0)
            ev_ebitda = (ev / ebitda) if ebitda and ebitda > 0 else None
            fcf_yield = (fcf / cap) if fcf is not None and cap > 0 else None
            updates.append((cap, pe, ev_ebitda, fcf_yield, cik))
        conn.executemany(
            "UPDATE results SET market_cap=?, pe=?, ev_ebitda=?, fcf_yield=? "
            "WHERE cik=?", updates)
        conn.commit()
        conn.close()
        if verbose:
            print("  market data attached for %d companies" % len(updates))

    # ------------------------------------------------------------- ranking
    def rank_within_sectors(self):
        """Percentile ranks computed inside each sector, not across the market."""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql("SELECT * FROM results", conn)
        if df.empty:
            conn.close()
            return
        conn.execute("DROP TABLE IF EXISTS ranks")
        cols = ["health_score", "revenue_growth", "operating_margin",
                "fcf_margin", "roic", "roe"]
        out = df[["cik", "ticker", "sector"]].copy()
        for c in cols:
            out[c + "_pct"] = (df.groupby("sector")[c]
                               .rank(pct=True, na_option="keep"))
        # Mean of metric ranks, then rank that within sector — see the note in
        # web/store.compute_sector_percentiles. A mean of percentiles is not a
        # percentile; ranking it makes the reported figure mean what it says.
        out["score"] = out[[c + "_pct" for c in cols]].mean(axis=1, skipna=True)
        out["composite_pct"] = out.groupby("sector")["score"].rank(
            pct=True, na_option="keep")
        out.to_sql("ranks", conn, index=False)
        conn.commit()
        conn.close()

    # --------------------------------------------------------------- views
    def table(self, where: str = "", limit: int = 50) -> pd.DataFrame:
        conn = sqlite3.connect(self.db_path)
        q = ("SELECT r.ticker, r.name, r.sector, r.health_passed, r.health_total, "
             "r.revenue, r.revenue_growth, r.operating_margin, r.fcf_margin, "
             "r.roic, r.net_debt_to_ebitda, r.piotroski, r.altman, "
             "r.passes_screen, k.composite_pct "
             "FROM results r LEFT JOIN ranks k ON r.cik = k.cik ")
        if where:
            q += "WHERE " + where + " "
        q += "ORDER BY r.health_score DESC, r.revenue DESC LIMIT %d" % limit
        df = pd.read_sql(q, conn)
        conn.close()
        return df

    def passing(self, limit: int = 100) -> pd.DataFrame:
        return self.table("r.passes_screen = 1", limit)

    def by_sector(self, sector: str, limit: int = 50) -> pd.DataFrame:
        return self.table("r.sector = '%s'" % sector.replace("'", "''"), limit)

    def sector_summary(self) -> pd.DataFrame:
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql(
            "SELECT sector, COUNT(*) AS companies, "
            "SUM(passes_screen) AS passing, AVG(health_score) AS avg_health, "
            "AVG(operating_margin) AS avg_op_margin, AVG(roic) AS avg_roic "
            "FROM results GROUP BY sector ORDER BY companies DESC", conn)
        conn.close()
        return df

    def stats(self) -> dict:
        conn = sqlite3.connect(self.db_path)
        n = conn.execute("SELECT COUNT(*) FROM results").fetchone()[0]
        p = conn.execute("SELECT COUNT(*) FROM results WHERE passes_screen=1").fetchone()[0]
        f = conn.execute("SELECT COUNT(*) FROM failures").fetchone()[0]
        s = conn.execute("SELECT COUNT(DISTINCT sector) FROM results").fetchone()[0]
        conn.close()
        return {"companies": n, "passing": p, "skipped": f, "sectors": s}
