"""Database layer for the finmodel web app.

Works against either backend, chosen by DATABASE_URL:

    postgresql://user:pass@host/db     production
    sqlite:///finmodel_web.db          local development (the default)

The same SQL is used for both wherever possible; only the DDL and the
placeholder style differ. Keeping SQLite as a first-class backend means you can
develop and test the whole stack with no services running, then point
DATABASE_URL at Postgres for deployment without touching application code.

The worker writes; the API only reads.
"""
from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd

DEFAULT_URL = "sqlite:///finmodel_web.db"

SQLITE_DDL = """
CREATE TABLE IF NOT EXISTS companies (
    cik INTEGER PRIMARY KEY, ticker TEXT NOT NULL, name TEXT NOT NULL,
    sector TEXT NOT NULL DEFAULT 'Unknown', fiscal_year INTEGER,
    health_passed INTEGER, health_total INTEGER, health_score REAL,
    revenue REAL, revenue_growth REAL, ebitda REAL, fcf REAL, net_debt REAL,
    gross_margin REAL, operating_margin REAL, net_margin REAL, fcf_margin REAL,
    roe REAL, roic REAL, net_debt_to_ebitda REAL, interest_coverage REAL,
    current_ratio REAL, debt_to_equity REAL, cash_conversion REAL,
    piotroski INTEGER, altman REAL, beneish REAL, accruals REAL,
    shares_diluted REAL,
    data_quality TEXT, passes_screen INTEGER NOT NULL DEFAULT 0,
    sector_percentile REAL, updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE UNIQUE INDEX IF NOT EXISTS companies_ticker_key ON companies (ticker);
CREATE INDEX IF NOT EXISTS companies_sector_idx  ON companies (sector);
CREATE INDEX IF NOT EXISTS companies_health_idx  ON companies (health_score DESC);
CREATE INDEX IF NOT EXISTS companies_revenue_idx ON companies (revenue DESC);
CREATE INDEX IF NOT EXISTS companies_passes_idx  ON companies (passes_screen);

CREATE TABLE IF NOT EXISTS financial_history (
    cik INTEGER NOT NULL, fiscal_year INTEGER NOT NULL,
    revenue REAL, gross_profit REAL, operating_income REAL, net_income REAL,
    ebitda REAL, operating_cash_flow REAL, capex REAL, fcf REAL,
    total_assets REAL, total_equity REAL, total_debt REAL,
    PRIMARY KEY (cik, fiscal_year)
);
CREATE INDEX IF NOT EXISTS history_cik_idx ON financial_history (cik);

CREATE TABLE IF NOT EXISTS market_data (
    cik INTEGER PRIMARY KEY, price REAL, shares_diluted REAL, market_cap REAL,
    enterprise_value REAL, pe REAL, forward_pe REAL, pb REAL, ps REAL,
    ev_ebitda REAL, ev_sales REAL, ev_ebit REAL, peg REAL,
    fcf_yield REAL, earnings_yield REAL, dividend_yield REAL,
    beta REAL, volatility REAL, return_1y REAL, return_total REAL,
    max_drawdown REAL, price_observations INTEGER, source TEXT,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS sector_benchmarks (
    sector TEXT NOT NULL, metric TEXT NOT NULL, n_companies INTEGER,
    count INTEGER, median REAL, mean REAL, p25 REAL, p75 REAL, std REAL,
    price_derived INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (sector, metric)
);
CREATE INDEX IF NOT EXISTS bench_sector_idx ON sector_benchmarks (sector);

CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT DEFAULT CURRENT_TIMESTAMP, finished_at TEXT,
    companies_loaded INTEGER, companies_skipped INTEGER, source TEXT, notes TEXT
);
"""

COMPANY_COLS = [
    "cik", "ticker", "name", "sector", "fiscal_year",
    "health_passed", "health_total", "health_score",
    "revenue", "revenue_growth", "ebitda", "fcf", "net_debt",
    "gross_margin", "operating_margin", "net_margin", "fcf_margin",
    "roe", "roic", "net_debt_to_ebitda", "interest_coverage",
    "current_ratio", "debt_to_equity", "cash_conversion",
    "piotroski", "altman", "beneish", "accruals", "shares_diluted",
    "data_quality", "passes_screen", "sector_percentile",
]

MARKET_COLS = [
    "cik", "price", "shares_diluted", "market_cap", "enterprise_value",
    "pe", "forward_pe", "pb", "ps", "ev_ebitda", "ev_sales", "ev_ebit",
    "peg", "fcf_yield", "earnings_yield", "dividend_yield", "beta",
    "volatility", "return_1y", "return_total", "max_drawdown",
    "price_observations", "source",
]

BENCHMARK_COLS = [
    "sector", "metric", "n_companies", "count", "median", "mean",
    "p25", "p75", "std", "price_derived",
]

HISTORY_COLS = [
    "cik", "fiscal_year", "revenue", "gross_profit", "operating_income",
    "net_income", "ebitda", "operating_cash_flow", "capex", "fcf",
    "total_assets", "total_equity", "total_debt",
]

# Columns a client is allowed to sort or filter on. Whitelisted because these
# names are interpolated into SQL; never let a request supply a raw column.
SORTABLE = {
    "health_score", "revenue", "revenue_growth", "ebitda", "fcf",
    "gross_margin", "operating_margin", "net_margin", "fcf_margin",
    "roe", "roic", "net_debt_to_ebitda", "interest_coverage",
    "current_ratio", "debt_to_equity", "cash_conversion",
    "piotroski", "altman", "beneish", "accruals", "sector_percentile",
    "ticker", "name", "sector",
}

NUMERIC_FILTERS = {
    "health_score", "revenue", "revenue_growth", "operating_margin",
    "gross_margin", "net_margin", "fcf_margin", "roe", "roic",
    "net_debt_to_ebitda", "interest_coverage", "current_ratio",
    "debt_to_equity", "piotroski", "altman",
}


def _clean(v):
    """NaN/NaT -> None so the DB stores a real NULL."""
    if v is None:
        return None
    if isinstance(v, float) and (v != v):
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(v, "item"):          # numpy scalar
        v = v.item()
    return v


@dataclass
class Database:
    url: str = ""
    _conn: Any = None
    is_postgres: bool = False

    def __post_init__(self):
        self.url = self.url or os.environ.get("DATABASE_URL", DEFAULT_URL)
        self.is_postgres = self.url.startswith(("postgres://", "postgresql://"))

    # ---- connection ------------------------------------------------------
    def connect(self):
        if self._conn is not None:
            return self._conn
        if self.is_postgres:
            import psycopg2
            import psycopg2.extras
            self._conn = psycopg2.connect(self.url)
        else:
            path = self.url.replace("sqlite:///", "") or "finmodel_web.db"
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def close(self):
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    @property
    def ph(self) -> str:
        """Parameter placeholder for this backend."""
        return "%s" if self.is_postgres else "?"

    def init_schema(self):
        conn = self.connect()
        cur = conn.cursor()
        if self.is_postgres:
            ddl = (Path(__file__).parent / "schema.sql").read_text()
            # pg_trgm may be unavailable on a managed instance without
            # superuser; the index is an optimisation, not a requirement.
            try:
                cur.execute(ddl)
            except Exception:
                conn.rollback()
                stripped = "\n".join(
                    ln for ln in ddl.splitlines()
                    if "pg_trgm" not in ln and "gin_trgm_ops" not in ln)
                cur.execute(stripped)
        else:
            cur.executescript(SQLITE_DDL)
        conn.commit()

    # ---- writes ----------------------------------------------------------
    def upsert_companies(self, rows: list[dict]):
        if not rows:
            return 0
        conn = self.connect()
        cols = COMPANY_COLS
        marks = ",".join([self.ph] * len(cols))
        if self.is_postgres:
            updates = ",".join("%s=EXCLUDED.%s" % (c, c) for c in cols if c != "cik")
            sql = ("INSERT INTO companies (%s) VALUES (%s) "
                   "ON CONFLICT (cik) DO UPDATE SET %s, updated_at=now()"
                   % (",".join(cols), marks, updates))
        else:
            sql = ("INSERT OR REPLACE INTO companies (%s) VALUES (%s)"
                   % (",".join(cols), marks))
        data = [tuple(_clean(r.get(c)) for c in cols) for r in rows]
        conn.cursor().executemany(sql, data)
        conn.commit()
        return len(data)

    def upsert_history(self, rows: list[dict]):
        if not rows:
            return 0
        conn = self.connect()
        cols = HISTORY_COLS
        marks = ",".join([self.ph] * len(cols))
        if self.is_postgres:
            updates = ",".join("%s=EXCLUDED.%s" % (c, c)
                               for c in cols if c not in ("cik", "fiscal_year"))
            sql = ("INSERT INTO financial_history (%s) VALUES (%s) "
                   "ON CONFLICT (cik, fiscal_year) DO UPDATE SET %s"
                   % (",".join(cols), marks, updates))
        else:
            sql = ("INSERT OR REPLACE INTO financial_history (%s) VALUES (%s)"
                   % (",".join(cols), marks))
        data = [tuple(_clean(r.get(c)) for c in cols) for r in rows]
        conn.cursor().executemany(sql, data)
        conn.commit()
        return len(data)

    def upsert_market(self, rows: list[dict]):
        if not rows:
            return 0
        conn = self.connect()
        cols = MARKET_COLS
        marks = ",".join([self.ph] * len(cols))
        if self.is_postgres:
            updates = ",".join("%s=EXCLUDED.%s" % (c, c) for c in cols if c != "cik")
            sql = ("INSERT INTO market_data (%s) VALUES (%s) "
                   "ON CONFLICT (cik) DO UPDATE SET %s, updated_at=now()"
                   % (",".join(cols), marks, updates))
        else:
            sql = ("INSERT OR REPLACE INTO market_data (%s) VALUES (%s)"
                   % (",".join(cols), marks))
        data = [tuple(_clean(r.get(c)) for c in cols) for r in rows]
        conn.cursor().executemany(sql, data)
        conn.commit()
        return len(data)

    def upsert_benchmarks(self, rows: list[dict]):
        if not rows:
            return 0
        conn = self.connect()
        cols = BENCHMARK_COLS
        marks = ",".join([self.ph] * len(cols))
        if self.is_postgres:
            updates = ",".join("%s=EXCLUDED.%s" % (c, c)
                               for c in cols if c not in ("sector", "metric"))
            sql = ("INSERT INTO sector_benchmarks (%s) VALUES (%s) "
                   "ON CONFLICT (sector, metric) DO UPDATE SET %s"
                   % (",".join(cols), marks, updates))
        else:
            sql = ("INSERT OR REPLACE INTO sector_benchmarks (%s) VALUES (%s)"
                   % (",".join(cols), marks))
        data = [tuple(_clean(r.get(c)) for c in cols) for r in rows]
        conn.cursor().executemany(sql, data)
        conn.commit()
        return len(data)

    def benchmarks(self, sector: str | None = None) -> list[dict]:
        conn = self.connect()
        cur = conn.cursor()
        if sector:
            cur.execute("SELECT * FROM sector_benchmarks WHERE sector = %s"
                        % self.ph, (sector,))
        else:
            cur.execute("SELECT * FROM sector_benchmarks ORDER BY sector, metric")
        return self._rows_to_dicts(cur)

    def peer_values(self, sector: str, metrics: list[str]) -> dict:
        """Raw peer readings for exact percentiles. Joins the price-derived
        columns so valuation multiples compare against real peers too."""
        safe = [m for m in metrics
                if m in NUMERIC_FILTERS or m in MARKET_COLS or m in COMPANY_COLS]
        if not safe:
            return {}
        company_cols = [m for m in safe if m in COMPANY_COLS]
        market_cols = [m for m in safe if m in MARKET_COLS and m != "cik"]
        sel = ([("c." + m) for m in company_cols]
               + [("m." + m) for m in market_cols])
        if not sel:
            return {}
        conn = self.connect()
        cur = conn.cursor()
        cur.execute(
            "SELECT %s FROM companies c LEFT JOIN market_data m ON c.cik = m.cik "
            "WHERE c.sector = %s" % (",".join(sel), self.ph), (sector,))
        rows = self._rows_to_dicts(cur)
        out: dict[str, list] = {}
        for m in company_cols + market_cols:
            vals = [r.get(m) for r in rows]
            out[m] = [float(v) for v in vals if v is not None]
        return out

    def universe_rows(self, limit: int = 2000) -> list[dict]:
        """Flat company+market rows — the input to the portfolio builder."""
        conn = self.connect()
        cur = conn.cursor()
        cur.execute(
            "SELECT c.*, m.market_cap, m.enterprise_value, m.pe, m.pb, m.ps, "
            "m.ev_ebitda, m.ev_sales, m.peg, m.fcf_yield, m.earnings_yield, "
            "m.dividend_yield, m.beta, m.volatility, m.return_1y, "
            "m.return_total, m.max_drawdown "
            "FROM companies c LEFT JOIN market_data m ON c.cik = m.cik "
            "LIMIT %s" % self.ph, (limit,))
        return self._rows_to_dicts(cur)

    def compute_sector_percentiles(self):
        """Rank each company inside its own sector, not against the market.

        A 30% gross-margin rule is meaningless for a grocery distributor and
        trivial for software, so the headline rank is peer-relative."""
        conn = self.connect()
        df = pd.read_sql_query(
            "SELECT cik, sector, health_score, revenue_growth, operating_margin,"
            " fcf_margin, roic, roe FROM companies", conn)
        if df.empty:
            return 0
        cols = ["health_score", "revenue_growth", "operating_margin",
                "fcf_margin", "roic", "roe"]
        pct = pd.DataFrame(index=df.index)
        for c in cols:
            pct[c] = df.groupby("sector")[c].rank(pct=True, na_option="keep")
        # Average the six metric ranks into one composite score, then rank THAT
        # within sector. The second step matters: a mean of percentiles is not
        # itself a percentile — it compresses toward 0.5 and never reaches 1,
        # so reporting it as "Nth percentile" would misstate the company's
        # standing. Ranking the composite makes the number mean what the label
        # says: this company scores above N% of its sector peers.
        df["score"] = pct.mean(axis=1, skipna=True)
        df["composite"] = df.groupby("sector")["score"].rank(
            pct=True, na_option="keep")
        pairs = [(_clean(r.composite), int(r.cik))
                 for r in df.itertuples() if _clean(r.composite) is not None]
        conn.cursor().executemany(
            "UPDATE companies SET sector_percentile=%s WHERE cik=%s"
            % (self.ph, self.ph), pairs)
        conn.commit()
        return len(pairs)

    def record_run(self, loaded: int, skipped: int, source: str, notes: str = ""):
        conn = self.connect()
        ph = self.ph
        conn.cursor().execute(
            "INSERT INTO runs (companies_loaded, companies_skipped, source, notes,"
            " finished_at) VALUES (%s,%s,%s,%s,%s)"
            % (ph, ph, ph, ph, "now()" if self.is_postgres else "CURRENT_TIMESTAMP"),
            (loaded, skipped, source, notes))
        conn.commit()

    # ---- reads (API hot path) -------------------------------------------
    def _rows_to_dicts(self, cur) -> list[dict]:
        names = [d[0] for d in cur.description]
        return [dict(zip(names, r)) for r in cur.fetchall()]

    def screen(self, filters: Optional[dict] = None, sector: Optional[str] = None,
               search: Optional[str] = None, passing_only: bool = False,
               sort_by: str = "sector_percentile", desc: bool = True,
               limit: int = 50, offset: int = 0) -> dict:
        """Filter + sort + paginate. Returns {'rows': [...], 'total': n}."""
        conn = self.connect()
        ph = self.ph
        where, params = [], []

        if sector:
            where.append("sector = %s" % ph)
            params.append(sector)
        if passing_only:
            where.append("passes_screen = %s" % ph)
            params.append(True if self.is_postgres else 1)
        if search:
            where.append("(UPPER(ticker) LIKE %s OR UPPER(name) LIKE %s)" % (ph, ph))
            like = "%" + search.upper() + "%"
            params += [like, like]
        for key, bound in (filters or {}).items():
            if bound is None:
                continue
            base, _, op = key.rpartition("__")
            if base not in NUMERIC_FILTERS or op not in ("min", "max"):
                continue
            where.append("%s %s %s" % (base, ">=" if op == "min" else "<=", ph))
            params.append(bound)

        clause = (" WHERE " + " AND ".join(where)) if where else ""
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM companies" + clause, params)
        total = cur.fetchone()[0]

        col = sort_by if sort_by in SORTABLE else "sector_percentile"
        direction = "DESC" if desc else "ASC"
        nulls = "NULLS LAST" if self.is_postgres else ""
        cur.execute(
            "SELECT * FROM companies%s ORDER BY %s %s %s, revenue DESC "
            "LIMIT %s OFFSET %s" % (clause, col, direction, nulls, ph, ph),
            params + [limit, offset])
        return {"rows": self._rows_to_dicts(cur), "total": total}

    def company(self, ticker: str) -> Optional[dict]:
        conn = self.connect()
        cur = conn.cursor()
        cur.execute("SELECT * FROM companies WHERE UPPER(ticker) = %s" % self.ph,
                    (ticker.upper(),))
        rows = self._rows_to_dicts(cur)
        if not rows:
            return None
        company = rows[0]
        cur.execute(
            "SELECT * FROM financial_history WHERE cik = %s ORDER BY fiscal_year"
            % self.ph, (company["cik"],))
        company["history"] = self._rows_to_dicts(cur)
        cur.execute("SELECT * FROM market_data WHERE cik = %s" % self.ph,
                    (company["cik"],))
        md = self._rows_to_dicts(cur)
        company["market"] = md[0] if md else None
        return company

    def sectors(self) -> list[dict]:
        conn = self.connect()
        cur = conn.cursor()
        cur.execute(
            "SELECT sector, COUNT(*) AS companies, "
            "SUM(CASE WHEN passes_screen THEN 1 ELSE 0 END) AS passing, "
            "AVG(health_score) AS avg_health, AVG(operating_margin) AS avg_op_margin, "
            "AVG(roic) AS avg_roic FROM companies GROUP BY sector "
            "ORDER BY companies DESC")
        return self._rows_to_dicts(cur)

    def stats(self) -> dict:
        conn = self.connect()
        cur = conn.cursor()
        cur.execute(
            "SELECT COUNT(*) AS companies, "
            "SUM(CASE WHEN passes_screen THEN 1 ELSE 0 END) AS passing, "
            "COUNT(DISTINCT sector) AS sectors, MAX(fiscal_year) AS latest_year "
            "FROM companies")
        out = self._rows_to_dicts(cur)[0]
        cur.execute("SELECT MAX(finished_at) AS last_run FROM runs")
        out["last_run"] = self._rows_to_dicts(cur)[0]["last_run"]
        cur.execute("SELECT COUNT(*) AS priced FROM market_data")
        out["priced"] = self._rows_to_dicts(cur)[0]["priced"]
        return out
