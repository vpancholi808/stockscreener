-- finmodel web schema (PostgreSQL)
--
-- Design notes:
--  * `companies` is deliberately WIDE and denormalized. A screener's hot path is
--    "filter + sort + paginate across the whole market", which is one indexed
--    scan over a few thousand rows. Normalizing would add joins for no gain.
--  * Price-derived fields live in their own table. SEC fundamentals are US
--    government work and public-domain, so they can be served publicly; price
--    data from a scraped feed usually cannot be redistributed. Keeping them
--    separate means the public API can serve `companies` alone and switch the
--    price layer on only when a redistributable feed is licensed.
--  * `financial_history` holds the per-year series that drive sparklines.
--  * Everything is written by the Python worker; the API only ever reads.

CREATE TABLE IF NOT EXISTS companies (
    cik                 INTEGER PRIMARY KEY,
    ticker              TEXT        NOT NULL,
    name                TEXT        NOT NULL,
    sector              TEXT        NOT NULL DEFAULT 'Unknown',
    fiscal_year         INTEGER,
    -- health scorecard
    health_passed       SMALLINT,
    health_total        SMALLINT,
    health_score        REAL,
    -- size & growth
    revenue             DOUBLE PRECISION,
    revenue_growth      REAL,
    ebitda              DOUBLE PRECISION,
    fcf                 DOUBLE PRECISION,
    net_debt            DOUBLE PRECISION,
    -- margins
    gross_margin        REAL,
    operating_margin    REAL,
    net_margin          REAL,
    fcf_margin          REAL,
    -- returns & leverage
    roe                 REAL,
    roic                REAL,
    net_debt_to_ebitda  REAL,
    interest_coverage   REAL,
    current_ratio       REAL,
    debt_to_equity      REAL,
    cash_conversion     REAL,
    -- forensic scores
    piotroski           SMALLINT,
    altman              REAL,
    beneish             REAL,
    accruals            REAL,
    shares_diluted      DOUBLE PRECISION,
    -- meta
    data_quality        TEXT,
    passes_screen       BOOLEAN     NOT NULL DEFAULT FALSE,
    sector_percentile   REAL,
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS companies_ticker_key ON companies (ticker);
CREATE INDEX IF NOT EXISTS companies_sector_idx       ON companies (sector);
CREATE INDEX IF NOT EXISTS companies_health_idx       ON companies (health_score DESC NULLS LAST);
CREATE INDEX IF NOT EXISTS companies_revenue_idx      ON companies (revenue DESC NULLS LAST);
CREATE INDEX IF NOT EXISTS companies_roic_idx         ON companies (roic DESC NULLS LAST);
CREATE INDEX IF NOT EXISTS companies_passes_idx       ON companies (passes_screen);
-- trigram index makes the search box fast on name as well as ticker
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE INDEX IF NOT EXISTS companies_name_trgm_idx    ON companies USING gin (name gin_trgm_ops);


CREATE TABLE IF NOT EXISTS financial_history (
    cik                 INTEGER     NOT NULL REFERENCES companies(cik) ON DELETE CASCADE,
    fiscal_year         INTEGER     NOT NULL,
    revenue             DOUBLE PRECISION,
    gross_profit        DOUBLE PRECISION,
    operating_income    DOUBLE PRECISION,
    net_income          DOUBLE PRECISION,
    ebitda              DOUBLE PRECISION,
    operating_cash_flow DOUBLE PRECISION,
    capex               DOUBLE PRECISION,
    fcf                 DOUBLE PRECISION,
    total_assets        DOUBLE PRECISION,
    total_equity        DOUBLE PRECISION,
    total_debt          DOUBLE PRECISION,
    PRIMARY KEY (cik, fiscal_year)
);

CREATE INDEX IF NOT EXISTS history_cik_idx ON financial_history (cik);


-- Optional, licence-gated layer. Populated only when a redistributable price
-- feed is configured; the public API omits these fields otherwise.
CREATE TABLE IF NOT EXISTS market_data (
    cik                 INTEGER PRIMARY KEY REFERENCES companies(cik) ON DELETE CASCADE,
    price               DOUBLE PRECISION,
    shares_diluted      DOUBLE PRECISION,
    market_cap          DOUBLE PRECISION,
    enterprise_value    DOUBLE PRECISION,
    pe                  REAL,
    forward_pe          REAL,
    pb                  REAL,
    ps                  REAL,
    ev_ebitda           REAL,
    ev_sales            REAL,
    ev_ebit             REAL,
    peg                 REAL,
    fcf_yield           REAL,
    earnings_yield      REAL,
    dividend_yield      REAL,
    beta                REAL,
    volatility          REAL,
    return_1y           REAL,
    return_total        REAL,
    max_drawdown        REAL,
    price_observations  INTEGER,
    source              TEXT,
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);


-- Per-sector reference statistics. `price_derived` marks rows built from
-- market prices, so the API can withhold them when the price feed's licence
-- does not permit redistribution.
CREATE TABLE IF NOT EXISTS sector_benchmarks (
    sector              TEXT    NOT NULL,
    metric              TEXT    NOT NULL,
    n_companies         INTEGER,
    count               INTEGER,
    median              DOUBLE PRECISION,
    mean                DOUBLE PRECISION,
    p25                 DOUBLE PRECISION,
    p75                 DOUBLE PRECISION,
    std                 DOUBLE PRECISION,
    price_derived       BOOLEAN NOT NULL DEFAULT FALSE,
    PRIMARY KEY (sector, metric)
);

CREATE INDEX IF NOT EXISTS bench_sector_idx ON sector_benchmarks (sector);


-- One row per worker run, so the site can show data freshness honestly.
CREATE TABLE IF NOT EXISTS runs (
    id                  SERIAL PRIMARY KEY,
    started_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at         TIMESTAMPTZ,
    companies_loaded    INTEGER,
    companies_skipped   INTEGER,
    source              TEXT,
    notes               TEXT
);
