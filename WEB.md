# finmodel web — public screener

Three pieces:

```
worker.py            Python. Talks to SEC. Runs on a schedule, writes to the DB.
finmodel/web/api.py  FastAPI. Read-only JSON. Never touches SEC.
web/                 Next.js 14. Server components, reads the API.
```

The separation is not stylistic. SEC rate-limits to roughly 10 requests/second
and the bulk dataset is ~1.5 GB, so fetching inside a request handler would be
slow and would get you blocked. Everything is pre-computed; the site only reads.

---

## Run it locally in three commands

```bash
pip install -r requirements.txt -r requirements-web.txt
python worker.py --seed                       # demo data, no network needed
uvicorn finmodel.web.api:app --port 8000
```

Then in a second terminal:

```bash
cd web && npm install && npm run dev
```

Open http://localhost:3000.

### With real SEC data instead of the demo set

```bash
export EDGAR_USER_AGENT="Your Name you@email.com"
python worker.py AAPL MSFT NVDA MU            # a few tickers
python worker.py --universe --limit 500       # try 500 before the full market
python worker.py --universe                   # everything (~1.5 GB first run)
```

### Or the whole stack in Docker

```bash
docker compose up --build
docker compose run --rm api python worker.py --seed
```

---

## Two decisions worth understanding before you deploy publicly

### 1. Price data is withheld by default

SEC filings are U.S. government work and in the public domain, so fundamentals
and every ratio derived from them can be republished freely.

Scraped price data is different. Yahoo's terms prohibit redistribution, so
serving yfinance-derived market caps, P/E or EV/EBITDA to the public is a terms
problem, not just a reliability one. The API therefore strips those fields
unless `SERVE_MARKET_DATA=true`.

If you want valuation multiples on a public site, move to a feed whose licence
permits redistribution — Tiingo and Polygon both do on paid tiers — populate the
`market_data` table from it, and then flip the flag.

### 2. Annual data is a product problem, not just a model limitation

The model reads 10-K filings, so every figure is as-of the last filed fiscal
year. Privately that is a known caveat. Publicly it is different: a visitor who
knows a company mid-upcycle will see a revenue figure far below its current
run-rate and conclude the site is broken.

Every page states "FY20XX as filed" for this reason, but the real fix is
quarterly/TTM parsing. **Treat that as a prerequisite for launch, not a
follow-up.**

---

## Architecture

```
            ┌──────────────┐
  SEC ─────►│  worker.py   │ scheduled (cron / systemd timer / GitHub Action)
            │  (finmodel)  │ streams bulk zip, scores, writes rows
            └──────┬───────┘
                   ▼
            ┌──────────────┐
            │  PostgreSQL  │  companies · financial_history · market_data · runs
            └──────┬───────┘
                   ▼
            ┌──────────────┐
            │   FastAPI    │  read-only, rate-limited, CORS-scoped
            └──────┬───────┘
                   ▼
            ┌──────────────┐
            │   Next.js    │  server components; browser never sees the API host
            └──────────────┘
```

**Why SQLite and Postgres both work.** `DATABASE_URL` picks the backend. SQLite
keeps local development dependency-free; Postgres is the deployment target.
Application code is identical.

**Why filters live in the URL.** The screener page is a server component with no
client-side fetching or state library — filter state is query params. Every view
is therefore a shareable link, which matters for a tool people send to each
other.

---

## API

| Endpoint | Purpose |
|---|---|
| `GET /api/companies` | screen: filter, sort, paginate |
| `GET /api/companies/{ticker}` | one company + history |
| `GET /api/sectors` | sector counts and averages |
| `GET /api/stats` | totals and data freshness |
| `GET /api/meta` | thresholds and definitions (frontend never hardcodes them) |
| `GET /api/health` | liveness |

Filters: `min_roic`, `min_operating_margin`, `min_fcf_margin`, `min_revenue`,
`min_revenue_growth`, `max_net_debt_to_ebitda`, `min_interest_coverage`,
`min_current_ratio`, `max_debt_to_equity`, `min_piotroski`, `min_altman`,
`min_health_score`, plus `sector`, `search`, `passing_only`, `sort_by`, `desc`,
`limit`, `offset`.

```bash
curl "localhost:8000/api/companies?min_roic=0.15&sector=Semiconductors&sort_by=roic"
```

---

## Deploying

**Frontend** — Vercel. Set `API_BASE` to the API's URL. It is read server-side
only, so the API host is never exposed to browsers.

**API + worker** — one small box (Hetzner CX22 is plenty) or any container host.
The worker needs ~2 GB free disk for the SEC bulk file.

**Database** — managed Postgres, or on the same box. The dataset is small:
roughly 8,000 companies is a few hundred MB including history.

Before going public:

- [ ] Put a real rate limiter in front of the API (nginx, Cloudflare). The
      built-in one is in-process and does not survive restarts or coordinate
      across replicas.
- [ ] Set `ALLOWED_ORIGINS` to your actual domain, not `*`.
- [ ] Add TTM/quarterly data (see above).
- [ ] Keep the disclaimer visible. It is in the footer and in every API
      response; do not strip it.

---

## Adding a field end to end

1. `finmodel/web/schema.sql` and `SQLITE_DDL` in `store.py` — add the column
2. `store.py` — add to `COMPANY_COLS`, and to `SORTABLE` / `NUMERIC_FILTERS` if
   users should sort or filter on it
3. `loader.py` — populate it in `model_to_rows`
4. `api.py` — add a query parameter if it is filterable
5. `web/lib/api.ts` — add to the `Company` type
6. `web/components/ScreenTable.tsx` — add a column

Re-run the worker; the upsert handles existing rows.

---

## Relative valuation, research notes and portfolios

Three layers were added on top of the screener.

### Relative valuation (`finmodel/valuation.py`)

P/E, P/B, P/S, EV/EBITDA, EV/Sales, EV/EBIT, PEG, FCF yield, earnings yield and
dividend yield. Market cap is price × diluted shares from the filings, so the
numerator and denominator of every multiple sit on the same basis.

Multiples return `None` on a non-positive denominator. A P/E computed on
negative earnings is undefined, not cheap, and letting it through would corrupt
every sector median it lands in.

Forward P/E stays `None` everywhere: it needs analyst consensus estimates, which
the SEC does not publish.

### Sector benchmarks (`finmodel/sectors.py`)

Per-sector median, mean, p25, p75 and standard deviation for twenty metrics,
with each company placed in its sector's distribution.

**Comparisons use the median, not the mean.** Valuation multiples are heavily
right-skewed — in the demo data the semiconductor P/E median is 24.6 against a
mean of 27.5, and net debt/EBITDA runs −0.02 median against +0.43 mean. Compare
against a mean and the answer is wrong for most of the sector. The mean is
stored and shown, because the gap between the two is itself informative.

Sectors with fewer than five constituents are marked unavailable rather than
given a noisy benchmark.

### Research notes (`finmodel/research.py`)

Structured written assessments: headline, summary, strengths, concerns, watch
items, valuation context, and limitations. Every finding carries the figure it
rests on.

**There is deliberately no buy/sell/hold verdict, and adding one would be a
mistake.** A public site that tells people whether to invest is giving
investment advice, a regulated activity in most jurisdictions. It is also
indefensible on this data: every figure comes from annual filings and can be up
to twelve months stale. The note reports what the numbers show; the reader draws
the conclusion. The test suite asserts no verdict field exists and that no
buy/sell language appears outside the disclaimer.

### Portfolio builder (`finmodel/portfolio_builder.py`)

Six strategies (quality, value, growth, income, defensive, balanced) crossed
with three risk profiles (conservative, balanced, aggressive) and three
allocation methods.

Risk is enforced as **constraints**, not as a score: position caps, sector caps,
solvency floors, minimum holdings. The two caps are satisfied *jointly* —
applying them in sequence lets sector redistribution push a position back over
its limit, which the tests caught. Where they cannot both be met, the position
cap holds and the result says so.

Reported: weighted characteristics, sector exposure, Herfindahl concentration
and effective number of holdings (1/HHI), which is a more honest read of
diversification than a raw position count.

`assess_portfolio()` takes existing holdings and flags concentration, sector
dominance and weak-balance-sheet exposure.

### Five years of history

Filings default to five fiscal years. Prices are fetched as five years of
monthly total-return history in one pass, yielding last price, annualised
volatility, Blume-adjusted beta, one-year and total return, and maximum
drawdown together — rather than a second round trip per ticker.

### S&P 500

```bash
python worker.py --sp500
```

Constituents are fetched from the Wikipedia table and cached for seven days.
No hardcoded list ships with the project: a stale membership list silently drops
recent additions and keeps removed companies, quietly biasing every sector
benchmark built on it. `load_constituents_csv(path)` takes your own list for any
other index, or to pin membership to a date for a backtest.

### New endpoints

| Endpoint | Purpose |
|---|---|
| `GET /api/sectors/{sector}/benchmarks` | sector medians, means and quartiles |
| `GET /api/companies/{ticker}/research` | research note + peer comparisons |
| `GET /api/portfolio/options` | strategies, risk profiles, allocation methods |
| `POST /api/portfolio/build` | construct a portfolio from criteria |
| `POST /api/portfolio/assess` | evaluate existing holdings |

Price-derived fields and benchmarks are withheld from all of these unless
`SERVE_MARKET_DATA=true`.
