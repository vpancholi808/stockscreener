# stockscreener

Equity health screening and portfolio construction, computed from SEC filings.

Pulls fundamentals from SEC EDGAR, rebuilds linked financial statements, scores
financial health against twelve balance-sheet and cash-flow tests, runs four
forensic accounting checks, compares every company against its own sector peers,
and builds portfolios under explicit risk constraints.

No analyst estimates, no price targets, no buy/sell verdicts — see
[Why there are no recommendations](#why-there-are-no-recommendations).

---

## What it does

| | |
|---|---|
| **Statements** | Linked 3-statement model from XBRL filings, 5 fiscal years |
| **Health** | 12 checks across solvency, liquidity, cash-flow quality, profitability |
| **Forensic** | Piotroski F-score, Altman Z, Beneish M, Sloan accruals, ROIC−WACC spread |
| **Valuation** | P/E, P/B, P/S, EV/EBITDA, EV/Sales, PEG, FCF yield, earnings yield |
| **Sector** | Per-sector medians and quartiles; every company placed in its peer distribution |
| **Research** | Written evidence summaries — strengths, concerns, valuation context, limits |
| **Portfolio** | 6 strategies × 3 risk profiles × 3 allocation methods, with hard constraints |
| **Quant** | DCF, CAPM/beta, Markowitz, VaR/CVaR, Black-Scholes, stochastic processes, Merton credit |

---

## Quick start

Three commands to a working site:

```bash
pip install -r requirements.txt
python worker.py --seed                 # demo data — no network, no SEC key
streamlit run streamlit_app.py          # http://localhost:8501
```

For real filings instead of the demo set (SEC requires a real name and email —
there is no API key and nothing to sign up for):

```bash
python check_edgar.py                  # confirm access, diagnose it if not

export EDGAR_USER_AGENT="Your Name you@email.com"
python worker.py AAPL MSFT NVDA MU     # specific tickers
python worker.py --sp500               # S&P 500 constituents
python worker.py --universe            # whole US market (~1.5 GB first run)
```

Single-company analysis with no web stack at all:

```bash
python run.py MU                        # console summary + HTML tearsheet
python run.py AAPL MSFT NVDA            # ranked screen dashboard
```

<details>
<summary>Or the JSON API + Next.js frontend instead of Streamlit</summary>

Streamlit and the API/Next.js stack read the same database and are
interchangeable. Use the API stack if you want JSON endpoints other things can
call; see [`WEB.md`](WEB.md).

```bash
pip install -r requirements-web.txt
uvicorn finmodel.web.api:app --port 8000
cd web && npm install && npm run dev     # http://localhost:3000
```
</details>

Everything in one command:

```bash
docker compose up --build
docker compose run --rm api python worker.py --seed
```

---

## Architecture

```
            ┌──────────────┐
  SEC ─────►│  worker.py   │  scheduled — streams bulk XBRL, scores, writes rows
 Yahoo ────►│  (finmodel)  │  the ONLY component that talks to a data source
            └──────┬───────┘
                   ▼
            ┌──────────────┐
            │  PostgreSQL  │  companies · history · market_data · sector_benchmarks
            └──────┬───────┘
                   │
        ┌──────────┴───────────┐     two interchangeable front ends,
        ▼                      ▼     both read-only
┌───────────────┐      ┌──────────────┐
│   Streamlit   │      │   FastAPI    │  rate-limited, CORS-scoped
│ one process,  │      └──────┬───────┘
│ no API layer  │             ▼
└───────────────┘      ┌──────────────┐
                       │   Next.js    │  browser never sees the API host
                       └──────────────┘
```

**Nothing fetches on request.** SEC rate-limits to roughly 10 requests/second
and the bulk dataset is ~1.5 GB. The worker precomputes on a schedule; the API
only reads. `DATABASE_URL` selects SQLite (local development) or Postgres
(deployment) with no application changes.

---

## Why there are no recommendations

The research notes report evidence — margins, leverage, cash conversion,
valuation against peers — and stop there. There is no buy/sell/hold field, and
this is deliberate:

1. A public site telling people whether to invest is giving investment advice,
   which is a regulated activity in most jurisdictions.
2. Every figure comes from annual filings and can be up to twelve months stale.
   A verdict built on that would be confidently wrong often enough to cause
   real losses.
3. Financial health and price attractiveness are different questions. One label
   collapses them into a false single answer.

The test suite enforces this: it asserts no verdict field exists and that no
buy/sell language appears outside the disclaimer.

---

## Known limitations

- **Annual data only.** Figures come from the last filed 10-K. For a company
  mid-cycle the gap to current trading can be large. Quarterly/TTM parsing is
  the top-priority next addition.
- **No forward estimates.** Forward P/E and growth forecasts need analyst
  consensus data, which the SEC does not publish.
- **Price data licensing.** SEC filings are U.S. government work and public
  domain. Scraped price data generally is not redistributable, so price-derived
  fields are withheld unless `SERVE_MARKET_DATA=true`. Use a feed licensed for
  redistribution before serving multiples publicly.
- **US SEC filers only.** Foreign issuers filing Form 20-F have partial coverage.
- **No backtester yet**, so no machine learning on signals — walk-forward
  validation with purging and embargo has to exist first.

---

## Documentation

| File | Contents |
|---|---|
| [`STREAMLIT.md`](STREAMLIT.md) | Streamlit app: run locally, deploy, troubleshoot |
| [`WEB.md`](WEB.md) | JSON API + Next.js stack, API reference, deployment |
| [`QUICKSTART.md`](QUICKSTART.md) | Plain-English setup and troubleshooting |
| [`docs/MODELS.md`](docs/MODELS.md) | Every model, parameter and formula |
| [`CLAUDE.md`](CLAUDE.md) | Architecture notes and deliberate design decisions |

---

## Tests

```bash
for t in test_pipeline test_capm test_dcf test_quant test_stoch \
         test_foundation test_screener test_web test_valuation \
         test_streamlit test_yahoo test_returns test_performance \
         test_edgar test_profiles; do
  python -W ignore $t.py
done
```

Every suite is a known-answer or analytic-identity check on synthetic data and
runs with no network: Black-Scholes against textbook values with put-call
parity, beta recovered from simulated returns, VaR against z·σ, analytic bond
prices against Monte Carlo, portfolio constraints against their caps.

CI runs all of them on every push, plus a frontend type-check, build and
dependency audit.

---

## Disclaimer

This is analysis tooling, not investment advice. It applies published accounting
tests to public filings and reports the result. It does not know your
circumstances, does not forecast prices, and is not a recommendation to buy or
sell any security. Verify every figure against the original filing before acting
on it.
