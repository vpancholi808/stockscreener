# CLAUDE.md — finmodel

Guidance for Claude Code working in this repository.

## What this is

`finmodel` is an equity-analysis toolkit in Python. It pulls live public data
(SEC EDGAR for fundamentals, Yahoo Finance for prices), builds a linked
3-statement model, and layers valuation, factor, portfolio, risk, and options
models on top. Two tracks:

- **Fundamental:** 3-statement model → DCF (intrinsic value vs. market price)
- **Market/risk:** CAPM/beta → WACC → factors → Markowitz → VaR → options

## Setup & run

```bash
pip install -r requirements.txt
export EDGAR_USER_AGENT="Real Name email@example.com"   # SEC blocks requests without this
python example_live.py AAPL     # financials + 12-metric health scorecard
python example_dcf.py  AAPL     # CAPM/WACC + DCF intrinsic value vs price
python example_quant.py         # universe → Markowitz, VaR, options demo
```

`EDGAR_USER_AGENT` must contain a real email or SEC throttles/blocks the pull.
Network access is required (EDGAR + Yahoo). This repo's earlier development was
done in a sandbox without network, so live data paths are validated by the user;
the test suites use synthetic data and run fully offline.

## Architecture (read these to orient)

```
finmodel/
  data/edgar.py        SEC EDGAR client: ticker->CIK, throttled companyfacts pull
  data/concepts.py     XBRL tag -> line-item maps with fallbacks (EXTEND HERE for coverage)
  data/prices.py       PriceFeed interface; YFinancePriceFeed (default) + FramePriceFeed (offline)
  data/store.py        local disk cache for companyfacts (DataStore), pluggable client
  data/quality.py      data-quality report: coverage, missing items, BS articulation
  statements.py        raw facts -> clean annual statements (handles duration vs instant periods)
  projections.py       driver-based linked projection; cash is the balancing plug
  scorecard.py         12-metric health scorecard with tunable thresholds (ScorecardConfig)
  capm.py              beta (excess-return OLS, Blume-adjusted) -> cost of equity -> WACC
  dcf.py               unlevered FCFF DCF + terminal value + sensitivity grid
  model.py             ThreeStatementModel orchestrator (from_ticker / from_facts / from_file)
  screener.py          run engine across a watchlist -> ranked, filterable table (Screener)
  valuation.py         relative valuation multiples (P/E, EV/EBITDA, P/B, PEG, yields)
  sectors.py           per-sector MEDIAN benchmarks + company-vs-peer percentiles
  research.py          evidence-based research notes (NO buy/sell verdict, by design)
  portfolio_builder.py strategies x risk profiles -> constrained allocation + assessment
  data/sp500.py        S&P 500 constituents (fetched + cached; no stale hardcoded list)
  report.py            self-contained HTML screening dashboard (dashboard_html)
  quant/universe.py    multi-ticker aligned returns panel (Universe)
  quant/factors.py     statement factors (value/quality/growth/profitability), cross-sectional z-scores
  quant/portfolio.py   Markowitz: Ledoit-Wolf cov, frontier, min-var, max-Sharpe (analytic + SLSQP)
  quant/risk.py        VaR (hist/parametric/Cornish-Fisher), CVaR, drawdown, stress tests
  quant/options.py     Black-Scholes price, Greeks, implied vol
  quant/stochastic.py  Brownian/GBM/OU/jump-diffusion/Heston simulators + Euler-Maruyama
  quant/rates.py       Vasicek & CIR short-rate models (analytic ZCB + simulation)
  quant/credit.py      Merton structural credit (equity as call on assets)
  web/store.py         DB layer; DATABASE_URL picks sqlite (dev) or postgres (prod)
  web/schema.sql       Postgres DDL (SQLITE_DDL in store.py mirrors it)
  web/loader.py        ThreeStatementModel -> DB rows; the scheduled worker's core
  web/api.py           FastAPI read-only JSON API (rate-limited, CORS-scoped)
  web/seed.py          demo data (real Micron + generated peers), no network needed
../web/                Next.js 14 frontend (app router, server components)
../worker.py           CLI: --seed | --universe | TICKER...  (the only SEC caller)
```

Entry point: `from finmodel import ThreeStatementModel, CapmModel, DcfModel`
and `from finmodel.quant import Universe, MarkowitzOptimizer, risk_report, ...`.

## Key design decisions (don't "fix" these — they're deliberate)

- **DCF discounts unlevered FCFF**, not the projection's levered `free_cash_flow`
  line. WACC already prices the debt tax shield; discounting levered CF at WACC
  would double-count it. See `dcf._fcff_from_projection`.
- **CAPM feeds DCF.** Cost of equity → WACC → discount rate. Beta is computed
  before the DCF even though it appears "later" in the conceptual list.
- **`driver_basis` defaults to "latest"** in Assumptions (not average). Averaging
  ratios across a cyclical swing (e.g. Micron FY23 -9% GM -> FY25 +40% GM) yields
  incoherent drivers. Options: latest | median | average. Assumptions.infer() also
  emits warnings when implied operating margin <= 0 or capex intensity > 25%.
- **Cash is the balance-sheet plug** in projections; `projection.balance_check`
  should be ~0 every year.
- **Beta uses Blume adjustment** (0.67·raw + 0.33) for cost of equity.
- **Factors are z-scored cross-sectionally** with signs normalized so higher =
  more attractive.
- **Missing debt tags read as zero debt** — verify leverage on unfamiliar names.
- **Point-in-time via `as_of`** — `build_financials(..., as_of=DATE)` and
  `from_ticker(..., as_of=DATE)` use only data filed on/before DATE (restatements
  included only once public). Use it for any backtest to avoid look-ahead bias.
- **The web app NEVER fetches from SEC in a request.** SEC rate-limits ~10 req/s
  and the bulk file is 1.5 GB. `worker.py` precomputes into the DB on a schedule;
  the API only reads. Do not "optimise" this into on-demand fetching.
- **Price fields are withheld unless SERVE_MARKET_DATA=true.** SEC data is public
  domain and republishable; scraped price data generally is not.
- **Sector percentile is a two-step rank**: mean of six metric percentile ranks,
  then rank THAT within sector. A mean of percentiles is not itself a percentile
  (it compresses toward 0.5 and never reaches 1), so skipping step two would
  misstate a company's standing. Applies to web/store.py and universe.py alike.
- **Sector comparisons use the MEDIAN, never the mean.** Valuation multiples are
  right-skewed; one company at 400x P/E drags a sector mean far above anything a
  typical constituent trades at. The mean is stored for reference only.
- **Multiples return None on non-positive denominators.** A P/E on negative
  earnings is undefined, not 'cheap'. Returning it would corrupt every sector
  median it lands in. See valuation._d.
- **research.py has NO buy/sell/hold field and must not gain one.** A public
  site issuing verdicts is giving regulated investment advice, and a verdict
  built on data up to 12 months stale would be confidently wrong often enough
  to cause losses. Findings state evidence; readers conclude.
- **Portfolio position and sector caps are satisfied JOINTLY**, by alternating
  until both hold (portfolio_builder._enforce_constraints). Applying them in
  sequence lets sector redistribution push a position back over its cap — this
  was a real bug caught by the test suite. The position cap wins ties.
- **Weighted portfolio metrics exclude out-of-bounds readings** (METRIC_BOUNDS)
  and report how many were dropped. Companies with near-zero book equity after
  buybacks show ROE in the hundreds of percent and would destroy the average.
- **Default history is 5 years** (n_years=5) for filings, and 5y monthly for
  prices. One price request yields last price, volatility, Blume-adjusted beta,
  1y/total return and max drawdown together.
- **Caching is default** — `from_ticker` reads/writes a disk cache via DataStore;
  pass `use_cache=False` to force a live pull.

## Default parameters (override per use; these are the dials that matter)

Scorecard (`ScorecardConfig`): D/E ≤ 2.0, NetDebt/EBITDA ≤ 3.0, interest cov ≥ 3.0,
current ≥ 1.2, quick ≥ 1.0, FCF margin ≥ 5%, cash conversion ≥ 0.8, gross ≥ 30%,
operating ≥ 10%, ROE ≥ 12%, ROIC ≥ 8%, positive FCF required.

CAPM (`CapmConfig`): risk_free_rate 4.2%, equity_risk_premium 5.0%, market ^GSPC,
monthly returns, 5y lookback. **Set rf and ERP deliberately each run.**

DCF (`DcfConfig`): terminal_growth 2.5%, Gordon method (or exit_ev_ebitda 10×),
mid-year on, tax inferred from history. **terminal_growth must be < WACC.**

## Testing

```bash
python test_pipeline.py   # 3-statement build, tag fallback, distressed case
python test_capm.py       # beta recovery on synthetic data, WACC integration
python test_dcf.py        # DCF recomputed independently + sensitivity monotonicity
python test_quant.py      # BS textbook value + parity, optimizer, VaR=z·σ, factor ranks
```

All tests are known-answer or analytic-identity checks on synthetic data and run
offline. When adding a model, add a matching known-answer test in the same style.

## Conventions

- Python 3.10+, deps: numpy, pandas, scipy, requests, yfinance. Add heavy deps
  (statsmodels, arch, sklearn, xgboost) only with the staged batch below.
- Keep modules dependency-light; options.py uses `math.erf` rather than scipy.
- Type hints + dataclasses for results; `__str__`/`summary()` for human-readable output.
- Do NOT create nested `finmodel/finmodel/finmodel` folders. The project root is
  `finmodel/` and the package is `finmodel/finmodel/`. One level of nesting only.

## Staged next (not yet built)

Fama-French (size/value/profitability/investment/momentum vs Ken French data),
GARCH(1,1) volatility, pairs trading / cointegration (Engle-Granger), ML
(XGBoost/RF/NN with walk-forward + leakage controls), execution/market-impact
(Almgren-Chriss). Heston has an MC pricer; a Fourier/characteristic-function
version and HJM rates are also open. These plug into `Universe` and `FactorModel` and feed the
eventual long/short portfolio model.
