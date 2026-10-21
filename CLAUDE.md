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
  data/yahoo.py        hardened price import: shape normalisation, vintage-stamped
                       cache, coverage report, bad-print audit (audit_prices)
  returns.py           return distribution: both annualizations, JB, tail counts,
                       rolling vol, kurtosis_concentration
  performance.py       TWR / MWR(XIRR) with external cash flows, rebalancing
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

## Streamlit app (streamlit_app.py + views/)

```
streamlit_app.py       entry point: st.set_page_config + st.navigation ONLY
views/screener.py      filter/sort/rank the universe
views/company.py       statements, peer comparison, research note
views/portfolio.py     build under constraints, or assess existing holdings
views/methodology.py   thresholds, formulas, limitations
finmodel/web/ui.py     cached DB reads, formatting, the licensing gate, footer
```

Run it: `streamlit run streamlit_app.py`. Tested by `test_streamlit.py`
(offline, via Streamlit's AppTest harness).

- **Navigation is declared, not discovered.** `st.navigation` in the entry
  point, with explicit `url_path` per page. There is no `pages/` directory:
  that convention derives sidebar labels from filenames ("streamlit app") and
  URL paths that change when a file is renamed.
- **`st.set_page_config` belongs to the entry point only.** Calling it from a
  view under `st.navigation` raises. Views set no page config.
- **Every DB read in ui.py is cached** — `@st.cache_resource` for the
  connection, `@st.cache_data` with a TTL for queries. Streamlit re-runs the
  whole script on every widget interaction, so an uncached query runs again on
  every click. `clear_caches()` is wired to the sidebar's Refresh button.
- **Caches persist across AppTest runs in one process.** Any test that changes
  the database or flips `SERVE_MARKET_DATA` must call `ui.clear_caches()`, or
  it replays the previous phase's result and silently passes.
- **Sliders that mean percentages hold integers**, not 0-1 fractions. A
  `"%d%%"` format on 0.35 renders as "0%" — this shipped once.
- **`assess_portfolio` takes `{ticker: weight}`**, not a list of dicts.
- **The same licensing gate as the API**: `ui.strip_market()` and the
  `PRICE_DERIVED_METRICS` filters. SEC data is public domain and
  republishable; scraped price data is not. Off by default, and the footer
  says so.
- **requirements.txt is the superset** (core + web + streamlit) because
  Streamlit Cloud only reads that filename. `requirements-core.txt` is the
  models alone; `requirements-web.txt` adds FastAPI. CI installs the superset
  so an undeclared dependency cannot pass.

## Return statistics and performance (returns.py, performance.py, data/yahoo.py)

Ported from the FINA4472 return-statistics and multi-asset notebooks, with the
corrections that moving from teaching code to production requires. Tested by
`test_returns.py`, `test_performance.py`, `test_yahoo.py` — all offline.

- **TWR is invariant to flow size only for pro-rata contributions.** Adding
  money in proportion to current holdings leaves the asset mix untouched, so
  returns after the flow are unchanged and TWR measures the strategy alone —
  which is the entire point of TWR. `contribution_weights` defaults to
  `'pro_rata'` for that reason. `'equal'`, `'initial'` or an explicit array
  genuinely change the mix and therefore every subsequent return; that is real
  portfolio behaviour, not an artefact, but it does break the invariance.
  The test suite asserts both halves.
- **With no external flows, MWR equals annualized TWR exactly.** Asserted to
  1e-8. If that identity ever breaks, the cash-flow signing or the day count
  is wrong.
- **Cash flows dated before the price history raise.** The notebook version
  silently applied them on day one, where they are indistinguishable from the
  opening capital and quietly inflate the capital base.
- **XIRR returns NaN rather than a number when no root is bracketed**, and
  `sign_changes` warns when the flows alternate more than once, because the
  NPV equation can then have several roots.
- **variance_drag is the Jensen gap**, `periods x (mean(r) - mean(log(1+r)))`,
  which is non-negative by construction and is what the sigma^2/2 result
  refers to. It is NOT the difference between the two annualized returns —
  that can be negative, because compounding is convex (`exp(x)-1 > x`) and
  for a high-mean low-variance series the convexity gain exceeds the drag.
  Do not "fix" a negative annualized gap; it is arithmetic.
- **kurtosis_concentration is the reason returns.py exists.** Excess kurtosis
  is a fourth moment, so one 12-sigma observation contributes 20,736 where an
  average day contributes 1. A 2,500-day sample can post excess kurtosis above
  30 on the strength of two observations. The share of the fourth moment held
  by the top k observations separates a genuinely fat-tailed series (share
  spread wide) from a bad dividend adjustment (share above ~85%). Measured
  over 40 samples of 2,512 draws: t(3) 59%, t(5) 28%, t(6) 17%, two planted
  spikes 89%. A high share also flags distributions whose population kurtosis
  does not exist (t with df <= 4), where the statistic is simply not estimable.
- **Never use t(3) as a "fat tails" fixture.** Its population kurtosis is
  infinite, so the sample statistic never converges and concentration is as
  high as a bad print's. t(5) or t(6) have finite kurtosis and behave.
- **Jarque-Bera is tested by its rejection RATE, not one sample.** A correctly
  calibrated 5% test rejects 5% of true nulls, so asserting that one normal
  draw passes is a flaky test by construction. The suite runs 400 samples and
  checks the rate lands near 5%.
- **audit_prices before trusting any moment.** `reversal` is the strongest
  flag: a real crash is not undone the next day, a mis-adjusted price is,
  almost exactly. `jump` and `sigma` alone fire on genuine crisis days too.
- **The price cache records its vintage.** Yahoo restates history
  retroactively, so the same request on two dates returns different numbers.
  Keying a cache only on the date range serves a stale vintage forever without
  saying so — the FINA4472 notebook has exactly this bug. `max_age_hours`
  defaults to 24 and the cache key separates `auto_adjust` settings, because
  adjusted and quoted prices must never share an entry.
