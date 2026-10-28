# finmodel — equity analysis toolkit

Foundation layer: an automated **3-statement model** that pulls a US public
company's fundamentals from **SEC EDGAR** (free), builds linked historical
financials, projects them forward, and grades the company on a tunable
financial-health scorecard.

This is steps 1 + 2 of the pipeline:

```
FUNDAMENTAL:  3-statement (built) → DCF (built) → comps → ratio analysis
MARKET/RISK:  CAPM/beta (built) → statement factors (built) → Markowitz (built)
              → VaR/stress (built) → options/Greeks (built)
STOCHASTIC:   Brownian/GBM/OU/jump-diffusion (built) → Vasicek/CIR rates (built)
              → Merton credit (built) → Heston stochastic-vol MC (built)
STAGED NEXT:  Fama-French → GARCH → pairs/cointegration → ML → execution
```

Every later layer is designed to consume `model.financials` and
`model.projection` (e.g. the DCF discounts `projection.cash_flow.loc['free_cash_flow']`).

## Install

```bash
pip install -r requirements.txt
```

## Quick start

SEC requires a descriptive User-Agent (your name + email) or it throttles you.

```python
from finmodel import ThreeStatementModel

m = ThreeStatementModel.from_ticker("AAPL", user_agent="Jane Doe jane@firm.com")

print(m.summary())            # revenue / EBITDA / FCF / net debt + health score
print(m.scorecard_frame())    # every metric, threshold, pass/fail
print(m.financials.income_statement)   # historical IS (annual)
print(m.projection.cash_flow)          # projected CF incl. free_cash_flow
```

Set the User-Agent once via environment variable instead:

```bash
export EDGAR_USER_AGENT="Jane Doe jane@firm.com"
```

## Tuning the scorecard

Thresholds are how you encode your strategy. Tighten leverage, demand higher FCF:

```python
from finmodel import ThreeStatementModel, ScorecardConfig

cfg = ScorecardConfig(
    max_net_debt_to_ebitda=2.0,   # stricter than the 3.0 default
    min_fcf_margin=0.10,
    min_roic=0.12,
)
m = ThreeStatementModel.from_ticker("MSFT", config=cfg)
```

## Overriding projection assumptions

By default every driver is inferred from history (CAGR, average margins,
effective tax rate, payout). Override any of them:

```python
from finmodel import ThreeStatementModel, Assumptions

a = Assumptions(years=5, revenue_growth=0.08, gross_margin=0.45, tax_rate=0.21)
m = ThreeStatementModel.from_ticker("NVDA", assumptions=a)
```

## Metrics computed

| Group | Metrics |
|---|---|
| Leverage/solvency | Debt/Equity, Net Debt/EBITDA, Interest coverage |
| Liquidity | Current ratio, Quick ratio |
| Cash flow quality | FCF margin, Cash conversion (OCF/NI), Positive FCF |
| Profitability | Gross margin, Operating margin, ROE, ROIC |

The projected balance sheet balances by construction (cash from the cash-flow
statement reconciles with retained earnings and debt). Check it yourself with
`m.projection.balance_check` (values ~0).

## Data foundation: caching, point-in-time, quality

The base data layer does three things beyond a plain pull:

**Caching (on by default).** Each company's data is cached on disk, so repeat
runs are instant and reproducible and you can work offline.

```python
m = ThreeStatementModel.from_ticker("AAPL")   # first call fetches + caches
m = ThreeStatementModel.from_ticker("AAPL")   # subsequent calls hit the cache
# manage it directly:
from finmodel import DataStore
store = DataStore()
store.cached_ciks(); store.get_facts("MSFT", force_refresh=True); store.clear()
```

**Point-in-time (look-ahead-free).** Pass `as_of` to see only data that was
publicly filed on or before that date — restatements included only once they
were public. This is essential for honest backtesting; without it, historical
signals "know" numbers before they existed.

```python
m = ThreeStatementModel.from_ticker("AAPL", as_of="2022-06-30")
```

**Data quality.** Every model carries a quality report that flags coverage gaps,
missing key items, and whether the historical balance sheet articulates
(A = L + E). A missing debt tag reads as zero debt, so this catches it.

```python
print(m.quality_summary())   # grade + coverage + flags
```

Offline from a saved file still works via `ThreeStatementModel.from_file(path)`.

## Caveats (read before trusting a number)

- **US filers only.** EDGAR XBRL covers SEC registrants. Foreign issuers
  filing 20-F have partial coverage.
- **Tag coverage.** If a company uses an unusual XBRL tag, a line item may come
  back NaN. Add the tag to `finmodel/data/concepts.py` — that's the one place to
  extend mappings. Missing debt tags are treated as zero debt, so verify
  leverage metrics on unfamiliar names.
- **Projections are simplified.** Single revenue growth rate, margins held at
  history, interest held flat, working capital as % of revenue. Good for
  screening; refine drivers before using a DCF for a real position.
- Not investment advice. This is analytical tooling; you make the decisions.

## CAPM / beta and WACC

Beta needs return history, which EDGAR doesn't carry — so this layer adds a
**price feed** (yfinance by default, swappable). It computes beta, cost of
equity, and a WACC built from your 3-statement model's capital structure.

```python
from finmodel import ThreeStatementModel, CapmModel, CapmConfig

tsm = ThreeStatementModel.from_ticker("AAPL", user_agent="You you@email.com")

cfg = CapmConfig(
    risk_free_rate=0.042,        # SET to the current 10Y Treasury yield
    equity_risk_premium=0.05,    # your assumed market premium
    frequency="M", lookback_years=5,
)
capm = CapmModel.from_ticker("AAPL", config=cfg, three_statement=tsm)

print(capm.beta)              # raw + Blume-adjusted beta, R², alpha, n
print(f"ke = {capm.cost_of_equity:.2%}")
print(capm.wacc)              # WACC, weights, after-tax cost of debt
```

Beta methodology: 5y monthly total returns, regressed as **excess** returns
over the risk-free rate. Raw beta is the OLS slope; **adjusted beta** applies
the Blume shrinkage `0.67*raw + 0.33` toward 1.0 (what Bloomberg reports) and is
used for cost of equity. WACC pulls debt and interest from the balance sheet
(cost of debt ≈ interest/debt, after-tax), equity value from price × diluted
shares (falls back to book equity), and the tax rate from the model's
assumptions.

Two inputs you should set deliberately: `risk_free_rate` (current 10Y) and
`equity_risk_premium` — they move cost of equity directly. The cost-of-debt
proxy (interest/debt) is rough; override it for leveraged names.

### Swapping the price feed

```python
from finmodel import CapmModel, FramePriceFeed
import pandas as pd
# prices: a dict of {ticker: pd.Series of adjusted close indexed by date}
feed = FramePriceFeed({"AAPL": aapl_px, "^GSPC": spx_px})
capm = CapmModel.from_ticker("AAPL", feed=feed)
```

## DCF valuation

Discounts unlevered free cash flow at the WACC from the CAPM layer, adds a
terminal value, and bridges to an intrinsic value per share you can compare to
the market price. This is the first layer that produces a buy/sell signal.

```python
from finmodel import ThreeStatementModel, CapmModel, CapmConfig, DcfModel, DcfConfig

tsm  = ThreeStatementModel.from_ticker("AAPL", user_agent="You you@email.com")
capm = CapmModel.from_ticker("AAPL", config=CapmConfig(risk_free_rate=0.042),
                             three_statement=tsm)

dcf = DcfModel.from_models(tsm, capm=capm,
                           config=DcfConfig(terminal_growth=0.025))
print(dcf.result.summary())        # value/share, upside vs price, EV bridge
print(dcf.result.table)            # per-year FCFF, discount factor, PV
print(dcf.sensitivity_table)       # value/share across WACC x terminal growth
```

Supply WACC manually instead of using CAPM:

```python
dcf = DcfModel.from_models(tsm, wacc=0.09, current_price=185.0)
```

Method notes:
- **Unlevered FCFF** = EBIT×(1−tax) + D&A − CapEx − ΔNWC, discounted at WACC.
  (The projection's `free_cash_flow` line is *levered*; the DCF deliberately
  recomputes the unlevered figure so the financing benefit isn't double-counted.)
- **Terminal value**: Gordon growth perpetuity by default (`terminal_growth`
  must be below WACC), or set `terminal_method="exit_multiple"` with
  `exit_ev_ebitda`.
- **Mid-year convention** is on by default.
- Watch **terminal share of EV** in the summary — above ~75% means the value
  rests heavily on terminal assumptions, so lean on the sensitivity table.
- Net cash (negative net debt) correctly raises equity value above EV.

## Quant layer (`finmodel.quant`)

Portfolio construction, risk, statement factors, and options — all sitting on a
shared `Universe` (multi-ticker returns panel from the same price feed).

```python
from finmodel.quant import (Universe, MarkowitzOptimizer, risk_report,
                            stress_test, bs_price, bs_greeks, FactorModel)

# --- Markowitz: the frontier ---
u = Universe.from_tickers(["AAPL","MSFT","NVDA","XOM","JNJ"], lookback_years=5)
opt = MarkowitzOptimizer.from_universe(u, rf=0.042, shrinkage=True)
print(opt.summary())                    # A/B/C/D, GMV, tangency, CAL slope
print(opt.geometry())                   # the four scalars the frontier reduces to
print(opt.min_variance())               # needs no expected returns at all
print(opt.optimize("min_variance", long_only=True))

frontier = opt.efficient_frontier(points=25)                  # efficient branch only
both    = opt.efficient_frontier(points=25, include_inefficient=True)  # for plotting

# --- The CAL: adding a risk-free asset ---
cal = opt.capital_allocation_line()     # raises NoTangencyPortfolio if rf >= GMV return
print(cal.tangency)                     # the one risky mix everyone holds
print(cal.slope)                        # == tangency Sharpe == max attainable
print(cal.return_at(0.15))              # rf + slope * sigma

# Risk aversion sets the split and nothing else (two-fund separation).
print(opt.complete_portfolio(risk_aversion=3.0))
print(opt.complete_portfolio(target_volatility=0.10))

# --- Risk: VaR / CVaR / stress ---
from finmodel.quant import portfolio_returns
port = portfolio_returns(u.returns, opt.max_sharpe().weights)
print(risk_report(port, confidence=0.95))   # hist/param/Cornish-Fisher VaR, CVaR, DD
print(stress_test(opt.max_sharpe().weights,
                  {"-10% market": {t: -0.10 for t in u.tickers}}))

# --- Options & Greeks ---
print(bs_price(100, 105, 0.05, 0.22, 0.5, "call"))
print(bs_greeks(100, 105, 0.05, 0.22, 0.5, "call"))   # delta, gamma, vega, theta, rho

# --- Statement factor scoring across a universe ---
models = {t: ThreeStatementModel.from_ticker(t) for t in ["AAPL","MSFT","NVDA"]}
caps   = {t: CapmModel.from_ticker(t).last_price * shares[t] for t in models}
fm = FactorModel.from_models(models, caps)
print(fm.composites)        # value/quality/growth/profitability + composite + rank
print(fm.long_short(n=1))   # top (long) / bottom (short) names
```

Method notes worth knowing:
- **Covariance** defaults to Ledoit-Wolf shrinkage (better conditioned than raw
  sample covariance when names ≈ observations). Annualized via the universe
  frequency.
- **Markowitz** has an exact analytic path (shorts allowed) used to validate the
  SLSQP constrained path (long-only / custom bounds). The whole frontier reduces
  to four scalars, A = 1'Σ⁻¹1, B = 1'Σ⁻¹μ, C = μ'Σ⁻¹μ and D = AC − B², exposed
  as `geometry()`: GMV sits at return B/A and variance 1/A, and
  σ²(r) = (Ar² − 2Br + C)/D traces the rest.
- **`efficient_frontier` returns the efficient branch only.** Sampling target
  returns from min(μ) upward puts half the points on the lower branch, where
  every point is dominated by one directly above it at the same volatility.
  `include_inefficient=True` returns the full hyperbola with an `efficient`
  flag, for plotting.
- **The tangency portfolio does not always exist.** w ∝ Σ⁻¹(μ − rf·1) is
  normalised by κ = 1'Σ⁻¹(μ − rf·1) = B − rf·A, which is positive exactly when
  rf is below the GMV return. Above it, dividing by a negative κ flips every
  weight and returns the *minimum*-Sharpe portfolio as the maximum, so
  `tangency()` raises `NoTangencyPortfolio` instead. The Sharpe ratio there
  climbs toward the frontier's asymptotic slope √(D/A) without reaching it —
  a supremum, not a portfolio. `tangency(long_only=True)` searches the simplex,
  which is compact, so a maximum always exists.
- **Max attainable Sharpe = √((μ−rf·1)'Σ⁻¹(μ−rf·1))**, equivalently
  √(C − 2·rf·B + rf²·A). The test suite asserts the solved tangency portfolio
  matches this closed form.
- **The CAL beats every frontier portfolio but one.** `capital_allocation_line()`
  gives E[r] = rf + S_t·σ, a straight line from (0, rf) whose slope is the
  tangency Sharpe ratio. It touches the frontier at exactly one point; the test
  suite verifies the gap bottoms out at the tangency volatility. The same line
  is the Capital Market Line when the tangency portfolio is the market
  portfolio.
- **Complete portfolios** split between rf and the tangency portfolio:
  y* = (E[r_t] − rf)/(A·σ_t²) maximises E[r] − (A/2)σ². Sharpe is constant along
  the line — y changes how much risk you carry, never how well you are paid for
  it.
- **VaR** is a positive loss fraction. Cornish-Fisher adjusts the Gaussian
  quantile for skew and fat tails — use it over parametric for real return data.
- **Greeks** units: vega per 1.00 vol (÷100 for per 1%), theta per year (÷365 for
  per day). Implied vol uses safeguarded Newton/bisection.
- **Factors** are z-scored cross-sectionally with sign normalized so higher =
  more attractive; the composite drives the eventual long/short book.

Validated against known answers in `test_quant.py` (BS textbook price + put-call
parity, analytic-vs-numeric optimizer agreement, parametric VaR = z·σ, factor
ranking).

## Stochastic processes, rates & credit (`finmodel.quant`)

The Itô/Bachelier simulation engine plus the models that build on it.

```python
from finmodel.quant import (gbm, ornstein_uhlenbeck, jump_diffusion,
                            Vasicek, CIR, solve_merton, heston_mc_price)

# Simulate processes (shape: n_paths x n_steps+1)
paths = gbm(S0=100, mu=0.08, sigma=0.2, T=1.0, n_steps=252, n_paths=10_000)

# Interest rates: analytic zero-coupon bond + term structure
v = Vasicek(kappa=0.5, theta=0.04, sigma=0.02, r0=0.03)
print(v.zcb_price(5.0), v.yield_to(5.0))     # closed-form
c = CIR(kappa=0.5, theta=0.04, sigma=0.08, r0=0.03)
print(c.feller_satisfied)                     # 2*kappa*theta >= sigma^2 -> r stays > 0

# Merton credit: equity as a call on firm assets (ties to the balance sheet)
m = solve_merton(equity_value=30e9, equity_vol=0.40, debt_face=70e9, r=0.03)
print(m)   # asset value/vol, distance-to-default, PD, credit spread (bps)

# Heston stochastic-vol option price by Monte Carlo
print(heston_mc_price(100, 100, 0.03, v0=0.04, kappa=2.0, theta=0.04,
                      xi=0.3, rho=-0.7, T=1.0, option="call"))
```

Integrating Merton with your fundamentals: pass a built `ThreeStatementModel`
and the market equity value/vol to `merton_from_model(...)` — it pulls total debt
off the balance sheet as the default barrier.

Method notes:
- Simulators use exact schemes where available (log-Euler GBM, exact OU);
  Heston/CIR use full-truncation Euler to keep variance/rates non-negative.
- Vasicek and CIR bond prices are the affine closed forms, validated against the
  Monte-Carlo average of exp(−∫r ds).
- Merton recovers unobservable asset value and asset vol by solving the two
  structural equations simultaneously; PD is risk-neutral N(−distance-to-default).

Validated in `test_stoch.py`: GBM mean = S₀e^{μT}, OU reverts to θ, analytic =
MC bond prices, Merton residuals ≈ 0 with PD monotonic in vol, and Heston → BS
in the zero-vol-of-vol limit.

## Screening & dashboard

Run the whole engine across a watchlist, then showcase it as a visual dashboard.

```python
from finmodel import Screener, ScreenConfig, dashboard_html

# thresholds define the screen; a company passes only if it meets all of them
cfg = ScreenConfig(min_health_score=0.6, min_fcf_margin=0.05,
                   max_net_debt_to_ebitda=3.0, min_roic=0.08,
                   min_operating_margin=0.10, min_revenue_growth=0.0)

scr = Screener.from_tickers(["AAPL","MSFT","NVDA","XOM","JNJ","JPM"], config=cfg)
print(scr.summary())          # "Screened 6 companies — N pass the screen"
print(scr.table())            # ranked metrics table (also .passing() for filtered)

open("screen.html","w").write(dashboard_html(scr, title="My Watchlist"))
```

The dashboard is a single self-contained HTML file (opens offline, no libraries):
a sortable ranking table with a **health meter** and pass/fail-colored cells per
company, plus detail cards showing key values and **multi-year sparklines** that
track each metric over time. `Screener.from_models(...)` works offline from
already-built models; pass `market_caps=` to add valuation multiples (P/E,
EV/EBITDA, FCF yield).

## Staged next

The remaining models need heavier dependencies and careful methodology, so
they're a separate batch: **Fama-French** (size/value/profitability/investment/
momentum regressions vs the Ken French data library), **GARCH(1,1)** volatility
forecasting, **pairs trading / cointegration** (Engle-Granger), **machine
learning** (XGBoost / random forest / NN with walk-forward and leakage controls),
and **execution / market impact** (Almgren-Chriss). These plug into the same
`Universe` and `FactorModel` so signals flow into the long/short portfolio.
