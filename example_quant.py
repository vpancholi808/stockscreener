"""Live quant example: pull a universe from yfinance, optimize, measure risk.

    python example_quant.py
"""
from finmodel.quant import (Universe, MarkowitzOptimizer, NoTangencyPortfolio,
                            risk_report, stress_test, portfolio_returns,
                            bs_greeks)

tickers = ["AAPL", "MSFT", "NVDA", "XOM", "JNJ", "JPM"]
u = Universe.from_tickers(tickers, frequency="M", lookback_years=5)

opt = MarkowitzOptimizer.from_universe(u, rf=0.042, shrinkage=True)
print(opt.summary(), "\n")

# A realistic risk-free rate is often above the global minimum-variance
# return of a six-stock universe over a bad five years, and then no tangency
# portfolio exists at all. Fall back to the long-only problem, whose feasible
# set is compact and therefore always has a maximum.
try:
    tan = opt.tangency()
    print("Tangency portfolio:\n", tan, "\n")
    print(opt.capital_allocation_line(), "\n")
    print("Complete portfolio at risk aversion 3:\n",
          opt.complete_portfolio(risk_aversion=3.0), "\n")
except NoTangencyPortfolio as exc:
    print(f"{exc}\n")
    tan = opt.tangency(long_only=True)
    print("Long-only best Sharpe instead:\n", tan, "\n")

print("Long-only min-variance:\n", opt.optimize("min_variance", long_only=True), "\n")

port = portfolio_returns(u.returns, tan.weights)
print(risk_report(port, confidence=0.95, periods_per_year=u.periods_per_year), "\n")
print(stress_test(tan.weights, {
    "-15% broad selloff": {t: -0.15 for t in u.tickers},
    "energy shock":       {"XOM": 0.10, "AAPL": -0.05, "NVDA": -0.08},
}))

print("\nSample option Greeks (S=100,K=105,22% vol,6m call):")
print(bs_greeks(100, 105, 0.042, 0.22, 0.5, "call"))
