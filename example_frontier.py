#!/usr/bin/env python3
"""The efficient frontier, the CAL and the tangency portfolio, end to end.

    python example_frontier.py                 # built-in four-asset example
    python example_frontier.py AAPL MSFT NVDA XOM JNJ JPM

With no tickers it runs on a fixed four-asset covariance, so it needs no
network and prints the same numbers every time. With tickers it pulls prices
and does the same thing on real data.

What it shows, in the order the theory builds it:

  1. the frontier geometry (A, B, C, D) and the global minimum-variance
     portfolio, which needs no expected returns at all;
  2. the efficient branch only — the lower branch is dominated and excluded;
  3. the tangency portfolio and the CAL, which beats every frontier portfolio
     except the one point it touches;
  4. the complete portfolio, where risk aversion sets the split between the
     risk-free asset and the tangency portfolio and nothing else.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from finmodel.quant import MarkowitzOptimizer, NoTangencyPortfolio, Universe

RF = 0.02

ASSETS = ["A", "B", "C", "D"]
MU = pd.Series([0.04, 0.08, 0.12, 0.10], index=ASSETS)
SD = np.array([0.10, 0.18, 0.30, 0.22])
CORR = np.array([[1.00, 0.30, 0.20, 0.25],
                 [0.30, 1.00, 0.45, 0.35],
                 [0.20, 0.45, 1.00, 0.40],
                 [0.25, 0.35, 0.40, 1.00]])
COV = pd.DataFrame(np.outer(SD, SD) * CORR, index=ASSETS, columns=ASSETS)


def build(tickers: list[str]) -> MarkowitzOptimizer:
    if not tickers:
        print("Four-asset example (no network needed).\n")
        return MarkowitzOptimizer(MU, COV, rf=RF)
    print(f"Pulling {len(tickers)} tickers...\n")
    u = Universe.from_tickers(tickers, frequency="M", lookback_years=5)
    return MarkowitzOptimizer.from_universe(u, rf=RF, shrinkage=True)


def main(argv: list[str]) -> int:
    opt = build(argv[1:])
    geo = opt.geometry()

    print("=" * 66)
    print("1) FRONTIER GEOMETRY")
    print(geo)
    print(f"  condition number of the covariance: {opt.condition_number:,.1f}")
    gmv = opt.min_variance()
    print("\n  Global minimum-variance portfolio:")
    print("   ", str(gmv).replace("\n", "\n    "))

    print("\n" + "=" * 66)
    print("2) THE EFFICIENT BRANCH")
    fr = opt.efficient_frontier(points=9)
    print(fr.assign(**{c: fr[c].map("{:.2%}".format)
                       for c in ("return", "volatility")})
          [["return", "volatility", "sharpe", "efficient"]].to_string(index=False))
    full = opt.efficient_frontier(points=9, include_inefficient=True)
    dom = full[~full.efficient]
    if not dom.empty:
        worst = dom.iloc[0]
        mirror = geo.volatility_at(2 * geo.gmv_return - worst["return"])
        print(f"\n  The lower branch is excluded by default. Its worst point "
              f"sits at\n  {worst['volatility']:.2%} volatility for "
              f"{worst['return']:.2%} return, where the efficient branch "
              f"offers\n  {geo.gmv_return * 2 - worst['return']:.2%} at the "
              f"same {mirror:.2%} risk — strictly more return, identical "
              f"volatility.")

    print("\n" + "=" * 66)
    print("3) THE TANGENCY PORTFOLIO AND THE CAL")
    try:
        cal = opt.capital_allocation_line()
    except NoTangencyPortfolio as exc:
        print(f"  {exc}")
        return 0
    print("  Tangency:")
    print("   ", str(cal.tangency).replace("\n", "\n    "))
    print(f"\n  CAL: E[r] = {opt.rf:.2%} + {cal.slope:.4f} x sigma")
    print(f"  Best Sharpe on the frontier is the tangency Sharpe, "
          f"{cal.slope:.4f};\n  the GMV portfolio manages only {gmv.sharpe:.4f}.")
    print(f"\n  At the GMV portfolio's {gmv.volatility:.2%} volatility the CAL "
          f"returns\n  {cal.return_at(gmv.volatility):.2%} against the "
          f"frontier's {gmv.expected_return:.2%} — "
          f"{cal.return_at(gmv.volatility) - gmv.expected_return:+.2%} for "
          f"the same risk.")

    print("\n" + "=" * 66)
    print("4) THE COMPLETE PORTFOLIO")
    print("  Risk aversion sets the split, and nothing else: every investor")
    print("  holds the same risky portfolio (Tobin's two-fund separation).\n")
    rows = []
    for a in (1.5, 2.0, 3.0, 5.0, 8.0):
        cp = cal.complete_portfolio(risk_aversion=a)
        rows.append({"risk aversion": a,
                     "in tangency": f"{cp.risky_weight:.1%}",
                     "at rf": f"{cp.risk_free_weight:.1%}",
                     "E[r]": f"{cp.expected_return:.2%}",
                     "vol": f"{cp.volatility:.2%}",
                     "Sharpe": f"{cp.sharpe:.3f}"})
    print(pd.DataFrame(rows).to_string(index=False))
    print("\n  The Sharpe ratio is identical down the column: moving along the")
    print("  CAL changes how much risk you take, never how well you are paid")
    print("  for it.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
