#!/usr/bin/env python3
"""Download prices, audit them, describe the distribution, measure performance.

    python example_performance.py                       # AAPL MSFT NVDA
    python example_performance.py MCD PFE AAPL HD AMZN --start 2016-01-01
    python example_performance.py AAPL --capital 10000 --deposit 2024-07-01:4000

This is the FINA4472 return-statistics and multi-asset notebooks run on the
project's own stack, with the two things a notebook leaves out:

  * the prices are audited before any moment is computed, because one bad
    dividend adjustment can carry an entire kurtosis figure, and
  * both annualizations are reported, because mean x 252 is not a compound
    growth rate.

Needs network access to Yahoo Finance. The download is cached for a day under
.finmodel_prices/ with the vintage recorded, so re-runs are instant and you can
always see how old the numbers are.
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from finmodel.data.yahoo import audit_prices, coverage, download_prices
from finmodel.performance import performance_report
from finmodel.returns import (aggregation_kurtosis, compute_returns,
                              return_statistics)


def parse_flow(text: str) -> tuple[str, float]:
    date, _, amount = text.partition(":")
    if not amount:
        raise argparse.ArgumentTypeError(
            "deposits look like 2024-07-01:4000 (negative to withdraw)")
    return date.strip(), float(amount)


def rule(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tickers", nargs="*", default=["AAPL", "MSFT", "NVDA"])
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default=None, help="exclusive, as in yfinance")
    ap.add_argument("--capital", type=float, default=10_000.0)
    ap.add_argument("--deposit", action="append", type=parse_flow, default=None,
                    help="DATE:AMOUNT, repeatable; negative withdraws")
    ap.add_argument("--rebalance", default=None,
                    help="ME, QE, YE ... (default: buy and hold)")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    tickers = [t.upper() for t in (args.tickers or ["AAPL", "MSFT", "NVDA"])]

    rule("1. DOWNLOAD")
    try:
        got = download_prices(tickers, start=args.start, end=args.end,
                              use_cache=not args.no_cache)
    except (ImportError, RuntimeError) as exc:
        print("  %s" % exc)
        print("\n  This script needs live access to Yahoo Finance.")
        return 1
    print(got.summary())

    prices = got.prices
    if prices.empty:
        print("  no usable price history")
        return 1

    print("\n  Coverage:")
    print(coverage(prices).to_string())

    rule("2. AUDIT — look before trusting any moment")
    flags = audit_prices(prices)
    if flags.empty:
        print("  Nothing flagged. Moments computed below rest on clean data.")
    else:
        print(flags.to_string(index=False))
        print("\n  'reversal' is the one to check first: a real crash is not")
        print("  undone the next day, a mis-adjusted price is. Open those")
        print("  dates against the market record before reporting kurtosis.")

    rule("3. RETURN DISTRIBUTION")
    returns = compute_returns(prices)
    report = return_statistics(returns)
    print(report.summary())

    print("\n  Kurtosis concentration — what share of the fourth moment comes")
    print("  from the two most extreme days:")
    print(report.concentration[["excess_kurtosis", "top_k_share",
                                "dates"]].to_string())

    print("\n  Kurtosis by horizon (it falls as returns are aggregated —")
    print("  fat tails are a property of the horizon, not just the asset):")
    print(aggregation_kurtosis(prices, ("D", "W", "ME")).round(2).to_string())

    rule("4. PERFORMANCE — time-weighted vs money-weighted")
    flows = dict(args.deposit or [])
    if flows:
        print("  External cash flows: %s\n" % ", ".join(
            f"{d} {a:+,.0f}" for d, a in flows.items()))
    else:
        print("  No external cash flows, so TWR and MWR must agree.\n")

    try:
        performance = performance_report(
            prices, args.capital, flows, rebalance=args.rebalance)
    except ValueError as exc:
        print("  %s" % exc)
        return 1
    print(performance.summary())

    if flows:
        print("\n  Investor cash flows:")
        print(performance.flows_frame.to_string(index=False))
        print("\n  TWR judges the strategy and ignores when money arrived.")
        print("  MWR judges the investor's experience and does not.")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
