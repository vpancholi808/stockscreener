"""Validate CAPM math offline: build returns with a KNOWN beta and confirm the
regression recovers it, then test WACC wired to a 3-statement model."""
import sys
sys.path.insert(0, "/home/claude/finmodel")

import numpy as np
import pandas as pd

from finmodel import (CapmModel, CapmConfig, FramePriceFeed, ThreeStatementModel,
                      compute_beta, to_returns)

rng = np.random.default_rng(42)

# 6 years of monthly dates.
dates = pd.date_range("2019-01-31", periods=72, freq="ME")

TRUE_BETA, TRUE_ALPHA_M = 1.35, 0.002  # monthly alpha
# Market monthly returns ~ N(0.7%, 4%).
mkt_ret = rng.normal(0.007, 0.04, len(dates))
# Stock returns generated from CAPM + idiosyncratic noise.
noise = rng.normal(0, 0.02, len(dates))
stock_ret = TRUE_ALPHA_M + TRUE_BETA * mkt_ret + noise

# Build price series from returns (start at 100).
mkt_px = pd.Series(100 * np.cumprod(1 + mkt_ret), index=dates, name="^GSPC")
stock_px = pd.Series(50 * np.cumprod(1 + stock_ret), index=dates, name="TEST")

feed = FramePriceFeed({"TEST": stock_px, "^GSPC": mkt_px})

print("=" * 60)
print(f"TRUE beta = {TRUE_BETA}, TRUE monthly alpha = {TRUE_ALPHA_M}")
cfg = CapmConfig(risk_free_rate=0.042, equity_risk_premium=0.05, lookback_years=6)

# Direct regression check.
s = to_returns(stock_px, "M")
m = to_returns(mkt_px, "M")
b = compute_beta(s, m, rf_per_period=(1.042 ** (1/12) - 1), frequency="M")
print("RECOVERED:", b)
assert abs(b.raw_beta - TRUE_BETA) < 0.1, "beta not recovered!"
print(f"  -> raw beta within {abs(b.raw_beta - TRUE_BETA):.3f} of truth  OK")

# Full CapmModel via the feed.
cm = CapmModel.from_ticker("TEST", config=cfg, feed=feed)
print(f"\nCost of equity (CAPM, adj beta): {cm.cost_of_equity:.2%}")
print(f"Last price: {cm.last_price:.2f}")

# Now WACC integrated with a 3-statement model.
def dur(s): return {"units": {"USD": [{"val": v, "fy": y, "fp": "FY", "form": "10-K",
        "end": f"{y}-12-31", "start": f"{y}-01-01"} for y, v in s.items()]}}
def inst(s): return {"units": {"USD": [{"val": v, "fy": y, "fp": "FY", "form": "10-K",
        "end": f"{y}-12-31"} for y, v in s.items()]}}
def shr(s): return {"units": {"shares": [{"val": v, "fy": y, "fp": "FY", "form": "10-K",
        "end": f"{y}-12-31", "start": f"{y}-01-01"} for y, v in s.items()]}}
ys = [2022, 2023, 2024]
facts = {"entityName": "Test Corp", "facts": {"us-gaap": {
    "Revenues": dur(dict(zip(ys, [80e9, 95e9, 110e9]))),
    "CostOfRevenue": dur(dict(zip(ys, [40e9, 47e9, 54e9]))),
    "OperatingIncomeLoss": dur(dict(zip(ys, [22e9, 27e9, 31e9]))),
    "NetIncomeLoss": dur(dict(zip(ys, [16e9, 20e9, 23e9]))),
    "IncomeTaxExpenseBenefit": dur(dict(zip(ys, [4e9, 5e9, 6e9]))),
    "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest":
        dur(dict(zip(ys, [20e9, 25e9, 29e9]))),
    "InterestExpense": dur(dict(zip(ys, [0.9e9, 0.9e9, 1.0e9]))),
    "NetCashProvidedByUsedInOperatingActivities": dur(dict(zip(ys, [24e9, 29e9, 33e9]))),
    "PaymentsToAcquirePropertyPlantAndEquipment": dur(dict(zip(ys, [6e9, 7e9, 8e9]))),
    "DepreciationDepletionAndAmortization": dur(dict(zip(ys, [5e9, 5.5e9, 6e9]))),
    "WeightedAverageNumberOfDilutedSharesOutstanding": shr(dict(zip(ys, [4.0e9, 3.95e9, 3.9e9]))),
    "CashAndCashEquivalentsAtCarryingValue": inst(dict(zip(ys, [20e9, 26e9, 32e9]))),
    "AssetsCurrent": inst(dict(zip(ys, [45e9, 52e9, 60e9]))),
    "LiabilitiesCurrent": inst(dict(zip(ys, [25e9, 27e9, 29e9]))),
    "LongTermDebtNoncurrent": inst(dict(zip(ys, [18e9, 17e9, 16e9]))),
    "StockholdersEquity": inst(dict(zip(ys, [45e9, 58e9, 70e9]))),
    "Assets": inst(dict(zip(ys, [95e9, 110e9, 125e9]))),
    "PropertyPlantAndEquipmentNet": inst(dict(zip(ys, [30e9, 31e9, 33e9]))),
}}}
tsm = ThreeStatementModel.from_facts(facts, ticker="TEST", n_years=3)
cm2 = CapmModel.from_ticker("TEST", config=cfg, feed=feed, three_statement=tsm)
print("\n--- WACC integrated with 3-statement model ---")
print(cm2.wacc)
w = cm2.wacc
print(f"  market cap = price {cm2.last_price:.2f} x {3.9e9/1e9:.2f}B shares "
      f"= {w.equity_value/1e9:.1f}B")
print(f"  debt = {w.debt_value/1e9:.1f}B, pretax kd = {w.pretax_cost_of_debt:.2%}, "
      f"tax = {w.tax_rate:.1%}")
print("=" * 60)
print("ALL CAPM CHECKS PASSED")
