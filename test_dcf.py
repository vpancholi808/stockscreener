"""Validate the DCF: independently recompute every number and check the
sensitivity grid is monotonic in the right directions."""
import sys
sys.path.insert(0, "/home/claude/finmodel")

import numpy as np
import pandas as pd

from finmodel import ThreeStatementModel, DcfConfig, DcfModel, run_dcf, sensitivity
from finmodel.dcf import _fcff_from_projection


def dur(s): return {"units": {"USD": [{"val": v, "fy": y, "fp": "FY", "form": "10-K",
        "end": f"{y}-12-31", "start": f"{y}-01-01"} for y, v in s.items()]}}
def inst(s): return {"units": {"USD": [{"val": v, "fy": y, "fp": "FY", "form": "10-K",
        "end": f"{y}-12-31"} for y, v in s.items()]}}
def shr(s): return {"units": {"shares": [{"val": v, "fy": y, "fp": "FY", "form": "10-K",
        "end": f"{y}-12-31", "start": f"{y}-01-01"} for y, v in s.items()]}}

ys = [2022, 2023, 2024]
facts = {"entityName": "DCF Test Corp", "facts": {"us-gaap": {
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
    "ShortTermInvestments": inst(dict(zip(ys, [5e9, 6e9, 7e9]))),
    "AssetsCurrent": inst(dict(zip(ys, [45e9, 52e9, 60e9]))),
    "LiabilitiesCurrent": inst(dict(zip(ys, [25e9, 27e9, 29e9]))),
    "LongTermDebtNoncurrent": inst(dict(zip(ys, [18e9, 17e9, 16e9]))),
    "StockholdersEquity": inst(dict(zip(ys, [45e9, 58e9, 70e9]))),
    "Assets": inst(dict(zip(ys, [95e9, 110e9, 125e9]))),
    "PropertyPlantAndEquipmentNet": inst(dict(zip(ys, [30e9, 31e9, 33e9]))),
}}}

tsm = ThreeStatementModel.from_facts(facts, ticker="DCFT", n_years=3)
WACC = 0.10
cfg = DcfConfig(terminal_growth=0.025, mid_year=True)
res = run_dcf(tsm, WACC, cfg, current_price=200.0)

print("=" * 60)
print(res.summary())
print("\n--- FCFF schedule (USD bn) ---")
print((res.table[["fcff", "discount_factor", "pv"]] / 1).assign(
    fcff_bn=(res.table["fcff"] / 1e9).round(2),
    pv_bn=(res.table["pv"] / 1e9).round(2))[["fcff_bn", "discount_factor", "pv_bn"]]
    .round(4).to_string())

# ---- Independent recomputation ----
tax = tsm.projection.assumptions.tax_rate
fcff = _fcff_from_projection(tsm.projection, tax).to_numpy()
n = len(fcff)
per = np.arange(1, n + 1) - 0.5            # mid-year
disc = 1 / (1 + WACC) ** per
pv_exp = (fcff * disc).sum()
tv = fcff[-1] * (1 + 0.025) / (WACC - 0.025)
pv_tv = tv / (1 + WACC) ** (n - 0.5)
ev = pv_exp + pv_tv
net_debt = 16e9 - (32e9 + 7e9)            # LTD - (cash + ST investments)
eq = ev - net_debt
vps = eq / 3.9e9

print("\n--- Independent recomputation ---")
checks = {
    "pv_explicit": (pv_exp, res.pv_explicit),
    "pv_terminal": (pv_tv, res.pv_terminal),
    "enterprise_value": (ev, res.enterprise_value),
    "net_debt": (net_debt, res.net_debt),
    "value_per_share": (vps, res.value_per_share),
}
for k, (mine, theirs) in checks.items():
    ok = abs(mine - theirs) < max(1.0, abs(theirs) * 1e-9)
    print(f"  {k:18s} mine={mine:,.2f}  module={theirs:,.2f}  {'OK' if ok else 'MISMATCH'}")
    assert ok, k

# ---- Sensitivity monotonicity ----
print("\n--- Sensitivity (value/share): WACC down rows, g across cols ---")
grid = sensitivity(tsm, WACC, cfg, current_price=200.0)
print(grid.to_string())
vals = grid.to_numpy(dtype=float)
# value should DECREASE down rows (rising WACC) and INCREASE across cols (rising g)
assert np.all(np.diff(vals, axis=0) < 0), "not monotonic in WACC"
assert np.all(np.diff(vals, axis=1) > 0), "not monotonic in growth"
print("  monotonic in WACC (down) and growth (up)  OK")
print("=" * 60)
print("ALL DCF CHECKS PASSED")
