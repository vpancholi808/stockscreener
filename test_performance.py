"""Time-weighted and money-weighted return — analytic identities.

Every check here is either a closed-form rate or an identity that must hold
regardless of the data, so a failure points at the code and not at the sample.
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from finmodel.performance import (align_flows, contribution_to_return,
                                  normalize_weights, performance_report,
                                  sign_changes, simulate_portfolio,
                                  weighted_returns, xirr, xnpv)

fails = []


def check(name, cond, detail=""):
    print("  [%s] %s%s" % ("OK" if cond else "FAIL", name,
                           (" — " + detail) if detail else ""))
    if not cond:
        fails.append(name)


DAY = pd.Timedelta(days=1)
T0 = pd.Timestamp("2024-01-01")

print("=" * 64)
print("1) XIRR AGAINST CLOSED FORM")
print("=" * 64)

r = xirr([(T0, -1000.0), (T0 + 365 * DAY, 1100.0)])
check("-1000 -> +1100 over 365 days is exactly 10%", abs(r - 0.10) < 1e-8,
      "%.10f" % r)

r = xirr([(T0, -1000.0), (T0 + 730 * DAY, 1210.0)])
check("-1000 -> +1210 over 730 days is exactly 10%", abs(r - 0.10) < 1e-8,
      "%.10f" % r)

r = xirr([(T0, -1000.0), (T0 + 365 * DAY, 500.0)])
check("halving over a year is -50%", abs(r + 0.50) < 1e-8, "%.10f" % r)

flows = [(T0, -5000.0), (T0 + 180 * DAY, -2000.0), (T0 + 365 * DAY, 7700.0)]
solved = xirr(flows)
check("xnpv at the solved rate is zero", abs(xnpv(solved, flows)) < 1e-6,
      "npv=%.2e at r=%.6f" % (xnpv(solved, flows), solved))

check("all-negative flows give NaN, not a number",
      np.isnan(xirr([(T0, -100.0), (T0 + 365 * DAY, -100.0)])))
check("a single flow gives NaN", np.isnan(xirr([(T0, -100.0)])))

check("one sign change counted", sign_changes(flows) == 1)
check("three sign changes counted",
      sign_changes([(T0, -1.0), (T0 + DAY, 2.0), (T0 + 2 * DAY, -3.0),
                    (T0 + 3 * DAY, 4.0)]) == 3)

print()
print("=" * 64)
print("2) TWR AND MWR IDENTITIES")
print("=" * 64)

idx = pd.date_range("2024-01-01", periods=253, freq="B")
rng = np.random.default_rng(11)
prices = pd.DataFrame({
    "A": 100 * np.exp(np.cumsum(rng.normal(0.0006, 0.012, len(idx)))),
    "B": 50 * np.exp(np.cumsum(rng.normal(0.0002, 0.020, len(idx)))),
    "C": 200 * np.exp(np.cumsum(rng.normal(0.0004, 0.008, len(idx)))),
}, index=idx)

no_flows = performance_report(prices, 10_000.0)
check("with no external flows, MWR equals annualized TWR",
      abs(no_flows.mwr - no_flows.twr_annualized) < 1e-8,
      "TWR_ann %.8f vs MWR %.8f" % (no_flows.twr_annualized, no_flows.mwr))

# The defining property of TWR: it measures the strategy, not the deposits.
base = no_flows.twr
spreads = []
for amount in (1_000.0, 50_000.0, 500_000.0, -3_000.0):
    got = performance_report(prices, 10_000.0,
                             {"2024-07-01": amount}).twr
    spreads.append(abs(got - base))
check("TWR is invariant to the size of a pro-rata flow",
      max(spreads) < 1e-9, "max deviation %.2e" % max(spreads))

# ...but only pro-rata. Fixed-weight contributions change the asset mix, and
# therefore every return after the flow. That is real, and documented.
drifted = performance_report(prices, 10_000.0, {"2024-07-01": 50_000.0},
                             contribution_weights="equal").twr
check("fixed-weight contributions do move TWR (they change the mix)",
      abs(drifted - base) > 1e-6, "deviation %.2e" % abs(drifted - base))

single = pd.DataFrame({"A": prices["A"]})
total = float(prices["A"].iloc[-1] / prices["A"].iloc[0] - 1)
one = performance_report(single, 10_000.0)
check("single asset: TWR equals the price return",
      abs(one.twr - total) < 1e-10, "%.10f vs %.10f" % (one.twr, total))

print()
print("=" * 64)
print("3) CASH-FLOW TIMING MOVES MWR IN THE RIGHT DIRECTION")
print("=" * 64)

flat_then_up = np.concatenate([np.full(126, 0.0001), np.full(127, 0.004)])
up_then_flat = np.concatenate([np.full(126, 0.004), np.full(127, 0.0001)])
rally = pd.DataFrame({"A": 100 * np.exp(np.cumsum(flat_then_up))}, index=idx)
fade = pd.DataFrame({"A": 100 * np.exp(np.cumsum(up_then_flat))}, index=idx)

good = performance_report(rally, 10_000.0, {"2024-06-28": 50_000.0})
check("deposit before a rally: MWR beats TWR",
      good.mwr > good.twr_annualized,
      "MWR %.4f vs TWR %.4f" % (good.mwr, good.twr_annualized))

bad = performance_report(fade, 10_000.0, {"2024-06-28": 50_000.0})
check("deposit after the gains: MWR trails TWR",
      bad.mwr < bad.twr_annualized,
      "MWR %.4f vs TWR %.4f" % (bad.mwr, bad.twr_annualized))

check("TWR itself is unmoved by which of the two it was",
      abs(performance_report(rally, 10_000.0, {"2024-06-28": 50_000.0}).twr
          - performance_report(rally, 10_000.0).twr) < 1e-9)

print()
print("=" * 64)
print("4) REBALANCING IDENTITY")
print("=" * 64)

# Rebalanced to fixed weights every period, the portfolio return must be the
# weighted average of asset returns in each period, compounded.
asset_returns = prices.pct_change().dropna()
expected = float((1.0 + weighted_returns(asset_returns)).prod() - 1.0)
daily = performance_report(prices, 10_000.0, rebalance="D").twr
check("daily rebalancing reproduces the weighted-average return",
      abs(daily - expected) < 1e-9, "%.10f vs %.10f" % (daily, expected))

manual = asset_returns.to_numpy() @ np.full(3, 1 / 3)
check("weighted_returns equals the explicit dot product",
      np.allclose(weighted_returns(asset_returns).to_numpy(), manual))

shares = contribution_to_return(asset_returns)["share_of_total"]
check("contribution shares sum to 1", abs(shares.sum() - 1.0) < 1e-9,
      "%.10f" % shares.sum())

print()
print("=" * 64)
print("5) BAD INPUT IS REFUSED, NOT ABSORBED")
print("=" * 64)


def raises(fn, fragment):
    try:
        fn()
    except (ValueError, TypeError) as exc:
        return fragment in str(exc)
    return False


check("flow dated before the price history raises",
      raises(lambda: performance_report(prices, 10_000.0,
                                        {"2023-01-01": 1_000.0}), "precedes"))
check("flow dated after the price history raises",
      raises(lambda: performance_report(prices, 10_000.0,
                                        {"2030-01-01": 1_000.0}), "after"))
check("withdrawal larger than the portfolio raises",
      raises(lambda: performance_report(prices, 10_000.0,
                                        {"2024-02-01": -1e9}), "exceeds"))
check("NaN prices are refused rather than silently dropped",
      raises(lambda: simulate_portfolio(prices.mask(prices > 150), 10_000.0),
             "NaN"))
check("non-positive opening capital raises",
      raises(lambda: performance_report(prices, 0.0), "positive"))
check("wrong number of weights raises",
      raises(lambda: normalize_weights(3, [0.5, 0.5]), "expected 3"))
check("negative weights need allow_short",
      raises(lambda: normalize_weights(2, [-0.5, 1.5]), "allow_short"))
check("weights are normalised, not required to sum to 1",
      np.allclose(normalize_weights(2, [2.0, 2.0]), [0.5, 0.5]))

multi = performance_report(prices, 10_000.0,
                           {"2024-03-01": 5_000.0, "2024-06-03": -2_000.0,
                            "2024-09-02": 3_000.0})
check("multiple IRR roots are warned about",
      any("sign" in w for w in multi.warnings))

print()
print("=" * 64)
print("6) BOOKKEEPING")
print("=" * 64)

history = simulate_portfolio(prices, 10_000.0, {"2024-07-01": 4_000.0})
identity = (history["Value after flow"]
            - history["Value before flow"]
            - history["External flow"]).abs().max()
check("value after flow = value before flow + flow", identity < 1e-6,
      "max break %.2e" % identity)

linked = float((1.0 + history["Period return"].dropna()).prod() - 1.0)
check("cumulative TWR equals the linked period returns",
      abs(linked - history["Cumulative TWR"].iloc[-1]) < 1e-10)

report = performance_report(prices, 10_000.0, {"2024-07-01": 4_000.0})
check("net contributions = capital + deposits",
      abs(report.net_contributions - 14_000.0) < 1e-9)
check("dollar gain = ending value - net contributions",
      abs(report.dollar_gain
          - (report.ending_value - report.net_contributions)) < 1e-9)
check("summary renders", len(report.summary()) > 80)

aligned = align_flows(prices.index, {"2024-07-06": 1_000.0})   # a Saturday
check("a weekend flow lands on the next trading day",
      list(aligned)[0] == pd.Timestamp("2024-07-08"),
      str(list(aligned)[0].date()))
aligned = align_flows(prices.index, {"2024-07-01": 100.0, "2024-07-02": 50.0})
check("two flows on different days stay separate", len(aligned) == 2)
aligned = align_flows(prices.index, {"2024-07-06": 100.0, "2024-07-07": 50.0})
check("two flows snapping to the same day are summed",
      len(aligned) == 1 and abs(list(aligned.values())[0] - 150.0) < 1e-9)

print()
if fails:
    print("FAILED (%d): %s" % (len(fails), "; ".join(fails)))
    sys.exit(1)
print("test_performance: all checks passed")
