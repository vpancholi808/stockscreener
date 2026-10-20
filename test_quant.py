"""Known-answer validation for finmodel.quant: options, Markowitz, risk, factors."""
import sys
sys.path.insert(0, "/home/claude/finmodel")

import numpy as np
import pandas as pd

from finmodel import ThreeStatementModel
from finmodel.quant import (Universe, MarkowitzOptimizer, ledoit_wolf_cov,
                            risk_report, stress_test, bs_price, bs_greeks,
                            implied_vol, FactorModel)

ok = lambda c: "OK" if c else "FAIL"
fails = []
def check(name, cond):
    print(f"  [{ok(cond)}] {name}")
    if not cond: fails.append(name)

print("=" * 64)
print("1) BLACK-SCHOLES & GREEKS")
# Textbook value: S=K=100, r=5%, sigma=20%, T=1 -> call 10.4506, put 5.5735
c = bs_price(100, 100, 0.05, 0.20, 1.0, "call")
p = bs_price(100, 100, 0.05, 0.20, 1.0, "put")
check(f"call ~ 10.4506 (got {c:.4f})", abs(c - 10.4506) < 1e-3)
check(f"put ~ 5.5735 (got {p:.4f})", abs(p - 5.5735) < 1e-3)
# Put-call parity: C - P = S - K e^{-rT}
parity = c - p - (100 - 100 * np.exp(-0.05))
check(f"put-call parity (resid {parity:.2e})", abs(parity) < 1e-6)
g = bs_greeks(100, 100, 0.05, 0.20, 1.0, "call")
check(f"call delta in (0,1) (={g.delta:.3f})", 0 < g.delta < 1)
check(f"gamma>0 (={g.gamma:.4f})", g.gamma > 0)
check(f"vega>0 (={g.vega:.3f})", g.vega > 0)
iv = implied_vol(c, 100, 100, 0.05, 1.0, "call")
check(f"implied vol round-trips to 0.20 (got {iv:.4f})", abs(iv - 0.20) < 1e-4)

print("\n2) MARKOWITZ OPTIMIZATION")
assets = ["A", "B", "C"]
mu = pd.Series([0.10, 0.12, 0.08], index=assets)
cov = pd.DataFrame([[0.04, 0.01, 0.00],
                    [0.01, 0.09, 0.02],
                    [0.00, 0.02, 0.0625]], index=assets, columns=assets)
opt = MarkowitzOptimizer(mu, cov, rf=0.02)
mv = opt.min_variance()
check(f"min-var weights sum to 1 ({mv.weights.sum():.6f})", abs(mv.weights.sum() - 1) < 1e-9)
# analytic min-var should match numeric (shorts allowed)
mv_num = opt.optimize("min_variance", long_only=False, bounds=((-2, 2),)*3)
check(f"analytic vs numeric min-var vol ({mv.volatility:.5f} vs {mv_num.volatility:.5f})",
      abs(mv.volatility - mv_num.volatility) < 1e-4)
ms = opt.max_sharpe()
check(f"max-Sharpe weights sum to 1 ({ms.weights.sum():.6f})", abs(ms.weights.sum() - 1) < 1e-9)
check(f"max-Sharpe Sharpe >= min-var Sharpe ({ms.sharpe:.3f} >= {mv.sharpe:.3f})",
      ms.sharpe >= mv.sharpe - 1e-9)
tr = opt.target_return(0.11)
check(f"target-return hits 11% (got {tr.expected_return:.4f})", abs(tr.expected_return - 0.11) < 1e-6)
lo = opt.optimize("min_variance", long_only=True)
check(f"long-only weights >= 0 (min {lo.weights.min():.4f})", lo.weights.min() >= -1e-6)
check(f"long-only var >= unconstrained var", lo.volatility >= mv.volatility - 1e-6)
# Ledoit-Wolf PSD
R = pd.DataFrame(np.random.default_rng(0).normal(0, 0.05, (60, 5)),
                 columns=list("VWXYZ"))
lw = ledoit_wolf_cov(R)
eig = np.linalg.eigvalsh(lw.to_numpy())
check(f"Ledoit-Wolf cov symmetric & PSD (min eig {eig.min():.2e})",
      eig.min() > -1e-10 and np.allclose(lw, lw.T))

print("\n3) VaR & STRESS TESTING")
rng = np.random.default_rng(7)
norm_ret = pd.Series(rng.normal(0.0005, 0.01, 20000))
rr = risk_report(norm_ret, confidence=0.95, periods_per_year=252)
z_expected = -(0.0005 + (-1.6449) * 0.01)        # mean + z*sigma
check(f"parametric 95% VaR ~ z*sigma ({rr.var_parametric:.4f} vs {z_expected:.4f})",
      abs(rr.var_parametric - z_expected) < 5e-4)
check(f"historical ~ parametric on normal data ({rr.var_historical:.4f})",
      abs(rr.var_historical - rr.var_parametric) < 1e-3)
check(f"CVaR >= VaR ({rr.cvar_historical:.4f} >= {rr.var_historical:.4f})",
      rr.cvar_historical >= rr.var_historical)
check(f"Cornish-Fisher ~ parametric when ~normal ({rr.var_cornish_fisher:.4f})",
      abs(rr.var_cornish_fisher - rr.var_parametric) < 2e-3)
check(f"max drawdown <= 0 ({rr.max_drawdown:.3f})", rr.max_drawdown <= 0)
w = pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})
st = stress_test(w, {"-10% mkt": {"A": -0.10, "B": -0.10, "C": -0.10},
                     "rates spike": {"A": -0.05, "B": 0.02, "C": -0.08}})
check(f"stress -10% mkt = -10% port ({st.loc['-10% mkt','portfolio_return']:.4f})",
      abs(st.loc["-10% mkt", "portfolio_return"] + 0.10) < 1e-9)

print("\n4) STATEMENT FACTOR MODELS")
def dur(s): return {"units": {"USD": [{"val": v, "fy": y, "fp": "FY", "form": "10-K",
        "end": f"{y}-12-31", "start": f"{y}-01-01"} for y, v in s.items()]}}
def inst(s): return {"units": {"USD": [{"val": v, "fy": y, "fp": "FY", "form": "10-K",
        "end": f"{y}-12-31"} for y, v in s.items()]}}
def shr(s): return {"units": {"shares": [{"val": v, "fy": y, "fp": "FY", "form": "10-K",
        "end": f"{y}-12-31", "start": f"{y}-01-01"} for y, v in s.items()]}}
ys = [2022, 2023, 2024]
def make(scale_growth, margin, debt):
    base = 50e9
    rev = dict(zip(ys, [base, base*(1+scale_growth), base*(1+scale_growth)**2]))
    ni = {y: rev[y]*margin for y in ys}
    gp = {y: rev[y]*0.5 for y in ys}
    oi = {y: rev[y]*(margin+0.05) for y in ys}
    return {"entityName": "X", "facts": {"us-gaap": {
        "Revenues": dur(rev), "CostOfRevenue": dur({y: rev[y]*0.5 for y in ys}),
        "OperatingIncomeLoss": dur(oi), "NetIncomeLoss": dur(ni),
        "IncomeTaxExpenseBenefit": dur({y: ni[y]*0.25 for y in ys}),
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest":
            dur({y: ni[y]*1.33 for y in ys}),
        "InterestExpense": dur({y: 0.5e9 for y in ys}),
        "NetCashProvidedByUsedInOperatingActivities": dur({y: ni[y]*1.2 for y in ys}),
        "PaymentsToAcquirePropertyPlantAndEquipment": dur({y: rev[y]*0.06 for y in ys}),
        "DepreciationDepletionAndAmortization": dur({y: rev[y]*0.05 for y in ys}),
        "WeightedAverageNumberOfDilutedSharesOutstanding": shr({y: 1e9 for y in ys}),
        "CashAndCashEquivalentsAtCarryingValue": inst({y: 10e9 for y in ys}),
        "AssetsCurrent": inst({y: 25e9 for y in ys}),
        "LiabilitiesCurrent": inst({y: 12e9 for y in ys}),
        "LongTermDebtNoncurrent": inst({y: debt for y in ys}),
        "StockholdersEquity": inst({y: 30e9 for y in ys}),
        "Assets": inst({y: 60e9 for y in ys}),
        "PropertyPlantAndEquipmentNet": inst({y: 20e9 for y in ys}),
    }}}
# strong: high growth, high margin, low debt; weak: opposite
models = {
    "STRONG": ThreeStatementModel.from_facts(make(0.20, 0.20, 5e9), "STRONG", 3),
    "MID":    ThreeStatementModel.from_facts(make(0.10, 0.12, 15e9), "MID", 3),
    "WEAK":   ThreeStatementModel.from_facts(make(0.02, 0.05, 30e9), "WEAK", 3),
}
mcaps = {"STRONG": 200e9, "MID": 200e9, "WEAK": 200e9}  # same price -> fundamentals drive
fm = FactorModel.from_models(models, mcaps)
print(fm.composites.round(2).to_string())
zmean = fm.zscores.mean().abs().max()
check(f"z-scores ~ mean 0 (max |mean| {zmean:.2e})", zmean < 1e-9)
check("STRONG ranks #1", fm.composites.index[0] == "STRONG")
check("WEAK ranks last", fm.composites.index[-1] == "WEAK")
ls = fm.long_short(n=1)
check(f"long/short picks STRONG/WEAK ({ls})",
      ls["long"] == ["STRONG"] and ls["short"] == ["WEAK"])

print("\n5) UNIVERSE")
dates = pd.date_range("2020-01-31", periods=48, freq="ME")
px = pd.DataFrame({"A": 100*np.cumprod(1+rng.normal(0.01,0.04,48)),
                   "B": 100*np.cumprod(1+rng.normal(0.008,0.05,48))}, index=dates)
u = Universe.from_prices(px, frequency="M")
check(f"returns shape ({u.returns.shape})", u.returns.shape[1] == 2)
check(f"cov matrix 2x2", u.cov_matrix().shape == (2, 2))

print("=" * 64)
print("ALL PASSED" if not fails else f"FAILURES: {fails}")
sys.exit(1 if fails else 0)
