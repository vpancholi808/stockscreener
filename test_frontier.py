"""Known-answer validation for the Markowitz frontier, the CAL and the
tangency portfolio.

Every check here is an analytic identity rather than a stored number, so the
suite stays valid if the inputs change. The three regression guards at the
end cover defects that shipped: a sign-flipped tangency portfolio, a frontier
that spanned its own inefficient branch, and a missing CAL.

Offline; no network.
"""
import sys
sys.path.insert(0, "/home/claude/finmodel")

import warnings

import numpy as np
import pandas as pd

from finmodel.quant.portfolio import (CapitalAllocationLine, FrontierGeometry,
                                      MarkowitzOptimizer, NoTangencyPortfolio,
                                      ledoit_wolf_cov)

ok = lambda c: "OK" if c else "FAIL"
fails = []
def check(name, cond):
    print(f"  [{ok(cond)}] {name}")
    if not cond:
        fails.append(name)

# A four-asset problem with a genuine risk/return trade-off: C has the
# highest return and the highest volatility, D is the better risk-adjusted
# holding, and the correlations are high enough that the frontier bends.
ASSETS = ["A", "B", "C", "D"]
MU = pd.Series([0.04, 0.08, 0.12, 0.10], index=ASSETS)
SD = np.array([0.10, 0.18, 0.30, 0.22])
CORR = np.array([[1.00, 0.30, 0.20, 0.25],
                 [0.30, 1.00, 0.45, 0.35],
                 [0.20, 0.45, 1.00, 0.40],
                 [0.25, 0.35, 0.40, 1.00]])
COV = pd.DataFrame(np.outer(SD, SD) * CORR, index=ASSETS, columns=ASSETS)
RF = 0.02

opt = MarkowitzOptimizer(MU, COV, rf=RF)
geo = opt.geometry()
SINV = np.linalg.inv(COV.to_numpy())

print("=" * 64)
print("1) FRONTIER GEOMETRY  (A, B, C, D and the hyperbola)")
gmv = opt.min_variance()
check(f"GMV return = B/A ({gmv.expected_return:.6f} vs {geo.gmv_return:.6f})",
      abs(gmv.expected_return - geo.gmv_return) < 1e-10)
check(f"GMV vol = sqrt(1/A) ({gmv.volatility:.6f} vs {geo.gmv_volatility:.6f})",
      abs(gmv.volatility - geo.gmv_volatility) < 1e-10)
check(f"weights sum to 1 ({gmv.weights.sum():.9f})",
      abs(gmv.weights.sum() - 1) < 1e-9)
check(f"D = AC - B^2 > 0 ({geo.d:.4f})", geo.d > 0)

# sigma^2(r) = (A r^2 - 2 B r + C) / D must reproduce every solved portfolio.
err = max(abs(geo.volatility_at(t) - opt.target_return(t).volatility)
          for t in np.linspace(0.02, 0.16, 40))
check(f"closed-form sigma(r) matches solved portfolios (max err {err:.2e})",
      err < 1e-10)

# The GMV portfolio is the vertex: no target return has lower variance.
vols = [opt.target_return(t).volatility for t in np.linspace(0.00, 0.20, 200)]
check(f"GMV is the global minimum ({gmv.volatility:.6f} <= {min(vols):.6f})",
      gmv.volatility <= min(vols) + 1e-12)

# The hyperbola is symmetric about r = B/A: reflecting a return across the
# vertex gives a portfolio with identical volatility and less return. That
# symmetry is exactly why the lower branch is dominated.
refl = max(abs(geo.volatility_at(t) - geo.volatility_at(2 * geo.gmv_return - t))
           for t in np.linspace(geo.gmv_return, 0.18, 40))
check(f"hyperbola symmetric about the vertex (max err {refl:.2e})", refl < 1e-12)

print("\n2) TANGENCY PORTFOLIO")
tan = opt.tangency()
excess = MU.to_numpy() - RF
identity = float(np.sqrt(excess @ SINV @ excess))
check(f"Sharpe = sqrt((mu-rf)'S^-1(mu-rf)) ({tan.sharpe:.10f} vs {identity:.10f})",
      abs(tan.sharpe - identity) < 1e-9)
check(f"geometry.max_sharpe agrees ({geo.max_sharpe(RF):.10f})",
      abs(geo.max_sharpe(RF) - identity) < 1e-9)
check(f"weights sum to 1 ({tan.weights.sum():.9f})",
      abs(tan.weights.sum() - 1) < 1e-9)
check(f"Sharpe exceeds GMV's ({tan.sharpe:.4f} > {gmv.sharpe:.4f})",
      tan.sharpe > gmv.sharpe)

# No portfolio on the frontier beats it, and the numeric solver finds it too.
fr = opt.efficient_frontier(points=400)
check(f"no frontier point beats it ({fr.sharpe.max():.8f} <= {tan.sharpe:.8f})",
      fr.sharpe.max() <= tan.sharpe + 1e-9)
num = opt.optimize("max_sharpe", long_only=False, bounds=(-5.0, 5.0))
check(f"analytic matches SLSQP ({tan.sharpe:.8f} vs {num.sharpe:.8f})",
      abs(tan.sharpe - num.sharpe) < 1e-6)

# Random long-only portfolios cannot beat the long-only tangency portfolio.
lo_tan = opt.tangency(long_only=True)
rng = np.random.default_rng(11)
W = rng.dirichlet(np.ones(4), size=4000)
best = max(((w @ MU.to_numpy()) - RF) / np.sqrt(w @ COV.to_numpy() @ w)
           for w in W)
check(f"long-only tangency beats 4000 random simplex draws "
      f"({lo_tan.sharpe:.4f} >= {best:.4f})", lo_tan.sharpe >= best - 1e-6)
check(f"long-only weights non-negative (min {lo_tan.weights.min():.2e})",
      lo_tan.weights.min() >= -1e-8)

print("\n3) EFFICIENT FRONTIER  (no dominated points)")
check(f"lowest return is the GMV return ({fr['return'].min():.6f} vs "
      f"{geo.gmv_return:.6f})", fr["return"].min() >= geo.gmv_return - 1e-9)
check("every point flagged efficient", bool(fr.efficient.all()))
check("volatility increases monotonically along the frontier",
      bool((np.diff(fr.volatility.to_numpy()) >= -1e-12).all()))
check("return increases monotonically along the frontier",
      bool((np.diff(fr["return"].to_numpy()) >= -1e-12).all()))

# The defining property: no returned point is dominated by another returned
# point at the same or lower volatility.
v, r = fr.volatility.to_numpy(), fr["return"].to_numpy()
dominated = sum(bool(((v <= vi + 1e-9) & (r >= ri + 1e-6)).any())
                for vi, ri in zip(v, r))
check(f"no point is dominated by another ({dominated} found)", dominated == 0)

full = opt.efficient_frontier(points=201, include_inefficient=True)
check(f"include_inefficient returns both branches "
      f"({int(full.efficient.sum())} efficient, "
      f"{int((~full.efficient).sum())} not)",
      full.efficient.any() and (~full.efficient).any())
# Each inefficient point must be dominated by an efficient one at equal vol.
lower = full[~full.efficient]
beaten = all(any(abs(ev - row.volatility) < 1e-6 and er > row["return"] + 1e-9
                 for ev, er in zip(full.volatility, full["return"]))
             or row["return"] < geo.gmv_return for _, row in lower.iterrows())
check("every inefficient point is dominated", beaten)

lo_fr = opt.efficient_frontier(points=40, long_only=True)
check("long-only frontier volatility monotone",
      bool((np.diff(lo_fr.volatility.to_numpy()) >= -1e-6).all()))
check(f"long-only frontier is inside the unconstrained one "
      f"({lo_fr.volatility.min():.4f} >= {gmv.volatility:.4f})",
      lo_fr.volatility.min() >= gmv.volatility - 1e-6)

print("\n4) CAPITAL ALLOCATION LINE")
cal = opt.capital_allocation_line()
check(f"slope = tangency Sharpe ({cal.slope:.10f})",
      abs(cal.slope - tan.sharpe) < 1e-12)
check(f"passes through (0, rf) ({cal.return_at(0.0):.10f} vs {RF})",
      abs(cal.return_at(0.0) - RF) < 1e-12)
check("return_at and volatility_at are inverses",
      abs(cal.volatility_at(cal.return_at(0.17)) - 0.17) < 1e-12)

# The CAL lies above the frontier everywhere and touches it exactly once.
gap = np.array([cal.return_at(vi) - ri for vi, ri in zip(v, r)])
touch = v[int(np.argmin(gap))]
check(f"CAL is never below the frontier (min gap {gap.min():.2e})",
      gap.min() >= -1e-8)
check(f"CAL touches the frontier at the tangency vol "
      f"({touch:.4f} vs {tan.volatility:.4f})",
      abs(touch - tan.volatility) < 2e-3)
check(f"CAL strictly beats the GMV portfolio "
      f"({cal.return_at(gmv.volatility):.4f} > {gmv.expected_return:.4f})",
      cal.return_at(gmv.volatility) > gmv.expected_return + 1e-6)

print("\n5) TWO-FUND SEPARATION & THE COMPLETE PORTFOLIO")
cp = cal.complete_portfolio(risk_aversion=3.0)
w = cp.asset_weights.to_numpy()
direct_r = float(w @ MU.to_numpy()) + cp.risk_free_weight * RF
direct_v = float(np.sqrt(w @ COV.to_numpy() @ w))
check(f"E[r] from holdings matches ({cp.expected_return:.10f} vs {direct_r:.10f})",
      abs(cp.expected_return - direct_r) < 1e-12)
check(f"vol from holdings matches ({cp.volatility:.10f} vs {direct_v:.10f})",
      abs(cp.volatility - direct_v) < 1e-12)
check(f"weights sum to 1 including cash "
      f"({cp.risky_weight + cp.risk_free_weight:.10f})",
      abs(cp.risky_weight + cp.risk_free_weight - 1) < 1e-12)
check(f"Sharpe is preserved along the line ({cp.sharpe:.10f} vs {tan.sharpe:.10f})",
      abs(cp.sharpe - tan.sharpe) < 1e-9)

# y* maximizes E[r] - (A/2) sigma^2. Confirm against a grid search.
A_RISK = 3.0
ystar = cal.optimal_allocation(A_RISK)
util = lambda y: (RF + y * (tan.expected_return - RF)
                  - 0.5 * A_RISK * (y * tan.volatility) ** 2)
grid = np.linspace(0, 3, 300001)
check(f"y*={ystar:.6f} maximizes mean-variance utility "
      f"(grid best {grid[int(np.argmax([util(g) for g in grid]))]:.6f})",
      abs(ystar - grid[int(np.argmax(util(grid)))]) < 1e-4)
check("higher risk aversion holds less of the risky portfolio",
      cal.optimal_allocation(6.0) < cal.optimal_allocation(3.0)
      < cal.optimal_allocation(1.5))

tv = cal.complete_portfolio(target_volatility=0.10)
check(f"target_volatility is hit ({tv.volatility:.10f})",
      abs(tv.volatility - 0.10) < 1e-12)
tr = cal.complete_portfolio(target_return=0.09)
check(f"target_return is hit ({tr.expected_return:.10f})",
      abs(tr.expected_return - 0.09) < 1e-12)
try:
    cal.complete_portfolio(risk_aversion=3.0, target_volatility=0.1)
    check("two criteria at once must raise", False)
except ValueError:
    check("two criteria at once raises", True)

tbl = cal.table(points=11, max_volatility=0.30)
check("CAL table is linear in volatility",
      np.allclose(np.diff(tbl["return"].to_numpy(), 2), 0, atol=1e-12))

print("\n6) NO TANGENCY WHEN rf >= THE GMV RETURN")
# With rf above the GMV return, kappa = 1'S^-1(mu - rf 1) goes negative.
# Normalising by it flips every sign and lands on the inefficient branch:
# the old code returned the *minimum*-Sharpe portfolio as the maximum.
hi_rf = 0.05
hi = MarkowitzOptimizer(MU, COV, rf=hi_rf)
hgeo = hi.geometry()
check(f"rf={hi_rf:.2%} is above the GMV return {hgeo.gmv_return:.4%}",
      hi_rf > hgeo.gmv_return)
raised = False
try:
    hi.tangency()
except NoTangencyPortfolio as exc:
    raised = True
    check(f"exception carries the asymptotic Sharpe ({exc.asymptotic_sharpe:.4f})",
          abs(exc.asymptotic_sharpe - np.sqrt(hgeo.d / hgeo.a)) < 1e-12)
check("tangency() raises NoTangencyPortfolio instead of sign-flipping", raised)

# Regression guard: reproduce the old normalisation and show it was wrong.
naive = SINV @ (MU.to_numpy() - hi_rf)
naive = naive / naive.sum()
naive_sharpe = ((naive @ MU.to_numpy()) - hi_rf) / np.sqrt(naive @ COV.to_numpy() @ naive)
best_frontier = hi.efficient_frontier(points=400).sharpe.max()
check(f"the old w/w.sum() gave {naive_sharpe:.4f} where {best_frontier:.4f} "
      f"was available", naive_sharpe < 0 < best_frontier)
check(f"and it is the sign-flip of a real portfolio (kappa={hgeo.b - hi_rf * hgeo.a:.4f} < 0)",
      hgeo.b - hi_rf * hgeo.a < 0)
check(f"the supremum is the asymptote, never attained "
      f"({best_frontier:.4f} < {hgeo.asymptotic_sharpe:.4f})",
      best_frontier < hgeo.asymptotic_sharpe)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    lo_hi = hi.tangency(long_only=True)
check(f"long_only=True still returns a maximum ({lo_hi.sharpe:.4f})",
      np.isfinite(lo_hi.sharpe))

print("\n7) DIAGNOSING A HELD PORTFOLIO")
# A portfolio already on the frontier must report no gap; one on the CAL must
# report no Sharpe shortfall. Everything else must report a non-negative gap,
# because the frontier is a bound.
d_gmv = opt.diagnose(gmv.weights)
check(f"a frontier portfolio shows no return gap ({d_gmv.return_gap:.2e})",
      d_gmv.return_gap < 1e-9 and d_gmv.is_efficient)
check(f"and no volatility gap ({d_gmv.volatility_gap:.2e})",
      d_gmv.volatility_gap < 1e-9)
check(f"but does fall short of the CAL ({d_gmv.sharpe_gap:.4f})",
      abs(d_gmv.sharpe_gap - (tan.sharpe - gmv.sharpe)) < 1e-9)

d_tan = opt.diagnose(tan.weights)
check(f"the tangency portfolio is on the CAL ({d_tan.sharpe_gap:.2e})",
      d_tan.is_on_cal and d_tan.is_efficient)

# Equal weights: a real, inefficient portfolio.
eq = pd.Series([0.25] * 4, index=ASSETS)
d_eq = opt.diagnose(eq)
check(f"equal weights is not efficient (gives up "
      f"{d_eq.return_gap:.2%} at {d_eq.volatility:.2%} vol)",
      not d_eq.is_efficient and d_eq.return_gap > 0)
check("every gap is non-negative",
      d_eq.return_gap >= 0 and d_eq.volatility_gap >= 0
      and d_eq.cal_gap >= 0 and d_eq.sharpe_gap >= 0)
check(f"the efficient portfolio at that risk really is on the frontier "
      f"({abs(geo.volatility_at(d_eq.frontier_return) - d_eq.volatility):.2e})",
      abs(geo.volatility_at(d_eq.frontier_return) - d_eq.volatility) < 1e-9)
rep = opt.diagnose(d_eq.efficient_weights)
check(f"and diagnosing it back reports no gap ({rep.return_gap:.2e})",
      rep.return_gap < 1e-8)
check(f"least risk for that return is lower ({d_eq.frontier_volatility:.4f} "
      f"< {d_eq.volatility:.4f})", d_eq.frontier_volatility < d_eq.volatility)
check("the CAL gap exceeds the frontier gap at the same risk",
      d_eq.cal_gap >= d_eq.return_gap - 1e-12)

# The trade list must actually land on the CAL.
moved = d_eq.weights + d_eq.trades()
check("trades reproduce the CAL-matched weights",
      np.allclose(moved.to_numpy(), d_eq.cal_weights.to_numpy(), atol=1e-12))
d_moved = opt.diagnose(moved, cash=d_eq.cal_cash_weight)
check(f"and that portfolio is on the CAL ({d_moved.sharpe_gap:.2e})",
      d_moved.is_on_cal)
check(f"at the same volatility it started with "
      f"({d_moved.volatility:.6f} vs {d_eq.volatility:.6f})",
      abs(d_moved.volatility - d_eq.volatility) < 1e-9)

# A cash position can sit below the GMV volatility, where no fully-invested
# risky portfolio exists and the frontier comparison is undefined.
half = opt.diagnose(tan.weights * 0.4)
check(f"a heavy cash position drops below GMV volatility "
      f"({half.volatility:.4f} < {geo.gmv_volatility:.4f})",
      half.volatility < geo.gmv_volatility)
check("the frontier comparison is None there, not a wrong number",
      half.return_gap is None and half.frontier_return is None)
check(f"the CAL comparison still applies and is tight ({half.sharpe_gap:.2e})",
      half.sharpe_gap is not None and half.sharpe_gap < 1e-9)
check(f"cash weight is inferred ({half.cash_weight:.4f})",
      abs(half.cash_weight - 0.6) < 1e-9)

# Input shapes and a name that is not in the universe.
check("accepts a dict of weights",
      abs(opt.diagnose({a: 0.25 for a in ASSETS}).volatility
          - d_eq.volatility) < 1e-12)
check("assets left out are held at zero",
      abs(opt.diagnose({"A": 1.0}).volatility - SD[0]) < 1e-9)
try:
    opt.diagnose({"NOPE": 1.0})
    check("an unknown asset must raise", False)
except ValueError:
    check("an unknown asset raises", True)

# With rf above the GMV return there is no CAL to measure against.
no_cal = hi.diagnose(eq)
check("no CAL leaves the CAL fields None, and the frontier ones intact",
      no_cal.sharpe_gap is None and no_cal.cal_weights is None
      and no_cal.return_gap is not None)

print("\n8) COVARIANCE AND INPUT HANDLING")
R = pd.DataFrame(np.random.default_rng(3).normal(0, 0.05, (60, 8)),
                 columns=list("ABCDEFGH"))
lw = ledoit_wolf_cov(R)
eig = np.linalg.eigvalsh(lw.to_numpy())
check(f"Ledoit-Wolf is PSD (min eig {eig.min():.2e})", eig.min() > -1e-12)
check("Ledoit-Wolf is symmetric", np.allclose(lw, lw.T))
sample_cond = np.linalg.cond(R.cov().to_numpy())
check(f"better conditioned than the sample covariance "
      f"({np.linalg.cond(lw.to_numpy()):.1f} < {sample_cond:.1f})",
      np.linalg.cond(lw.to_numpy()) < sample_cond)
# Shrinkage intensity is scale-invariant, so annualizing afterwards is safe.
check("annualization is a pure scaling",
      np.allclose(ledoit_wolf_cov(R, 12).to_numpy(),
                  ledoit_wolf_cov(R).to_numpy() * 12))
check("scaling the returns scales the covariance exactly",
      np.allclose(ledoit_wolf_cov(R * 3).to_numpy(),
                  ledoit_wolf_cov(R).to_numpy() * 9))

# A single (lo, hi) pair must apply to every asset, not be read as n bounds.
one_pair = opt.optimize("min_variance", long_only=False, bounds=(-2.0, 2.0))
explicit = opt.optimize("min_variance", long_only=False, bounds=((-2.0, 2.0),) * 4)
check(f"a single bounds pair broadcasts ({one_pair.volatility:.8f} vs "
      f"{explicit.volatility:.8f})",
      abs(one_pair.volatility - explicit.volatility) < 1e-8)

# Constant expected returns leave no frontier to trace.
flat = MarkowitzOptimizer(pd.Series([0.07] * 4, index=ASSETS), COV, rf=RF)
try:
    flat.geometry()
    check("constant mu must raise", False)
except ValueError as exc:
    check(f"constant mu raises with an explanation ({str(exc)[:40]}...)", True)

try:
    MarkowitzOptimizer(pd.Series([0.05, np.nan, 0.1, 0.1], index=ASSETS), COV)
    check("non-finite mu must raise", False)
except ValueError:
    check("non-finite mu raises", True)

print("=" * 64)
print("ALL PASSED" if not fails else f"FAILURES: {fails}")
sys.exit(1 if fails else 0)
