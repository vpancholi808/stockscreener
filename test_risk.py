"""Known-answer validation for VaR, expected shortfall and VaR backtesting.

Grew out of the FINA4472 value-at-risk notebook. The notebook compares a
historical and a Gaussian VaR by magnitude and stops there, which cannot
say whether either number is any good. A VaR is a forecast: it is tested by
counting how often losses breach it, and whether those breaches arrive
scattered or in clusters.

Offline; no network.
"""
import sys

sys.path.insert(0, "/home/claude/stockscreener")

import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

from finmodel.quant.risk import (backtest_var, cal_volatility_for_loss_tolerance,
                                 risk_report, volatility_for_loss_tolerance)

fails = []


def check(name, cond, detail=""):
    print("  [%s] %s%s" % ("OK" if cond else "FAIL", name,
                           (" — " + detail) if detail else ""))
    if not cond:
        fails.append(name)


rng = np.random.default_rng(23)
MU, SIGMA, N = 0.0006, 0.013, 6000
NORMAL = pd.Series(rng.normal(MU, SIGMA, N))

print("=" * 66)
print("1) ONE-DAY VaR AGAINST THE CLOSED FORM")
rep = risk_report(NORMAL, confidence=0.95, horizon_days=1)
m, s = NORMAL.mean(), NORMAL.std(ddof=1)
z = stats.norm.ppf(0.05)
check("parametric VaR = -(mu + z sigma)",
      abs(rep.var_parametric - (-(m + z * s))) < 1e-12,
      "%.8f vs %.8f" % (rep.var_parametric, -(m + z * s)))
check("historical VaR is the empirical quantile",
      abs(rep.var_historical - (-np.quantile(NORMAL, 0.05))) < 1e-12)
check("the two agree on normal data",
      abs(rep.var_parametric - rep.var_historical) < 0.002,
      "%.4f vs %.4f" % (rep.var_parametric, rep.var_historical))
check("CVaR exceeds VaR", rep.cvar_historical > rep.var_historical)
check("annual volatility is sigma*sqrt(252)",
      abs(rep.volatility_annual - s * np.sqrt(252)) < 1e-12)

print("\n2) HORIZON SCALING: THE MEAN GROWS WITH h, NOT sqrt(h)")
# Regression guard. The old code scaled the whole quantile by sqrt(h), which
# inflates the mean term. At h=1 the two are identical, which is exactly why
# a one-day test never caught it.
for h in (1, 5, 10, 20):
    got = risk_report(NORMAL, confidence=0.95, horizon_days=h).var_parametric
    want = -(m * h + z * s * np.sqrt(h))
    old = -(m + z * s) * np.sqrt(h)
    ok = abs(got - want) < 1e-12
    check("h=%-2d parametric VaR is correct" % h, ok,
          "%.6f vs %.6f (old sqrt-scaling gave %.6f)" % (got, want, old))
    if h > 1:
        check("h=%-2d differs from the old sqrt-scaled mean" % h,
              abs(want - old) > 1e-6,
              "the bug was worth %.4f of portfolio value" % abs(want - old))

# Independent confirmation: aggregate the series into real h-day returns and
# compare against the empirical quantile of those.
for h in (5, 10):
    agg = NORMAL.to_numpy()[:N // h * h].reshape(-1, h).sum(axis=1)
    empirical = -np.quantile(agg, 0.05)
    got = risk_report(NORMAL, confidence=0.95, horizon_days=h).var_parametric
    old = -(m + z * s) * np.sqrt(h)
    check("h=%d matches genuinely aggregated returns" % h,
          abs(got - empirical) < abs(old - empirical),
          "new err %.5f vs old err %.5f" % (abs(got - empirical),
                                            abs(old - empirical)))

print("\n3) BACKTESTING: DOES THE VaR HOLD UP OUT OF SAMPLE")
b = backtest_var(NORMAL, confidence=0.95, window=500, method="parametric")
check("a correct model passes coverage on normal data", b.coverage_ok,
      "%d breaches vs %.0f expected, p=%.3f"
      % (b.breaches, b.expected_breaches, b.kupiec_p))
check("and passes independence", b.independent, "p=%.3f" % b.christoffersen_p)
check("breach rate lands near nominal", abs(b.breach_rate - 0.05) < 0.015,
      "%.3f" % b.breach_rate)
check("expected breaches = n * alpha",
      abs(b.expected_breaches - b.observations * 0.05) < 1e-9)

# Fat tails: the Gaussian VaR is breached too often in the far tail, while
# the empirical quantile and the Cornish-Fisher correction survive.
t5 = pd.Series(rng.standard_t(5, N) * SIGMA / np.sqrt(5 / 3) + MU)
bp = backtest_var(t5, confidence=0.99, window=500, method="parametric")
bh = backtest_var(t5, confidence=0.99, window=500, method="historical")
check("Gaussian VaR is rejected on fat tails at 99%", not bp.coverage_ok,
      "%d breaches vs %.0f expected, p=%.3f"
      % (bp.breaches, bp.expected_breaches, bp.kupiec_p))
check("the historical quantile survives the same data", bh.coverage_ok,
      "%d breaches vs %.0f expected, p=%.3f"
      % (bh.breaches, bh.expected_breaches, bh.kupiec_p))
check("and the Gaussian breaches more often than the empirical one",
      bp.breaches > bh.breaches, "%d vs %d" % (bp.breaches, bh.breaches))

# Volatility clustering: the breach COUNT can be perfect while the timing is
# badly wrong. Only the independence test sees this.
v = np.full(N, SIGMA ** 2)
sh = np.zeros(N)
e = rng.normal(0, 1, N)
for i in range(1, N):
    v[i] = 1e-6 + 0.12 * sh[i - 1] ** 2 + 0.86 * v[i - 1]
    sh[i] = np.sqrt(v[i]) * e[i]
garch = pd.Series(sh + MU)
bg = backtest_var(garch, confidence=0.95, window=500, method="historical")
check("clustered volatility is caught by the independence test",
      not bg.independent,
      "coverage p=%.3f (%s) but independence p=%.3f"
      % (bg.kupiec_p, "passes" if bg.coverage_ok else "fails",
         bg.christoffersen_p))
check("the overall verdict fails even if coverage passed", not bg.passes)

print("\n4) BACKTEST EDGE CASES")
calm = pd.Series(rng.normal(0, 1e-6, 1200))
b0 = backtest_var(calm, confidence=0.95, window=500, method="historical")
check("a degenerate series does not produce nan p-values",
      np.isfinite(b0.kupiec_p) and np.isfinite(b0.christoffersen_p),
      "kupiec %.3f, christoffersen %.3f" % (b0.kupiec_p, b0.christoffersen_p))
try:
    backtest_var(NORMAL.iloc[:100], window=250)
    check("too-short series must raise", False)
except ValueError:
    check("too-short series raises", True)
try:
    backtest_var(NORMAL, method="nonsense")
    check("an unknown method must raise", False)
except ValueError:
    check("an unknown method raises", True)

print("\n5) LOSS TOLERANCE INVERTS THE VaR")
for loss in (0.10, 0.20, 0.35):
    for conf in (0.95, 0.99):
        mu_a = 0.07
        sig = volatility_for_loss_tolerance(loss, conf, mu_a)
        za = stats.norm.ppf(1 - conf)
        back = -(mu_a + za * sig)
        check("fully invested: %.0f%% at %.0f%% confidence round-trips"
              % (loss * 100, conf * 100), abs(back - loss) < 1e-12,
              "recovered %.6f" % back)
check("a positive drift buys volatility",
      volatility_for_loss_tolerance(0.20, 0.95, 0.08)
      > volatility_for_loss_tolerance(0.20, 0.95, 0.0))
check("fat tails cut the budget",
      volatility_for_loss_tolerance(0.20, 0.95, 0.08, skew=-0.7,
                                    excess_kurtosis=3.0)
      < volatility_for_loss_tolerance(0.20, 0.95, 0.08))

print("\n6) THE CAL VERSION ACCOUNTS FOR THE CASH POSITION")
# Sizing against the risky fund's own expected return overshoots, because the
# blended portfolio earns less. This is the identity that catches it.
RF, SHARPE, TAN_VOL = 0.02, 0.457, 0.1756
for loss in (0.08, 0.15, 0.22, 0.45):
    sig = cal_volatility_for_loss_tolerance(loss, RF, SHARPE, 0.95)
    y = sig / TAN_VOL
    r_c = RF + y * SHARPE * TAN_VOL
    delivered = -(r_c + stats.norm.ppf(0.05) * sig)
    check("a %.0f%% tolerance delivers exactly that" % (loss * 100),
          abs(delivered - loss) < 1e-12, "delivered %.6f" % delivered)
    # Sizing against the risky fund's own expected return misses the target.
    # The direction flips at y=1: an investor holding cash earns LESS than
    # the tangency portfolio, so the naive budget overshoots the loss limit;
    # a levered one earns more and undershoots. The overshoot is the
    # dangerous half, and it is the half that holds for every profile
    # conservative enough to keep cash — which is the default.
    naive = volatility_for_loss_tolerance(loss, 0.95, RF + SHARPE * TAN_VOL)
    y_n = naive / TAN_VOL
    d_naive = -((RF + y_n * SHARPE * TAN_VOL)
                + stats.norm.ppf(0.05) * naive)
    check("  and the naive sizing misses it", abs(d_naive - loss) > 1e-3,
          "naive delivers %.4f against %.4f asked (%s, y=%.2f)"
          % (d_naive, loss, "overshoots" if d_naive > loss else "undershoots",
             y_n))
    if y_n < 1.0:
        check("  the miss is an overshoot whenever cash is held",
              d_naive > loss + 1e-6,
              "%.4f delivered against %.4f agreed" % (d_naive, loss))

try:
    cal_volatility_for_loss_tolerance(0.15, 0.02, sharpe=2.0, confidence=0.95)
    check("an unreachable tolerance must raise", False)
except ValueError:
    check("a Sharpe above |z| raises rather than returning nonsense", True)

print("=" * 66)
print("ALL PASSED" if not fails else "FAILURES (%d): %s" % (len(fails), fails))
sys.exit(1 if fails else 0)
