"""Known-answer validation for the Monte Carlo engine.

A simulation is the easiest kind of code to get confidently wrong: it
always returns numbers, and the numbers always look plausible. Every check
here is an identity or a calibration against an answer known in advance.

The two that matter most:

  * **Gaussian MC VaR must equal the closed form.** If it does not, the
    drift adjustment, the Cholesky factor, the horizon arithmetic or the
    aggregation is wrong, and no amount of fat-tailed modelling on top of
    it will help. This is the wiring check.
  * **The block bootstrap must preserve autocorrelation the iid bootstrap
    destroys.** That is the only reason the block model exists. Resampling
    single days from a series with volatility clustering gives back the
    right marginal distribution and the wrong dynamics, which is exactly
    the failure that makes a one-day VaR pass and a ten-day VaR lie.

Offline; no network.
"""
import sys

sys.path.insert(0, "/home/claude/stockscreener")

import math
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from scipy import stats

from finmodel.quant.montecarlo import (MODELS, monte_carlo_var,
                                       quantile_standard_error,
                                       simulate_asset_returns,
                                       simulate_portfolio)

fails = []


def check(name, cond, detail=""):
    print("  [%s] %s%s" % ("OK" if cond else "FAIL", name,
                           (" — " + detail) if detail else ""))
    if not cond:
        fails.append(name)


ASSETS = ["A", "B", "C"]
MU = pd.Series([0.08, 0.12, 0.05], index=ASSETS)
SD = np.array([0.18, 0.30, 0.10])
RHO = 0.35
CORR = np.full((3, 3), RHO)
np.fill_diagonal(CORR, 1.0)
COV = pd.DataFrame(np.outer(SD, SD) * CORR, index=ASSETS, columns=ASSETS)
W = pd.Series([0.4, 0.4, 0.2], index=ASSETS)


# --------------------------------------------------------------------------
print("\n1. Moments of the simulated process")

# The generator works in log space, so the drift it is handed must be
# ln(1+mu_step) - s^2/2 for the ARITHMETIC mean of the simple returns to
# land on mu. Feeding mu straight in understates every expected return by
# half a variance — small daily, compounding to percentage points a year.
r = simulate_asset_returns(MU, COV, n_paths=200_000, n_steps=1,
                           periods_per_year=252, seed=1)[:, 0, :]
ann_mean = r.mean(axis=0) * 252
check("arithmetic mean matches mu (drift adjustment)",
      np.allclose(ann_mean, MU.to_numpy(), atol=0.004),
      "got %s vs %s" % (np.round(ann_mean, 4), MU.to_numpy()))

ann_vol = r.std(axis=0, ddof=1) * math.sqrt(252)
check("volatility matches sigma",
      np.allclose(ann_vol, SD, rtol=0.02),
      "got %s vs %s" % (np.round(ann_vol, 4), SD))

got_corr = np.corrcoef(r.T)[np.triu_indices(3, 1)]
check("correlation survives the Cholesky factor",
      np.allclose(got_corr, RHO, atol=0.01),
      "got %s vs %.2f" % (np.round(got_corr, 4), RHO))

# Simulating in log space is what keeps wealth positive. A normal simple
# return can be below -100%, which is a share price through zero.
w5 = simulate_portfolio(MU, COV, W, horizon_years=5.0, n_paths=5_000,
                        seed=2).wealth
check("wealth is strictly positive on every path",
      bool((w5 > 0).all()), "min %.6f" % w5.min())


# --------------------------------------------------------------------------
print("\n2. Gaussian VaR against the closed form")

# A SINGLE asset compounds to an exactly lognormal wealth, so there is a
# closed form to hit at every horizon rather than only at one day:
#   log W_h ~ N(h*m, h*s^2),  m = ln(1+mu/k) - s^2/2,  s = sigma/sqrt(k)
# and the alpha-quantile of the horizon return is exp(h*m + z*s*sqrt(h))-1.
# This pins the drift adjustment, the log-space construction and the
# horizon arithmetic together. Nothing else in the suite is this strict.
ONE = pd.Series([0.08], index=["A"])
ONE_COV = pd.DataFrame([[0.18 ** 2]], index=["A"], columns=["A"])
ONE_W = pd.Series([1.0], index=["A"])
s_step = 0.18 / math.sqrt(252)
m_step = math.log1p(0.08 / 252) - 0.5 * s_step ** 2
z05 = stats.norm.ppf(0.05)

for h_days in (1, 5, 10, 20, 60, 252):
    # Paths traded against horizon: the memory is paths x steps, and the
    # tolerance below is scaled by the estimator's own standard error, so
    # a shorter run simply gets a wider (and still honest) band.
    paths = 400_000 if h_days <= 20 else 120_000
    sim = simulate_portfolio(ONE, ONE_COV, ONE_W, horizon_years=h_days / 252,
                             n_paths=paths, seed=100 + h_days)
    got = float(-np.quantile(sim.horizon_return, 0.05))
    exact = -(math.exp(h_days * m_step + z05 * s_step * math.sqrt(h_days)) - 1.0)
    # The tolerance is the estimator's own standard error, not a flat
    # number: the quantile of a 252-day return is spread over a far wider
    # range than a one-day one, so the same absolute band is strict at the
    # short end and arbitrary at the long end. A fixed 2e-4 failed h=252
    # on noise alone while passing a horizon where the engine had 20x more
    # room to be wrong.
    se = quantile_standard_error(sim.horizon_return, 0.05)
    tol = max(3.0 * se, 1e-4)
    check("h=%3dd: VaR %.5f vs exact lognormal %.5f" % (h_days, got, exact),
          abs(got - exact) < tol,
          "gap %+.6f, tol %.6f" % (got - exact, tol))

# At one day, compounding has no room to act, so the normal approximation
# and the simulation coincide: that is the wiring check on the multi-asset
# path, where the weighted sum of lognormals has no closed form.
mc1 = monte_carlo_var(MU, COV, W, confidence=0.95, horizon_years=1 / 252,
                      n_paths=400_000, seed=21)
check("1-day multi-asset VaR matches the normal closed form",
      abs(mc1.var - mc1.parametric_var) < max(3.0 * mc1.var_stderr, 2e-4),
      "MC %.5f vs parametric %.5f" % (mc1.var, mc1.parametric_var))

# Beyond one day the two must DISAGREE, in a known direction. Compounding
# is multiplicative, so the horizon return is right-skewed and its left
# tail is thinner than the normal's: the parametric number overstates the
# loss, and by more as the horizon lengthens. Asserting agreement here
# would be asserting that the engine has the same flaw as the shortcut.
gaps = []
for h_days in (10, 60, 252):
    m = monte_carlo_var(MU, COV, W, confidence=0.95, horizon_years=h_days / 252,
                        n_paths=200_000 if h_days <= 10 else 60_000,
                        seed=30 + h_days)
    gaps.append(m.parametric_var - m.var)
    check("h=%3dd: parametric overstates VaR (lognormal left tail)" % h_days,
          m.parametric_var > m.var,
          "parametric %.5f vs MC %.5f (+%.3fpp)"
          % (m.parametric_var, m.var, 100 * gaps[-1]))
check("the overstatement grows with the horizon",
      gaps[0] < gaps[1] < gaps[2],
      "%s pp" % np.round(np.array(gaps) * 100, 3))

# ES for a normal is phi(z)/alpha * sigma - mu. Checked at one day, where
# the normal reference is the right one.
mu_p = float(W @ MU)
sd_p = float(math.sqrt(W @ COV @ W))
es_expected = (sd_p * math.sqrt(1 / 252) * stats.norm.pdf(z05) / 0.05) - mu_p / 252
check("1-day ES against the normal closed form",
      abs(mc1.expected_shortfall - es_expected) < max(4.0 * mc1.es_stderr, 3e-4),
      "MC %.5f vs %.5f" % (mc1.expected_shortfall, es_expected))

check("ES is never below VaR", mc1.expected_shortfall >= mc1.var,
      "ES %.5f, VaR %.5f" % (mc1.expected_shortfall, mc1.var))


# --------------------------------------------------------------------------
print("\n3. Fat tails do what they claim")

# Student-t is rescaled by sqrt((df-2)/df) so its covariance is still COV.
# Without that the t draws are inflated by that factor and the "fat tail"
# is really just a higher volatility, which is a different claim.
rt = simulate_asset_returns(MU, COV, n_paths=200_000, n_steps=1,
                            periods_per_year=252, model="t", df=6.0,
                            seed=3)[:, 0, :]
t_vol = rt.std(axis=0, ddof=1) * math.sqrt(252)
check("t draws preserve the target covariance",
      np.allclose(t_vol, SD, rtol=0.04),
      "got %s vs %s" % (np.round(t_vol, 4), SD))

# Same variance, heavier tail: the 99% VaR must be higher and the 50%
# "VaR" (the median) must not be.
n_var = monte_carlo_var(MU, COV, W, confidence=0.99, horizon_years=1 / 252,
                        n_paths=400_000, model="normal", seed=4)
t_var = monte_carlo_var(MU, COV, W, confidence=0.99, horizon_years=1 / 252,
                        n_paths=400_000, model="t", df=5.0, seed=4)
check("t 99% VaR exceeds normal at equal variance",
      t_var.var > n_var.var * 1.05,
      "t %.4f vs normal %.4f (+%.1f%%)"
      % (t_var.var, n_var.var, 100 * (t_var.var / n_var.var - 1)))

# The normal run must still agree with its own closed form, which is the
# wiring check for the t code path's scaffolding.
n95 = monte_carlo_var(MU, COV, W, confidence=0.95, horizon_years=1 / 252,
                      n_paths=400_000, model="normal", seed=41)
# The tolerance carries a floor rather than being 3 standard errors flat.
# Even at one day the compounded return is lognormal, not normal, so the
# simulation sits a consistent ~1.3e-4 BELOW the parametric figure here —
# a real difference in the right direction, not sampling noise, and one
# that no number of paths will shrink. The floor is that gap; the 3-SE
# term is what catches an actual wiring fault.
gap95 = n95.var - n95.parametric_var
check("normal still matches its closed form at 95%",
      abs(gap95) < max(3 * n95.var_stderr, 2e-4) and gap95 < 0,
      "gap %+.6f (3se %.6f)" % (gap95, 3 * n95.var_stderr))

# Fat tails do NOT raise VaR at every confidence level, and assuming they
# do is the trap. Holding variance fixed, the mass a t puts in the extreme
# tails comes out of its shoulders, so inside the crossover the t quantile
# is CLOSER to the mean than the normal's. Measured on this portfolio the
# crossover sits between 95% and 97.5%:
#
#     conf    normal      t(5)     t - normal
#     0.900  0.01329   0.01182   -0.00147
#     0.950  0.01710   0.01616   -0.00094
#     0.975  0.02037   0.02062   +0.00025
#     0.990  0.02414   0.02692   +0.00279
#     0.995  0.02676   0.03225   +0.00549
#
# So switching to a fat-tailed model and reporting 95% VaR LOWERS the
# number by about 6%, which reads as a safer portfolio and is nothing of
# the kind. The regulatory 99% is past the crossover; the conventional
# 95% is not. The suite asserts the sign on both sides so neither can
# silently flip.
lo_c = [(c, monte_carlo_var(MU, COV, W, confidence=c, horizon_years=1 / 252,
                            n_paths=400_000, model=m, df=5.0, seed=42).var)
        for c in (0.90, 0.95) for m in ("normal", "t")]
check("inside the crossover the t VaR is LOWER than normal",
      lo_c[1][1] < lo_c[0][1] and lo_c[3][1] < lo_c[2][1],
      "90%%: t %.5f vs n %.5f | 95%%: t %.5f vs n %.5f"
      % (lo_c[1][1], lo_c[0][1], lo_c[3][1], lo_c[2][1]))

hi_c = [(c, monte_carlo_var(MU, COV, W, confidence=c, horizon_years=1 / 252,
                            n_paths=400_000, model=m, df=5.0, seed=42).var)
        for c in (0.99, 0.995) for m in ("normal", "t")]
check("beyond the crossover the t VaR is HIGHER",
      hi_c[1][1] > hi_c[0][1] and hi_c[3][1] > hi_c[2][1],
      "99%%: t %.5f vs n %.5f | 99.5%%: t %.5f vs n %.5f"
      % (hi_c[1][1], hi_c[0][1], hi_c[3][1], hi_c[2][1]))

# Expected shortfall has no crossover: it averages everything past the
# cut, so the extra tail mass always shows up. That is the argument for
# reporting ES alongside VaR rather than instead of it.
es_n = monte_carlo_var(MU, COV, W, confidence=0.95, horizon_years=1 / 252,
                       n_paths=400_000, model="normal", seed=43)
es_t = monte_carlo_var(MU, COV, W, confidence=0.95, horizon_years=1 / 252,
                       n_paths=400_000, model="t", df=5.0, seed=43)
check("ES rises with fat tails even where 95% VaR falls",
      es_t.var < es_n.var and es_t.expected_shortfall > es_n.expected_shortfall,
      "VaR %.5f<%.5f but ES %.5f>%.5f"
      % (es_t.var, es_n.var, es_t.expected_shortfall, es_n.expected_shortfall))

k_n = stats.kurtosis(n_var_r := simulate_portfolio(
    MU, COV, W, horizon_years=1 / 252, n_paths=200_000, seed=5).horizon_return)
k_t = stats.kurtosis(simulate_portfolio(
    MU, COV, W, horizon_years=1 / 252, n_paths=200_000, model="t", df=6.0,
    seed=5).horizon_return)
check("excess kurtosis: normal ~0, t positive",
      abs(k_n) < 0.1 and k_t > 0.5, "normal %.3f, t %.3f" % (k_n, k_t))


# --------------------------------------------------------------------------
print("\n4. Bootstrap models against their own history")

# Build a history with volatility clustering, so the two bootstraps can be
# told apart. An iid resample of it has the right marginal distribution and
# no autocorrelation in absolute returns; a block resample keeps some.
rng = np.random.default_rng(42)
n_hist = 2_500
chol = np.linalg.cholesky(CORR).T

# A real GARCH(1,1) recursion: sigma2_t = omega + alpha*r2_{t-1} +
# beta*sigma2_{t-1}, with the variance feeding back on the realised shock.
# A first attempt drove it with an independent noise term instead, which
# gave a series with a time-varying-looking volatility and no
# autocorrelation in |r| at all — so both bootstraps passed the test that
# was supposed to separate them. alpha+beta = 0.95 is the persistence
# equity index returns actually show.
omega, alpha, beta = 1e-6, 0.10, 0.85
sig2 = np.empty(n_hist)
sig2[0] = omega / (1 - alpha - beta)
shock = np.empty((n_hist, 3))
for i in range(n_hist):
    if i:
        sig2[i] = omega + alpha * shock[i - 1, 0] ** 2 + beta * sig2[i - 1]
    shock[i] = (rng.normal(size=3) @ chol) * math.sqrt(sig2[i])
hist = pd.DataFrame(shock, index=pd.RangeIndex(n_hist), columns=ASSETS)

hist_ac = pd.Series(np.abs(hist @ W.to_numpy())).autocorr(lag=1)
check("the fixture itself has volatility clustering",
      hist_ac > 0.05, "|r| lag-1 autocorr %.3f" % hist_ac)

boot = simulate_portfolio(MU, COV, W, horizon_years=250 / 252, n_paths=4_000,
                          model="bootstrap", history=hist, seed=6)
blk = simulate_portfolio(MU, COV, W, horizon_years=250 / 252, n_paths=4_000,
                         model="block", history=hist, block=25, seed=6)


def mean_abs_autocorr(sim):
    """Lag-1 autocorrelation of |returns|, averaged over paths."""
    x = np.abs(sim.step_returns)
    x = x - x.mean(axis=1, keepdims=True)
    num = (x[:, :-1] * x[:, 1:]).sum(axis=1)
    den = (x * x).sum(axis=1)
    return float(np.mean(num / np.where(den > 0, den, np.nan)))


ac_boot, ac_blk = mean_abs_autocorr(boot), mean_abs_autocorr(blk)
check("iid bootstrap destroys autocorrelation (as it must)",
      abs(ac_boot) < 0.03, "%.4f" % ac_boot)
check("block bootstrap retains it",
      ac_blk > ac_boot + 0.04, "block %.4f vs iid %.4f" % (ac_blk, ac_boot))

# Both must still reproduce the history's own volatility, or they are
# resampling something other than what they were given.
hist_vol = float(np.std(hist @ W.to_numpy(), ddof=1))
for nm, sim in (("bootstrap", boot), ("block", blk)):
    got = float(sim.step_returns.std(ddof=1))
    check("%s reproduces history volatility" % nm,
          abs(got / hist_vol - 1) < 0.10,
          "%.5f vs %.5f" % (got, hist_vol))

for m in MODELS:
    kw = {"history": hist} if m in ("bootstrap", "block") else {}
    s = simulate_portfolio(MU, COV, W, horizon_years=0.5, n_paths=500,
                           model=m, seed=7, **kw)
    check("model %r runs and is finite" % m,
          np.isfinite(s.wealth).all() and s.wealth.shape[0] >= 500)


# --------------------------------------------------------------------------
print("\n5. The cash leg")

# All cash at a known rate is a closed form with no randomness in it: a
# degenerate case, and the one that catches a cash leg wired to the wrong
# clock. 2% for one year must be exactly 2%, on every path.
allcash = simulate_portfolio(MU, COV, pd.Series(0.0, index=ASSETS),
                             cash_weight=1.0, risk_free=0.02,
                             horizon_years=1.0, n_paths=500, seed=8)
term = allcash.terminal
check("100% cash is deterministic", float(term.std()) < 1e-12,
      "sd %.2e" % term.std())
check("100% cash earns exactly the risk-free rate",
      abs(float(term.mean()) - 1.02) < 1e-9, "%.9f vs 1.02" % term.mean())

# Half in cash must sit between all-cash and fully invested on volatility.
half = simulate_portfolio(MU, COV, W * 0.5, risk_free=0.02,
                          horizon_years=1.0, n_paths=40_000, seed=9)
full = simulate_portfolio(MU, COV, W, risk_free=0.02,
                          horizon_years=1.0, n_paths=40_000, seed=9)
check("cash dilutes risk",
      0 < half.terminal.std() < full.terminal.std(),
      "half %.4f, full %.4f" % (half.terminal.std(), full.terminal.std()))
check("cash_weight inferred from weights that do not sum to one",
      abs(half.cash_weight - 0.5) < 1e-12, "%.6f" % half.cash_weight)


# --------------------------------------------------------------------------
print("\n6. Simulation error is reported honestly")

# se ~ 1/sqrt(n): quadrupling the paths must halve it. A VaR printed to
# four decimals from a thousand paths is mostly noise, and the only way a
# reader can know that is if the number comes with its error.
ses = {}
for n in (10_000, 40_000, 160_000):
    m = monte_carlo_var(MU, COV, W, horizon_years=1 / 252, n_paths=n, seed=11)
    ses[n] = m.var_stderr
r1 = ses[10_000] / ses[40_000]
r2 = ses[40_000] / ses[160_000]
check("standard error scales as 1/sqrt(n)",
      1.6 < r1 < 2.5 and 1.6 < r2 < 2.5,
      "ratios %.2f and %.2f (expect ~2)" % (r1, r2))

# The 99% quantile is estimated from far fewer effective observations than
# the 95%, so its error is larger at the same path count.
se95 = monte_carlo_var(MU, COV, W, confidence=0.95, horizon_years=1 / 252,
                       n_paths=100_000, seed=12).var_stderr
se99 = monte_carlo_var(MU, COV, W, confidence=0.99, horizon_years=1 / 252,
                       n_paths=100_000, seed=12).var_stderr
check("99% VaR is noisier than 95% at equal n", se99 > se95,
      "se99 %.6f vs se95 %.6f" % (se99, se95))

lo, hi = mc1.var_ci
check("the confidence interval brackets the estimate",
      lo < mc1.var < hi, "%.5f < %.5f < %.5f" % (lo, mc1.var, hi))

sample = np.random.default_rng(1).normal(size=200_000)
se_known = quantile_standard_error(sample, 0.05)
# Closed form for a standard normal: sqrt(p(1-p)/n) / phi(z_p).
se_true = math.sqrt(0.05 * 0.95 / 200_000) / stats.norm.pdf(stats.norm.ppf(0.05))
check("quantile_standard_error against its closed form",
      abs(se_known / se_true - 1) < 0.15,
      "%.6f vs %.6f" % (se_known, se_true))


# --------------------------------------------------------------------------
print("\n7. Mechanics")

a = simulate_portfolio(MU, COV, W, horizon_years=1.0, n_paths=2_000, seed=777)
b = simulate_portfolio(MU, COV, W, horizon_years=1.0, n_paths=2_000, seed=777)
c = simulate_portfolio(MU, COV, W, horizon_years=1.0, n_paths=2_000, seed=778)
check("same seed reproduces the run exactly",
      np.array_equal(a.wealth, b.wealth))
check("a different seed does not",
      not np.array_equal(a.wealth, c.wealth))

# Antithetic sampling makes the shock sample exactly mean-zero, which
# removes the sampling error in the mean estimate entirely. The caller
# asks for n paths and gets n paths: half are drawn and half are mirrors,
# rather than n being doubled behind the caller's back. That matters
# because the independent-draw count is what the error scales on, so a
# path count that silently doubled would make every reported standard
# error optimistic.
anti = simulate_asset_returns(MU, COV, n_paths=5_000, n_steps=1,
                              periods_per_year=252, antithetic=True, seed=13)
check("antithetic returns the requested path count",
      anti.shape[0] == 5_000, "%d paths" % anti.shape[0])
log_anti = np.log1p(anti[:, 0, :])
centred = log_anti - log_anti.mean(axis=0)
check("antithetic shocks are exactly symmetric",
      abs(float(np.sort(centred[:, 0])[:2500].sum()
                + np.sort(centred[:, 0])[-2500:].sum())) < 1e-8)

# A correlation matrix with a negative eigenvalue describes no possible set
# of assets. The engine must repair it rather than return NaN from a failed
# Cholesky, which is what numpy does unguarded.
bad = pd.DataFrame([[1.0, 0.95, -0.95],
                    [0.95, 1.0, 0.95],
                    [-0.95, 0.95, 1.0]], index=ASSETS, columns=ASSETS)
bad_cov = pd.DataFrame(np.outer(SD, SD) * bad.to_numpy(),
                       index=ASSETS, columns=ASSETS)
check("the fixture is genuinely not PSD",
      float(np.linalg.eigvalsh(bad_cov.to_numpy()).min()) < 0)
rep = simulate_asset_returns(MU, bad_cov, n_paths=1_000, n_steps=1, seed=14)
check("non-PSD covariance is repaired, not NaN",
      np.isfinite(rep).all())

dd = a.max_drawdown()
check("max drawdown is a fraction in [0, 1)",
      bool(((dd >= 0) & (dd < 1)).all()),
      "range %.4f to %.4f" % (dd.min(), dd.max()))

p = a.probability_below(1.0)
check("probability of loss is a probability", 0.0 <= p <= 1.0, "%.3f" % p)
check("summary() renders", "VaR" in a.summary())

try:
    simulate_portfolio(MU, COV, pd.Series([1.0], index=["ZZZ"]))
    check("unknown assets raise", False)
except ValueError:
    check("unknown assets raise", True)

try:
    simulate_asset_returns(MU, COV, n_paths=100, n_steps=1, model="bootstrap")
    check("bootstrap without history raises", False)
except ValueError:
    check("bootstrap without history raises", True)


# --------------------------------------------------------------------------
print("\n" + "=" * 62)
if fails:
    print("FAILED %d check(s):" % len(fails))
    for f in fails:
        print("  - %s" % f)
    sys.exit(1)
print("All Monte Carlo checks passed.")
