"""Known-answer validation for correlation estimation.

Ported from the FINA4472 correlation-matrix notebook. Every check is an
identity or a calibration against a distribution whose answer is known in
advance, so the suite stays valid if the inputs change.

Offline; no network.
"""
import sys

sys.path.insert(0, "/home/claude/stockscreener")

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from finmodel.quant.correlation import (average_correlation, coverage,
                                        correlation_matrix, correlation_report,
                                        covariance_from_correlation,
                                        diversification_ratio,
                                        exceedance_correlation, is_psd,
                                        log_to_simple, min_eigenvalue,
                                        nearest_psd, prepare_returns,
                                        rolling_correlation,
                                        tail_dependence_report)

fails = []


def check(name, cond, detail=""):
    print("  [%s] %s%s" % ("OK" if cond else "FAIL", name,
                           (" — " + detail) if detail else ""))
    if not cond:
        fails.append(name)


rng = np.random.default_rng(17)
ASSETS = list("ABCD")
N = 3000
RHO = 0.4
C_TRUE = np.full((4, 4), RHO)
np.fill_diagonal(C_TRUE, 1.0)
L = np.linalg.cholesky(C_TRUE)
SD = np.array([0.010, 0.014, 0.020, 0.016])
R = pd.DataFrame((rng.normal(size=(N, 4)) @ L.T) * SD, columns=ASSETS,
                 index=pd.date_range("2015-01-01", periods=N, freq="B"))
PRICES = 100 * (1 + R).cumprod()

print("=" * 66)
print("1) RETURNS: THE LOG / SIMPLE DISTINCTION")
simple = prepare_returns(PRICES, "simple")
logs = prepare_returns(PRICES, "log")
# prepare_returns drops the first row, so compounding starts from the SECOND
# price, not from the initial 100.
check("simple returns reproduce the prices",
      np.allclose(PRICES.iloc[0].to_numpy() * (1 + simple).cumprod().iloc[-1],
                  PRICES.iloc[-1].to_numpy(), rtol=1e-9),
      "end price %.4f vs %.4f"
      % ((PRICES.iloc[0] * (1 + simple).cumprod().iloc[-1]).iloc[0],
         PRICES.iloc[-1].iloc[0]))
check("log_to_simple inverts the log transform",
      np.allclose(log_to_simple(logs).to_numpy(), simple.to_numpy(), atol=1e-12),
      "max diff %.2e" % np.abs(log_to_simple(logs).to_numpy()
                               - simple.to_numpy()).max())
# A portfolio return is a weighted sum of SIMPLE returns. Doing it with log
# returns is wrong, and this is how wrong at daily frequency.
w = np.array([0.25] * 4)
port_simple = simple @ w
port_wrong = logs @ w
gap = float(np.abs(np.expm1(port_wrong) - port_simple).max())
check("weighting log returns is not the portfolio return", gap > 1e-6,
      "max daily error %.2e — small, but it compounds" % gap)
check("correlation is nearly identical either way",
      abs(average_correlation(simple.corr())
          - average_correlation(logs.corr())) < 0.01,
      "%.4f vs %.4f" % (average_correlation(simple.corr()),
                        average_correlation(logs.corr())))

print("\n2) THE MATRIX IS VALID BEFORE IT REACHES AN OPTIMISER")
corr = correlation_matrix(R)
check("sample correlation recovers the truth",
      abs(average_correlation(corr) - RHO) < 0.03,
      "%.4f vs %.2f" % (average_correlation(corr), RHO))
check("diagonal is exactly one",
      np.allclose(np.diag(corr.to_numpy()), 1.0))
check("symmetric", np.allclose(corr.to_numpy(), corr.to_numpy().T))
check("positive semi-definite", is_psd(corr),
      "min eigenvalue %.2e" % min_eigenvalue(corr))

# A matrix someone typed in, or one built by pairwise deletion, need not be
# PSD. It describes no possible set of assets, and an optimiser inverting it
# returns confident nonsense.
bad = pd.DataFrame([[1.0, 0.9, -0.9],
                    [0.9, 1.0, 0.9],
                    [-0.9, 0.9, 1.0]], index=list("XYZ"), columns=list("XYZ"))
check("an impossible correlation matrix is detected", not is_psd(bad),
      "min eigenvalue %.4f" % min_eigenvalue(bad))
fixed = nearest_psd(bad)
check("nearest_psd repairs it", is_psd(fixed),
      "min eigenvalue %.2e" % min_eigenvalue(fixed))
check("and keeps a unit diagonal",
      np.allclose(np.diag(fixed.to_numpy()), 1.0))
# That matrix is violently invalid (min eigenvalue -0.8), so a large shift is
# correct. The meaningful property is that the repair is NEARER than the
# obvious fallback of giving up and shrinking to the identity.
_d_fix = np.linalg.norm(fixed.to_numpy() - bad.to_numpy())
_d_eye = np.linalg.norm(np.eye(3) - bad.to_numpy())
check("the repair beats shrinking to the identity", _d_fix < _d_eye,
      "Frobenius %.3f vs %.3f" % (_d_fix, _d_eye))
# A mildly invalid matrix — the realistic case from pairwise deletion — must
# barely move.
mild = corr.to_numpy().copy()
mild[0, 1] = mild[1, 0] = mild[0, 1] + 0.30
mild = pd.DataFrame(mild, index=ASSETS, columns=ASSETS)
if not is_psd(mild):
    shift = np.abs(nearest_psd(mild).to_numpy() - mild.to_numpy()).max()
    check("a mildly invalid matrix barely moves", shift < 0.10,
          "max shift %.4f" % shift)
else:
    check("a mildly invalid matrix barely moves", True, "already PSD")
check("a valid matrix is left untouched by repair",
      np.allclose(nearest_psd(corr).to_numpy(), corr.to_numpy(), atol=1e-10))

print("\n3) COVARIANCE: SCALE-FREE TIMES SCALED")
vols = pd.Series(SD * np.sqrt(252), index=ASSETS)
cov = covariance_from_correlation(corr, vols)
D = np.diag(vols.to_numpy())
check("Sigma = D C D exactly",
      np.allclose(cov.to_numpy(), D @ corr.to_numpy() @ D, atol=1e-15),
      "max diff %.2e" % np.abs(cov.to_numpy()
                               - D @ corr.to_numpy() @ D).max())
check("the diagonal is variance",
      np.allclose(np.sqrt(np.diag(cov.to_numpy())), vols.to_numpy()))
daily = covariance_from_correlation(corr, pd.Series(SD, index=ASSETS))
annual = covariance_from_correlation(corr, pd.Series(SD, index=ASSETS),
                                     periods_per_year=252)
check("annualising scales the covariance by the period count",
      np.allclose(annual.to_numpy(), daily.to_numpy() * 252),
      "ratio %.1f" % (annual.to_numpy()[0, 0] / daily.to_numpy()[0, 0]))
check("correlation is unchanged by annualising",
      abs(average_correlation(corr) - average_correlation(
          pd.DataFrame(annual.to_numpy()
                       / np.outer(np.sqrt(np.diag(annual.to_numpy())),
                                  np.sqrt(np.diag(annual.to_numpy()))),
                       index=ASSETS, columns=ASSETS))) < 1e-12)

print("\n4) DIVERSIFICATION RATIO")
# n equal-weight, equal-volatility, uncorrelated assets: portfolio vol is
# sigma/sqrt(n) while the weighted average vol is sigma, so the ratio is
# exactly sqrt(n).
for n in (2, 4, 9):
    names = [str(i) for i in range(n)]
    ident = pd.DataFrame(np.eye(n) * 0.04, index=names, columns=names)
    eq = pd.Series([1.0 / n] * n, index=names)
    got = diversification_ratio(eq, ident)
    check("%d uncorrelated assets give sqrt(%d)" % (n, n),
          abs(got - np.sqrt(n)) < 1e-10, "%.6f vs %.6f" % (got, np.sqrt(n)))
ones = pd.DataFrame(np.full((3, 3), 0.04), index=list("XYZ"),
                    columns=list("XYZ"))
check("perfectly correlated assets give exactly 1",
      abs(diversification_ratio(pd.Series([1/3] * 3, index=list("XYZ")),
                                ones) - 1.0) < 1e-10)

print("\n5) EXCEEDANCE CORRELATION IS CALIBRATED")
# On genuinely Gaussian data the empirical joint-tail correlation must match
# the Gaussian benchmark. If it does not, the measure is reporting its own
# selection bias as a finding.
worst = 0.0
for rho in (0.2, 0.5, 0.8):
    Ct = np.array([[1.0, rho], [rho, 1.0]])
    g = pd.DataFrame(rng.normal(size=(20000, 2)) @ np.linalg.cholesky(Ct).T,
                     columns=["A", "B"])
    for th in (0.5, 1.0):
        e = exceedance_correlation(g, ("A", "B"), threshold=th, draws=100_000)
        worst = max(worst, abs(e.excess_lower))
check("Gaussian data shows no excess tail dependence", worst < 0.12,
      "largest |excess| %.3f across rho and threshold" % worst)

# Clayton copula: same normal marginals, genuine lower tail dependence.
from scipy.stats import norm
u = rng.random(8000)
theta = 1.5
v = (u ** (-theta) * (rng.random(8000) ** (-theta / (1 + theta)) - 1) + 1) \
    ** (-1 / theta)
clayton = pd.DataFrame({"A": norm.ppf(np.clip(u, 1e-9, 1 - 1e-9)),
                        "B": norm.ppf(np.clip(v, 1e-9, 1 - 1e-9))})
e = exceedance_correlation(clayton, ("A", "B"), threshold=1.0, draws=100_000)
check("genuine tail dependence is detected", e.excess_lower > 0.2,
      "excess %.3f" % e.excess_lower)
check("and shows downside asymmetry", e.asymmetry > 0.2,
      "down %.3f vs up %.3f" % (e.lower_corr, e.upper_corr))
check("a symmetric Gaussian pair shows no asymmetry",
      abs(exceedance_correlation(
          pd.DataFrame(rng.normal(size=(20000, 2)) @ np.linalg.cholesky(
              np.array([[1.0, 0.5], [0.5, 1.0]])).T, columns=["A", "B"]),
          ("A", "B"), threshold=1.0, draws=100_000).asymmetry) < 0.12)

report = tail_dependence_report(clayton, threshold=1.0, draws=50_000)
check("the pair report ranks by excess", len(report) == 1
      and report.iloc[0]["excess_lower"] > 0.2)

print("\n6) ROLLING CORRELATION AND COVERAGE")
roll = rolling_correlation(R, window=250)
check("rolling average correlation has the right length",
      len(roll) == len(R) - 250 + 1, "%d" % len(roll))
check("and sits near the true correlation",
      abs(roll.mean() - RHO) < 0.05, "mean %.3f vs %.2f" % (roll.mean(), RHO))
pair = rolling_correlation(R, window=250, pair=("A", "B"))
check("a single pair rolls too", len(pair) > 0 and abs(pair.mean() - RHO) < 0.06)

short = PRICES.copy()
short.iloc[:500, 2] = np.nan          # C listed late
cov_rep = coverage(short)
check("coverage reports what alignment costs",
      int(cov_rep.loc["A", "lost_to_alignment"]) == 500,
      "A loses %d rows to C's short history"
      % cov_rep.loc["A", "lost_to_alignment"])

print("\n7) THE FULL REPORT")
rep = correlation_report(R, periods_per_year=252)
check("report recovers the average correlation",
      abs(rep.average_correlation - RHO) < 0.03)
check("annual volatility is daily times sqrt(252)",
      np.allclose(rep.volatility_annual.to_numpy(), SD * np.sqrt(252),
                  rtol=0.08))
check("covariance matches the correlation and vols",
      np.allclose(rep.covariance_annual.to_numpy(),
                  np.diag(rep.volatility_annual) @ rep.correlation.to_numpy()
                  @ np.diag(rep.volatility_annual), atol=1e-14))
check("a clean sample needs no repair", not rep.was_repaired)
check("most and least correlated pairs are reported",
      rep.most_correlated[2] >= rep.least_correlated[2])
check("Pearson and Spearman agree on normal data",
      rep.spearman_gap < 0.1, "gap %.3f" % rep.spearman_gap)

print("=" * 66)
print("ALL PASSED" if not fails else "FAILURES (%d): %s" % (len(fails), fails))
sys.exit(1 if fails else 0)
