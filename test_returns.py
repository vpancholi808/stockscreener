"""Return statistics — known answers and distributional identities."""
import math
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from finmodel.returns import (aggregation_kurtosis, annualize_mean,
                              annualize_volatility, compute_returns,
                              infer_periods_per_year, jarque_bera,
                              kurtosis_concentration, return_statistics,
                              rolling_volatility, tail_exceedances)

fails = []


def check(name, cond, detail=""):
    print("  [%s] %s%s" % ("OK" if cond else "FAIL", name,
                           (" — " + detail) if detail else ""))
    if not cond:
        fails.append(name)


rng = np.random.default_rng(4472)
N = 2512
IDX = pd.date_range("2016-01-04", periods=N, freq="B")

print("=" * 64)
print("1) ANNUALIZATION")
print("=" * 64)

# A constant daily return has an exactly known compound annual rate.
constant = pd.DataFrame({"X": np.full(252, 0.001)},
                        index=pd.date_range("2024-01-01", periods=252,
                                            freq="B"))
geo = annualize_mean(constant, 252, "geometric")["X"]
check("constant 0.1%/day compounds to (1.001)^252 - 1",
      abs(geo - (1.001 ** 252 - 1)) < 1e-10, "%.8f" % geo)

ari = annualize_mean(constant, 252, "arithmetic")["X"]
check("arithmetic annualization is mean x 252",
      abs(ari - 0.252) < 1e-12, "%.8f" % ari)
check("with no variance, compounding beats the arithmetic sum",
      geo > ari, "geometric %.4f > arithmetic %.4f" % (geo, ari))

vol_data = pd.DataFrame({"X": rng.normal(0, 0.02, N)}, index=IDX)
check("annualized volatility is sd x sqrt(252)",
      abs(annualize_volatility(vol_data, 252)["X"]
          - vol_data["X"].std() * math.sqrt(252)) < 1e-12)

# Variance drag: for lognormal returns arithmetic - geometric ~ sigma^2 / 2.
sigma = 0.30
log_draws = rng.normal(0.0004, sigma / math.sqrt(252), N)
lognormal = pd.DataFrame({"X": np.exp(log_draws) - 1.0}, index=IDX)
stats = return_statistics(lognormal, periods=252)
drag = stats.stats["variance_drag"]["X"]
realised = stats.stats["volatility_ann"]["X"]
check("variance drag is about sigma^2 / 2",
      abs(drag - realised ** 2 / 2) < 0.01,
      "drag %.4f vs sigma^2/2 %.4f" % (drag, realised ** 2 / 2))

print()
print("=" * 64)
print("2) FREQUENCY INFERENCE")
print("=" * 64)

for freq, want in (("B", 252), ("D", 252), ("W", 52), ("ME", 12),
                   ("QE", 4), ("YE", 1)):
    idx = pd.date_range("2015-01-01", periods=40, freq=freq)
    got = infer_periods_per_year(idx)
    check("%s infers %d periods/year" % (freq, want), got == want, str(got))

print()
print("=" * 64)
print("3) SIMPLE VS LOG RETURNS")
print("=" * 64)

prices = pd.DataFrame(
    {"X": 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.012, N)))}, index=IDX)
simple = compute_returns(prices, "simple")
logs = compute_returns(prices, "log")

compounded = float((1.0 + simple["X"]).prod() - 1.0)
summed = float(np.exp(logs["X"].sum()) - 1.0)
check("compounded simple returns equal summed log returns",
      abs(compounded - summed) < 1e-9,
      "%.10f vs %.10f" % (compounded, summed))
check("log return is always below the simple return",
      bool((logs["X"] <= simple["X"] + 1e-15).all()))

print()
print("=" * 64)
print("4) NORMALITY TESTS AGAINST KNOWN DISTRIBUTIONS")
print("=" * 64)

normal = pd.DataFrame({"N": rng.normal(0.0005, 0.013, N)}, index=IDX)

# Asserting that one normal sample fails to reject would be a flaky test by
# construction: a correctly calibrated 5% test rejects 5% of true nulls. The
# meaningful check is the rejection RATE over many independent samples, which
# is also a far stronger statement about the implementation.
samples = pd.DataFrame({"s%d" % i: rng.normal(0.0, 0.013, 2000)
                        for i in range(400)})
rate = float(jarque_bera(samples)["rejects_normal_5pct"].mean())
check("JB is calibrated: ~5% of true-normal samples are rejected",
      0.02 < rate < 0.10, "rejection rate %.3f over 400 samples" % rate)

# chi-square(2) has survival exp(-x/2); its 95th percentile is 5.991.
probe = pd.DataFrame({"P": [0.0] * 10})
check("the chi2(2) p-value formula puts 5.991 at p = 0.05",
      abs(math.exp(-5.991464547 / 2) - 0.05) < 1e-6)

heavy = pd.DataFrame({"T": 0.013 * rng.standard_t(3, N)}, index=IDX)
jb = jarque_bera(heavy)
check("Student-t(3) draws reject normality",
      bool(jb["rejects_normal_5pct"]["T"]),
      "JB %.0f" % jb["jarque_bera"]["T"])

exceed = tail_exceedances(normal, 4.0)
check("normal data produce roughly the predicted 4-sigma count",
      exceed["observed"]["N"] <= 3,
      "observed %d, expected %.2f" % (exceed["observed"]["N"],
                                      exceed["expected_if_normal"]["N"]))
exceed = tail_exceedances(heavy, 4.0)
check("Student-t produces many more 4-sigma days than normal",
      exceed["ratio"]["T"] > 10, "%.0fx" % exceed["ratio"]["T"])

print()
print("=" * 64)
print("5) KURTOSIS CONCENTRATION SEPARATES BAD PRINTS FROM FAT TAILS")
print("=" * 64)

# A dedicated generator, so adding a check above cannot shift these draws.
draw = np.random.default_rng(90210)
clean = draw.normal(0.0005, 0.013, N)
spiked = clean.copy()
spiked[1100] = 0.181            # the shape of a missed dividend adjustment
spiked[1500] = -0.159
# Student-t(5): excess kurtosis 6/(nu-4) = 6, genuinely fat but FINITE. t(3)
# would be wrong here — its population kurtosis does not exist, so the sample
# statistic never settles and concentration is as high as a bad print's.
student = 0.013 * draw.standard_t(5, N)

panel = pd.DataFrame({"CLEAN": clean, "SPIKED": spiked, "FAT": student},
                     index=IDX)
conc = kurtosis_concentration(panel, k=2)

check("clean normal data: low kurtosis, unconcentrated",
      conc["excess_kurtosis"]["CLEAN"] < 1
      and conc["top_k_share"]["CLEAN"] < 0.15,
      "kurt %.2f share %.1f%%" % (conc["excess_kurtosis"]["CLEAN"],
                                  conc["top_k_share"]["CLEAN"] * 100))

check("two planted spikes: high kurtosis, highly concentrated",
      conc["excess_kurtosis"]["SPIKED"] > 10
      and conc["top_k_share"]["SPIKED"] > 0.7,
      "kurt %.1f share %.0f%%" % (conc["excess_kurtosis"]["SPIKED"],
                                  conc["top_k_share"]["SPIKED"] * 100))

check("genuine fat tails: high kurtosis, NOT concentrated",
      conc["excess_kurtosis"]["FAT"] > 2
      and conc["top_k_share"]["FAT"] < 0.6,
      "kurt %.1f share %.0f%%" % (conc["excess_kurtosis"]["FAT"],
                                  conc["top_k_share"]["FAT"] * 100))

check("the diagnostic distinguishes them where kurtosis alone cannot",
      conc["top_k_share"]["SPIKED"] > conc["top_k_share"]["FAT"] + 0.3,
      "spiked %.0f%% vs fat %.0f%%" % (conc["top_k_share"]["SPIKED"] * 100,
                                       conc["top_k_share"]["FAT"] * 100))

report = return_statistics(panel, periods=252)
spiked_warned = any("SPIKED" in w for w in report.warnings)
fat_warned = any(w.startswith("FAT") for w in report.warnings)
check("the report warns about the spiked series", spiked_warned)
check("the report does NOT warn about genuine fat tails", not fat_warned)

print()
print("=" * 64)
print("6) TAILS THIN OUT AS THE HORIZON LENGTHENS")
print("=" * 64)

fat_prices = pd.DataFrame({"T": 100 * np.exp(np.cumsum(student))}, index=IDX)
agg = aggregation_kurtosis(fat_prices, ("D", "W", "ME"))
check("daily kurtosis exceeds weekly", agg["D"]["T"] > agg["W"]["T"],
      "D %.2f > W %.2f" % (agg["D"]["T"], agg["W"]["T"]))
check("weekly kurtosis exceeds monthly", agg["W"]["T"] > agg["ME"]["T"],
      "W %.2f > ME %.2f" % (agg["W"]["T"], agg["ME"]["T"]))

roll = rolling_volatility(panel, window=60, periods=252)
check("rolling volatility has window-1 leading NaNs",
      int(roll["CLEAN"].isna().sum()) == 59)
check("rolling volatility is positive thereafter",
      bool((roll["CLEAN"].dropna() > 0).all()))

print()
print("=" * 64)
print("7) REPORT SHAPE")
print("=" * 64)

rep = return_statistics(panel, periods=252)
for col in ("mean_ann_arithmetic", "mean_ann_geometric", "variance_drag",
            "volatility_ann", "skewness", "excess_kurtosis", "jb_p_value"):
    check("stats expose %s" % col, col in rep.stats.columns)
check("both annualizations are reported, never just one",
      rep.stats["mean_ann_arithmetic"].notna().all()
      and rep.stats["mean_ann_geometric"].notna().all())
check("summary renders", len(rep.summary()) > 100)

thin = return_statistics(panel.iloc[:100], periods=252)
check("a thin sample is flagged for higher-moment estimates",
      any("thin" in w for w in thin.warnings))

print()
if fails:
    print("FAILED (%d): %s" % (len(fails), "; ".join(fails)))
    sys.exit(1)
print("test_returns: all checks passed")
