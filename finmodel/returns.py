"""Distributional statistics of returns, and how much to trust them.

Most of portfolio theory assumes returns are normal. They are not, and the
ways they fail matter for risk: the left tail is heavier than a normal
distribution implies, volatility is not constant, and the sample estimates of
the higher moments are fragile.

What this module adds beyond the textbook four moments:

**Both annualizations, always.** The usual `mean x 252` is an *arithmetic*
annualization and is not a compound growth rate. Per period, the arithmetic
mean exceeds the geometric mean by roughly half the variance — for a
30%-volatility stock that drag is about 4.5 percentage points a year, larger
than most of the differences these numbers get used to argue about. Both are
reported, along with `variance_drag`, which is the per-period gap scaled up:
`periods x (mean(r) - mean(log(1+r)))`. By Jensen's inequality that is never
negative.

Note that the *annualized* figures can still cross over, with the geometric
above the arithmetic, and that is not a bug. Compounding is convex:
`exp(x) - 1 > x`. When the mean return is large relative to the variance,
that convexity gain outweighs the drag, and a stock compounding at 30% will
show a geometric annual return above `mean x 252`. The drag column measures
the Jensen gap itself, which is the quantity the textbook result is about.

**Kurtosis concentration.** Excess kurtosis is a fourth moment, so a single
12-sigma observation contributes 12^4 = 20,736 while a typical day contributes
about 1. A sample of 2,500 days can therefore post an excess kurtosis above 30
on the strength of two observations. That number is then reported as "fat
tails" when the honest statement is "two days, which may be a crash or may be
a bad dividend adjustment". `kurtosis_concentration` measures exactly what
share of the fourth moment comes from the k most extreme observations, so the
two cases can be told apart — and `data.yahoo.audit_prices` says which it is.

**Tail counts.** Comparing how many 4-sigma days actually occurred with how
many a normal distribution predicts is the most direct statement of the same
point, and needs no distributional theory to read.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd

TRADING_DAYS = 252

# Trading periods per year by pandas-style frequency.
PERIODS = {"D": 252, "B": 252, "W": 52, "SM": 24, "M": 12, "ME": 12,
           "Q": 4, "QE": 4, "A": 1, "Y": 1, "YE": 1}


def _as_frame(data) -> pd.DataFrame:
    if isinstance(data, pd.Series):
        return data.to_frame(data.name or "returns")
    return pd.DataFrame(data)


# --------------------------------------------------------------------------
# returns and frequency
# --------------------------------------------------------------------------
def compute_returns(prices, kind: str = "simple", dropna: bool = True):
    """Simple or log returns from a price series or frame.

    Simple returns aggregate across assets (a portfolio's return is the
    weighted average of its holdings'); log returns aggregate across time
    (they add). Use simple for cross-sectional work and log for compounding,
    and do not mix them in one calculation.
    """
    prices = prices.astype(float)
    if kind == "simple":
        out = prices.pct_change(fill_method=None)
    elif kind == "log":
        out = np.log(prices / prices.shift(1))
    else:
        raise ValueError("kind must be 'simple' or 'log'")
    return out.dropna() if dropna else out


def infer_periods_per_year(index: pd.DatetimeIndex) -> int:
    """Observations per year, from the spacing of the index itself.

    Reading the calendar beats trusting a caller-supplied constant: passing
    252 for a monthly series inflates annualized volatility by sqrt(21).
    """
    if len(index) < 3:
        return TRADING_DAYS
    gap = float(np.median(np.diff(index.values).astype("timedelta64[D]")
                          .astype(float)))
    if gap <= 0:
        return TRADING_DAYS
    for days, periods in ((1.5, 252), (4.0, 252), (9.0, 52), (20.0, 24),
                          (45.0, 12), (135.0, 4), (270.0, 2)):
        if gap <= days:
            return periods
    return 1


# --------------------------------------------------------------------------
# annualization
# --------------------------------------------------------------------------
def annualize_mean(returns, periods: int, method: str = "geometric"):
    """Annualize a mean return.

    arithmetic: mean x periods. Simple, and what most notebooks print, but it
    is not a growth rate — compounding is ignored.

    geometric: (prod(1+r))^(periods/n) - 1. The rate that actually turns the
    starting value into the ending value, which is the one an investor earns.
    """
    frame = _as_frame(returns)
    if method == "arithmetic":
        return frame.mean() * periods
    if method == "geometric":
        n = len(frame)
        if n == 0:
            return pd.Series(np.nan, index=frame.columns)
        growth = (1.0 + frame).prod()
        # A holding that goes to zero has no real-valued geometric rate.
        return growth.where(growth > 0).pow(periods / n) - 1.0
    raise ValueError("method must be 'arithmetic' or 'geometric'")


def variance_drag(returns, periods: int):
    """Annualized gap between the arithmetic and geometric *per-period* mean.

    periods x (mean(r) - mean(log(1+r))). Jensen's inequality makes this
    non-negative for any sample, and a second-order expansion of log(1+r)
    puts it at roughly sigma^2 / 2, which is the textbook result.

    This is not the same as the difference between the two *annualized*
    returns, which can take either sign because compounding is convex.
    """
    frame = _as_frame(returns)
    safe = frame.where(frame > -1.0)          # a total loss has no log return
    return (safe.mean() - np.log1p(safe).mean()) * periods


def annualize_volatility(returns, periods: int):
    """Scale by sqrt(periods), which assumes returns are serially independent.

    Variance adds across independent periods, so volatility grows with the
    square root of time. Real returns show volatility clustering, so the
    assumption is wrong in detail — see `rolling_volatility`.
    """
    return _as_frame(returns).std() * math.sqrt(periods)


# --------------------------------------------------------------------------
# normality
# --------------------------------------------------------------------------
def jarque_bera(returns) -> pd.DataFrame:
    """Jarque-Bera statistic and p-value per column.

    JB = n/6 * (S^2 + K^2/4), asymptotically chi-square with 2 degrees of
    freedom, whose survival function is exactly exp(-x/2). With a few thousand
    daily observations the statistic is almost always enormous and the p-value
    underflows to zero: the test has power to detect departures far too small
    to matter, so treat it as a yes/no on normality and judge magnitude from
    the skewness and kurtosis themselves.
    """
    frame = _as_frame(returns)
    n = len(frame)
    skew = frame.skew()
    kurt = frame.kurt()                      # already excess
    stat = n / 6.0 * (skew ** 2 + kurt ** 2 / 4.0)
    with np.errstate(over="ignore"):
        p = np.exp(-stat / 2.0)
    return pd.DataFrame({"jarque_bera": stat, "p_value": p,
                         "rejects_normal_5pct": p < 0.05})


def _normal_sf(z: float) -> float:
    """Two-sided normal tail probability, via erfc (no scipy needed)."""
    return math.erfc(abs(z) / math.sqrt(2.0))


def tail_exceedances(returns, threshold: float = 4.0) -> pd.DataFrame:
    """Observed vs normal-implied count of |z| > threshold days.

    The most legible evidence against normality in the whole module: no
    distribution theory to read, just two counts side by side.
    """
    frame = _as_frame(returns)
    n = len(frame)
    z = (frame - frame.mean()) / frame.std()
    observed = (z.abs() > threshold).sum()
    expected = n * _normal_sf(threshold)
    out = pd.DataFrame({"observed": observed})
    out["expected_if_normal"] = expected
    out["ratio"] = out["observed"] / expected if expected > 0 else np.nan
    return out


def kurtosis_concentration(returns, k: int = 2) -> pd.DataFrame:
    """How much of the fourth moment comes from the k most extreme days.

    Excess kurtosis weights each observation by z^4. One 12-sigma day
    contributes 20,736 where an average day contributes about 1, so a handful
    of points can carry the entire statistic.

    A high share is not proof of a data error, but it localises the claim:
    "this distribution has fat tails" becomes "these k specific days produced
    this number", and those days can then be checked against the market record
    or against `data.yahoo.audit_prices`. A genuinely fat-tailed series spreads
    its fourth moment over many observations and shows a low share.

    There is a third case the share also catches: a distribution whose
    population kurtosis does not exist. Student-t with 4 or fewer degrees of
    freedom has infinite kurtosis, so the sample statistic never converges —
    it keeps rising with more data and is always dominated by the largest
    observations. Such a series shows a high, unstable share, and the right
    reading is not "bad data" but "this statistic is not estimable here".
    Measured across 40 samples of 2,512 draws: t(3) gives a median share of
    59%, t(5) 28%, t(6) 17%, and a clean series with two planted spikes 89%.
    """
    frame = _as_frame(returns)
    n = len(frame)
    rows = []
    for col in frame.columns:
        r = frame[col].dropna()
        if len(r) < 4 or r.std() == 0:
            rows.append({"ticker": col, "excess_kurtosis": np.nan,
                         "top_k_share": np.nan, "dates": "", "max_abs_z": np.nan})
            continue
        z = (r - r.mean()) / r.std()
        z4 = z ** 4
        top = z4.nlargest(k)
        rows.append({
            "ticker": col,
            "excess_kurtosis": float(r.kurt()),
            "top_k_share": float(top.sum() / z4.sum()),
            "dates": ", ".join(str(pd.Timestamp(d).date()) for d in top.index),
            "max_abs_z": float(z.abs().max()),
        })
    out = pd.DataFrame(rows).set_index("ticker")
    out.attrs["k"] = k
    out.attrs["n"] = n
    return out


# --------------------------------------------------------------------------
# time variation
# --------------------------------------------------------------------------
def rolling_volatility(returns, window: int = 60,
                       periods: Optional[int] = None) -> pd.DataFrame:
    """Annualized volatility over a moving window.

    A flat line would mean constant volatility. Real series cluster — calm
    stretches and violent ones — which is why a single whole-sample sigma
    understates risk in a crisis and overstates it the rest of the time.
    """
    frame = _as_frame(returns)
    periods = periods or infer_periods_per_year(frame.index)
    return frame.rolling(window).std() * math.sqrt(periods)


def aggregation_kurtosis(prices, freqs: Sequence[str] = ("D", "W", "ME")
                         ) -> pd.DataFrame:
    """Excess kurtosis of returns measured at several horizons.

    Kurtosis falls as the horizon lengthens, because a longer return is a sum
    of more independent-ish daily moves and the central limit theorem pulls
    sums toward normality. Fat tails are therefore a property of the *horizon*
    as much as of the asset, and a daily-calibrated tail model should not be
    read as a statement about annual outcomes.
    """
    rows = {}
    for freq in freqs:
        px = prices if freq in ("D", "B") else prices.resample(freq).last()
        r = compute_returns(px)
        if len(r) > 3:
            rows[freq] = r.kurt()
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------
@dataclass
class DistributionReport:
    stats: pd.DataFrame
    exceedances: pd.DataFrame
    concentration: pd.DataFrame
    periods_per_year: int
    threshold: float
    warnings: list = field(default_factory=list)

    def summary(self) -> str:
        lines = ["Return distribution (%d observations, %d periods/year)"
                 % (self.stats["observations"].max(), self.periods_per_year),
                 ""]
        show = self.stats[["mean_ann_arithmetic", "mean_ann_geometric",
                           "variance_drag", "volatility_ann", "skewness",
                           "excess_kurtosis"]]
        lines.append(show.round(4).to_string())
        lines.append("")
        lines.append("Days beyond %.0f sigma (normal would give %.2f):"
                     % (self.threshold,
                        self.exceedances["expected_if_normal"].iloc[0]))
        lines.append(self.exceedances[["observed", "ratio"]].round(1).to_string())
        for w in self.warnings:
            lines.append("  warning: %s" % w)
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


def return_statistics(returns, periods: Optional[int] = None,
                      threshold: float = 4.0, k: int = 2
                      ) -> DistributionReport:
    """Four moments, both annualizations, normality tests and tail diagnostics."""
    frame = _as_frame(returns).dropna(how="all")
    if frame.empty:
        raise ValueError("no returns given")
    periods = periods or infer_periods_per_year(frame.index)

    arithmetic = annualize_mean(frame, periods, "arithmetic")
    geometric = annualize_mean(frame, periods, "geometric")
    vol = annualize_volatility(frame, periods)
    jb = jarque_bera(frame)

    stats = pd.DataFrame({
        "observations": frame.notna().sum(),
        "mean_period": frame.mean(),
        "mean_ann_arithmetic": arithmetic,
        "mean_ann_geometric": geometric,
        "variance_drag": variance_drag(frame, periods),
        "volatility_period": frame.std(),
        "volatility_ann": vol,
        "skewness": frame.skew(),
        "excess_kurtosis": frame.kurt(),
        "min": frame.min(),
        "max": frame.max(),
        "jarque_bera": jb["jarque_bera"],
        "jb_p_value": jb["p_value"],
    })
    stats["sharpe_naive"] = stats["mean_ann_geometric"] / stats["volatility_ann"]

    exceedances = tail_exceedances(frame, threshold)
    concentration = kurtosis_concentration(frame, k)

    warnings: list[str] = []
    if len(frame) < 250:
        warnings.append(
            "%d observations is thin for third and fourth moments; skewness "
            "and kurtosis need hundreds of points to be stable" % len(frame))
    fragile = concentration[concentration["top_k_share"] > 0.5]
    for ticker, row in fragile.iterrows():
        warnings.append(
            "%s: %.0f%% of the fourth moment comes from %d day(s) (%s). "
            "Check those dates before reporting the kurtosis of %.1f as a "
            "property of the distribution."
            % (ticker, row["top_k_share"] * 100, k, row["dates"],
               row["excess_kurtosis"]))

    return DistributionReport(
        stats=stats, exceedances=exceedances, concentration=concentration,
        periods_per_year=periods, threshold=threshold, warnings=warnings)
