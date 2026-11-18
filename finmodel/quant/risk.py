"""Portfolio downside risk: Value-at-Risk, Conditional VaR, and stress testing.

VaR is reported as a positive loss fraction at a given confidence (e.g. 0.95).
Three estimators:
  - historical: empirical quantile of realized returns (no distribution assumed);
  - parametric (Gaussian): mean - z*sigma;
  - Cornish-Fisher: parametric adjusted for skew and excess kurtosis (fat tails).
CVaR (expected shortfall) is the mean loss beyond the VaR threshold.

Stress testing applies named shock scenarios (or replays a historical window) to
a portfolio and reports the implied P&L.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats


def portfolio_returns(asset_returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    w = weights.reindex(asset_returns.columns).fillna(0.0)
    return asset_returns @ w


@dataclass
class RiskReport:
    confidence: float
    horizon_days: int
    var_historical: float
    var_parametric: float
    var_cornish_fisher: float
    cvar_historical: float
    volatility_annual: float
    max_drawdown: float
    skew: float
    excess_kurtosis: float

    def __str__(self) -> str:
        c = int(self.confidence * 100)
        return (f"{c}% VaR ({self.horizon_days}d): hist={self.var_historical:.2%}  "
                f"param={self.var_parametric:.2%}  CF={self.var_cornish_fisher:.2%}\n"
                f"{c}% CVaR: {self.cvar_historical:.2%} | "
                f"ann vol {self.volatility_annual:.2%} | "
                f"max DD {self.max_drawdown:.2%} | "
                f"skew {self.skew:.2f} kurt(excess) {self.excess_kurtosis:.2f}")


def _max_drawdown(returns: pd.Series) -> float:
    curve = (1 + returns).cumprod()
    peak = curve.cummax()
    return float((curve / peak - 1).min())


def risk_report(returns: pd.Series, confidence: float = 0.95,
                horizon_days: int = 1, periods_per_year: int = 252) -> RiskReport:
    r = returns.dropna().to_numpy()
    alpha = 1 - confidence
    z = stats.norm.ppf(alpha)
    mu, sigma = r.mean(), r.std(ddof=1)
    s = stats.skew(r)
    k = stats.kurtosis(r)  # excess kurtosis (Fisher)

    # Over h periods the MEAN scales with h and the standard deviation with
    # sqrt(h) — they are different powers. Scaling the whole quantile by
    # sqrt(h), as this did, inflates the mean term by sqrt(h) instead of h.
    # At h=1 the two agree exactly, which is why a one-day test never caught
    # it; at the Basel 10-day horizon it overstated VaR by about half a
    # percentage point of portfolio value on a typical equity series.
    h = max(int(horizon_days), 1)
    root = np.sqrt(h)
    mu_h, sigma_h = mu * h, sigma * root

    # The empirical quantile has no closed-form horizon rule, so it keeps the
    # square-root convention on the dispersion and carries the mean
    # separately. This is an approximation and is documented as one: real
    # multi-period returns are not iid, so the honest alternative is to
    # re-estimate on overlapping h-day returns.
    var_hist = -(np.quantile(r, alpha) - mu) * root - mu_h
    var_param = -(mu_h + z * sigma_h)
    # Cornish-Fisher adjusted quantile: the Gaussian quantile corrected for
    # the sample's own skew and excess kurtosis.
    z_cf = (z + (z**2 - 1) * s / 6 + (z**3 - 3*z) * k / 24
            - (2*z**3 - 5*z) * s**2 / 36)
    var_cf = -(mu_h + z_cf * sigma_h)
    tail = r[r <= np.quantile(r, alpha)]
    cvar = (-(tail.mean() - mu) * root - mu_h) if tail.size else float("nan")

    return RiskReport(
        confidence=confidence, horizon_days=horizon_days,
        var_historical=var_hist, var_parametric=var_param,
        var_cornish_fisher=var_cf, cvar_historical=cvar,
        volatility_annual=sigma * np.sqrt(periods_per_year),
        max_drawdown=_max_drawdown(returns), skew=float(s), excess_kurtosis=float(k))


@dataclass
class VarBacktest:
    """Did the VaR model actually hold up out of sample?

    A VaR estimate is a prediction: at 95% confidence, losses should breach
    it on about 5% of days. Reporting a VaR without testing that is reporting
    an untested forecast. Two things can go wrong and they are separate
    failures:

      * **Coverage** — too many breaches (the model understates risk) or too
        few (it overstates, which wastes capital). Kupiec's proportion-of-
        failures test is a likelihood ratio against the binomial null.
      * **Independence** — breaches arriving in clusters. Ten scattered
        breaches in 250 days is a working model; ten in one week is a model
        that misses volatility regimes entirely, even though the *count* is
        perfect. Christoffersen's test looks at whether a breach makes the
        next one more likely.

    Both statistics are chi-squared: one degree of freedom each, and their
    sum (conditional coverage) has two.
    """
    confidence: float
    observations: int
    breaches: int
    expected_breaches: float
    breach_rate: float
    kupiec_stat: float
    kupiec_p: float
    christoffersen_stat: float
    christoffersen_p: float
    conditional_p: float
    worst_breach: float
    mean_breach_excess: float

    @property
    def coverage_ok(self) -> bool:
        return self.kupiec_p >= 0.05

    @property
    def independent(self) -> bool:
        return self.christoffersen_p >= 0.05

    @property
    def passes(self) -> bool:
        return self.coverage_ok and self.independent

    def __str__(self) -> str:
        c = int(self.confidence * 100)
        verdict = "passes" if self.passes else "FAILS"
        lines = [
            f"{c}% VaR backtest over {self.observations} periods: {verdict}",
            f"  breaches {self.breaches} vs {self.expected_breaches:.1f} "
            f"expected ({self.breach_rate:.2%} vs {1 - self.confidence:.2%})",
            f"  Kupiec coverage p={self.kupiec_p:.3f}"
            f"{'' if self.coverage_ok else '  <- rejected'}",
            f"  Christoffersen independence p={self.christoffersen_p:.3f}"
            f"{'' if self.independent else '  <- clustered'}",
        ]
        if self.breaches:
            lines.append(f"  worst breach {self.worst_breach:.2%}, "
                         f"average excess beyond VaR "
                         f"{self.mean_breach_excess:.2%}")
        return "\n".join(lines)


def _lr_safe(*terms) -> float:
    """Sum of x*log(p) terms, treating 0*log(0) as 0 rather than nan."""
    total = 0.0
    for count, p in terms:
        if count == 0:
            continue
        if p <= 0 or p >= 1:
            return float("nan")
        total += count * np.log(p)
    return total


def backtest_var(returns: pd.Series, confidence: float = 0.95,
                 window: int = 250, method: str = "historical"
                 ) -> VarBacktest:
    """Re-estimate VaR on a rolling window and count out-of-sample breaches.

    The estimate for day t uses only the `window` days before it, so every
    breach is genuinely out of sample. Testing a VaR against the same data
    that produced it is circular: the empirical quantile of a sample breaches
    that sample at exactly the nominal rate by construction, which proves
    nothing about tomorrow.

    `method` is 'historical' (empirical quantile), 'parametric' (Gaussian) or
    'cornish_fisher' (Gaussian corrected for the window's skew and kurtosis).
    Comparing them on the same series is the honest version of the
    historical-vs-parametric table: the question is not which number is
    larger but which one breaches at the rate it promises.
    """
    r = returns.dropna()
    if len(r) <= window + 10:
        raise ValueError(
            f"need more than {window + 10} observations to backtest a "
            f"{window}-day window; got {len(r)}")
    alpha = 1.0 - confidence
    z = stats.norm.ppf(alpha)
    vals = r.to_numpy()

    forecasts = np.empty(len(vals) - window)
    for i in range(window, len(vals)):
        hist = vals[i - window:i]
        if method == "historical":
            q = np.quantile(hist, alpha)
        elif method == "parametric":
            q = hist.mean() + z * hist.std(ddof=1)
        elif method == "cornish_fisher":
            s, k = stats.skew(hist), stats.kurtosis(hist)
            z_cf = (z + (z**2 - 1) * s / 6 + (z**3 - 3*z) * k / 24
                    - (2*z**3 - 5*z) * s**2 / 36)
            q = hist.mean() + z_cf * hist.std(ddof=1)
        else:
            raise ValueError("method must be 'historical', 'parametric' or "
                             "'cornish_fisher'")
        forecasts[i - window] = q

    actual = vals[window:]
    breach = actual < forecasts
    n = len(actual)
    x = int(breach.sum())
    pi_hat = x / n
    p = alpha

    # Kupiec proportion-of-failures: likelihood ratio against breaches being
    # Bernoulli(alpha).
    ll_null = _lr_safe((n - x, 1 - p), (x, p))
    ll_alt = _lr_safe((n - x, 1 - pi_hat), (x, pi_hat))
    if np.isnan(ll_alt) or np.isnan(ll_null):
        lr_uc = float("nan")
    else:
        lr_uc = -2.0 * (ll_null - ll_alt)
    p_uc = float(1 - stats.chi2.cdf(lr_uc, 1)) if np.isfinite(lr_uc) else float("nan")

    # Christoffersen independence: does a breach today change the odds of one
    # tomorrow? Counts of transitions between the breach/no-breach states.
    prev, cur = breach[:-1], breach[1:]
    n00 = int(np.sum(~prev & ~cur)); n01 = int(np.sum(~prev & cur))
    n10 = int(np.sum(prev & ~cur));  n11 = int(np.sum(prev & cur))
    pi01 = n01 / (n00 + n01) if (n00 + n01) else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) else 0.0
    pi_all = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    ll_ind_null = _lr_safe((n00 + n10, 1 - pi_all), (n01 + n11, pi_all))
    ll_ind_alt = _lr_safe((n00, 1 - pi01), (n01, pi01),
                          (n10, 1 - pi11), (n11, pi11))
    if np.isnan(ll_ind_alt) or np.isnan(ll_ind_null):
        # Degenerate: no breaches at all, or every breach isolated. There is
        # no evidence of clustering, so independence is not rejected.
        lr_ind, p_ind = 0.0, 1.0
    else:
        lr_ind = -2.0 * (ll_ind_null - ll_ind_alt)
        p_ind = float(1 - stats.chi2.cdf(lr_ind, 1))

    lr_cc = (lr_uc if np.isfinite(lr_uc) else 0.0) + lr_ind
    p_cc = float(1 - stats.chi2.cdf(lr_cc, 2))

    excess = forecasts[breach] - actual[breach]
    return VarBacktest(
        confidence=confidence, observations=n, breaches=x,
        expected_breaches=n * p, breach_rate=pi_hat,
        kupiec_stat=float(lr_uc), kupiec_p=p_uc,
        christoffersen_stat=float(lr_ind), christoffersen_p=p_ind,
        conditional_p=p_cc,
        worst_breach=float(actual[breach].min()) if x else float("nan"),
        mean_breach_excess=float(excess.mean()) if x else float("nan"))


def volatility_for_loss_tolerance(
        max_loss: float, confidence: float = 0.95,
        expected_return: float = 0.0, skew: float = 0.0,
        excess_kurtosis: float = 0.0) -> float:
    """The annual volatility an investor can carry for a stated loss limit.

    Investors cannot state a risk-aversion coefficient, but they can answer
    "in a bad year, how much could you lose and still hold on?". That answer
    is a VaR constraint, and inverting it gives a volatility budget:

        VaR = -(mu + z_alpha * sigma) = max_loss   =>   sigma = (max_loss + mu) / |z_alpha|

    A positive expected return buys volatility: the drift offsets part of the
    loss, so the same tolerance supports a slightly riskier book.

    Passing the sample's `skew` and `excess_kurtosis` swaps the Gaussian
    quantile for the Cornish-Fisher one. That matters: equity returns are
    left-skewed and fat-tailed, so the Gaussian quantile understates the bad
    year and a portfolio sized on it carries more risk than the investor
    agreed to. Sizing on the corrected quantile is the conservative choice
    and the gap between the two is worth showing.
    """
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    if max_loss <= 0:
        raise ValueError("max_loss must be a positive fraction, e.g. 0.15")
    z = stats.norm.ppf(1 - confidence)
    if skew or excess_kurtosis:
        z = (z + (z**2 - 1) * skew / 6
             + (z**3 - 3*z) * excess_kurtosis / 24
             - (2*z**3 - 5*z) * skew**2 / 36)
    if z >= 0:
        raise ValueError("the adjusted quantile is non-negative; the skew and "
                         "kurtosis supplied do not describe a loss tail")
    sigma = (max_loss + expected_return) / abs(z)
    return max(float(sigma), 0.0)


def cal_volatility_for_loss_tolerance(
        max_loss: float, risk_free: float, sharpe: float,
        confidence: float = 0.95, skew: float = 0.0,
        excess_kurtosis: float = 0.0) -> float:
    """Volatility budget for a portfolio that mixes cash with a risky fund.

    `volatility_for_loss_tolerance` answers the fully-invested question: how
    much volatility can a portfolio with *this* expected return carry. On the
    capital allocation line the two move together — holding less of the risky
    fund cuts the volatility and the expected return at once — so using the
    risky fund's own expected return overstates what the blended portfolio
    will earn, and the loss comes out worse than the investor asked for.

    Solve it jointly instead. With y in the tangency portfolio:

        r_c = rf + y(mu_t - rf),  sigma_c = y sigma_t,  S = (mu_t - rf)/sigma_t

    and requiring -(r_c + z sigma_c) = L gives

        sigma_c = (L + rf) / (|z| - S)

    which depends on the risky fund only through its Sharpe ratio — the one
    thing that does not change as you move along the line.

    Raises when |z| <= S: the line's expected return then grows at least as
    fast as its loss quantile, so no amount of leverage ever reaches the
    stated loss and the constraint does not bind.
    """
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    if max_loss <= 0:
        raise ValueError("max_loss must be a positive fraction, e.g. 0.15")
    z = stats.norm.ppf(1 - confidence)
    if skew or excess_kurtosis:
        z = (z + (z**2 - 1) * skew / 6
             + (z**3 - 3*z) * excess_kurtosis / 24
             - (2*z**3 - 5*z) * skew**2 / 36)
    if z >= 0:
        raise ValueError("the adjusted quantile is non-negative; the skew and "
                         "kurtosis supplied do not describe a loss tail")
    denom = abs(z) - sharpe
    if denom <= 1e-9:
        raise ValueError(
            f"a Sharpe ratio of {sharpe:.3f} at {confidence:.0%} confidence "
            f"never reaches a {max_loss:.0%} loss: expected return grows at "
            f"least as fast as the loss quantile, so the tolerance does not "
            f"bind and leverage is unbounded. Raise the confidence level or "
            f"treat the inputs as implausible.")
    return max(float((max_loss + risk_free) / denom), 0.0)


def stress_test(weights: pd.Series, scenarios: dict[str, dict[str, float]],
                portfolio_value: float = 1.0) -> pd.DataFrame:
    """Apply named per-asset shock scenarios. scenarios = {name: {ticker: ret}}."""
    rows = []
    for name, shocks in scenarios.items():
        shock_vec = pd.Series(shocks).reindex(weights.index).fillna(0.0)
        pnl = float((weights * shock_vec).sum())
        rows.append({"scenario": name, "portfolio_return": pnl,
                     "pnl": pnl * portfolio_value})
    return pd.DataFrame(rows).set_index("scenario")


def historical_scenario(asset_returns: pd.DataFrame, weights: pd.Series,
                        start: str, end: str) -> dict:
    """Replay a historical window (e.g. a crash) on the current weights."""
    window = asset_returns.loc[start:end]
    port = portfolio_returns(window, weights)
    cumulative = float((1 + port).prod() - 1)
    return {"start": start, "end": end, "cumulative_return": cumulative,
            "worst_day": float(port.min()), "days": len(port)}
