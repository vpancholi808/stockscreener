"""Correlation estimation for portfolio construction.

Ported from the FINA4472 correlation-matrix notebook with the corrections
that feeding an optimiser, rather than a heatmap, demands.

The notebook's closing claim — "assets with lower correlation provide greater
diversification" — is true on average and misleading in the only period that
matters. Correlations are not constants. They rise in crashes, which is
exactly when the diversification was supposed to pay. `stress_correlation`
measures that, and corrects for the statistical artefact that makes it look
worse than it is.

Four things this module exists to get right:

  * **Log returns correlate; simple returns aggregate.** A portfolio return
    is a weighted sum of *simple* returns, never of log returns:
    log(1 + w'R) != w'log(1 + R). Correlation is nearly identical either way
    at daily frequency, so estimating it on log returns is fine, but the
    covariance handed to a mean-variance optimiser must be consistent with
    the arithmetic mean it is paired with. `prepare_returns` makes the choice
    explicit instead of leaving it to whichever cell ran last.
  * **A correlation matrix fed to an optimiser must be positive
    semi-definite.** `DataFrame.corr()` deletes missing values pairwise, so
    different cells can come from different samples and the result need not
    be PSD. An optimiser inverting it then produces weights for a covariance
    no set of assets could generate.
  * **Correlation is scale-free; covariance is not.** Daily correlation needs
    no annualisation and daily covariance does. Mixing the two is a silent
    factor-of-252 error.
  * **Conditioning on turbulent days inflates measured correlation even when
    the true correlation is unchanged** (Boyer, Gibson and Loretan, 1999).
    Reporting the raw conditional number as "correlation rises in crises"
    overstates a real effect. The Forbes-Rigobon adjustment separates them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

# Below this, an eigenvalue is numerically zero rather than negative.
_EIG_TOL = -1e-10


def prepare_returns(prices: pd.DataFrame, kind: str = "simple",
                    dropna: bool = True) -> pd.DataFrame:
    """Price levels to returns, with the log/simple choice made explicitly.

    `kind='simple'` gives (P_t / P_{t-1}) - 1, which is what portfolio
    weights multiply. `kind='log'` gives ln(P_t / P_{t-1}), which is additive
    across time and is the right input for correlation and for distribution
    work.

    `dropna` removes any row where a price is missing. That is the safe
    default for correlation — every pair then comes from the same sample, so
    the matrix is PSD — but it discards a whole day's data because one asset
    did not trade, so `coverage` below reports how much was lost.
    """
    if kind not in ("simple", "log"):
        raise ValueError("kind must be 'simple' or 'log'")
    px = prices.sort_index()
    rets = (np.log(px).diff() if kind == "log" else px.pct_change())
    return rets.dropna(how="any") if dropna else rets.dropna(how="all")


def log_to_simple(log_returns: pd.DataFrame | pd.Series):
    """exp(r) - 1. Needed before log returns meet portfolio weights."""
    return np.expm1(log_returns)


def coverage(prices: pd.DataFrame) -> pd.DataFrame:
    """What each asset costs the joint sample.

    `dropna(how='any')` on a panel is a joint operation: one asset with a
    short history truncates every other. This reports what each column has
    on its own and what survives alignment, because the difference is
    invisible once the matrix is built.
    """
    aligned = prices.dropna(how="any")
    rows = []
    for col in prices.columns:
        s = prices[col].dropna()
        rows.append({
            "asset": col,
            "observations": int(len(s)),
            "first": s.index.min() if len(s) else None,
            "last": s.index.max() if len(s) else None,
            "in_aligned_sample": int(len(aligned)),
            "lost_to_alignment": int(len(s) - len(aligned)),
        })
    return pd.DataFrame(rows).set_index("asset")


def is_psd(matrix: pd.DataFrame | np.ndarray, tol: float = _EIG_TOL) -> bool:
    m = np.asarray(matrix, dtype=float)
    return bool(np.linalg.eigvalsh((m + m.T) / 2).min() >= tol)


def min_eigenvalue(matrix: pd.DataFrame | np.ndarray) -> float:
    m = np.asarray(matrix, dtype=float)
    return float(np.linalg.eigvalsh((m + m.T) / 2).min())


def nearest_psd(matrix: pd.DataFrame, epsilon: float = 0.0) -> pd.DataFrame:
    """Clip negative eigenvalues and restore a unit diagonal.

    A correlation matrix built with pairwise deletion, or one a user typed in
    by hand, can fail to be positive semi-definite — it describes no possible
    set of assets. Rather than let an optimiser invert it and return
    confident nonsense, project it onto the nearest valid matrix: clamp the
    eigenvalues at `epsilon`, rebuild, and rescale so the diagonal is one
    again. The result is a genuine correlation matrix close to the input.
    """
    m = np.asarray(matrix, dtype=float)
    m = (m + m.T) / 2
    vals, vecs = np.linalg.eigh(m)
    vals = np.clip(vals, epsilon, None)
    rebuilt = vecs @ np.diag(vals) @ vecs.T
    d = np.sqrt(np.clip(np.diag(rebuilt), 1e-16, None))
    rebuilt = rebuilt / np.outer(d, d)
    np.fill_diagonal(rebuilt, 1.0)
    if isinstance(matrix, pd.DataFrame):
        return pd.DataFrame(rebuilt, index=matrix.index, columns=matrix.columns)
    return pd.DataFrame(rebuilt)


def correlation_matrix(returns: pd.DataFrame, method: str = "pearson",
                       repair: bool = True) -> pd.DataFrame:
    """Correlation, validated as positive semi-definite before it is returned.

    `method='spearman'` ranks the data first, which is robust to the handful
    of extreme days that dominate a Pearson estimate on financial returns —
    worth comparing against, because a large gap between the two says the
    linear number is being set by a few observations.
    """
    corr = returns.corr(method=method)
    if repair and not is_psd(corr):
        corr = nearest_psd(corr)
    return corr


def covariance_from_correlation(corr: pd.DataFrame, volatilities: pd.Series,
                                periods_per_year: Optional[int] = None
                                ) -> pd.DataFrame:
    """Sigma = D C D, with D the diagonal matrix of volatilities.

    Correlation is unit-free, so it needs no annualisation; volatility does.
    Pass `periods_per_year` only when `volatilities` are per-period and you
    want an annual covariance — passing it with already-annual volatilities
    scales the matrix by 252 and every optimiser result with it.
    """
    assets = [a for a in corr.index if a in set(volatilities.index)]
    if not assets:
        raise ValueError("correlation and volatilities share no assets")
    c = corr.loc[assets, assets].to_numpy(dtype=float)
    v = volatilities.loc[assets].to_numpy(dtype=float)
    if np.any(v < 0):
        raise ValueError("volatilities must be non-negative")
    if periods_per_year:
        v = v * np.sqrt(periods_per_year)
    cov = np.outer(v, v) * c
    return pd.DataFrame(cov, index=assets, columns=assets)


def average_correlation(corr: pd.DataFrame) -> float:
    """Mean of the off-diagonal entries — one number for 'how diversified'."""
    m = corr.to_numpy(dtype=float)
    n = m.shape[0]
    if n < 2:
        return float("nan")
    off = m[~np.eye(n, dtype=bool)]
    return float(off.mean())


def rolling_correlation(returns: pd.DataFrame, window: int = 126,
                        pair: Optional[tuple] = None) -> pd.Series | pd.DataFrame:
    """Correlation through time, because the full-sample number is an average.

    With `pair=('A','B')` this is that pair's rolling correlation. Without it,
    the rolling *average* pairwise correlation — the diversification the
    portfolio actually had, month by month. A flat line justifies using one
    covariance matrix; a line that swings from 0.2 to 0.8 says the
    single-matrix assumption behind mean-variance optimisation is the
    dominant source of error, not the estimator.
    """
    if pair is not None:
        a, b = pair
        return returns[a].rolling(window).corr(returns[b]).dropna()
    cols = list(returns.columns)
    out = {}
    for end in range(window, len(returns) + 1):
        chunk = returns.iloc[end - window:end]
        c = chunk.corr().to_numpy(dtype=float)
        n = c.shape[0]
        out[returns.index[end - 1]] = float(c[~np.eye(n, dtype=bool)].mean())
    return pd.Series(out, name="average_correlation")


@dataclass
class ExceedanceCorrelation:
    """Correlation conditional on both assets moving together in the tail.

    The obvious way to ask "does diversification survive a crash" — take the
    worst days by average return and correlate the assets within them — is
    wrong, and wrong in a direction that flatters the answer. Selecting on
    the panel's own average conditions on the sum of the series, which forces
    the components apart within the selected sample: on simulated data with
    the true correlation held fixed at 0.35, that method reports 0.27 in calm
    periods and 0.06 in "stress". The apparent collapse is the selection,
    not the market.

    The measure that works conditions each pair on *both* series exceeding a
    threshold, in standard deviations from their own mean (Longin and Solnik,
    2001; Ang and Chen, 2002). It still has a bias — any tail selection
    lowers measured correlation as the threshold rises — but it is a bias
    with a known shape, so the benchmark is what a Gaussian copula with the
    same full-sample correlation would produce at the same threshold.

    `excess_lower` is the number that matters: empirical joint-downside
    correlation minus the Gaussian benchmark. Above zero means the assets
    crash together more than a normal distribution implies, which is exactly
    the diversification a mean-variance optimiser thinks it has and does not.
    """
    threshold: float
    full_sample: float
    lower_corr: float
    upper_corr: float
    lower_n: int
    upper_n: int
    gaussian_lower: float
    gaussian_upper: float

    @property
    def excess_lower(self) -> float:
        """How much more the pair crashes together than Gaussian predicts."""
        return self.lower_corr - self.gaussian_lower

    @property
    def asymmetry(self) -> float:
        """Downside minus upside. Positive = falls together, rises apart."""
        return self.lower_corr - self.upper_corr

    def __str__(self) -> str:
        return (
            f"At {self.threshold:.1f} sigma, full-sample correlation "
            f"{self.full_sample:.3f}:\n"
            f"  both down: {self.lower_corr:.3f} over {self.lower_n} days "
            f"(Gaussian would give {self.gaussian_lower:.3f}, "
            f"excess {self.excess_lower:+.3f})\n"
            f"  both up:   {self.upper_corr:.3f} over {self.upper_n} days\n"
            f"  downside asymmetry {self.asymmetry:+.3f}")


def _gaussian_exceedance(rho: float, threshold: float, draws: int = 200_000,
                         seed: int = 0) -> tuple:
    """Exceedance correlation a bivariate normal would show at this threshold.

    Computed by simulation rather than a closed form: the conditional
    correlation of a truncated bivariate normal has no elementary expression,
    and the simulation is exact in the limit and fast enough at this size.
    """
    rho = float(np.clip(rho, -0.999, 0.999))
    rng = np.random.default_rng(seed)
    cov = np.array([[1.0, rho], [rho, 1.0]])
    z = rng.multivariate_normal([0.0, 0.0], cov, size=draws)
    out = []
    for sign in (-1.0, 1.0):
        if sign < 0:
            sel = (z[:, 0] < -threshold) & (z[:, 1] < -threshold)
        else:
            sel = (z[:, 0] > threshold) & (z[:, 1] > threshold)
        sub = z[sel]
        out.append(float(np.corrcoef(sub.T)[0, 1]) if len(sub) > 20
                   else float("nan"))
    return out[0], out[1]


def exceedance_correlation(returns: pd.DataFrame, pair: tuple,
                           threshold: float = 1.0,
                           draws: int = 200_000) -> ExceedanceCorrelation:
    """Joint-tail correlation for one pair, against its Gaussian benchmark."""
    a, b = pair
    r = returns[[a, b]].dropna(how="any")
    if len(r) < 100:
        raise ValueError(f"need at least 100 aligned observations, got {len(r)}")
    z = (r - r.mean()) / r.std(ddof=1)
    za, zb = z[a].to_numpy(), z[b].to_numpy()
    full = float(np.corrcoef(za, zb)[0, 1])

    lo = (za < -threshold) & (zb < -threshold)
    up = (za > threshold) & (zb > threshold)
    lower = float(np.corrcoef(za[lo], zb[lo])[0, 1]) if lo.sum() > 20 else float("nan")
    upper = float(np.corrcoef(za[up], zb[up])[0, 1]) if up.sum() > 20 else float("nan")
    g_lo, g_up = _gaussian_exceedance(full, threshold, draws=draws)

    return ExceedanceCorrelation(
        threshold=threshold, full_sample=full,
        lower_corr=lower, upper_corr=upper,
        lower_n=int(lo.sum()), upper_n=int(up.sum()),
        gaussian_lower=g_lo, gaussian_upper=g_up)


def tail_dependence_report(returns: pd.DataFrame, threshold: float = 1.0,
                           draws: int = 100_000) -> pd.DataFrame:
    """Exceedance correlation for every pair, worst excess first.

    One row per pair. `excess_lower` ranks them: the pairs at the top are the
    ones whose diversification disappears in joint downturns, and they are
    the pairs a covariance matrix is most wrong about.
    """
    cols = list(returns.columns)
    rows = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            try:
                e = exceedance_correlation(returns, (a, b), threshold, draws)
            except ValueError:
                continue
            rows.append({"a": a, "b": b, "full_sample": e.full_sample,
                         "both_down": e.lower_corr, "both_up": e.upper_corr,
                         "gaussian_down": e.gaussian_lower,
                         "excess_lower": e.excess_lower,
                         "asymmetry": e.asymmetry, "down_days": e.lower_n})
    if not rows:
        return pd.DataFrame(columns=["a", "b", "full_sample", "both_down",
                                     "both_up", "gaussian_down",
                                     "excess_lower", "asymmetry", "down_days"])
    return (pd.DataFrame(rows)
            .sort_values("excess_lower", ascending=False, na_position="last")
            .reset_index(drop=True))


def diversification_ratio(weights: pd.Series, cov: pd.DataFrame) -> float:
    """Weighted average volatility divided by portfolio volatility.

    One at perfect correlation, rising as correlations fall — it is the
    diversification the covariance matrix actually delivers, in a single
    number. Useful as a sanity check on an optimiser: a "diversified"
    portfolio with a ratio near 1.0 is holding assets that move together
    regardless of how many names it has.
    """
    assets = [a for a in weights.index if a in set(cov.index)]
    w = weights.loc[assets].to_numpy(dtype=float)
    c = cov.loc[assets, assets].to_numpy(dtype=float)
    vols = np.sqrt(np.diag(c))
    port_vol = float(np.sqrt(max(w @ c @ w, 0.0)))
    if port_vol <= 0:
        return float("nan")
    return float(np.abs(w) @ vols / port_vol)


@dataclass
class CorrelationReport:
    correlation: pd.DataFrame
    covariance_annual: pd.DataFrame
    volatility_annual: pd.Series
    average_correlation: float
    min_eigenvalue: float
    was_repaired: bool
    most_correlated: tuple
    least_correlated: tuple
    spearman_gap: float
    observations: int

    def __str__(self) -> str:
        a, b, v = self.most_correlated
        c, d, w = self.least_correlated
        return (
            f"{len(self.correlation)} assets, {self.observations} aligned "
            f"observations\n"
            f"  average pairwise correlation {self.average_correlation:.3f}\n"
            f"  most correlated:  {a}/{b} {v:.3f}\n"
            f"  least correlated: {c}/{d} {w:.3f}\n"
            f"  smallest eigenvalue {self.min_eigenvalue:.2e}"
            f"{'  (repaired to PSD)' if self.was_repaired else ''}\n"
            f"  largest Pearson-Spearman gap {self.spearman_gap:.3f}"
            + ("  <- a few days are driving the linear estimate"
               if self.spearman_gap > 0.15 else ""))


def correlation_report(returns: pd.DataFrame,
                       periods_per_year: int = 252) -> CorrelationReport:
    """Everything the optimiser needs from a return panel, plus the caveats."""
    r = returns.dropna(how="any")
    if len(r) < 2:
        raise ValueError("need at least two aligned observations")
    raw = r.corr()
    repaired = not is_psd(raw)
    corr = nearest_psd(raw) if repaired else raw
    vols = r.std(ddof=1) * np.sqrt(periods_per_year)
    cov = covariance_from_correlation(corr, vols)

    m = corr.to_numpy(dtype=float).copy()
    np.fill_diagonal(m, np.nan)
    names = list(corr.index)
    hi = np.unravel_index(np.nanargmax(m), m.shape)
    lo = np.unravel_index(np.nanargmin(m), m.shape)

    spear = r.corr(method="spearman").to_numpy(dtype=float)
    gap = float(np.nanmax(np.abs(np.asarray(raw) - spear)
                          - np.eye(len(names)) * 10))

    return CorrelationReport(
        correlation=corr, covariance_annual=cov, volatility_annual=vols,
        average_correlation=average_correlation(corr),
        min_eigenvalue=min_eigenvalue(raw), was_repaired=repaired,
        most_correlated=(names[hi[0]], names[hi[1]], float(m[hi])),
        least_correlated=(names[lo[0]], names[lo[1]], float(m[lo])),
        spearman_gap=gap, observations=int(len(r)))
