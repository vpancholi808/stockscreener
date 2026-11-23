"""Monte Carlo simulation for portfolios: paths, VaR, and terminal wealth.

`risk.py` already computes VaR three ways — the empirical quantile, the
Gaussian closed form, and Cornish-Fisher. This module exists for the
questions those cannot answer, and it is worth being blunt about which
those are.

**A Gaussian Monte Carlo VaR is a slow parametric VaR.** Drawing a million
normals and taking a quantile estimates, with simulation error, the number
`var_parametric` returns exactly. It is not a better estimate; it is a
noisier one. `test_montecarlo.py` asserts the two agree, precisely so the
equivalence is on the record rather than being rediscovered as a finding.

Monte Carlo earns its place where no closed form exists:

  * **Fat tails and the real joint distribution.** `model="t"` keeps the
    covariance but gives the tails a Student-t's weight; `model="bootstrap"`
    resamples whole historical rows, so the cross-sectional dependence and
    every fat tail in the sample come along without a distributional
    assumption at all.
  * **Volatility clustering.** `model="block"` resamples contiguous blocks,
    which keeps calm and turbulent stretches intact. An iid bootstrap
    destroys exactly the clustering that `backtest_var` shows is the hardest
    thing for a VaR model to survive.
  * **Path-dependent questions.** The distribution of the worst drawdown on
    the way, the chance of ever falling below a floor, the spread of wealth
    after ten years. None of these is a quantile of a one-period
    distribution and none has an elementary closed form.

Two implementation points carry most of the correctness:

  * **Portfolios aggregate simple returns, so the simulation must too.**
    Paths are generated in log space (wealth cannot go negative there), then
    each asset's log return is converted to a simple return before the
    weights are applied: r_p = sum_i w_i (exp(x_i) - 1). Weighting log
    returns instead would be the error `correlation.py` warns about, applied
    a few million times.
  * **The drift needs adjusting or the simulation quietly earns less than
    you asked for.** If a step's log return is N(m, s^2) then the expected
    simple return is exp(m + s^2/2) - 1, so setting m = mu_step gives a mean
    of roughly mu + s^2/2, not mu. Setting m = ln(1 + mu_step) - s^2/2 makes
    the arithmetic mean come out at mu_step exactly, which is what the
    optimiser's inputs mean. The test suite checks this to three decimals
    on 200,000 paths.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

_TOL = 1e-12

MODELS = ("normal", "t", "bootstrap", "block")

# Ceiling on the float cells held in one generated chunk, ~200 MB at 8
# bytes. `simulate_portfolio` sizes its chunks against this so peak memory
# depends on the horizon and the asset count but not on the path count.
CHUNK_CELLS = 25_000_000


def _rng(seed: Optional[int]) -> np.random.Generator:
    return np.random.default_rng(seed)


def _cholesky(cov: np.ndarray) -> np.ndarray:
    """Lower-triangular factor, repaired if the matrix is not quite PSD.

    A covariance built from shrinkage or from a hand-typed correlation can
    sit a hair below positive definite, where `cholesky` raises. Clipping the
    eigenvalues is the same repair `correlation.nearest_psd` applies, and it
    is better than refusing to simulate over a rounding error.
    """
    try:
        return np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        vals, vecs = np.linalg.eigh((cov + cov.T) / 2)
        vals = np.clip(vals, 1e-14, None)
        repaired = vecs @ np.diag(vals) @ vecs.T
        warnings.warn(
            "covariance was not positive definite and has been repaired by "
            "clipping its negative eigenvalues; check the inputs",
            RuntimeWarning, stacklevel=3)
        return np.linalg.cholesky(repaired)


def _antithetic(draws: np.ndarray) -> np.ndarray:
    """Pair every draw with its mirror image.

    For a symmetric driving distribution, -z is as likely as z, so the pair
    averages out half the sampling noise in any symmetric statistic. It
    halves the number of independent draws, so the saving is real but
    smaller than the path count suggests, and it does nothing for a
    statistic deep in one tail.
    """
    return np.concatenate([draws, -draws], axis=0)


def simulate_asset_returns(
        mu: pd.Series, cov: pd.DataFrame, n_paths: int, n_steps: int,
        periods_per_year: int = 252, model: str = "normal",
        df: float = 5.0, history: Optional[pd.DataFrame] = None,
        block: int = 20, antithetic: bool = True,
        seed: Optional[int] = None) -> np.ndarray:
    """Simple per-step returns, shaped (n_paths, n_steps, n_assets).

    `mu` and `cov` are annual, matching what the optimiser is given. Each
    step covers 1/`periods_per_year` of a year.

    **`mu` must be the ARITHMETIC annual mean**, the per-period expectation
    times the period count — not the geometric (compound) return. The two
    differ by about sigma^2/2, which is 13 percentage points a year on a
    52%-vol name, and passing the wrong one runs every path light by that
    much. It does not raise; it returns a plausible distribution centred
    in the wrong place. `returns.annualize_mean` DEFAULTS TO GEOMETRIC, so
    callers must pass ``method="arithmetic"`` explicitly.

    The identity that catches it: the median compound return of a
    simulation must reproduce the geometric mean implied by its own inputs,
    `(1+mu)*exp(-sigma^2/2) - 1`. `test_montecarlo.py` asserts it.

    `model`:
      * ``normal`` — correlated lognormal steps, drift-adjusted so the mean
        simple return is mu/k.
      * ``t`` — the same, with the normal replaced by a Student-t with `df`
        degrees of freedom, rescaled to keep the covariance. Below df=4 the
        kurtosis does not exist and below df=2 neither does the variance, so
        `df` is floored at 2.5 with a warning.
      * ``bootstrap`` — resample whole rows of `history`, which keeps the
        observed cross-sectional dependence and tails exactly.
      * ``block`` — resample contiguous blocks of `history` of length
        `block`, which additionally keeps volatility clustering.

    The bootstraps ignore `mu` and `cov` entirely: the sample is the model.
    """
    if model not in MODELS:
        raise ValueError("model must be one of %s" % ", ".join(MODELS))
    if n_paths < 2 or n_steps < 1:
        raise ValueError("need at least 2 paths and 1 step")

    assets = [a for a in mu.index if a in set(cov.index)]
    if not assets:
        raise ValueError("mu and cov share no assets")
    m = mu.loc[assets].to_numpy(dtype=float)
    S = cov.loc[assets, assets].to_numpy(dtype=float)
    n_assets = len(assets)
    k = int(periods_per_year)
    rng = _rng(seed)

    if model in ("bootstrap", "block"):
        if history is None or history.empty:
            raise ValueError(
                "%r needs a `history` of past returns; it resamples the "
                "sample rather than assuming a distribution" % model)
        h = history.reindex(columns=assets).dropna(how="any").to_numpy(float)
        if len(h) < 30:
            raise ValueError(
                "need at least 30 aligned observations to resample, got %d"
                % len(h))
        if model == "bootstrap":
            idx = rng.integers(0, len(h), size=(n_paths, n_steps))
            return h[idx]
        b = max(int(block), 1)
        n_blocks = int(np.ceil(n_steps / b))

        # CIRCULAR block bootstrap: starts span the whole sample and wrap.
        # Drawing starts from [0, n-b) instead — the obvious reading of
        # "a block that fits" — underweights both ends badly, because an
        # observation at index i is only reachable from starts in
        # [i-b+1, i] and near an edge most of those do not exist. Measured
        # at n=1253, b=20, as a fraction of fair weight:
        #
        #     first three   0.052  0.105  0.157
        #     middle        1.024
        #     last three    0.103  0.052  0.000
        #
        # The final observation is never sampled at all. That is not a
        # rounding detail: it biased the resampled mean by -3% on a test
        # series whose worst stretch sat at the start, and on a live
        # 5-year NVDA run it quietly discounted the 2022 drawdown and
        # moved the whole simulated distribution right — median +14pp, a
        # BETTER 5th percentile and a lower probability of loss, which
        # reads as "volatility clustering improves outcomes" and is pure
        # edge effect. Wrapping gives every observation weight exactly b
        # and is mean-unbiased by construction (Politis & Romano 1992).
        starts = rng.integers(0, len(h), size=(n_paths, n_blocks))
        offs = np.arange(b)
        # (paths, blocks, b) -> flatten the block axis, then trim to n_steps.
        idx = (starts[:, :, None] + offs[None, None, :]) % len(h)
        return h[idx.reshape(n_paths, -1)[:, :n_steps]]

    # --- parametric paths -------------------------------------------------
    half = n_paths // 2 if antithetic else n_paths
    step_cov = S / k
    L = _cholesky(step_cov)
    s2 = np.diag(step_cov)

    if model == "t":
        if df <= 2.5:
            warnings.warn(
                "df=%.2f has no finite variance (df<=2) or no finite "
                "kurtosis (df<=4); floored at 2.5 so the covariance still "
                "means something" % df, RuntimeWarning, stacklevel=2)
            df = 2.5
        # A Student-t is a normal scaled by sqrt(df/chi2_df). Scaling the
        # normal by sqrt((df-2)/df) first keeps the covariance at step_cov,
        # so switching model only changes the tails, never the risk the
        # optimiser was handed.
        z = rng.standard_normal((half, n_steps, n_assets))
        g = rng.chisquare(df, size=(half, n_steps, 1))
        z = z * np.sqrt(df / g) * math.sqrt((df - 2.0) / df)
    else:
        z = rng.standard_normal((half, n_steps, n_assets))

    if antithetic:
        z = _antithetic(z)
        if z.shape[0] < n_paths:            # odd n_paths
            z = np.concatenate([z, rng.standard_normal(
                (n_paths - z.shape[0], n_steps, n_assets))], axis=0)
        z = z[:n_paths]

    shocks = z @ L.T
    # Drift adjustment: E[exp(x) - 1] = exp(m + s^2/2) - 1, so to land the
    # arithmetic mean on mu/k the log drift must be ln(1 + mu/k) - s^2/2.
    step_mu = m / k
    if np.any(step_mu <= -1.0):
        raise ValueError("a per-step expected return of -100% or worse "
                         "leaves nothing to compound")
    drift = np.log1p(step_mu) - 0.5 * s2
    log_returns = drift[None, None, :] + shocks
    return np.expm1(log_returns)


@dataclass
class SimulationResult:
    """Simulated portfolio paths and what can be read off them."""
    wealth: np.ndarray             # (n_paths, n_steps+1), starts at 1.0
    step_returns: np.ndarray       # (n_paths, n_steps) portfolio simple returns
    model: str
    horizon_years: float
    periods_per_year: int
    assets: list
    weights: np.ndarray
    cash_weight: float
    risk_free: float

    @property
    def n_paths(self) -> int:
        return int(self.wealth.shape[0])

    @property
    def n_steps(self) -> int:
        return int(self.wealth.shape[1] - 1)

    @property
    def terminal(self) -> np.ndarray:
        return self.wealth[:, -1]

    @property
    def horizon_return(self) -> np.ndarray:
        """Total simple return over the whole horizon, per path."""
        return self.terminal - 1.0

    def quantile(self, q) -> np.ndarray:
        return np.quantile(self.terminal, q)

    def var(self, confidence: float = 0.95) -> float:
        """Loss on the horizon return, as a positive fraction."""
        return float(-np.quantile(self.horizon_return, 1.0 - confidence))

    def expected_shortfall(self, confidence: float = 0.95) -> float:
        r = self.horizon_return
        cut = np.quantile(r, 1.0 - confidence)
        tail = r[r <= cut]
        return float(-tail.mean()) if tail.size else float("nan")

    def probability_below(self, level: float = 1.0) -> float:
        """Chance of finishing below `level` times the starting value."""
        return float(np.mean(self.terminal < level))

    def max_drawdown(self) -> np.ndarray:
        """Worst peak-to-trough fall on each path, as a positive fraction.

        A path-dependent statistic: it depends on the order the returns
        arrived in, not just the terminal value, which is why a one-period
        VaR says nothing about it.
        """
        peak = np.maximum.accumulate(self.wealth, axis=1)
        return np.max(1.0 - self.wealth / peak, axis=1)

    def summary(self, confidence: float = 0.95) -> str:
        lo, mid, hi = self.quantile([0.05, 0.5, 0.95])
        dd = self.max_drawdown()
        return "\n".join([
            f"{self.n_paths:,} paths, {self.horizon_years:g}y, "
            f"model={self.model}",
            f"  terminal wealth per 1.00 invested: "
            f"5th {lo:.3f}  median {mid:.3f}  95th {hi:.3f}",
            f"  horizon VaR {confidence:.0%}: "
            f"{self.var(confidence):.2%}   ES: "
            f"{self.expected_shortfall(confidence):.2%}",
            f"  chance of ending below where it started: "
            f"{self.probability_below():.1%}",
            f"  worst drawdown along the way: median {np.median(dd):.1%}, "
            f"95th percentile {np.quantile(dd, 0.95):.1%}",
        ])


def simulate_portfolio(
        mu: pd.Series, cov: pd.DataFrame, weights: pd.Series,
        horizon_years: float = 1.0, periods_per_year: int = 252,
        n_paths: int = 20_000, model: str = "normal", df: float = 5.0,
        history: Optional[pd.DataFrame] = None, block: int = 20,
        cash_weight: Optional[float] = None, risk_free: float = 0.0,
        antithetic: bool = True, seed: Optional[int] = None
        ) -> SimulationResult:
    """Simulate a portfolio forward and return its paths.

    Weights summing to less than one leave the remainder in cash earning
    `risk_free`, matching `MarkowitzOptimizer.diagnose` and `for_investor`;
    more than one means borrowing at that rate. Pass `cash_weight` to state
    it instead of inferring it.
    """
    assets = [a for a in weights.index if a in set(cov.index)]
    if not assets:
        raise ValueError("weights and cov share no assets")
    w = weights.loc[assets].to_numpy(dtype=float)
    cash = float(cash_weight) if cash_weight is not None else 1.0 - w.sum()

    n_steps = max(int(round(horizon_years * periods_per_year)), 1)

    # Portfolios aggregate SIMPLE returns. The cash leg earns its own
    # per-step rate on the same clock.
    cash_step = (1.0 + risk_free) ** (1.0 / periods_per_year) - 1.0

    # Generated in chunks of paths, because the asset cube is n_assets
    # times larger than anything kept from it. A year of daily steps on
    # 200,000 paths and three assets is a 1.2 GB intermediate that
    # collapses immediately to the 400 MB of portfolio returns actually
    # wanted — and on a small container the process is OOM-killed before
    # it gets there, which gives the caller a dead process and no error to
    # read. Chunking bounds the intermediate at CHUNK_CELLS regardless of
    # how many paths are asked for; the results are identical, since each
    # path is independent of every other.
    port = np.empty((n_paths, n_steps), dtype=float)
    per_path = max(n_steps * max(len(assets), 1), 1)
    chunk = int(np.clip(CHUNK_CELLS // per_path, 256, n_paths))
    seeds = _rng(seed).integers(0, 2 ** 63 - 1, size=-(-n_paths // chunk))

    for i, lo in enumerate(range(0, n_paths, chunk)):
        hi = min(lo + chunk, n_paths)
        rets = simulate_asset_returns(
            mu=mu.loc[assets], cov=cov.loc[assets, assets],
            n_paths=max(hi - lo, 2), n_steps=n_steps,
            periods_per_year=periods_per_year, model=model, df=df,
            history=history, block=block, antithetic=antithetic,
            seed=int(seeds[i]))
        port[lo:hi] = (rets @ w + cash * cash_step)[:hi - lo]
        del rets

    # Built in place. `np.cumprod(1.0 + port, ...)` reads naturally but
    # allocates a third full-size array for the `1.0 + port` temporary, so
    # peak memory is 3x the result rather than 2x — enough to fail a long
    # horizon that the chunked generation above had already made feasible.
    wealth = np.empty((n_paths, n_steps + 1), dtype=float)
    wealth[:, 0] = 1.0
    np.add(port, 1.0, out=wealth[:, 1:])
    np.cumprod(wealth[:, 1:], axis=1, out=wealth[:, 1:])

    return SimulationResult(
        wealth=wealth, step_returns=port, model=model,
        horizon_years=float(horizon_years), periods_per_year=int(periods_per_year),
        assets=assets, weights=w, cash_weight=cash, risk_free=float(risk_free))


def quantile_standard_error(sample: np.ndarray, p: float) -> float:
    """Simulation error on an estimated quantile.

    se = sqrt(p(1-p)/n) / f(q), with the density at the quantile estimated
    from the spacing of the order statistics. Worth reporting: the same
    number of paths gives a far noisier 99% VaR than 95%, because the
    density out there is thinner and fewer draws land near it. A VaR printed
    to six decimals from a thousand paths is mostly noise.
    """
    x = np.sort(np.asarray(sample, dtype=float))
    n = x.size
    if n < 100 or not 0 < p < 1:
        return float("nan")
    q = np.quantile(x, p)
    # Density by a symmetric difference over a window that widens with n.
    h = max(int(round(n ** 0.8 * 0.05)), 10)
    i = int(np.clip(round(p * n), h, n - h - 1))
    span = x[i + h] - x[i - h]
    if span <= 0:
        return float("nan")
    density = (2.0 * h / n) / span
    return float(math.sqrt(p * (1 - p) / n) / density)


@dataclass
class MonteCarloVar:
    confidence: float
    horizon_years: float
    n_paths: int
    model: str
    var: float
    var_stderr: float
    expected_shortfall: float
    es_stderr: float
    parametric_var: float

    @property
    def var_ci(self) -> tuple:
        """95% interval for the VaR estimate itself, from simulation error."""
        if not np.isfinite(self.var_stderr):
            return (float("nan"), float("nan"))
        half = 1.959963984540054 * self.var_stderr
        return (self.var - half, self.var + half)

    def __str__(self) -> str:
        lo, hi = self.var_ci
        out = [
            f"{self.confidence:.0%} VaR over {self.horizon_years:g}y, "
            f"{self.n_paths:,} {self.model} paths",
            f"  VaR {self.var:.2%}  +/- {self.var_stderr:.2%} "
            f"(95% CI {lo:.2%} to {hi:.2%})",
            f"  ES  {self.expected_shortfall:.2%}  "
            f"+/- {self.es_stderr:.2%}",
        ]
        if np.isfinite(self.parametric_var):
            gap = self.var - self.parametric_var
            out.append(
                f"  Gaussian closed form says {self.parametric_var:.2%}"
                + (f" ({gap:+.2%} — the simulation is finding tail the "
                   f"normal misses)" if abs(gap) > 2 * self.var_stderr
                   else " (agrees within simulation error, as it must for "
                        "a Gaussian model)"))
        return "\n".join(out)


def monte_carlo_var(
        mu: pd.Series, cov: pd.DataFrame, weights: pd.Series,
        confidence: float = 0.95, horizon_years: float = 1.0 / 252,
        periods_per_year: int = 252, n_paths: int = 50_000,
        model: str = "normal", df: float = 5.0,
        history: Optional[pd.DataFrame] = None, block: int = 20,
        cash_weight: Optional[float] = None, risk_free: float = 0.0,
        antithetic: bool = True, seed: Optional[int] = None) -> MonteCarloVar:
    """VaR and expected shortfall from simulation, each with its own error.

    `parametric_var` comes back alongside for comparison. For
    ``model="normal"`` the two must agree within simulation error — that is
    the check that the machinery is wired up correctly, not a finding. A gap
    wider than a couple of standard errors means the model is finding tail
    weight the normal distribution does not have, which is the only reason
    to run the simulation at all.
    """
    sim = simulate_portfolio(
        mu=mu, cov=cov, weights=weights, horizon_years=horizon_years,
        periods_per_year=periods_per_year, n_paths=n_paths, model=model,
        df=df, history=history, block=block, cash_weight=cash_weight,
        risk_free=risk_free, antithetic=antithetic, seed=seed)

    r = sim.horizon_return
    alpha = 1.0 - confidence
    var = float(-np.quantile(r, alpha))
    se = quantile_standard_error(r, alpha)

    cut = np.quantile(r, alpha)
    tail = r[r <= cut]
    es = float(-tail.mean()) if tail.size else float("nan")
    es_se = (float(tail.std(ddof=1) / math.sqrt(tail.size))
             if tail.size > 1 else float("nan"))

    # Closed form on the same horizon, for comparison.
    assets = sim.assets
    w = sim.weights
    ann_mu = float(w @ mu.loc[assets].to_numpy()) + sim.cash_weight * risk_free
    ann_var = float(w @ cov.loc[assets, assets].to_numpy() @ w)
    h = horizon_years
    m_h = ann_mu * h
    s_h = math.sqrt(max(ann_var, 0.0) * h)
    param = float(-(m_h + stats.norm.ppf(alpha) * s_h))

    return MonteCarloVar(
        confidence=confidence, horizon_years=float(horizon_years),
        n_paths=sim.n_paths, model=model, var=var, var_stderr=se,
        expected_shortfall=es, es_stderr=es_se, parametric_var=param)
