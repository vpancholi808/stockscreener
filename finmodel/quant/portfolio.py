"""Markowitz mean-variance optimization: the efficient frontier and the CAL.

Three objects, in the order the theory builds them:

  * `FrontierGeometry` — the four scalars A, B, C, D that determine the entire
    frontier. Every closed-form result below is a function of them alone, so
    they are computed once and reused rather than re-derived per call.
  * `MarkowitzOptimizer` — portfolios of risky assets only: the global
    minimum-variance portfolio, target-return portfolios, the tangency
    portfolio, and the frontier itself. Two solver paths, the analytic
    (unconstrained, shorting allowed) and the numeric (long-only or custom
    bounds, SLSQP). The analytic path validates the numeric one in the tests.
  * `CapitalAllocationLine` — what happens once a risk-free asset is added.
    Mixing it with the tangency portfolio dominates every frontier portfolio
    except the single point the line touches (Tobin's two-fund separation).
    The CAL is called the Capital Market Line when the tangency portfolio is
    the market portfolio — the same line under the extra assumption that
    every investor sees the same inputs.

Covariance defaults to Ledoit-Wolf shrinkage toward a scaled identity, which
is far better conditioned than the raw sample covariance when assets are many
relative to observations. The optimizer inverts the covariance, so its
condition number is the amplification factor on estimation error in `mu`;
`condition_number` exposes it and construction warns when it is extreme.

Everything here is in the same units as its inputs. `from_universe` passes
annualized mu and cov, so the returns, volatilities and Sharpe ratios that
come back are annual.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .universe import Universe

# Below this, two returns are the same number and the tangency portfolio is
# numerically unbounded rather than merely large.
_TOL = 1e-10

# Inverting a covariance this badly conditioned amplifies estimation error in
# mu by the same factor. Not an error — real asset covariances are genuinely
# near-singular — but the weights that come out should not be read as precise.
_CONDITION_WARN = 1e6


def ledoit_wolf_cov(returns: pd.DataFrame, annualize: int = 1) -> pd.DataFrame:
    """Ledoit-Wolf (2004) shrinkage of the sample covariance toward mu*I.

    The sample covariance is unbiased but, with p assets and n observations,
    its extreme eigenvalues are biased outward by roughly p/n — the largest
    too large, the smallest too small. Inverting it therefore puts enormous
    weight on the directions estimated worst, which is why unconstrained
    Markowitz weights on sample moments are famously unusable. Shrinking
    toward a scaled identity pulls the eigenvalues back together by an
    intensity estimated from the data itself.

    The shrinkage intensity is scale-invariant, so applying `annualize`
    afterwards does not change it.
    """
    X = returns.to_numpy(dtype=float)
    n, p = X.shape
    if n < 2:
        raise ValueError("need at least two observations to estimate a covariance")
    Xc = X - X.mean(axis=0)
    sample = (Xc.T @ Xc) / n          # the LW derivation uses 1/n, not 1/(n-1)
    mu = np.trace(sample) / p
    target = mu * np.eye(p)

    d2 = float(np.sum((sample - target) ** 2))
    # b2 = (1/n^2) * sum_i ||x_i x_i' - S||_F^2, expanded so it is one pass
    # rather than n outer products:
    #   ||x_i x_i'||_F^2 = (x_i'x_i)^2,  <x_i x_i', S> = x_i' S x_i
    sq = np.einsum("ij,ij->i", Xc, Xc)                 # x_i'x_i
    quad = np.einsum("ij,jk,ik->", Xc, sample, Xc)     # sum_i x_i' S x_i
    b2 = float((np.sum(sq ** 2) - 2.0 * quad + n * np.sum(sample ** 2)) / n ** 2)

    b2 = min(b2, d2)
    shrink = b2 / d2 if d2 > 0 else 0.0
    cov = shrink * target + (1.0 - shrink) * sample
    return pd.DataFrame(cov * annualize, index=returns.columns,
                        columns=returns.columns)


# Loss tolerances, not risk-aversion coefficients. Each is the worst annual
# drawdown the investor says they could hold through, at 95% confidence — a
# one-in-twenty year. The labels are the ones people recognise; the numbers
# are what the optimiser actually uses.
RISK_PROFILES: dict[str, dict] = {
    "conservative": {
        "label": "Conservative",
        "max_annual_loss": 0.08,
        "summary": "Capital preservation first. A bad year costs under 8%.",
    },
    "moderate": {
        "label": "Moderate",
        "max_annual_loss": 0.15,
        "summary": "Some growth, limited drawdown. A bad year costs about 15%.",
    },
    "balanced": {
        "label": "Balanced",
        "max_annual_loss": 0.22,
        "summary": "Long horizon, ordinary market risk. A bad year costs about 22%.",
    },
    "growth": {
        "label": "Growth",
        "max_annual_loss": 0.32,
        "summary": "Growth priority, able to sit through a deep fall.",
    },
    "aggressive": {
        "label": "Aggressive",
        "max_annual_loss": 0.45,
        "summary": "Maximum long-run return, accepts severe drawdowns.",
    },
}


@dataclass
class InvestorAllocation:
    """A portfolio sized to a loss the investor said they could live with.

    `volatility_budget` is the annual volatility that loss tolerance implies.
    When a return history is attached it is computed from the tangency
    portfolio's own Cornish-Fisher quantile rather than the Gaussian one;
    `volatility_budget_gaussian` is what the normal assumption would have
    allowed, and the gap between them is the risk a Gaussian sizing would
    have taken without the investor agreeing to it.
    """
    profile: str
    max_annual_loss: float
    confidence: float
    volatility_budget: float
    volatility_budget_gaussian: float
    skew: float
    excess_kurtosis: float
    portfolio: "CompletePortfolio"
    tangency: "Portfolio"
    implied_risk_aversion: float
    leverage_capped: bool
    risk_free: float

    @property
    def tail_penalty(self) -> float:
        """Volatility the fat-tail adjustment removed from the budget."""
        return self.volatility_budget_gaussian - self.volatility_budget

    @property
    def expected_loss_at_confidence(self) -> float:
        """The bad-year loss this allocation actually implies, Gaussian."""
        from scipy import stats as _st
        z = _st.norm.ppf(1 - self.confidence)
        return -(self.portfolio.expected_return
                 + z * self.portfolio.volatility)

    def summary(self) -> str:
        lines = [
            f"{RISK_PROFILES.get(self.profile, {}).get('label', 'Custom')}: "
            f"tolerates {self.max_annual_loss:.0%} in a "
            f"1-in-{round(1 / (1 - self.confidence))} year",
            f"  volatility budget {self.volatility_budget:.2%}"
            + (f" (Gaussian would allow {self.volatility_budget_gaussian:.2%}; "
               f"fat tails cost {self.tail_penalty:.2%})"
               if abs(self.tail_penalty) > 1e-6 else ""),
            f"  {self.portfolio.risky_weight:.1%} in the tangency portfolio, "
            f"{self.portfolio.risk_free_weight:.1%} at "
            f"{self.risk_free:.2%}",
            f"  E[r]={self.portfolio.expected_return:.2%}  "
            f"vol={self.portfolio.volatility:.2%}  "
            f"Sharpe={self.portfolio.sharpe:.3f}",
            f"  implied risk aversion A={self.implied_risk_aversion:.2f}",
        ]
        if self.leverage_capped:
            lines.append(
                "  capped at 100% invested: the stated tolerance would "
                "support more risk than an unlevered book can carry")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


class NoTangencyPortfolio(ValueError):
    """No CAL from this risk-free rate touches the efficient branch.

    Raised when rf is at or above the global minimum-variance return. See
    `MarkowitzOptimizer.tangency` for why that case has no solution rather
    than a hard-to-find one.
    """

    def __init__(self, message: str, *, risk_free: float, gmv_return: float,
                 asymptotic_sharpe: float):
        super().__init__(message)
        self.risk_free = risk_free
        self.gmv_return = gmv_return
        self.asymptotic_sharpe = asymptotic_sharpe


@dataclass
class Portfolio:
    weights: pd.Series
    expected_return: float
    volatility: float
    sharpe: float

    def __str__(self) -> str:
        w = ", ".join(f"{k}={v:.2%}" for k, v in self.weights.items())
        return (f"E[r]={self.expected_return:.2%}  vol={self.volatility:.2%}  "
                f"Sharpe={self.sharpe:.2f}\n  weights: {w}")


@dataclass
class CompletePortfolio:
    """A split between the risk-free asset and the tangency portfolio.

    `risky_weight` (y) above 1 means borrowing at rf to lever the tangency
    portfolio; below 0 means shorting it, which is never optimal for a
    risk-averse investor facing a positive risk premium.
    """
    risky_weight: float
    risk_free_weight: float
    expected_return: float
    volatility: float
    sharpe: float
    asset_weights: pd.Series
    risk_free: float

    @property
    def is_levered(self) -> bool:
        return self.risky_weight > 1.0 + _TOL

    def __str__(self) -> str:
        w = ", ".join(f"{k}={v:.2%}" for k, v in self.asset_weights.items())
        return (f"E[r]={self.expected_return:.2%}  vol={self.volatility:.2%}  "
                f"Sharpe={self.sharpe:.2f}\n"
                f"  {self.risky_weight:.2%} in the tangency portfolio, "
                f"{self.risk_free_weight:.2%} at {self.risk_free:.2%}\n"
                f"  holdings: {w}")


@dataclass
class PortfolioDiagnosis:
    """Where a portfolio someone actually holds sits against the frontier.

    Three separate questions, because they have different answers and people
    conflate them:

      * **Return given up at the same risk.** Holding this volatility, the
        efficient branch would have paid `frontier_return`. The difference is
        what the current mix costs in return for risk already being carried.
      * **Risk carried for the same return.** The minimum-variance portfolio
        with this expected return has volatility `frontier_volatility`. The
        difference is risk taken for nothing.
      * **Shortfall against the CAL.** Once a risk-free asset exists the
        frontier stops being the benchmark: mixing cash with the tangency
        portfolio beats every frontier portfolio but one. `sharpe_gap` is the
        honest measure, because it does not change as you lever up or down.

    A portfolio holding cash can sit at a volatility below the global
    minimum-variance portfolio, where no fully-invested risky portfolio
    exists. The frontier comparison is then undefined and comes back None;
    the CAL comparison still applies, which is the point of the CAL.
    """
    weights: pd.Series
    cash_weight: float
    expected_return: float
    volatility: float
    sharpe: float
    risk_free: float

    frontier_return: Optional[float]        # best return at this volatility
    return_gap: Optional[float]             # how much is being left on the table
    efficient_weights: Optional[pd.Series]  # the mix that would earn it

    frontier_volatility: float              # least risk for this return
    volatility_gap: float                   # risk carried for nothing

    cal_return: Optional[float]             # the CAL's return at this volatility
    cal_gap: Optional[float]
    sharpe_gap: Optional[float]             # best Sharpe minus this one
    cal_weights: Optional[pd.Series]        # tangency scaled to this volatility
    cal_cash_weight: Optional[float]

    @property
    def is_efficient(self) -> bool:
        """True when the portfolio is on the efficient branch, to 1bp."""
        return self.return_gap is not None and self.return_gap <= 1e-4

    @property
    def is_on_cal(self) -> bool:
        return self.sharpe_gap is not None and self.sharpe_gap <= 1e-4

    def trades(self) -> pd.Series:
        """Weight changes that move this portfolio onto the CAL.

        Positive is buy, negative is sell. The cash leg is the balancing
        item and is reported separately as `cal_cash_weight`.
        """
        if self.cal_weights is None:
            return pd.Series(dtype=float)
        return (self.cal_weights - self.weights).rename("trade")

    def summary(self) -> str:
        lines = [
            f"Held:      E[r]={self.expected_return:.2%}  "
            f"vol={self.volatility:.2%}  Sharpe={self.sharpe:.3f}"
            + (f"  (cash {self.cash_weight:.1%})"
               if abs(self.cash_weight) > 1e-9 else ""),
        ]
        if self.return_gap is None:
            lines.append(
                "  Below the minimum-variance volatility, which only a cash "
                "position can reach — no fully-invested risky portfolio "
                "exists at this risk, so the frontier comparison does not "
                "apply. Compare against the CAL.")
        elif self.is_efficient:
            lines.append("  On the efficient frontier.")
        else:
            lines.append(
                f"  Giving up {self.return_gap:.2%} of return at this "
                f"volatility (efficient: {self.frontier_return:.2%}).")
        lines.append(
            f"  Carrying {self.volatility_gap:.2%} more volatility than "
            f"needed for this return (minimum: "
            f"{self.frontier_volatility:.2%}).")
        if self.sharpe_gap is None:
            lines.append("  No tangency portfolio exists at this risk-free "
                         "rate, so there is no CAL to compare against.")
        elif self.is_on_cal:
            lines.append("  On the capital allocation line.")
        else:
            lines.append(
                f"  {self.cal_gap:.2%} below the CAL at this volatility; "
                f"Sharpe shortfall {self.sharpe_gap:.3f}.")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


@dataclass
class FrontierGeometry:
    """The four scalars that determine the whole minimum-variance frontier.

        A = 1'S^-1 1      B = 1'S^-1 mu      C = mu'S^-1 mu      D = AC - B^2

    with S the covariance and mu the expected returns. A > 0 whenever S is
    positive definite; D > 0 unless mu is proportional to the unit vector, in
    which case every portfolio has the same expected return and the frontier
    collapses to a point.

    The frontier is the hyperbola sigma^2(r) = (A r^2 - 2 B r + C) / D, whose
    vertex is the global minimum-variance portfolio at r = B/A, sigma^2 = 1/A.
    Returns above B/A are the efficient branch; returns below it are dominated
    — same volatility, less return.
    """
    a: float
    b: float
    c: float
    d: float

    @property
    def gmv_return(self) -> float:
        return self.b / self.a

    @property
    def gmv_variance(self) -> float:
        return 1.0 / self.a

    @property
    def gmv_volatility(self) -> float:
        return math.sqrt(self.gmv_variance)

    @property
    def asymptotic_sharpe(self) -> float:
        """sqrt(D/A) — the slope the frontier approaches as return grows.

        The frontier is a hyperbola, so its asymptote has finite slope. No
        portfolio of risky assets alone attains a Sharpe ratio above this at
        any risk-free rate equal to the GMV return, which is exactly why that
        case has no tangency portfolio.
        """
        return math.sqrt(self.d / self.a)

    def variance_at(self, target_return: float) -> float:
        return (self.a * target_return ** 2 - 2.0 * self.b * target_return
                + self.c) / self.d

    def volatility_at(self, target_return: float) -> float:
        return math.sqrt(max(self.variance_at(target_return), 0.0))

    def return_at_volatility(self, volatility: float) -> Optional[float]:
        """The efficient branch's return at a given volatility, or None.

        Inverting sigma^2(r) = (A r^2 - 2B r + C)/D gives
        r = (B +/- sqrt(B^2 - A(C - D sigma^2))) / A, and the upper branch
        takes the plus. The discriminant is non-negative exactly when
        sigma >= sqrt(1/A), so a volatility below the global minimum-variance
        portfolio's returns None rather than a complex number: no portfolio
        of these assets reaches it.
        """
        disc = self.b ** 2 - self.a * (self.c - self.d * volatility ** 2)
        if disc < 0:
            # At exactly the global minimum-variance volatility the
            # discriminant is analytically zero, so floating point lands it a
            # few ulp either side depending on the BLAS build. Treating a tiny
            # negative value as "no portfolio reaches this volatility" made
            # diagnose() return None for the GMV portfolio itself on some
            # machines and a clean 0.0 on others. Snap the vertex to zero and
            # reject only genuinely unreachable volatilities.
            if disc > -_TOL * max(self.b ** 2, 1.0):
                disc = 0.0
            else:
                return None
        return (self.b + math.sqrt(disc)) / self.a

    def is_efficient(self, target_return: float) -> bool:
        return target_return >= self.gmv_return - _TOL

    def max_sharpe(self, risk_free: float) -> float:
        """The highest Sharpe ratio attainable, = sqrt((mu-rf)'S^-1(mu-rf)).

        Expanded in A, B, C this is sqrt(C - 2 rf B + rf^2 A). Valid only
        when a tangency portfolio exists (rf < the GMV return); otherwise the
        supremum is `asymptotic_sharpe` and is never attained.
        """
        return math.sqrt(max(self.c - 2.0 * risk_free * self.b
                             + risk_free ** 2 * self.a, 0.0))

    def __str__(self) -> str:
        return (f"A={self.a:.4f} B={self.b:.4f} C={self.c:.4f} D={self.d:.4f}\n"
                f"  GMV: E[r]={self.gmv_return:.2%} vol={self.gmv_volatility:.2%}\n"
                f"  asymptotic Sharpe={self.asymptotic_sharpe:.3f}")


@dataclass
class CapitalAllocationLine:
    """Combinations of the risk-free asset and one risky portfolio.

    Every point on the line is `y` in the tangency portfolio and `1-y` at rf:

        E[r] = rf + y (E[r_t] - rf)        sigma = y sigma_t

    Eliminating y gives E[r] = rf + S_t * sigma, a straight line from (0, rf)
    with slope equal to the tangency portfolio's Sharpe ratio. Because that
    slope is the highest attainable, the line lies above the efficient
    frontier everywhere except the single tangency point — which is Tobin's
    two-fund separation: the risky mix is the same for every investor, and
    risk tolerance only sets y.

    The same line is called the Capital Market Line when the tangency
    portfolio is the market portfolio.
    """
    risk_free: float
    tangency: Portfolio
    long_only: bool = False

    @property
    def slope(self) -> float:
        """Reward-to-volatility; identical to the tangency Sharpe ratio."""
        return self.tangency.sharpe

    def return_at(self, volatility: float) -> float:
        return self.risk_free + self.slope * volatility

    def volatility_at(self, expected_return: float) -> float:
        if abs(self.slope) < _TOL:
            return float("nan")
        return (expected_return - self.risk_free) / self.slope

    def optimal_allocation(self, risk_aversion: float) -> float:
        """y* = (E[r_t] - rf) / (A sigma_t^2), the maximizer of E[r] - A/2 sigma^2.

        The textbook result. `risk_aversion` is the coefficient A in that
        utility function, conventionally 2-4 for a typical investor; it is
        not a volatility target and not a percentage.
        """
        if risk_aversion <= 0:
            raise ValueError("risk_aversion must be positive")
        var = self.tangency.volatility ** 2
        if var < _TOL:
            raise ValueError("tangency portfolio has no variance to allocate against")
        return (self.tangency.expected_return - self.risk_free) / (risk_aversion * var)

    def allocate(self, risky_weight: float) -> CompletePortfolio:
        y = float(risky_weight)
        ret = self.risk_free + y * (self.tangency.expected_return - self.risk_free)
        vol = abs(y) * self.tangency.volatility
        sharpe = (ret - self.risk_free) / vol if vol > _TOL else float("nan")
        return CompletePortfolio(
            risky_weight=y, risk_free_weight=1.0 - y,
            expected_return=ret, volatility=vol, sharpe=sharpe,
            asset_weights=self.tangency.weights * y,
            risk_free=self.risk_free)

    def complete_portfolio(self, *, risk_aversion: Optional[float] = None,
                           target_volatility: Optional[float] = None,
                           target_return: Optional[float] = None
                           ) -> CompletePortfolio:
        """The point on the line picked by exactly one of three criteria."""
        given = [x is not None for x in
                 (risk_aversion, target_volatility, target_return)]
        if sum(given) != 1:
            raise ValueError("pass exactly one of risk_aversion, "
                             "target_volatility or target_return")
        if risk_aversion is not None:
            y = self.optimal_allocation(risk_aversion)
        elif target_volatility is not None:
            if target_volatility < 0:
                raise ValueError("target_volatility must be non-negative")
            y = target_volatility / self.tangency.volatility
        else:
            y = self.volatility_at(target_return) / self.tangency.volatility
        return self.allocate(y)

    def table(self, points: int = 25,
              max_volatility: Optional[float] = None) -> pd.DataFrame:
        """Sampled points along the line, for plotting against the frontier."""
        hi = max_volatility if max_volatility is not None \
            else self.tangency.volatility * 2.0
        vols = np.linspace(0.0, hi, points)
        return pd.DataFrame({
            "volatility": vols,
            "return": [self.return_at(v) for v in vols],
            "risky_weight": vols / self.tangency.volatility,
        })

    def __str__(self) -> str:
        return (f"CAL from rf={self.risk_free:.2%}, slope={self.slope:.3f}\n"
                f"  tangency: E[r]={self.tangency.expected_return:.2%} "
                f"vol={self.tangency.volatility:.2%}")


def _stats(w, mu, cov, rf):
    r = float(w @ mu)
    v = float(np.sqrt(max(w @ cov @ w, 0.0)))
    return r, v, (r - rf) / v if v > _TOL else float("nan")


def _as_bounds(bounds, n: int):
    """Accept either one (lo, hi) pair for every asset or a sequence of n."""
    if bounds is None:
        return None
    seq = list(bounds)
    if len(seq) == 2 and all(isinstance(x, (int, float)) or x is None
                             for x in seq):
        return tuple((seq[0], seq[1]) for _ in range(n))   # one pair for all
    if len(seq) != n:
        raise ValueError(f"bounds must be one (lo, hi) pair or {n} of them, "
                         f"got {len(seq)}")
    return tuple(tuple(b) for b in seq)


class MarkowitzOptimizer:
    def __init__(self, mu: pd.Series, cov: pd.DataFrame, rf: float = 0.0):
        common = [a for a in mu.index if a in set(cov.index)]
        if not common:
            raise ValueError("mu and cov share no assets")
        self.assets = list(common)
        self.mu = mu.loc[common].to_numpy(dtype=float)
        self.cov = cov.loc[common, common].to_numpy(dtype=float)
        if not np.all(np.isfinite(self.mu)) or not np.all(np.isfinite(self.cov)):
            raise ValueError("mu and cov must be finite; drop or fill assets "
                             "with missing moments before optimizing")
        self.rf = float(rf)
        self._inv = np.linalg.pinv(self.cov)
        self._geometry: Optional[FrontierGeometry] = None
        # Optional return panel, used only for the tail shape in
        # for_investor(). Never replaces the covariance passed in.
        self._history: Optional[pd.DataFrame] = None
        self._periods_per_year: int = 252

        cond = self.condition_number
        if cond > _CONDITION_WARN:
            warnings.warn(
                f"covariance condition number is {cond:.3g}: inverting it "
                f"amplifies estimation error in mu by about that factor, so "
                f"unconstrained weights will be large and unstable. Use "
                f"ledoit_wolf_cov, or constrain with optimize(long_only=True).",
                RuntimeWarning, stacklevel=2)

    @classmethod
    def from_universe(cls, u: Universe, rf: float = 0.0,
                      shrinkage: bool = True) -> "MarkowitzOptimizer":
        mu = u.mean_returns(annualized=True)
        cov = (ledoit_wolf_cov(u.returns, u.periods_per_year) if shrinkage
               else u.cov_matrix(annualized=True))
        return cls(mu, cov, rf)

    # ---- geometry ---------------------------------------------------------
    @property
    def condition_number(self) -> float:
        return float(np.linalg.cond(self.cov))

    def geometry(self) -> FrontierGeometry:
        """A, B, C, D — computed once and cached."""
        if self._geometry is not None:
            return self._geometry
        ones = np.ones(len(self.assets))
        a = float(ones @ self._inv @ ones)
        b = float(ones @ self._inv @ self.mu)
        c = float(self.mu @ self._inv @ self.mu)
        d = a * c - b * b
        if a <= _TOL:
            raise ValueError("covariance is singular or not positive definite; "
                             "A = 1'S^-1 1 must be strictly positive")
        if d <= _TOL * max(a * c, 1.0):
            raise ValueError(
                "D = AC - B^2 is zero: expected returns are proportional to "
                "the unit vector, so every portfolio has the same expected "
                "return and there is no frontier to trace. Check that mu "
                "varies across assets.")
        self._geometry = FrontierGeometry(a, b, c, d)
        return self._geometry

    def _wrap(self, w) -> Portfolio:
        r, v, s = _stats(w, self.mu, self.cov, self.rf)
        return Portfolio(pd.Series(w, index=self.assets), r, v, s)

    # ---- analytic (unconstrained) ----------------------------------------
    def min_variance(self) -> Portfolio:
        """Global minimum-variance portfolio: w = S^-1 1 / (1'S^-1 1).

        The one frontier portfolio that needs no expected returns at all,
        which is why it is far more stable out of sample than the tangency
        portfolio.
        """
        ones = np.ones(len(self.assets))
        w = self._inv @ ones
        return self._wrap(w / (ones @ w))

    def tangency(self, long_only: bool = False) -> Portfolio:
        """The portfolio the CAL touches: w proportional to S^-1 (mu - rf 1).

        The normalising constant is kappa = 1'S^-1(mu - rf 1) = B - rf A,
        which is positive exactly when rf is below the GMV return. Dividing
        by a negative kappa flips every weight and lands on the *inefficient*
        branch — the minimum-Sharpe portfolio, returned as if it were the
        maximum. That is a sign error, not an edge case, so this raises
        instead.

        When rf is at or above the GMV return, no tangency portfolio exists:
        the Sharpe ratio rises monotonically along the efficient branch toward
        the asymptotic slope sqrt(D/A) without ever reaching it. The supremum
        is real; the maximiser is not. `long_only=True` searches the simplex
        instead, which is compact, so a maximum always exists there.
        """
        if long_only:
            return self.optimize("max_sharpe", long_only=True)
        g = self.geometry()
        kappa = g.b - self.rf * g.a
        if kappa <= _TOL * max(abs(g.b), 1.0):
            raise NoTangencyPortfolio(
                f"No tangency portfolio exists: rf={self.rf:.4%} is at or "
                f"above the global minimum-variance return "
                f"{g.gmv_return:.4%}. Along the efficient branch the Sharpe "
                f"ratio then increases toward the frontier's asymptotic slope "
                f"{g.asymptotic_sharpe:.4f} without attaining it, so the "
                f"maximum is a supremum and not a portfolio. Lower rf, "
                f"re-examine the expected returns, or call "
                f"tangency(long_only=True) for the best Sharpe ratio on a "
                f"bounded feasible set.",
                risk_free=self.rf, gmv_return=g.gmv_return,
                asymptotic_sharpe=g.asymptotic_sharpe)
        w = (self._inv @ (self.mu - self.rf)) / kappa
        return self._wrap(w)

    def max_sharpe(self) -> Portfolio:
        """Alias for `tangency()`, kept because it is the familiar name."""
        return self.tangency(long_only=False)

    def target_return(self, target: float) -> Portfolio:
        """Minimum-variance portfolio with a given expected return.

        Below the GMV return this is a valid *minimum-variance* portfolio but
        not an efficient one — a portfolio with the same volatility and a
        higher return exists, by reflection about r = B/A.
        """
        g = self.geometry()
        ones = np.ones(len(self.assets))
        lam = (g.c - g.b * target) / g.d
        gam = (g.a * target - g.b) / g.d
        w = lam * (self._inv @ ones) + gam * (self._inv @ self.mu)
        return self._wrap(w)

    # ---- the capital allocation line --------------------------------------
    def capital_allocation_line(self, long_only: bool = False
                                ) -> CapitalAllocationLine:
        """The line from (0, rf) through the tangency portfolio."""
        return CapitalAllocationLine(risk_free=self.rf,
                                     tangency=self.tangency(long_only=long_only),
                                     long_only=long_only)

    def complete_portfolio(self, *, risk_aversion: Optional[float] = None,
                           target_volatility: Optional[float] = None,
                           target_return: Optional[float] = None,
                           long_only: bool = False) -> CompletePortfolio:
        """Optimal split between rf and the tangency portfolio."""
        return self.capital_allocation_line(long_only=long_only) \
            .complete_portfolio(risk_aversion=risk_aversion,
                                target_volatility=target_volatility,
                                target_return=target_return)

    # ---- allocating for an investor --------------------------------------
    def for_investor(self, profile: Optional[str] = None, *,
                     max_annual_loss: Optional[float] = None,
                     confidence: float = 0.95,
                     tail_adjust: bool = True,
                     allow_leverage: bool = False,
                     long_only: bool = False) -> "InvestorAllocation":
        """Build a portfolio from a loss the investor can live with.

        Mean-variance theory parameterises risk tolerance as a coefficient A
        in U = E[r] - (A/2)sigma^2. Nobody knows their A. People do know,
        roughly, how large a fall they could sit through without selling,
        and that is a VaR statement: "in a bad year, no worse than 20%".

        So the chain runs loss tolerance -> volatility budget -> a point on
        the capital allocation line, rather than starting from a coefficient
        nobody can supply. The implied A comes back in the result, because it
        is the number the textbook uses and it is worth seeing.

        `profile` is a shorthand for a tolerance — see `RISK_PROFILES`. Pass
        `max_annual_loss` instead to state it directly.

        `tail_adjust` sizes against the Cornish-Fisher quantile of the
        tangency portfolio's own return distribution rather than the Gaussian
        one, when a return history is available via `attach_history`. Equity
        returns are left-skewed and fat-tailed, so the Gaussian quantile
        understates the bad year; a portfolio sized on it carries more risk
        than the investor agreed to.

        `allow_leverage=False` caps the risky weight at 100%. Borrowing to
        reach a volatility target is a real option on the CAL and a poor fit
        for most people stating a loss limit, so it is off by default.
        """
        from .risk import cal_volatility_for_loss_tolerance

        if (profile is None) == (max_annual_loss is None):
            raise ValueError("pass exactly one of profile or max_annual_loss")
        if profile is not None:
            key = str(profile).strip().lower()
            if key not in RISK_PROFILES:
                raise ValueError(
                    "unknown profile %r; choose from %s"
                    % (profile, ", ".join(RISK_PROFILES)))
            max_annual_loss = RISK_PROFILES[key]["max_annual_loss"]
        loss = float(max_annual_loss)

        cal = self.capital_allocation_line(long_only=long_only)
        tan = cal.tangency

        skew = kurt = 0.0
        if tail_adjust and self._history is not None:
            port = self._history.reindex(columns=self.assets).dropna() \
                   @ tan.weights.to_numpy()
            if len(port) >= 60:
                from scipy import stats as _st
                # Annualise the shape of the distribution: skew and excess
                # kurtosis both shrink as independent periods are added.
                n = self._periods_per_year
                skew = float(_st.skew(port)) / math.sqrt(n)
                kurt = float(_st.kurtosis(port)) / n

        # The budget is solved against the blended portfolio, not the risky
        # fund alone: holding cash lowers the expected return as well as the
        # volatility, so sizing on the tangency portfolio's own return
        # overshoots the loss the investor agreed to.
        budget_gauss = cal_volatility_for_loss_tolerance(
            loss, self.rf, tan.sharpe, confidence)
        budget = cal_volatility_for_loss_tolerance(
            loss, self.rf, tan.sharpe, confidence, skew=skew,
            excess_kurtosis=kurt) if (skew or kurt) else budget_gauss

        y = budget / tan.volatility if tan.volatility > _TOL else 0.0
        capped = allow_leverage is False and y > 1.0
        if capped:
            y = 1.0
        cp = cal.allocate(y)

        implied_a = float("nan")
        if y > _TOL and tan.volatility > _TOL:
            implied_a = (tan.expected_return - self.rf) / (y * tan.volatility ** 2)

        return InvestorAllocation(
            profile=(profile.lower() if profile else "custom"),
            max_annual_loss=loss, confidence=confidence,
            volatility_budget=budget,
            volatility_budget_gaussian=budget_gauss,
            skew=skew, excess_kurtosis=kurt,
            portfolio=cp, tangency=tan, implied_risk_aversion=implied_a,
            leverage_capped=capped, risk_free=self.rf)

    def attach_history(self, returns: pd.DataFrame,
                       periods_per_year: int = 252) -> "MarkowitzOptimizer":
        """Give the optimizer the return panel its moments came from.

        Only the tail shape is taken from it, for `for_investor`. The
        covariance stays whatever was passed to the constructor, so a
        shrunk covariance is not quietly replaced by the sample one.
        """
        self._history = returns
        self._periods_per_year = int(periods_per_year)
        return self

    # ---- diagnosing a portfolio someone holds ----------------------------
    def diagnose(self, weights, cash: Optional[float] = None
                 ) -> PortfolioDiagnosis:
        """Measure a held portfolio against the frontier and the CAL.

        `weights` maps asset to weight — a dict, a Series, or a sequence in
        the optimizer's asset order. Assets left out are held at zero.

        Weights that sum to less than one leave the remainder in cash at the
        risk-free rate; more than one means borrowing there. Pass `cash`
        explicitly to state it rather than infer it, in which case the risky
        weights are used as given and the two need not sum to one.

        The answer is a gap, never a recommendation: it says what the stated
        assumptions imply about this mix, and those assumptions are the
        user's. Change mu and every number here changes.
        """
        if isinstance(weights, dict):
            w = pd.Series(weights, dtype=float)
        elif isinstance(weights, pd.Series):
            w = weights.astype(float)
        else:
            w = pd.Series(np.asarray(weights, dtype=float), index=self.assets)
        unknown = [a for a in w.index if a not in set(self.assets)]
        if unknown:
            raise ValueError("not in the universe: %s" % ", ".join(map(str, unknown)))
        w = w.reindex(self.assets).fillna(0.0)
        wv = w.to_numpy()

        risky = float(wv.sum())
        cash_w = float(cash) if cash is not None else 1.0 - risky

        r = float(wv @ self.mu) + cash_w * self.rf
        v = float(np.sqrt(max(wv @ self.cov @ wv, 0.0)))
        s = (r - self.rf) / v if v > _TOL else float("nan")

        g = self.geometry()

        # Best return available at this volatility, on the efficient branch.
        fr = g.return_at_volatility(v)
        gap = None if fr is None else max(fr - r, 0.0)
        eff_w = None
        if fr is not None:
            eff_w = self.target_return(fr).weights

        # Least volatility that still earns this return. Below the GMV return
        # the efficient answer is the GMV portfolio itself: it has less risk
        # AND more return, so that is the honest comparison.
        target = max(r, g.gmv_return)
        fv = g.volatility_at(target)
        vgap = max(v - fv, 0.0)

        cal_r = cal_gap = sgap = cal_w = cal_cash = None
        try:
            cal = self.capital_allocation_line()
        except NoTangencyPortfolio:
            pass
        else:
            cal_r = cal.return_at(v)
            cal_gap = max(cal_r - r, 0.0)
            sgap = max(cal.slope - s, 0.0) if s == s else None
            y = v / cal.tangency.volatility if cal.tangency.volatility > _TOL else 0.0
            cal_w = cal.tangency.weights * y
            cal_cash = 1.0 - y

        return PortfolioDiagnosis(
            weights=w, cash_weight=cash_w, expected_return=r, volatility=v,
            sharpe=s, risk_free=self.rf,
            frontier_return=fr, return_gap=gap, efficient_weights=eff_w,
            frontier_volatility=fv, volatility_gap=vgap,
            cal_return=cal_r, cal_gap=cal_gap, sharpe_gap=sgap,
            cal_weights=cal_w, cal_cash_weight=cal_cash)

    # ---- numeric (constrained, e.g. long-only) ---------------------------
    def optimize(self, objective: str = "max_sharpe", long_only: bool = True,
                 bounds: Optional[Sequence] = None,
                 target_return: Optional[float] = None) -> Portfolio:
        n = len(self.assets)
        bnds = _as_bounds(bounds, n) or \
            tuple((0.0, 1.0) if long_only else (-1.0, 1.0) for _ in range(n))

        # A naive 1/n start can sit outside the box; clip into it, then push
        # back onto the budget constraint so SLSQP starts feasible.
        w0 = np.clip(np.ones(n) / n,
                     [b[0] for b in bnds], [b[1] for b in bnds])
        total = w0.sum()
        if abs(total) > _TOL:
            w0 = w0 / total
            w0 = np.clip(w0, [b[0] for b in bnds], [b[1] for b in bnds])

        cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
        if target_return is not None:
            cons.append({"type": "eq",
                         "fun": lambda w: float(w @ self.mu) - target_return})

        if objective == "min_variance":
            def fun(w):
                return float(w @ self.cov @ w)
        elif objective == "max_sharpe":
            if long_only and np.all(self.mu - self.rf <= 0):
                warnings.warn(
                    f"every asset's expected return is at or below rf="
                    f"{self.rf:.2%}, so the best long-only Sharpe ratio is "
                    f"negative: this returns the least-bad portfolio, not an "
                    f"attractive one.", RuntimeWarning, stacklevel=2)

            def fun(w):
                _, _, s = _stats(w, self.mu, self.cov, self.rf)
                return -s if s == s else 1e6     # NaN at zero vol: push away
        else:
            raise ValueError("objective must be 'max_sharpe' or 'min_variance'")

        res = minimize(fun, w0, method="SLSQP", bounds=bnds, constraints=cons,
                       options={"maxiter": 500, "ftol": 1e-10})
        if not res.success:
            warnings.warn(f"SLSQP did not converge: {res.message}",
                          RuntimeWarning, stacklevel=2)
        return self._wrap(res.x)

    # ---- the frontier -----------------------------------------------------
    def efficient_frontier(self, points: int = 25, long_only: bool = False,
                           include_inefficient: bool = False,
                           max_return: Optional[float] = None) -> pd.DataFrame:
        """Points along the frontier, efficient branch only by default.

        The frontier is a hyperbola in (sigma, r) with its vertex at the GMV
        portfolio. Sampling target returns from min(mu) upward — the obvious
        choice, and the one this used to make — puts roughly half the points
        on the lower branch, where every point is dominated by one directly
        above it at the same volatility. Plotting that draws a sideways
        parabola rather than a frontier, and picking from it by volatility
        silently selects the worse of the two portfolios at that risk.

        So the default range starts at the GMV return. Pass
        `include_inefficient=True` to get the full hyperbola for plotting;
        the `efficient` column flags which branch each point is on.
        """
        if long_only:
            lo = self.optimize("min_variance", long_only=True).expected_return
            floor = float(self.mu.min())
        else:
            g = self.geometry()
            lo = g.gmv_return
            floor = min(float(self.mu.min()), 2.0 * lo - float(self.mu.max()))

        gmv_return = lo        # the branch boundary, kept before lo moves

        hi = max_return if max_return is not None else float(self.mu.max())
        if hi <= lo + _TOL:
            # Every asset's return is at or below the GMV return, which an
            # unconstrained frontier reaches by shorting. Give the curve a
            # span rather than returning a single point.
            hi = lo + max(abs(lo), 0.01) * 0.5
        if include_inefficient:
            lo = min(floor, lo)

        targets = np.linspace(lo, hi, points)
        rows = []
        for t in targets:
            try:
                p = (self.optimize("min_variance", long_only=True,
                                   target_return=float(t)) if long_only
                     else self.target_return(float(t)))
            except Exception:
                continue
            rows.append({"target_return": float(t),
                         "return": p.expected_return,
                         "volatility": p.volatility,
                         "sharpe": p.sharpe,
                         "efficient": bool(p.expected_return
                                           >= gmv_return - _TOL)})
        return pd.DataFrame(rows)

    # ---- reporting --------------------------------------------------------
    def summary(self) -> str:
        g = self.geometry()
        gmv = self.min_variance()
        out = [str(g), f"  rf={self.rf:.2%}"]
        try:
            cal = self.capital_allocation_line()
            out.append(f"  tangency: E[r]={cal.tangency.expected_return:.2%} "
                       f"vol={cal.tangency.volatility:.2%} "
                       f"Sharpe={cal.slope:.3f}")
            out.append(f"  CAL beats GMV by "
                       f"{cal.slope - gmv.sharpe:.3f} Sharpe")
        except NoTangencyPortfolio as exc:
            out.append(f"  no tangency portfolio: {exc}")
        return "\n".join(out)
