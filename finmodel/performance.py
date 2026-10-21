"""Portfolio performance through time, with external cash flows.

`portfolio_builder` answers "what should this portfolio hold and what does it
look like today". This module answers the other question an investor asks
first: "how did it actually do" — which needs a history, and needs to deal
with money going in and out along the way.

Two measures, and the difference between them matters:

**Time-weighted return (TWR)** values the portfolio before every cash flow,
takes each subperiod's return, and links them geometrically. Because the flow
is excluded from the subperiod it lands in, the result is unaffected by how
much money was contributed or when. That is what makes it the right measure of
the *strategy*, and why it is the basis of GIPS-compliant manager reporting:
a manager does not control when clients deposit.

**Money-weighted return (MWR)** is the internal rate of return on the dated
investor cash flows — XIRR. It is what the *investor* earned, because it does
weight each period by how much money was exposed to it. Deposit heavily just
before a rally and MWR beats TWR; deposit just before a drawdown and it
trails.

Neither is more correct. They answer different questions, and reporting one
while the reader assumes the other is how performance gets misrepresented.
The identity worth remembering: **with no external flows the two are equal**,
and the test suite asserts it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

DAYS_PER_YEAR = 365.0        # actual/365, the XIRR convention


# --------------------------------------------------------------------------
# weights
# --------------------------------------------------------------------------
def normalize_weights(n: int, values=None, allow_short: bool = False
                      ) -> np.ndarray:
    """Equal weights by default; otherwise validate what was given."""
    if values is None:
        return np.full(n, 1.0 / n)
    w = np.asarray(list(values), dtype=float)
    if w.size != n:
        raise ValueError("expected %d weights, got %d" % (n, w.size))
    if not allow_short and np.any(w < 0):
        raise ValueError("negative weights require allow_short=True")
    total = w.sum()
    if not np.isfinite(total) or np.isclose(total, 0.0):
        raise ValueError("weights must sum to a non-zero number")
    return w / total          # normalise rather than demand they sum to 1


# --------------------------------------------------------------------------
# XIRR
# --------------------------------------------------------------------------
def xnpv(rate: float, cash_flows: Sequence[tuple], basis: float = DAYS_PER_YEAR
         ) -> float:
    """Net present value of dated cash flows at an annual `rate`."""
    if rate <= -1.0:
        return float("inf")
    flows = sorted(cash_flows, key=lambda cf: cf[0])
    t0 = flows[0][0]
    return float(sum(
        amount / (1.0 + rate) ** ((pd.Timestamp(date) - pd.Timestamp(t0)).days / basis)
        for date, amount in flows))


def sign_changes(cash_flows: Sequence[tuple]) -> int:
    """Count sign alternations in the dated flows.

    By Descartes' rule of signs this bounds the number of positive real roots.
    One change means the IRR is unique; more than one means there may be
    several rates that zero the NPV, and the single number XIRR returns is
    then not the whole story.
    """
    amounts = [a for _, a in sorted(cash_flows, key=lambda cf: cf[0])
               if a != 0]
    return sum(1 for a, b in zip(amounts, amounts[1:]) if a * b < 0)


def xirr(cash_flows: Sequence[tuple], lower: float = -0.9999,
         upper: float = 10.0, tol: float = 1e-10, max_iter: int = 300,
         basis: float = DAYS_PER_YEAR) -> float:
    """Annualised internal rate of return on dated cash flows (actual/365).

    Bisection on a bracketed sign change: slower than Newton but it cannot
    diverge, and cash-flow NPV curves are badly behaved enough that robustness
    is worth more than iterations. Returns NaN when no bracket exists, which
    is the honest answer for flows that never cross zero.
    """
    flows = [(pd.Timestamp(d), float(a)) for d, a in cash_flows if a is not None]
    if len(flows) < 2:
        return float("nan")
    amounts = [a for _, a in flows]
    if not (any(a > 0 for a in amounts) and any(a < 0 for a in amounts)):
        return float("nan")          # all one sign: no rate solves it

    lo_val = xnpv(lower, flows, basis)
    hi_val = xnpv(upper, flows, basis)
    while lo_val * hi_val > 0 and upper < 1e6:
        upper *= 2.0
        hi_val = xnpv(upper, flows, basis)
    if lo_val * hi_val > 0:
        return float("nan")

    for _ in range(max_iter):
        mid = (lower + upper) / 2.0
        mid_val = xnpv(mid, flows, basis)
        if abs(mid_val) < tol or (upper - lower) < 1e-12:
            return mid
        if lo_val * mid_val <= 0:
            upper = mid
        else:
            lower, lo_val = mid, mid_val
    return (lower + upper) / 2.0


# --------------------------------------------------------------------------
# cash-flow alignment
# --------------------------------------------------------------------------
def align_flows(index: pd.DatetimeIndex, flows: Optional[Mapping]) -> dict:
    """Snap each flow to the first trading day on or after its date.

    A flow dated *before* the sample raises rather than silently landing on
    day one, where it would be indistinguishable from the opening investment
    and would quietly inflate the capital base.
    """
    aligned: dict = {}
    if not flows:
        return aligned
    first = index[0]
    for date, amount in flows.items():
        stamp = pd.Timestamp(date)
        if stamp < first:
            raise ValueError(
                "cash flow dated %s precedes the first price date %s; fold it "
                "into the opening capital or extend the price history"
                % (stamp.date(), first.date()))
        eligible = index[index >= stamp]
        if len(eligible) == 0:
            raise ValueError(
                "cash flow dated %s is after the last price date %s"
                % (stamp.date(), index[-1].date()))
        day = eligible[0]
        aligned[day] = aligned.get(day, 0.0) + float(amount)
    return aligned


def _rebalance_dates(index: pd.DatetimeIndex, freq: Optional[str]) -> set:
    """Last trading day of each period for the given pandas frequency."""
    if not freq:
        return set()
    marks = pd.Series(index, index=index).groupby(
        pd.Grouper(freq=freq)).last().dropna()
    return set(pd.DatetimeIndex(marks.to_numpy()))


# --------------------------------------------------------------------------
# simulation
# --------------------------------------------------------------------------
@dataclass
class PerformanceResult:
    history: pd.DataFrame
    twr: float
    twr_annualized: float
    mwr: float
    ending_value: float
    net_contributions: float
    dollar_gain: float
    cash_flows: list
    days: int
    warnings: list = field(default_factory=list)

    @property
    def flows_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.cash_flows,
                            columns=["Date", "Investor cash flow"])

    def summary(self) -> str:
        def pct(v):
            return "n/a" if v is None or v != v else "%.2f%%" % (v * 100)
        lines = [
            "Performance over %d days (%s to %s)" % (
                self.days, self.history.index[0].date(),
                self.history.index[-1].date()),
            f"  Ending value           {self.ending_value:>15,.2f}",
            f"  Net contributions      {self.net_contributions:>15,.2f}",
            f"  Dollar gain            {self.dollar_gain:>15,.2f}",
            "  Time-weighted return   %15s" % pct(self.twr),
            "    annualized           %15s" % pct(self.twr_annualized),
            "  Money-weighted (XIRR)  %15s" % pct(self.mwr),
        ]
        gap = (self.mwr - self.twr_annualized
               if self.mwr == self.mwr and self.twr_annualized == self.twr_annualized
               else None)
        if gap is not None and abs(gap) > 1e-9:
            lines.append(
                "  MWR %s annualized TWR by %s — cash-flow timing %s"
                % ("beat" if gap > 0 else "trailed", pct(abs(gap)),
                   "helped" if gap > 0 else "hurt"))
        for w in self.warnings:
            lines.append("  warning: %s" % w)
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


def simulate_portfolio(prices: pd.DataFrame, capital: float,
                       flows: Optional[Mapping] = None,
                       initial_weights=None, contribution_weights=None,
                       rebalance: Optional[str] = None,
                       allow_short: bool = False) -> pd.DataFrame:
    """Track units, value and the time-weighted return path day by day.

    Units are bought at the open position and added when a flow arrives, so
    weights drift with prices between flows — which is what an untouched
    portfolio actually does. Pass `rebalance` ('ME', 'QE', 'YE', ...) to reset
    to the initial weights at the end of each period instead.

    Each period's return is measured on the value *before* that day's flow,
    against the previous day's value *after* its flow. That is what removes
    cash-flow timing from the TWR.

    `contribution_weights` decides how new money is invested:

      None / 'pro_rata'  in proportion to what is already held, leaving the
                         asset mix untouched. This is the default because it
                         is what "add money to my portfolio" means, and
                         because it is the only choice that leaves TWR
                         genuinely invariant to the size of the flow.
      'initial'          at the opening weights.
      'equal'            equally across assets.
      array              explicit weights.

    The last three all *change the mix* when the portfolio has drifted, so
    they change every return after the flow. That is real portfolio
    behaviour, not an artefact — but it does mean TWR is then no longer
    independent of how much was deposited, which is usually not what someone
    reaching for TWR intends.
    """
    if prices.empty:
        raise ValueError("no prices given")
    if prices.isna().any().any():
        raise ValueError("prices contain NaN; align them first "
                         "(see data.yahoo.align_prices)")
    if capital <= 0:
        raise ValueError("opening capital must be positive")

    n = prices.shape[1]
    w0 = normalize_weights(n, initial_weights, allow_short)

    pro_rata = contribution_weights is None or (
        isinstance(contribution_weights, str)
        and contribution_weights == "pro_rata")
    if isinstance(contribution_weights, str):
        if contribution_weights == "initial":
            wc = w0
        elif contribution_weights == "equal":
            wc = np.full(n, 1.0 / n)
        elif contribution_weights == "pro_rata":
            wc = None
        else:
            raise ValueError(
                "contribution_weights must be 'pro_rata', 'initial', 'equal', "
                "an array, or None")
    elif contribution_weights is None:
        wc = None
    else:
        wc = normalize_weights(n, contribution_weights, allow_short)

    aligned = align_flows(prices.index, flows)
    rebalance_on = _rebalance_dates(prices.index, rebalance)

    units = capital * w0 / prices.iloc[0].to_numpy()
    records, cumulative, previous_value = [], 1.0, capital

    for position, (date, row) in enumerate(prices.iterrows()):
        price = row.to_numpy()
        value_before = float(units @ price)

        if position == 0:
            period_return = np.nan
        elif previous_value > 0:
            period_return = value_before / previous_value - 1.0
            cumulative *= 1.0 + period_return
        else:
            # A portfolio that reached zero has no defined return from there.
            period_return = np.nan

        flow = aligned.get(date, 0.0)
        if flow:
            if flow < 0 and -flow > value_before + 1e-9:
                raise ValueError(
                    "withdrawal of %.2f on %s exceeds the portfolio value of "
                    "%.2f" % (-flow, date.date(), value_before))
            if pro_rata and value_before > 0:
                # Scale every holding by the same factor: the mix is unchanged,
                # so returns after the flow do not depend on its size.
                units = units * (1.0 + flow / value_before)
            else:
                weights_now = (wc if wc is not None else w0)
                units = units + flow * weights_now / price
        if date in rebalance_on:
            value_now = float(units @ price)
            units = value_now * w0 / price

        value_after = float(units @ price)
        records.append((date, value_before, flow, value_after,
                        period_return, cumulative - 1.0))
        previous_value = value_after

    return pd.DataFrame(records, columns=[
        "Date", "Value before flow", "External flow", "Value after flow",
        "Period return", "Cumulative TWR"]).set_index("Date")


def performance_report(prices: pd.DataFrame, capital: float,
                       flows: Optional[Mapping] = None,
                       initial_weights=None, contribution_weights=None,
                       rebalance: Optional[str] = None,
                       allow_short: bool = False) -> PerformanceResult:
    """Full TWR / MWR report for a portfolio with external cash flows."""
    history = simulate_portfolio(
        prices, capital, flows, initial_weights, contribution_weights,
        rebalance, allow_short)

    ending_value = float(history["Value after flow"].iloc[-1])
    aligned = align_flows(prices.index, flows)
    net_contributions = float(capital + sum(aligned.values()))

    cash_flows = [(prices.index[0], -float(capital))]
    cash_flows += [(d, -float(a)) for d, a in aligned.items()]
    cash_flows.append((prices.index[-1], ending_value))
    cash_flows.sort(key=lambda cf: cf[0])

    days = int((prices.index[-1] - prices.index[0]).days)
    twr = float(history["Cumulative TWR"].iloc[-1])

    warnings: list[str] = []
    twr_annualized = float("nan")
    if days > 0 and (1.0 + twr) > 0:
        twr_annualized = (1.0 + twr) ** (DAYS_PER_YEAR / days) - 1.0
    elif days == 0:
        warnings.append("single observation: annualized figures undefined")

    mwr = xirr(cash_flows)
    changes = sign_changes(cash_flows)
    if changes > 1:
        warnings.append(
            "investor cash flows change sign %d times, so more than one rate "
            "can solve the NPV equation; read the XIRR alongside the dated "
            "flows rather than on its own" % changes)
    if mwr != mwr:
        warnings.append("XIRR did not converge to a bracketed root")
    if days and days < 90:
        warnings.append(
            "annualizing %d days compounds a short sample into a yearly rate; "
            "treat it as arithmetic, not a forecast" % days)

    return PerformanceResult(
        history=history, twr=twr, twr_annualized=twr_annualized, mwr=mwr,
        ending_value=ending_value, net_contributions=net_contributions,
        dollar_gain=ending_value - net_contributions, cash_flows=cash_flows,
        days=days, warnings=warnings)


# --------------------------------------------------------------------------
# convenience
# --------------------------------------------------------------------------
def weighted_returns(returns: pd.DataFrame, weights=None) -> pd.Series:
    """Return of a portfolio rebalanced to fixed weights every period.

    r_p,t = sum_i w_i r_i,t. Note what the constant weights imply: the
    portfolio is traded back to them at every observation. It is the right
    baseline for comparing weighting schemes and the wrong one for describing
    a portfolio nobody touched — use `simulate_portfolio` for that.
    """
    w = normalize_weights(returns.shape[1], weights)
    return pd.Series(returns.to_numpy() @ w, index=returns.index,
                     name="portfolio")


def contribution_to_return(returns: pd.DataFrame, weights=None) -> pd.DataFrame:
    """Each asset's share of the portfolio's return over the whole sample."""
    w = normalize_weights(returns.shape[1], weights)
    total = (1.0 + returns).prod() - 1.0
    contribution = pd.Series(w * total.to_numpy(), index=returns.columns)
    out = pd.DataFrame({
        "weight": pd.Series(w, index=returns.columns),
        "asset_return": total,
        "contribution": contribution,
    })
    denominator = contribution.sum()
    out["share_of_total"] = (contribution / denominator
                             if denominator else np.nan)
    return out.sort_values("contribution", ascending=False)
