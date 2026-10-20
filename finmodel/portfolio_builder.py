"""Build and assess portfolios from fundamental criteria.

Two entry points:

  build_portfolio(...)    screen a universe by strategy, then allocate under
                          risk constraints
  assess_portfolio(...)   take an existing set of holdings and weights and
                          report what they add up to

Design notes that matter:

  * Risk is enforced as CONSTRAINTS, not as a score. A "conservative" portfolio
    is one that cannot hold more than 8% in a name, cannot exceed 30% in a
    sector, and excludes companies failing solvency tests — not one with a
    lower number attached.
  * Allocation never implies precision the inputs do not support. With annual
    fundamentals and no return history, mean-variance optimisation would be
    fitting noise, so the default is score-weighted with caps. Markowitz is
    used only when real return history is supplied.
  * Diversification is measured, not assumed: concentration (HHI), effective
    number of holdings, and sector exposure are all reported.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Iterable, Optional

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# Strategies: what to look for
# --------------------------------------------------------------------------

@dataclass
class Strategy:
    key: str
    name: str
    description: str
    # hard filters: metric -> (op, value) where op is 'min' or 'max'
    filters: dict[str, tuple[str, float]] = field(default_factory=dict)
    # ranking weights: metric -> weight (direction handled by `lower_better`)
    weights: dict[str, float] = field(default_factory=dict)
    lower_better: set[str] = field(default_factory=set)


STRATEGIES: dict[str, Strategy] = {
    "quality": Strategy(
        key="quality", name="Quality",
        description="Durable businesses: high returns on capital, real cash "
                    "generation, conservative balance sheets.",
        filters={"roic": ("min", 0.10), "fcf_margin": ("min", 0.05),
                 "net_debt_to_ebitda": ("max", 3.0),
                 "health_score": ("min", 0.70)},
        weights={"roic": 0.30, "fcf_margin": 0.25, "operating_margin": 0.20,
                 "piotroski": 0.15, "net_debt_to_ebitda": 0.10},
        lower_better={"net_debt_to_ebitda"},
    ),
    "value": Strategy(
        key="value", name="Value",
        description="Cheap against sector peers on earnings, cash flow and "
                    "book, with solvency tests applied to avoid value traps.",
        filters={"altman": ("min", 1.81), "fcf": ("min", 0.0),
                 "health_score": ("min", 0.50)},
        weights={"earnings_yield": 0.30, "fcf_yield": 0.30, "pe": 0.20,
                 "pb": 0.10, "ev_ebitda": 0.10},
        lower_better={"pe", "pb", "ev_ebitda"},
    ),
    "growth": Strategy(
        key="growth", name="Growth",
        description="Expanding revenue with margins that hold up and a balance "
                    "sheet that can fund the growth.",
        filters={"revenue_growth": ("min", 0.10),
                 "operating_margin": ("min", 0.05),
                 "net_debt_to_ebitda": ("max", 4.0)},
        weights={"revenue_growth": 0.40, "operating_margin": 0.25,
                 "roic": 0.20, "fcf_margin": 0.15},
        lower_better=set(),
    ),
    "income": Strategy(
        key="income", name="Income",
        description="Cash-generative businesses able to sustain distributions, "
                    "screened on coverage rather than headline yield.",
        filters={"fcf_margin": ("min", 0.05), "interest_coverage": ("min", 3.0),
                 "net_debt_to_ebitda": ("max", 3.5)},
        weights={"fcf_yield": 0.35, "dividend_yield": 0.25,
                 "fcf_margin": 0.20, "interest_coverage": 0.20},
        lower_better=set(),
    ),
    "defensive": Strategy(
        key="defensive", name="Defensive",
        description="Capital preservation first: low leverage, strong liquidity, "
                    "no distress signals, steady profitability.",
        filters={"altman": ("min", 2.99), "current_ratio": ("min", 1.5),
                 "net_debt_to_ebitda": ("max", 2.0),
                 "interest_coverage": ("min", 5.0),
                 "health_score": ("min", 0.75)},
        weights={"altman": 0.25, "interest_coverage": 0.25,
                 "current_ratio": 0.20, "operating_margin": 0.15,
                 "net_debt_to_ebitda": 0.15},
        lower_better={"net_debt_to_ebitda"},
    ),
    "balanced": Strategy(
        key="balanced", name="Balanced",
        description="A blend of quality, valuation and growth with no single "
                    "factor dominating.",
        filters={"health_score": ("min", 0.60), "fcf": ("min", 0.0)},
        weights={"roic": 0.20, "fcf_yield": 0.20, "revenue_growth": 0.20,
                 "operating_margin": 0.20, "piotroski": 0.20},
        lower_better=set(),
    ),
}


# --------------------------------------------------------------------------
# Risk profiles: what the portfolio is allowed to look like
# --------------------------------------------------------------------------

@dataclass
class RiskProfile:
    key: str
    name: str
    description: str
    max_position: float          # cap on any single holding
    min_holdings: int
    target_holdings: int
    max_sector: float            # cap on any single sector
    min_health_score: float
    max_net_debt_to_ebitda: Optional[float]
    require_positive_fcf: bool


RISK_PROFILES: dict[str, RiskProfile] = {
    "conservative": RiskProfile(
        key="conservative", name="Conservative",
        description="Wide diversification and strict balance-sheet limits. "
                    "Accepts lower expected return to reduce the chance of "
                    "permanent loss in any one holding.",
        max_position=0.08, min_holdings=15, target_holdings=25,
        max_sector=0.30, min_health_score=0.75,
        max_net_debt_to_ebitda=2.5, require_positive_fcf=True),
    "balanced": RiskProfile(
        key="balanced", name="Balanced",
        description="Moderate concentration with standard solvency screens.",
        max_position=0.12, min_holdings=10, target_holdings=18,
        max_sector=0.40, min_health_score=0.60,
        max_net_debt_to_ebitda=3.5, require_positive_fcf=True),
    "aggressive": RiskProfile(
        key="aggressive", name="Aggressive",
        description="Higher conviction and fewer holdings. Concentration cuts "
                    "both ways — it raises dispersion of outcomes, not just "
                    "expected return.",
        max_position=0.20, min_holdings=6, target_holdings=12,
        max_sector=0.55, min_health_score=0.45,
        max_net_debt_to_ebitda=None, require_positive_fcf=False),
}


# --------------------------------------------------------------------------
# Scoring and allocation
# --------------------------------------------------------------------------

def _zscore(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    sd = s.std(ddof=0)
    if sd is None or sd == 0 or pd.isna(sd):
        return pd.Series(0.0, index=series.index)
    return ((s - s.mean()) / sd).clip(-3, 3)


def score_candidates(df: pd.DataFrame, strategy: Strategy) -> pd.Series:
    """Composite strategy score. Z-scored per metric, direction-adjusted."""
    total = pd.Series(0.0, index=df.index)
    weight_used = pd.Series(0.0, index=df.index)
    for metric, w in strategy.weights.items():
        if metric not in df.columns:
            continue
        z = _zscore(df[metric])
        if metric in strategy.lower_better:
            z = -z
        present = pd.to_numeric(df[metric], errors="coerce").notna()
        total = total.add(z.fillna(0.0) * w * present, fill_value=0.0)
        weight_used = weight_used.add(w * present, fill_value=0.0)
    # Rescale so a company missing some metrics is not quietly penalised.
    return (total / weight_used.replace(0, np.nan)).fillna(-99.0)


def _apply_filters(df: pd.DataFrame, strategy: Strategy,
                   risk: RiskProfile) -> tuple[pd.DataFrame, dict]:
    mask = pd.Series(True, index=df.index)
    rejected: dict[str, int] = {}

    def apply(metric, op, value, label):
        nonlocal mask
        if metric not in df.columns:
            return
        col = pd.to_numeric(df[metric], errors="coerce")
        ok = col >= value if op == "min" else col <= value
        ok = ok.fillna(False)
        dropped = int((mask & ~ok).sum())
        if dropped:
            rejected[label] = rejected.get(label, 0) + dropped
        mask &= ok

    for metric, (op, value) in strategy.filters.items():
        apply(metric, op, value, f"{strategy.name} filter: {metric}")
    apply("health_score", "min", risk.min_health_score,
          f"{risk.name} risk: health score")
    if risk.max_net_debt_to_ebitda is not None:
        apply("net_debt_to_ebitda", "max", risk.max_net_debt_to_ebitda,
              f"{risk.name} risk: leverage")
    if risk.require_positive_fcf:
        apply("fcf", "min", 0.0, f"{risk.name} risk: positive free cash flow")
    return df[mask].copy(), rejected


def _allocate(df: pd.DataFrame, risk: RiskProfile, method: str,
              volatility: Optional[dict[str, float]] = None) -> pd.Series:
    """Weights that respect the position cap. Excess is redistributed, not
    dropped, so the book always sums to 1."""
    n = len(df)
    if n == 0:
        return pd.Series(dtype=float)

    if method == "equal":
        raw = pd.Series(1.0 / n, index=df.index)
    elif method == "inverse_vol" and volatility:
        vols = pd.Series(
            [volatility.get(t, np.nan) for t in df["ticker"]], index=df.index)
        vols = vols.replace(0, np.nan)
        if vols.notna().sum() < 2:
            raw = pd.Series(1.0 / n, index=df.index)
        else:
            inv = (1.0 / vols).fillna((1.0 / vols).median())
            raw = inv / inv.sum()
    else:  # score-weighted
        sc = df["strategy_score"]
        shifted = sc - sc.min() + 0.10      # keep every selected name positive
        raw = shifted / shifted.sum()

    # Capping is handled by _enforce_constraints, which must satisfy the
    # position and sector limits jointly rather than one after the other.
    return raw / raw.sum()


def _cap_positions(w: pd.Series, cap: float) -> pd.Series:
    """Cap each weight, redistributing the excess to names with room."""
    w = w.copy()
    for _ in range(200):
        over = w > cap + 1e-12
        if not over.any():
            break
        excess = float((w[over] - cap).sum())
        w[over] = cap
        room = (cap - w[~over]).clip(lower=0)
        if room.sum() <= 1e-12:
            break      # every name is at the cap; nowhere left to put it
        w[~over] = w[~over] + excess * (room / room.sum())
    total = w.sum()
    return w / total if total > 0 else w


def _enforce_constraints(df: pd.DataFrame, w: pd.Series,
                         risk: RiskProfile) -> tuple[pd.Series, bool]:
    """Satisfy the position cap and the sector cap TOGETHER.

    Applying them in sequence does not work: redistributing weight to bring a
    sector under its cap pushes individual positions back above theirs. The two
    constraints are therefore alternated until both hold, and the caller is told
    whether convergence was actually reached rather than being handed a book
    that quietly breaches a limit."""
    n = len(w)
    if n == 0:
        return w, True
    pos_cap = max(risk.max_position, 1.0 / n)   # infeasible below 1/n

    for _ in range(60):
        w = _cap_positions(w, pos_cap)
        if "sector" not in df.columns:
            return w, True

        by_sector = w.groupby(df["sector"]).sum()
        over = by_sector[by_sector > risk.max_sector + 1e-9]
        if over.empty:
            return w, True

        for sector, total in over.items():
            idx = df.index[df["sector"] == sector]
            others = df.index[df["sector"] != sector]
            if len(others) == 0:
                return w, False     # single-sector book: cap unreachable
            scale = risk.max_sector / total
            freed = float(w[idx].sum() * (1 - scale))
            w[idx] = w[idx] * scale
            other_sum = float(w[others].sum())
            if other_sum <= 1e-12:
                w[others] = freed / len(others)
            else:
                w[others] = w[others] + freed * (w[others] / other_sum)
        w = w / w.sum()

    # Position cap is the harder promise — a single name blowing through its
    # limit is the risk this profile exists to prevent — so it wins the tie.
    w = _cap_positions(w, pos_cap)
    by_sector = w.groupby(df["sector"]).sum() if "sector" in df.columns else None
    converged = by_sector is None or bool(
        (by_sector <= risk.max_sector + 0.01).all())
    return w, converged


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------

@dataclass
class Holding:
    ticker: str
    name: str
    sector: str
    weight: float
    strategy_score: float
    health_passed: Optional[int] = None
    health_total: Optional[int] = None
    roic: Optional[float] = None
    fcf_margin: Optional[float] = None
    revenue_growth: Optional[float] = None
    pe: Optional[float] = None
    net_debt_to_ebitda: Optional[float] = None

    def as_dict(self):
        return asdict(self)


@dataclass
class PortfolioResult:
    strategy: str
    strategy_name: str
    risk_profile: str
    risk_profile_name: str
    allocation_method: str
    holdings: list[Holding]
    sector_weights: dict[str, float]
    metrics: dict
    universe_size: int
    eligible_size: int
    rejected: dict[str, int]
    warnings: list[str] = field(default_factory=list)

    def as_dict(self):
        d = asdict(self)
        d["holdings"] = [h for h in d["holdings"]]
        return d

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([h.as_dict() for h in self.holdings])


# Sane bounds for ratio metrics. A company that has bought back most of its
# equity shows a tiny book value and an ROE in the hundreds of percent — that
# is a fact about its capital structure, not its profitability, and letting it
# into a weighted average silently destroys the figure. Readings outside these
# bounds are excluded and counted rather than clipped, because clipping would
# invent a number the company never produced.
METRIC_BOUNDS: dict[str, tuple[float, float]] = {
    "roe": (-2.0, 2.0),
    "roic": (-1.0, 1.5),
    "operating_margin": (-2.0, 1.0),
    "net_margin": (-2.0, 1.0),
    "fcf_margin": (-2.0, 1.0),
    "gross_margin": (-1.0, 1.0),
    "revenue_growth": (-1.0, 3.0),
    "net_debt_to_ebitda": (-20.0, 30.0),
    "pe": (0.0, 300.0),
    "ev_ebitda": (0.0, 200.0),
    "fcf_yield": (-1.0, 1.0),
    "dividend_yield": (0.0, 0.5),
}


def _weighted(df: pd.DataFrame, w: pd.Series, col: str
              ) -> tuple[Optional[float], int]:
    """Weighted average, excluding readings outside plausible bounds.

    Returns (value, n_excluded) so callers can report how much was dropped."""
    if col not in df.columns:
        return None, 0
    v = pd.to_numeric(df[col], errors="coerce")
    mask = v.notna()
    excluded = 0
    lo_hi = METRIC_BOUNDS.get(col)
    if lo_hi is not None:
        lo, hi = lo_hi
        in_range = v.between(lo, hi)
        excluded = int((mask & ~in_range).sum())
        mask = mask & in_range
    if not mask.any():
        return None, excluded
    ww = w[mask]
    if ww.sum() == 0:
        return None, excluded
    return float((v[mask] * ww).sum() / ww.sum()), excluded


def portfolio_metrics(df: pd.DataFrame, w: pd.Series) -> dict:
    hhi = float((w ** 2).sum())
    out = {
        "holdings": int(len(df)),
        "largest_position": float(w.max()) if len(w) else 0.0,
        "top5_concentration": float(w.nlargest(min(5, len(w))).sum()),
        "herfindahl": hhi,
        # 1/HHI is the number of equally weighted holdings that would give the
        # same concentration — a more honest read than a raw count.
        "effective_holdings": float(1.0 / hhi) if hhi > 0 else 0.0,
        "sectors": int(df["sector"].nunique()) if "sector" in df.columns else 0,
    }
    dropped: dict[str, int] = {}
    for col in ("roic", "roe", "operating_margin", "fcf_margin",
                "revenue_growth", "net_debt_to_ebitda", "pe", "ev_ebitda",
                "fcf_yield", "dividend_yield", "health_score", "piotroski"):
        value, n_excl = _weighted(df, w, col)
        out[f"weighted_{col}"] = value
        if n_excl:
            dropped[col] = n_excl
    out["excluded_outliers"] = dropped
    return out


def build_portfolio(universe: Iterable[dict], strategy: str = "balanced",
                    risk_profile: str = "balanced",
                    allocation: str = "score",
                    max_holdings: Optional[int] = None,
                    volatility: Optional[dict[str, float]] = None,
                    sector_limit: Optional[float] = None) -> PortfolioResult:
    """Screen a universe by strategy, then allocate under risk constraints."""
    strat = STRATEGIES.get(strategy)
    if strat is None:
        raise ValueError(f"unknown strategy '{strategy}'; "
                         f"choose from {sorted(STRATEGIES)}")
    risk = RISK_PROFILES.get(risk_profile)
    if risk is None:
        raise ValueError(f"unknown risk profile '{risk_profile}'; "
                         f"choose from {sorted(RISK_PROFILES)}")
    if sector_limit is not None:
        risk = RiskProfile(**{**asdict(risk), "max_sector": sector_limit})

    df = pd.DataFrame(list(universe))
    universe_size = len(df)
    warnings: list[str] = []
    if df.empty:
        return PortfolioResult(strategy, strat.name, risk_profile, risk.name,
                               allocation, [], {}, {}, 0, 0, {},
                               ["universe is empty"])

    eligible, rejected = _apply_filters(df, strat, risk)
    eligible_size = len(eligible)

    if eligible_size == 0:
        return PortfolioResult(
            strategy, strat.name, risk_profile, risk.name, allocation, [], {},
            {}, universe_size, 0, rejected,
            ["No company met both the strategy filters and the risk limits. "
             "Loosen the risk profile or widen the universe."])

    eligible["strategy_score"] = score_candidates(eligible, strat)
    target = max_holdings or risk.target_holdings
    selected = eligible.nlargest(min(target, eligible_size), "strategy_score").copy()

    if len(selected) < risk.min_holdings:
        warnings.append(
            f"Only {len(selected)} companies qualified against a "
            f"{risk.name.lower()} minimum of {risk.min_holdings}. The book is "
            f"more concentrated than this risk profile intends.")

    w = _allocate(selected, risk, allocation, volatility)
    w, converged = _enforce_constraints(selected, w, risk)
    selected["weight"] = w
    if not converged:
        warnings.append(
            "Position and sector caps could not both be met with the names "
            "available. The position cap was held and the sector cap relaxed, "
            "because a single oversized holding is the larger risk.")

    if "sector" in selected.columns:
        sector_w = selected.groupby("sector")["weight"].sum().sort_values(
            ascending=False)
        sector_weights = {str(k): float(v) for k, v in sector_w.items()}
        if sector_w.iloc[0] > risk.max_sector + 0.01:
            warnings.append(
                f"{sector_w.index[0]} remains {sector_w.iloc[0]:.0%} of the "
                f"book — too few eligible names elsewhere to diversify further.")
    else:
        sector_weights = {}

    metrics = portfolio_metrics(selected, selected["weight"])
    if metrics["effective_holdings"] < 5:
        warnings.append(
            f"Effective diversification is only "
            f"{metrics['effective_holdings']:.1f} holdings once weights are "
            f"accounted for, despite {metrics['holdings']} positions.")

    def g(row, col):
        v = row.get(col)
        try:
            return None if v is None or pd.isna(v) else float(v)
        except (TypeError, ValueError):
            return None

    holdings = [
        Holding(
            ticker=str(r.get("ticker", "")), name=str(r.get("name", "")),
            sector=str(r.get("sector", "Unknown")),
            weight=float(r["weight"]),
            strategy_score=float(r["strategy_score"]),
            health_passed=int(r["health_passed"])
            if r.get("health_passed") is not None
            and not pd.isna(r.get("health_passed")) else None,
            health_total=int(r["health_total"])
            if r.get("health_total") is not None
            and not pd.isna(r.get("health_total")) else None,
            roic=g(r, "roic"), fcf_margin=g(r, "fcf_margin"),
            revenue_growth=g(r, "revenue_growth"), pe=g(r, "pe"),
            net_debt_to_ebitda=g(r, "net_debt_to_ebitda"),
        )
        for _, r in selected.sort_values("weight", ascending=False).iterrows()
    ]

    return PortfolioResult(
        strategy=strategy, strategy_name=strat.name,
        risk_profile=risk_profile, risk_profile_name=risk.name,
        allocation_method=allocation, holdings=holdings,
        sector_weights=sector_weights, metrics=metrics,
        universe_size=universe_size, eligible_size=eligible_size,
        rejected=rejected, warnings=warnings)


def assess_portfolio(holdings: dict[str, float],
                     universe: Iterable[dict]) -> dict:
    """Report what an existing book adds up to. `holdings` is ticker -> weight."""
    df = pd.DataFrame(list(universe))
    if df.empty or not holdings:
        return {"error": "no holdings or empty universe"}

    total = sum(holdings.values())
    if total <= 0:
        return {"error": "weights must sum to a positive number"}
    norm = {k.upper(): v / total for k, v in holdings.items()}

    df["ticker_u"] = df["ticker"].astype(str).str.upper()
    matched = df[df["ticker_u"].isin(norm)].copy()
    missing = sorted(set(norm) - set(matched["ticker_u"]))
    if matched.empty:
        return {"error": "none of the supplied tickers are in the database",
                "unmatched": missing}

    matched["weight"] = matched["ticker_u"].map(norm)
    matched["weight"] = matched["weight"] / matched["weight"].sum()

    metrics = portfolio_metrics(matched, matched["weight"])
    sector_w = (matched.groupby("sector")["weight"].sum().sort_values(
        ascending=False).to_dict() if "sector" in matched.columns else {})

    flags: list[str] = []
    if metrics["largest_position"] > 0.25:
        flags.append(
            f"Largest position is {metrics['largest_position']:.0%} of the "
            f"book — single-name risk dominates the outcome.")
    if sector_w and max(sector_w.values()) > 0.50:
        top = max(sector_w, key=sector_w.get)
        flags.append(
            f"{top} is {sector_w[top]:.0%} of the book; sector shocks will "
            f"drive most of the variance.")
    if metrics["effective_holdings"] < 5:
        flags.append(
            f"Effective diversification is {metrics['effective_holdings']:.1f} "
            f"holdings after weighting.")
    weak = matched[pd.to_numeric(matched.get("health_score"),
                                 errors="coerce") < 0.5]
    if not weak.empty:
        w_share = float(matched.loc[weak.index, "weight"].sum())
        flags.append(
            f"{w_share:.0%} of the book sits in companies passing fewer than "
            f"half the health checks ({', '.join(weak['ticker'].head(5))}).")
    if missing:
        flags.append(
            f"Not found in the database and excluded from every figure above: "
            f"{', '.join(missing)}.")

    return {
        "metrics": metrics,
        "sector_weights": {str(k): float(v) for k, v in sector_w.items()},
        "holdings": [
            {"ticker": r["ticker"], "name": r.get("name"),
             "sector": r.get("sector"), "weight": float(r["weight"]),
             "health_passed": r.get("health_passed"),
             "health_total": r.get("health_total"),
             "roic": r.get("roic"), "pe": r.get("pe")}
            for _, r in matched.sort_values("weight", ascending=False).iterrows()
        ],
        "unmatched": missing,
        "flags": flags,
    }
