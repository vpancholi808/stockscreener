"""Sector benchmarks — a company's numbers only mean something against peers.

A 32% gross margin is excellent for a grocery distributor and alarming for
enterprise software. Absolute thresholds therefore mostly rediscover which
sectors are structurally asset-light. This module builds per-sector reference
statistics and positions each company inside its own sector's distribution.

On mean vs median: the headline comparison uses the MEDIAN, not the mean.
Valuation multiples are severely right-skewed — one company trading at 400x
earnings drags a sector mean far above anything a typical constituent trades
at, and every comparison against it is then wrong. The mean is still computed
and exposed, because it is informative about skew, but it is not the yardstick.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from .valuation import HIGHER_IS_CHEAPER, LOWER_IS_CHEAPER, MULTIPLE_LABELS

# Metrics benchmarked per sector. Direction: +1 means higher is better for the
# company, -1 means lower is better. For valuation multiples "better" means
# cheaper, which is why most carry -1.
BENCHMARK_METRICS: dict[str, int] = {
    # operating quality
    "gross_margin": +1,
    "operating_margin": +1,
    "net_margin": +1,
    "fcf_margin": +1,
    "roe": +1,
    "roic": +1,
    "revenue_growth": +1,
    # balance sheet
    "net_debt_to_ebitda": -1,
    "debt_to_equity": -1,
    "interest_coverage": +1,
    "current_ratio": +1,
    # valuation
    "pe": -1,
    "pb": -1,
    "ps": -1,
    "ev_ebitda": -1,
    "ev_sales": -1,
    "ev_ebit": -1,
    "peg": -1,
    "fcf_yield": +1,
    "earnings_yield": +1,
}

METRIC_LABELS = {
    "gross_margin": "Gross margin", "operating_margin": "Operating margin",
    "net_margin": "Net margin", "fcf_margin": "FCF margin",
    "roe": "ROE", "roic": "ROIC", "revenue_growth": "Revenue growth",
    "net_debt_to_ebitda": "Net debt / EBITDA", "debt_to_equity": "Debt / equity",
    "interest_coverage": "Interest coverage", "current_ratio": "Current ratio",
    **MULTIPLE_LABELS,
}

PERCENT_METRICS = {
    "gross_margin", "operating_margin", "net_margin", "fcf_margin", "roe",
    "roic", "revenue_growth", "fcf_yield", "earnings_yield", "dividend_yield",
}

# Below this many observations a sector median is noise rather than a benchmark.
MIN_PEERS = 5


@dataclass
class MetricStats:
    metric: str
    count: int
    median: Optional[float] = None
    mean: Optional[float] = None
    p25: Optional[float] = None
    p75: Optional[float] = None
    std: Optional[float] = None

    @property
    def reliable(self) -> bool:
        return self.count >= MIN_PEERS


@dataclass
class SectorBenchmark:
    sector: str
    n_companies: int
    metrics: dict[str, MetricStats] = field(default_factory=dict)

    def stat(self, metric: str) -> Optional[MetricStats]:
        return self.metrics.get(metric)

    def to_rows(self) -> list[dict]:
        return [
            {"sector": self.sector, "metric": m, "n_companies": self.n_companies,
             "count": s.count, "median": s.median, "mean": s.mean,
             "p25": s.p25, "p75": s.p75, "std": s.std}
            for m, s in self.metrics.items()
        ]


@dataclass
class Comparison:
    """One metric, company versus its sector."""
    metric: str
    label: str
    value: Optional[float]
    sector_median: Optional[float]
    percentile: Optional[float]      # 0-1, already direction-adjusted
    direction: int                   # +1 higher better, -1 lower better
    peers: int
    is_percent: bool

    @property
    def available(self) -> bool:
        return (self.value is not None and self.sector_median is not None
                and self.peers >= MIN_PEERS)

    @property
    def premium(self) -> Optional[float]:
        """Fractional gap to the sector median (+0.25 = 25% above)."""
        if self.value is None or not self.sector_median:
            return None
        if self.sector_median == 0:
            return None
        return self.value / self.sector_median - 1.0

    def verdict(self) -> str:
        """Plain-language standing. Direction-aware."""
        if not self.available or self.percentile is None:
            return "no comparison"
        p = self.percentile
        if p >= 0.80:
            return "top quintile"
        if p >= 0.60:
            return "above median"
        if p >= 0.40:
            return "around median"
        if p >= 0.20:
            return "below median"
        return "bottom quintile"


def build_benchmarks(rows: Iterable[dict],
                     metrics: Optional[dict[str, int]] = None
                     ) -> dict[str, SectorBenchmark]:
    """Compute per-sector reference statistics from company rows.

    `rows` need only carry a 'sector' key plus whichever metrics are present;
    missing values are skipped rather than imputed."""
    metrics = metrics or BENCHMARK_METRICS
    df = pd.DataFrame(list(rows))
    if df.empty or "sector" not in df.columns:
        return {}

    out: dict[str, SectorBenchmark] = {}
    for sector, grp in df.groupby("sector"):
        bench = SectorBenchmark(sector=str(sector), n_companies=len(grp))
        for metric in metrics:
            if metric not in grp.columns:
                bench.metrics[metric] = MetricStats(metric=metric, count=0)
                continue
            s = pd.to_numeric(grp[metric], errors="coerce").replace(
                [np.inf, -np.inf], np.nan).dropna()
            if s.empty:
                bench.metrics[metric] = MetricStats(metric=metric, count=0)
                continue
            bench.metrics[metric] = MetricStats(
                metric=metric,
                count=int(s.size),
                median=float(s.median()),
                mean=float(s.mean()),
                p25=float(s.quantile(0.25)),
                p75=float(s.quantile(0.75)),
                std=float(s.std(ddof=0)) if s.size > 1 else 0.0,
            )
        out[str(sector)] = bench
    return out


def compare_to_sector(company: dict, benchmark: SectorBenchmark,
                      metrics: Optional[dict[str, int]] = None,
                      peer_values: Optional[dict[str, list[float]]] = None
                      ) -> dict[str, Comparison]:
    """Position one company against its sector, metric by metric.

    `peer_values` (metric -> list of peer readings) gives an exact empirical
    percentile. Without it the percentile is inferred from the quartiles, which
    is coarser but needs no peer list in memory."""
    metrics = metrics or BENCHMARK_METRICS
    out: dict[str, Comparison] = {}

    for metric, direction in metrics.items():
        stats = benchmark.stat(metric) if benchmark else None
        raw = company.get(metric)
        value = None
        if raw is not None:
            try:
                value = None if pd.isna(raw) else float(raw)
            except (TypeError, ValueError):
                value = None

        pctile = None
        if value is not None and stats is not None and stats.count:
            if peer_values and metric in peer_values and peer_values[metric]:
                peers = [p for p in peer_values[metric] if p is not None]
                if peers:
                    below = sum(1 for p in peers if p < value)
                    equal = sum(1 for p in peers if p == value)
                    pctile = (below + 0.5 * equal) / len(peers)
            else:
                pctile = _quartile_percentile(value, stats)
            # Flip so that higher percentile always means "more attractive".
            if pctile is not None and direction < 0:
                pctile = 1.0 - pctile

        out[metric] = Comparison(
            metric=metric,
            label=METRIC_LABELS.get(metric, metric),
            value=value,
            sector_median=stats.median if stats else None,
            percentile=pctile,
            direction=direction,
            peers=stats.count if stats else 0,
            is_percent=metric in PERCENT_METRICS,
        )
    return out


def _quartile_percentile(value: float, s: MetricStats) -> Optional[float]:
    """Approximate a percentile by linear interpolation between quartiles.

    Coarse by construction — it assumes a locally uniform distribution between
    breakpoints — but it needs only five summary numbers, which keeps the
    database small. Exact percentiles are used wherever peer values are to hand."""
    if s.p25 is None or s.median is None or s.p75 is None:
        return None
    p25, p50, p75 = s.p25, s.median, s.p75
    if value <= p25:
        if p25 == p50:
            return 0.25
        span = max(p50 - p25, 1e-12)
        return max(0.0, 0.25 - 0.25 * min(1.0, (p25 - value) / span))
    if value <= p50:
        span = max(p50 - p25, 1e-12)
        return 0.25 + 0.25 * (value - p25) / span
    if value <= p75:
        span = max(p75 - p50, 1e-12)
        return 0.50 + 0.25 * (value - p50) / span
    span = max(p75 - p50, 1e-12)
    return min(1.0, 0.75 + 0.25 * min(1.0, (value - p75) / span))


def valuation_summary(comparisons: dict[str, Comparison]) -> dict:
    """Collapse the valuation multiples into one cheap/expensive reading."""
    keys = [k for k in ("pe", "ev_ebitda", "ps", "pb", "ev_sales")
            if k in comparisons and comparisons[k].available]
    usable = [comparisons[k] for k in keys
              if comparisons[k].percentile is not None]
    if not usable:
        return {"available": False, "label": "no valuation data",
                "percentile": None, "metrics_used": []}

    # Percentiles are already direction-adjusted, so a HIGH value here means
    # cheap relative to peers.
    score = float(np.mean([c.percentile for c in usable]))
    if score >= 0.70:
        label = "cheap vs sector"
    elif score >= 0.55:
        label = "modestly cheap vs sector"
    elif score >= 0.45:
        label = "in line with sector"
    elif score >= 0.30:
        label = "modestly expensive vs sector"
    else:
        label = "expensive vs sector"
    return {"available": True, "label": label, "percentile": score,
            "metrics_used": [c.metric for c in usable]}
