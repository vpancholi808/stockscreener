"""Read-only JSON API for the finmodel screener.

    uvicorn finmodel.web.api:app --reload

Deliberately read-only: every figure is pre-computed by the worker
(`finmodel.web.loader`) and served from the database. No request ever touches
SEC or a price feed.

Two things that matter for a PUBLIC deployment:

  * SERVE_MARKET_DATA (env, default false). SEC filings are US government work
    and public domain, so fundamentals and ratios can be republished freely.
    Scraped price data generally cannot be redistributed under the provider's
    terms, so price-derived fields are withheld unless you have switched to a
    feed whose licence permits it.
  * Every list response carries a `disclaimer`. This is analysis tooling built
    from public filings, not investment advice, and the API says so rather than
    leaving it to the frontend to remember.
"""
from __future__ import annotations

import os
import time
from collections import defaultdict, deque
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .store import NUMERIC_FILTERS, SORTABLE, Database

DISCLAIMER = ("Computed from public SEC filings for informational purposes only. "
              "Not investment advice. Figures are as last filed and may be out of "
              "date; verify against primary sources before acting.")

SERVE_MARKET_DATA = os.environ.get("SERVE_MARKET_DATA", "false").lower() == "true"
RATE_LIMIT = int(os.environ.get("RATE_LIMIT_PER_MIN", "120"))
ALLOWED_ORIGINS = [o.strip() for o in
                   os.environ.get("ALLOWED_ORIGINS", "*").split(",") if o.strip()]

MARKET_FIELDS = ("price", "shares_diluted", "market_cap", "pe", "ev_ebitda",
                 "fcf_yield", "beta")

app = FastAPI(
    title="finmodel screener API",
    version="0.1.0",
    description="Financial health screening computed from SEC filings.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_db = Database()
_hits: dict[str, deque] = defaultdict(deque)


@app.middleware("http")
async def rate_limit(request: Request, call_next):
    """Coarse in-memory limiter — enough to stop casual abuse of a public
    endpoint. Put a real limiter (nginx, Cloudflare, API gateway) in front of
    this in production; an in-process counter does not survive restarts or
    coordinate across replicas."""
    if RATE_LIMIT > 0 and request.url.path.startswith("/api/"):
        who = request.client.host if request.client else "unknown"
        now = time.time()
        q = _hits[who]
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= RATE_LIMIT:
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Try again shortly."})
        q.append(now)
    return await call_next(request)


@app.on_event("startup")
def startup():
    _db.connect()


def _strip_market(row: dict) -> dict:
    if SERVE_MARKET_DATA:
        return row
    return {k: v for k, v in row.items() if k not in MARKET_FIELDS}


@app.get("/api/health")
def healthcheck():
    try:
        _db.connect()
        return {"status": "ok", "backend": "postgres" if _db.is_postgres else "sqlite"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/api/stats")
def stats():
    s = _db.stats()
    s["serves_market_data"] = SERVE_MARKET_DATA
    s["disclaimer"] = DISCLAIMER
    return s


@app.get("/api/sectors")
def sectors():
    return {"sectors": _db.sectors()}


@app.get("/api/meta")
def meta():
    """What the screener measures — so the frontend never hardcodes thresholds."""
    from ..screener import ScreenConfig
    from ..scorecard import ScorecardConfig
    sc, scr = ScorecardConfig(), ScreenConfig()
    return {
        "scorecard_thresholds": {
            "debt_to_equity": {"max": sc.max_debt_to_equity, "group": "solvency"},
            "net_debt_to_ebitda": {"max": sc.max_net_debt_to_ebitda, "group": "solvency"},
            "interest_coverage": {"min": sc.min_interest_coverage, "group": "solvency"},
            "current_ratio": {"min": sc.min_current_ratio, "group": "liquidity"},
            "quick_ratio": {"min": sc.min_quick_ratio, "group": "liquidity"},
            "fcf_margin": {"min": sc.min_fcf_margin, "group": "cash flow"},
            "cash_conversion": {"min": sc.min_cash_conversion, "group": "cash flow"},
            "gross_margin": {"min": sc.min_gross_margin, "group": "profitability"},
            "operating_margin": {"min": sc.min_operating_margin, "group": "profitability"},
            "roe": {"min": sc.min_roe, "group": "profitability"},
            "roic": {"min": sc.min_roic, "group": "profitability"},
        },
        "screen_defaults": {
            "min_health_score": scr.min_health_score,
            "min_revenue_growth": scr.min_revenue_growth,
            "min_fcf_margin": scr.min_fcf_margin,
            "max_net_debt_to_ebitda": scr.max_net_debt_to_ebitda,
            "min_operating_margin": scr.min_operating_margin,
            "min_roic": scr.min_roic,
        },
        "forensic_scores": {
            "piotroski": "0-9 binary tests of improving fundamentals; 8-9 strong",
            "altman": "Bankruptcy risk; >2.99 safe, <1.81 distress",
            "beneish": "Earnings-manipulation flag; above -1.78 warrants scrutiny",
            "accruals": "(net income - operating cash flow) / assets; >0.10 is high",
        },
        "sortable": sorted(SORTABLE),
        "filterable": sorted(NUMERIC_FILTERS),
        "disclaimer": DISCLAIMER,
    }


@app.get("/api/companies")
def companies(
    sector: Optional[str] = None,
    search: Optional[str] = Query(None, max_length=64),
    passing_only: bool = False,
    sort_by: str = "sector_percentile",
    desc: bool = True,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    min_health_score: Optional[float] = None,
    min_revenue: Optional[float] = None,
    min_revenue_growth: Optional[float] = None,
    min_operating_margin: Optional[float] = None,
    min_gross_margin: Optional[float] = None,
    min_fcf_margin: Optional[float] = None,
    min_roe: Optional[float] = None,
    min_roic: Optional[float] = None,
    max_net_debt_to_ebitda: Optional[float] = None,
    min_interest_coverage: Optional[float] = None,
    min_current_ratio: Optional[float] = None,
    max_debt_to_equity: Optional[float] = None,
    min_piotroski: Optional[float] = None,
    min_altman: Optional[float] = None,
):
    filters = {
        "health_score__min": min_health_score,
        "revenue__min": min_revenue,
        "revenue_growth__min": min_revenue_growth,
        "operating_margin__min": min_operating_margin,
        "gross_margin__min": min_gross_margin,
        "fcf_margin__min": min_fcf_margin,
        "roe__min": min_roe,
        "roic__min": min_roic,
        "net_debt_to_ebitda__max": max_net_debt_to_ebitda,
        "interest_coverage__min": min_interest_coverage,
        "current_ratio__min": min_current_ratio,
        "debt_to_equity__max": max_debt_to_equity,
        "piotroski__min": min_piotroski,
        "altman__min": min_altman,
    }
    try:
        result = _db.screen(filters=filters, sector=sector, search=search,
                            passing_only=passing_only, sort_by=sort_by,
                            desc=desc, limit=limit, offset=offset)
    except Exception as e:
        raise HTTPException(status_code=500, detail="query failed: %s" % e)
    return {
        "rows": [_strip_market(r) for r in result["rows"]],
        "total": result["total"],
        "limit": limit,
        "offset": offset,
        "disclaimer": DISCLAIMER,
    }


@app.get("/api/companies/{ticker}")
def company(ticker: str):
    row = _db.company(ticker)
    if row is None:
        raise HTTPException(status_code=404, detail="Company not found: %s" % ticker)
    if not SERVE_MARKET_DATA:
        row["market"] = None
    row["disclaimer"] = DISCLAIMER
    return row


# ---------------------------------------------------------------------------
# Relative valuation, sector benchmarks, research, portfolio
# ---------------------------------------------------------------------------

from pydantic import BaseModel, Field  # noqa: E402

PRICE_DERIVED_METRICS = {"pe", "forward_pe", "pb", "ps", "ev_ebitda",
                         "ev_sales", "ev_ebit", "peg", "fcf_yield",
                         "earnings_yield", "dividend_yield"}


def _benchmarks_for(sector: str) -> list[dict]:
    rows = _db.benchmarks(sector)
    if SERVE_MARKET_DATA:
        return rows
    return [r for r in rows if not r.get("price_derived")]


@app.get("/api/sectors/{sector}/benchmarks")
def sector_benchmarks(sector: str):
    rows = _benchmarks_for(sector)
    if not rows:
        raise HTTPException(status_code=404,
                            detail="No benchmarks for sector: %s" % sector)
    return {
        "sector": sector,
        "benchmarks": rows,
        "note": ("Comparisons use the sector MEDIAN. Valuation multiples are "
                 "right-skewed, so a mean would sit well above what a typical "
                 "constituent trades at. The mean is included for reference."),
        "disclaimer": DISCLAIMER,
    }


@app.get("/api/companies/{ticker}/research")
def company_research(ticker: str):
    """Evidence-based research note. Deliberately contains no buy/sell verdict."""
    from ..research import generate_research
    from ..sectors import BENCHMARK_METRICS, compare_to_sector, MetricStats, \
        SectorBenchmark

    rows = _db.universe_rows(limit=100000)
    match = [r for r in rows if str(r.get("ticker", "")).upper() == ticker.upper()]
    if not match:
        raise HTTPException(status_code=404, detail="Company not found: %s" % ticker)
    company = match[0]
    sector = company.get("sector", "Unknown")

    stored = _db.benchmarks(sector)
    bench = SectorBenchmark(
        sector=sector,
        n_companies=stored[0]["n_companies"] if stored else 0,
        metrics={r["metric"]: MetricStats(
            metric=r["metric"], count=r["count"] or 0, median=r["median"],
            mean=r["mean"], p25=r["p25"], p75=r["p75"], std=r["std"])
            for r in stored},
    )
    peers = _db.peer_values(sector, list(BENCHMARK_METRICS))
    comparisons = compare_to_sector(company, bench, peer_values=peers)

    if not SERVE_MARKET_DATA:
        comparisons = {k: v for k, v in comparisons.items()
                       if k not in PRICE_DERIVED_METRICS}

    note = generate_research(company, comparisons)
    return {
        "research": note.as_dict(),
        "comparisons": [
            {"metric": c.metric, "label": c.label, "value": c.value,
             "sector_median": c.sector_median, "percentile": c.percentile,
             "direction": c.direction, "peers": c.peers,
             "is_percent": c.is_percent, "verdict": c.verdict(),
             "premium": c.premium}
            for c in comparisons.values() if c.available
        ],
        "no_recommendation": (
            "This summary reports what the filed figures show. It contains no "
            "buy, sell or hold recommendation, and none should be inferred."),
        "disclaimer": DISCLAIMER,
    }


@app.get("/api/portfolio/options")
def portfolio_options():
    from ..portfolio_builder import RISK_PROFILES, STRATEGIES
    return {
        "strategies": [
            {"key": s.key, "name": s.name, "description": s.description,
             "filters": {k: {"op": v[0], "value": v[1]}
                         for k, v in s.filters.items()},
             "weights": s.weights}
            for s in STRATEGIES.values()
        ],
        "risk_profiles": [
            {"key": r.key, "name": r.name, "description": r.description,
             "max_position": r.max_position, "min_holdings": r.min_holdings,
             "target_holdings": r.target_holdings, "max_sector": r.max_sector,
             "min_health_score": r.min_health_score,
             "max_net_debt_to_ebitda": r.max_net_debt_to_ebitda,
             "require_positive_fcf": r.require_positive_fcf}
            for r in RISK_PROFILES.values()
        ],
        "allocation_methods": [
            {"key": "score", "name": "Strategy-weighted",
             "description": "Weight by strategy score, capped per position."},
            {"key": "equal", "name": "Equal weight",
             "description": "Every holding the same size. Hard to beat when "
                            "expected returns are uncertain."},
            {"key": "inverse_vol", "name": "Inverse volatility",
             "description": "Smaller weights to more volatile names. Requires "
                            "return history."},
        ],
        "disclaimer": DISCLAIMER,
    }


class BuildRequest(BaseModel):
    strategy: str = "balanced"
    risk_profile: str = "balanced"
    allocation: str = "score"
    max_holdings: Optional[int] = Field(default=None, ge=3, le=60)
    sector_limit: Optional[float] = Field(default=None, gt=0.05, le=1.0)
    sectors: Optional[list[str]] = None


@app.post("/api/portfolio/build")
def portfolio_build(req: BuildRequest):
    from ..portfolio_builder import build_portfolio
    rows = _db.universe_rows(limit=100000)
    if req.sectors:
        wanted = {s.lower() for s in req.sectors}
        rows = [r for r in rows if str(r.get("sector", "")).lower() in wanted]
    try:
        result = build_portfolio(
            rows, strategy=req.strategy, risk_profile=req.risk_profile,
            allocation=req.allocation, max_holdings=req.max_holdings,
            sector_limit=req.sector_limit)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    out = result.as_dict()
    if not SERVE_MARKET_DATA:
        for h in out["holdings"]:
            h.pop("pe", None)
        out["metrics"] = {k: v for k, v in out["metrics"].items()
                          if not any(p in k for p in PRICE_DERIVED_METRICS)}
    out["disclaimer"] = DISCLAIMER
    out["no_recommendation"] = (
        "A screening result, not a recommendation. Allocation here reflects "
        "the criteria you selected, not advice about what you should own.")
    return out


class AssessRequest(BaseModel):
    holdings: dict[str, float] = Field(
        ..., description="ticker -> weight; weights are normalised to sum to 1")


@app.post("/api/portfolio/assess")
def portfolio_assess(req: AssessRequest):
    from ..portfolio_builder import assess_portfolio
    if not req.holdings:
        raise HTTPException(status_code=400, detail="no holdings supplied")
    if len(req.holdings) > 200:
        raise HTTPException(status_code=400, detail="at most 200 holdings")
    rows = _db.universe_rows(limit=100000)
    result = assess_portfolio(req.holdings, rows)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    if not SERVE_MARKET_DATA:
        for h in result["holdings"]:
            h.pop("pe", None)
    result["disclaimer"] = DISCLAIMER
    return result
