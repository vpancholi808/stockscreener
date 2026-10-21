#!/usr/bin/env python3
"""Check a real, loaded database — the metrics and the app's structure.

    python worker.py --sp500
    python validate_real_data.py

The offline suites prove the models are arithmetically right on synthetic
data. They cannot tell you whether the real pipeline produced a *sensible*
universe: whether coverage is thin on some line item, whether a sector median
is implausible, whether the dividend yields came back on the wrong scale, or
whether every page still renders when the data is messy rather than generated.

Nothing here asserts a specific number — real data changes daily. It asserts
ranges that would have to be wrong for the result to be usable, and prints
the distribution so a human can judge the rest.

Under GitHub Actions it also emits workflow annotations, because log archives
are not always reachable.
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

CI = bool(os.environ.get("GITHUB_ACTIONS"))
problems: list[str] = []
notes: list[str] = []


def note(label: str, detail: str) -> None:
    line = "%s | %s" % (label, detail)
    notes.append(line)
    print("  %s" % line)
    if CI:
        print("::notice::V %s" % line[:380])


def bad(label: str, detail: str) -> None:
    line = "%s | %s" % (label, detail)
    problems.append(line)
    print("  PROBLEM  %s" % line)
    if CI:
        print("::error::V %s" % line[:380])


def expect(label: str, condition: bool, detail: str) -> bool:
    (note if condition else bad)(label, detail)
    return condition


def rule(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main() -> int:
    from finmodel.web.store import Database

    db = Database()
    stats = db.stats()
    total = stats.get("companies") or 0
    if total == 0:
        bad("empty", "no companies loaded; run worker.py --sp500 first")
        return 1

    conn = db.connect()
    companies = pd.read_sql_query("SELECT * FROM companies", conn)
    market = pd.read_sql_query("SELECT * FROM market_data", conn)
    merged = companies.merge(market, on="cik", how="left",
                             suffixes=("", "_mkt"))

    rule("1. UNIVERSE")
    note("loaded", "%d companies, %d sectors, latest FY %s"
         % (total, stats.get("sectors"), stats.get("latest_year")))
    # A partial run (worker.py --sp500 --limit N) is a legitimate smoke test,
    # so the expected size is a parameter rather than a constant.
    floor = int(os.environ.get("EXPECT_MIN_COMPANIES", "400"))
    expect("size", total >= floor,
           "%d companies against a floor of %d (full S&P 500 is ~503; well "
           "below means pulls failed)" % (total, floor))
    note("priced", "%d of %d have market data (%.0f%%)"
         % (stats.get("priced") or 0, total,
            100 * (stats.get("priced") or 0) / total))

    sectors = companies["sector"].value_counts()
    note("sectors", ", ".join("%s=%d" % (s, n) for s, n in sectors.items()))
    expect("sector spread", len(sectors) >= 8,
           "%d distinct sectors" % len(sectors))
    expect("no dominant sector", sectors.iloc[0] / total < 0.40,
           "largest is %s at %.0f%%" % (sectors.index[0],
                                        100 * sectors.iloc[0] / total))

    rule("2. METRIC COVERAGE")
    # Coverage floors are per metric, because several are not universal
    # concepts. A bank has no cost of revenue, so no gross margin; a REIT's
    # Altman Z is not meaningful and the inputs are often absent. Holding
    # every metric to one threshold would report a correct pipeline as broken.
    FLOORS = {
        "revenue": 95, "fcf": 85, "net_margin": 90, "roe": 85,
        "health_score": 95, "sector_percentile": 95, "piotroski": 70,
        "current_ratio": 55,          # banks present unclassified balance sheets
        "gross_margin": 50,           # no cost of revenue for financials/REITs
        "operating_margin": 70,
        "ebitda": 70, "roic": 70, "net_debt_to_ebitda": 65,
        "interest_coverage": 55,      # needs a separate interest expense line
        "altman": 40,                 # designed for manufacturers, not banks
        "beneish": 40,                # needs two consecutive years of 8 inputs
        "debt_to_equity": 80,
    }
    rows = []
    for col, floor in FLOORS.items():
        if col not in companies.columns:
            bad("missing column", col)
            continue
        rows.append((col, 100 * companies[col].notna().mean(), floor))
    for col, pct, floor in sorted(rows, key=lambda r: r[1]):
        expect("coverage", pct >= floor,
               "%-22s %5.1f%% (floor %d%%)" % (col, pct, floor))

    # Which sectors systematically lack which metric. This is structure, not
    # failure, but it belongs on the methodology page rather than being
    # discovered by a user staring at an empty column.
    for col in ("gross_margin", "altman", "interest_coverage"):
        if col not in companies.columns:
            continue
        by_sector = companies.groupby("sector")[col].apply(
            lambda s: 100 * s.notna().mean())
        absent = by_sector[by_sector < 50]
        if len(absent):
            note("not applicable", "%s is absent for %s"
                 % (col, ", ".join("%s (%.0f%%)" % (s, v)
                                   for s, v in absent.items())))

    priced_cols = ["price", "market_cap", "pe", "pb", "ps", "ev_ebitda",
                   "fcf_yield", "beta", "forward_pe", "dividend_yield",
                   "target_mean_price", "week52_high"]
    for col in priced_cols:
        if col not in market.columns:
            bad("missing market column", col)
            continue
        pct = 100 * market[col].notna().mean() if len(market) else 0.0
        note("market coverage", "%-20s %5.1f%%" % (col, pct))

    if "forward_pe" in market.columns:
        min_forward = int(os.environ.get("EXPECT_MIN_FORWARD_PE", "50"))
        expect("forward P/E arrived",
               market["forward_pe"].notna().sum() >= min_forward,
               "%d companies have it against a floor of %d (SEC cannot "
               "supply this; Yahoo can)"
               % (int(market["forward_pe"].notna().sum()), min_forward))

    rule("3. ARE THE NUMBERS PLAUSIBLE?")
    # Ranges a usable S&P 500 snapshot has to sit inside. Deliberately wide:
    # the point is to catch a unit error or a parsing fault, not to second
    # guess the market.
    bounds = {
        "gross_margin": (0.0, 1.0, "fraction"),
        "operating_margin": (-0.5, 0.80, "fraction"),
        "net_margin": (-0.5, 0.80, "fraction"),
        "roe": (-2.0, 3.0, "fraction"),
        "roic": (-1.0, 2.0, "fraction"),
        "current_ratio": (0.0, 20.0, "ratio"),
        "piotroski": (0.0, 9.0, "score"),
        "health_score": (0.0, 1.0, "fraction"),
        "sector_percentile": (0.0, 1.0, "fraction"),
    }
    for col, (lo, hi, kind) in bounds.items():
        if col not in companies.columns:
            continue
        s = companies[col].dropna()
        if s.empty:
            continue
        outside = int(((s < lo) | (s > hi)).sum())
        detail = ("median %.3f, range %.3f..%.3f, %d outside [%.1f, %.1f]"
                  % (s.median(), s.min(), s.max(), outside, lo, hi))
        # A handful of genuine extremes is normal; a large share is a bug.
        expect("range %s" % col, outside <= max(3, 0.02 * len(s)), detail)

    if "dividend_yield" in market.columns:
        s = market["dividend_yield"].dropna()
        if not s.empty:
            # The unit trap. Yahoo reports a percent in current versions; if
            # that slipped through unconverted, medians land near 2.0 not 0.02.
            expect("dividend yield scale", s.max() <= 0.25,
                   "median %.4f, max %.4f — a max above 0.25 means a percent "
                   "was stored as a fraction" % (s.median(), s.max()))
            expect("dividend yield magnitude", 0.0 < s.median() < 0.08,
                   "median %.4f (S&P 500 payers typically 1-4%%)" % s.median())

    if "pe" in market.columns:
        s = market["pe"].dropna()
        if not s.empty:
            expect("P/E positivity", bool((s > 0).all()),
                   "min %.1f — a negative P/E means the guard failed"
                   % s.min())
            note("P/E distribution", "median %.1f, p25 %.1f, p75 %.1f, max %.1f"
                 % (s.median(), s.quantile(.25), s.quantile(.75), s.max()))

    rule("4. SECTOR BENCHMARKS")
    benchmarks = pd.read_sql_query("SELECT * FROM sector_benchmarks", conn)
    if benchmarks.empty:
        bad("benchmarks", "none computed")
    else:
        note("benchmarks", "%d rows across %d sectors, %d metrics"
             % (len(benchmarks), benchmarks["sector"].nunique(),
                benchmarks["metric"].nunique()))
        thin = benchmarks[benchmarks["count"] < 5]
        note("thin benchmarks", "%d metric/sector pairs with fewer than 5 "
             "constituents (reported as unavailable by design)" % len(thin))
        # A NaN median means the metric does not exist for that sector, which
        # is the honest answer for a bank's gross margin — not a range error.
        gm = benchmarks[benchmarks["metric"] == "gross_margin"]
        unavailable = []
        for _, r in gm.iterrows():
            median = r["median"]
            if median is None or median != median:
                unavailable.append(r["sector"])
            elif not (0 <= median <= 1):
                bad("benchmark range", "%s gross margin median %.3f"
                    % (r["sector"], median))
        if unavailable:
            note("benchmark not applicable",
                 "gross margin has no median for %s — these sectors do not "
                 "report a cost of revenue" % ", ".join(unavailable))
        # The mean/median gap is the project's stated reason for using the
        # median; show it on real data rather than asserting it.
        pe = benchmarks[(benchmarks["metric"] == "pe")
                        & benchmarks["median"].notna()]
        if not pe.empty:
            for _, r in pe.sort_values("median").iterrows():
                note("sector P/E", "%-24s median %6.1f  mean %6.1f  (%+.0f%%)"
                     % (r["sector"], r["median"], r["mean"] or 0,
                        100 * ((r["mean"] or 0) / r["median"] - 1)
                        if r["median"] else 0))

    rule("5. SCORECARD BEHAVIOUR ON REAL DATA")
    if "passes_screen" in companies.columns:
        passing = companies.groupby("sector")["passes_screen"].agg(
            ["sum", "count"])
        passing["rate"] = 100 * passing["sum"] / passing["count"]
        for sector, r in passing.sort_values("rate", ascending=False).iterrows():
            note("pass rate", "%-24s %3d/%3d  %5.1f%%"
                 % (sector, r["sum"], r["count"], r["rate"]))
        spread = passing["rate"].max() - passing["rate"].min()
        note("sector bias", "pass rates span %.0f points across sectors — the "
             "reason the headline rank is peer-relative, not absolute"
             % spread)

    rule("6. DOES THE APP WORK ON THIS DATA?")
    screen = db.screen(limit=25)
    expect("screener query", screen["total"] > 0,
           "%d rows match an unfiltered screen" % screen["total"])
    filtered = db.screen(filters={"roic__min": 0.15}, limit=10)
    expect("screener filter", 0 < filtered["total"] < screen["total"],
           "ROIC >= 15%% narrows %d -> %d"
           % (screen["total"], filtered["total"]))

    ticker = screen["rows"][0]["ticker"]
    company = db.company(ticker)
    expect("company page data", bool(company and company.get("history")),
           "%s has %d history rows" % (ticker, len(company.get("history", []))))

    try:
        from finmodel.web.ui import load_research
        from finmodel.research import generate_research
        from finmodel.sectors import (BENCHMARK_METRICS, MetricStats,
                                      SectorBenchmark, compare_to_sector)
        universe = db.universe_rows(limit=100000)
        ok = 0
        for row in universe[:25]:
            sector = row.get("sector") or "Unknown"
            stored = db.benchmarks(sector)
            bench = SectorBenchmark(
                sector=sector,
                n_companies=stored[0]["n_companies"] if stored else 0,
                metrics={b["metric"]: MetricStats(
                    metric=b["metric"], count=b["count"] or 0,
                    median=b["median"], mean=b["mean"], p25=b["p25"],
                    p75=b["p75"], std=b["std"]) for b in stored})
            peers = db.peer_values(sector, list(BENCHMARK_METRICS))
            comparisons = compare_to_sector(row, bench, peer_values=peers)
            n = generate_research(row, comparisons)
            if n.summary and not hasattr(n, "verdict"):
                ok += 1
        expect("research notes", ok >= 20,
               "%d of 25 generated with no verdict field" % ok)
    except Exception as exc:
        bad("research notes", "%s: %s" % (type(exc).__name__, exc))

    try:
        from finmodel.portfolio_builder import (RISK_PROFILES, STRATEGIES,
                                                build_portfolio)
        universe = db.universe_rows(limit=100000)
        built = 0
        for strategy in STRATEGIES:
            for profile in RISK_PROFILES:
                try:
                    result = build_portfolio(universe, strategy=strategy,
                                             risk_profile=profile)
                    if result.holdings:
                        built += 1
                except ValueError:
                    pass
        expect("portfolios", built >= 15,
               "%d of %d strategy/profile combinations produced holdings"
               % (built, len(STRATEGIES) * len(RISK_PROFILES)))

        result = build_portfolio(universe, strategy="quality",
                                 risk_profile="balanced")
        weights = [h.weight for h in result.holdings]
        cap = RISK_PROFILES["balanced"].max_position
        expect("position cap honoured", max(weights) <= cap + 1e-6,
               "largest position %.2f%% against a %.0f%% cap"
               % (100 * max(weights), 100 * cap))
        expect("weights sum to 1", abs(sum(weights) - 1.0) < 1e-6,
               "%.8f" % sum(weights))
        sector_max = max(result.sector_weights.values())
        note("sector concentration", "largest sector %.1f%%" % (100 * sector_max))
    except Exception as exc:
        bad("portfolios", "%s: %s" % (type(exc).__name__, exc))

    rule("7. STREAMLIT PAGES AGAINST REAL DATA")
    try:
        from streamlit.testing.v1 import AppTest
        import finmodel.web.ui as ui
        ui.clear_caches()
        root = Path(__file__).resolve().parent
        for path, label in (("streamlit_app.py", "router/screener"),
                            ("views/company.py", "company"),
                            ("views/portfolio.py", "portfolio"),
                            ("views/methodology.py", "methodology")):
            at = AppTest.from_file(str(root / path), default_timeout=300)
            at.run()
            if at.exception:
                first = at.exception[0]
                bad("page %s" % label,
                    (first.value or "").strip().splitlines()[0][:200])
            else:
                rows = len(at.dataframe[0].value) if at.dataframe else 0
                note("page %s" % label, "renders, %d dataframes, %d rows"
                     % (len(at.dataframe), rows))
    except ImportError:
        note("streamlit", "not installed; page checks skipped")

    rule("SUMMARY")
    if problems:
        print("  %d problem(s):" % len(problems))
        for p in problems:
            print("    - %s" % p)
        if CI:
            print("::error::V SUMMARY %d problems" % len(problems))
        return 1
    print("  No problems. %d checks reported." % len(notes))
    if CI:
        print("::notice::V SUMMARY clean — %d checks" % len(notes))
    return 0


if __name__ == "__main__":
    sys.exit(main())
