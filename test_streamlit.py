"""Streamlit app suite — renders every page and exercises the interactive paths.

Offline and self-contained, like the other suites: it seeds a throwaway SQLite
database with the generated demo set, then drives the app through Streamlit's
AppTest harness. No network, no SEC, no browser.

What this is actually guarding against, having been caught by hand once each:

  * a page raising on render (an import, a column that is not there, a format
    string applied to the wrong scale)
  * passing the wrong shape into a model function — assess_portfolio takes
    ticker -> weight, and a list of dicts raised AttributeError
  * the SERVE_MARKET_DATA licensing gate leaking price-derived fields
"""
from __future__ import annotations

import os
import sys
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

_TMP = tempfile.mkdtemp(prefix="finmodel_st_")
os.environ["DATABASE_URL"] = "sqlite:///%s/test.db" % _TMP

from streamlit.testing.v1 import AppTest  # noqa: E402

VIEWS = ROOT / "views"
failures: list[str] = []


def fail(msg: str) -> None:
    failures.append(msg)
    print("  FAIL  %s" % msg)


def ok(msg: str) -> None:
    print("  ok    %s" % msg)


def run(path: str, timeout: int = 300, **query) -> AppTest:
    at = AppTest.from_file(str(ROOT / path), default_timeout=timeout)
    for k, v in query.items():
        at.query_params[k] = v
    at.run()
    return at


def check(at: AppTest, label: str) -> bool:
    if at.exception:
        first = at.exception[0]
        detail = (first.value or "").strip().splitlines()
        fail("%s raised: %s" % (label, detail[0] if detail else first))
        for line in (first.stack_trace or [])[-4:]:
            print("        %s" % line.rstrip())
        return False
    return True


def seed_database() -> int:
    from finmodel.web.loader import compute_benchmarks
    from finmodel.web.seed import seed, seed_prices
    from finmodel.web.store import Database

    db = Database()
    db.init_schema()
    seed(db)
    seed_prices(db)
    compute_benchmarks(db)
    db.compute_sector_percentiles()
    n = db.stats()["companies"]
    db.close()

    # AppTest runs share a process, so the caches populated while the database
    # was empty survive into the next phase and would report zero companies.
    import finmodel.web.ui as _u
    _u.clear_caches()
    _u.get_db.clear()
    return n


# --------------------------------------------------------------------------
print("\n[1] empty database -> first-run panel")
at = run("streamlit_app.py")
if check(at, "empty-db screener"):
    if not any("Load demo data" in str(b.label) for b in at.button):
        fail("empty database should offer the demo-data button")
    elif not at.warning:
        fail("empty database should warn rather than render empty tables")
    else:
        ok("empty state offers demo data instead of blank tables")

print("\n[2] seed")
count = seed_database()
if count < 100:
    fail("seed loaded only %d companies" % count)
else:
    ok("seeded %d companies" % count)

print("\n[3] every page renders")
PAGES = [("streamlit_app.py", "router"),
         ("views/screener.py", "screener"),
         ("views/company.py", "company"),
         ("views/portfolio.py", "portfolio"),
         ("views/methodology.py", "methodology")]
for path, label in PAGES:
    at = run(path)
    if check(at, label):
        ok("%s renders" % label)

print("\n[4] screener filters narrow the result set")
at = run("views/screener.py")
if check(at, "screener"):
    total = len(at.dataframe[0].value)
    at.toggle[0].set_value(True).run()
    if check(at, "passing-only toggle"):
        passing = len(at.dataframe[0].value)
        if passing >= total or passing == 0:
            fail("passing-only gave %d of %d" % (passing, total))
        else:
            ok("passing-only: %d of %d" % (passing, total))

at = run("views/screener.py")
if check(at, "screener"):
    at.text_input[0].set_value("zzzz-no-such-company").run()
    if check(at, "no-match search"):
        if not at.info:
            fail("a search with no matches should show an empty-state message")
        else:
            ok("no-match search shows an empty state")

print("\n[5] portfolio builds under every strategy and risk profile")
from finmodel.portfolio_builder import RISK_PROFILES, STRATEGIES  # noqa: E402

built = 0
for strategy in STRATEGIES:
    for profile in RISK_PROFILES:
        at = run("views/portfolio.py")
        at.selectbox[0].set_value(strategy).run()
        at.selectbox[1].set_value(profile).run()
        [b for b in at.button if "Build portfolio" in str(b.label)][0].click().run()
        if check(at, "build %s/%s" % (strategy, profile)):
            built += 1
if built == len(STRATEGIES) * len(RISK_PROFILES):
    ok("%d strategy x risk-profile combinations build" % built)

print("\n[6] sector cap slider is an integer percent")
at = run("views/portfolio.py")
if check(at, "portfolio"):
    cap = next((s for s in at.slider
                if "Max weight in any one sector" in str(s.label)), None)
    if cap is None:
        fail("sector cap slider missing")
    elif not isinstance(cap.value, int) or not 10 <= cap.value <= 100:
        # A 0-1 fraction here renders as "0%" under a %d%% format.
        fail("sector cap should be an integer percent, got %r" % (cap.value,))
    else:
        ok("sector cap is %d%%" % cap.value)

print("\n[7] assess accepts the formats people actually paste")
CASES = [
    ("plain", "%s, 0.6\n%s, 0.4", False),
    ("percent signs", "%s, 60%%\n%s, 40%%", False),
    ("duplicate ticker", "%s, 0.5\n%s, 0.3\n%s, 0.2", True),
]
from finmodel.web.store import Database  # noqa: E402

_db = Database()
tickers = [r["ticker"] for r in _db.screen(limit=5)["rows"]]
_db.close()

for label, template, is_dup in CASES:
    at = run("views/portfolio.py")
    if is_dup:
        text = template % (tickers[0], tickers[0], tickers[1])
    else:
        text = template % (tickers[0], tickers[1])
    at.text_area[0].set_value(text).run()
    [b for b in at.button if str(b.label) == "Assess"][0].click().run()
    if check(at, "assess (%s)" % label):
        if at.error:
            fail("assess (%s) errored: %s" % (label, at.error[0].value))
        else:
            ok("assess handles %s" % label)

at = run("views/portfolio.py")
at.text_area[0].set_value("ZZZZ, 1.0").run()
[b for b in at.button if str(b.label) == "Assess"][0].click().run()
if check(at, "assess unknown ticker"):
    if not at.error:
        fail("an unknown ticker should produce a visible error")
    else:
        ok("unknown ticker reports an error rather than failing silently")

print("\n[8] price-derived fields stay behind SERVE_MARKET_DATA")


def priced_metrics_visible() -> bool:
    # The research note is cached, so the previous flag's result would be
    # replayed and the gate would appear to work in both directions.
    import finmodel.web.ui as _u
    _u.clear_caches()
    at = run("views/company.py")
    if not check(at, "company page"):
        return False
    found = set()
    for frame in at.dataframe:
        cols = getattr(frame.value, "columns", [])
        if "Metric" in cols:
            found |= {str(m).lower() for m in frame.value["Metric"]}
    return any(k in m for m in found
               for k in ("p/e", "ev/", "yield", "p/b", "p/s", "peg"))


import finmodel.web.ui as _ui  # noqa: E402

_ui.SERVE_MARKET_DATA = False
if priced_metrics_visible():
    fail("price-derived metrics shown with SERVE_MARKET_DATA off — "
         "scraped price data is generally not redistributable")
else:
    ok("price-derived metrics withheld by default")

_ui.SERVE_MARKET_DATA = True
if not priced_metrics_visible():
    fail("price-derived metrics missing with SERVE_MARKET_DATA on")
else:
    ok("price-derived metrics served when explicitly enabled")
_ui.SERVE_MARKET_DATA = False

# --------------------------------------------------------------------------
print()
if failures:
    print("FAILED (%d)" % len(failures))
    for f in failures:
        print("  - %s" % f)
    sys.exit(1)
print("test_streamlit: all checks passed")
