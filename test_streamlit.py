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
         ("views/frontier.py", "frontier"),
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
# Frontier page. It needs no database, so it must render on a fresh
# deployment, and it must survive the parameter regimes that break the
# underlying mathematics rather than raising on them.
# --------------------------------------------------------------------------
print("\n[9] efficient frontier and the CAL")

SLIDERS = {"correlation": 0, "rf": 1, "aversion": 2}


def frontier_at(**sliders):
    at = AppTest.from_file(str(ROOT / "views" / "frontier.py"),
                           default_timeout=120).run()
    for name, val in sliders.items():
        at.slider[SLIDERS[name]].set_value(val)
    return at.run() if sliders else at


at = frontier_at()
if at.exception:
    fail("frontier page raised: %s" % at.exception[0].message)
else:
    labels = {m.label for m in at.metric}
    if {"Min-variance return", "Best Sharpe ratio"} <= labels:
        ok("frontier page renders with no database")
    else:
        fail("frontier page rendered without its headline metrics: %s" % labels)

# rf above the global minimum-variance return has no tangency portfolio. The
# old normalisation divided by a negative number and returned the *minimum*
# Sharpe portfolio as the maximum; the page must say so instead.
at = frontier_at(rf=9.0)
if at.exception:
    fail("frontier page raised with rf above the GMV return: %s"
         % at.exception[0].message)
elif not any("No tangency portfolio" in w.value for w in at.warning):
    fail("rf above the GMV return did not produce the no-tangency warning")
elif any(m.label == "Best Sharpe ratio" and m.value != "—" for m in at.metric):
    fail("a Sharpe ratio was reported where no tangency portfolio exists")
else:
    ok("rf above the GMV return is explained, not silently sign-flipped")

# Near-perfect correlation leaves almost nothing to diversify away.
at = frontier_at(correlation=0.95)
if at.exception:
    fail("frontier page raised at correlation 0.95: %s" % at.exception[0].message)
elif at.error:
    fail("frontier page errored at correlation 0.95: %s" % at.error[0].value)
else:
    ok("frontier page handles near-perfect correlation")

# Risk aversion must move the split and nothing else.
low = frontier_at(aversion=1.5)
high = frontier_at(aversion=10.0)
def risky(a):
    return next((m.value for m in a.metric
                 if m.label == "In the tangency portfolio"), None)
lo_w, hi_w = risky(low), risky(high)
if lo_w is None or hi_w is None:
    fail("complete-portfolio split not rendered")
elif float(lo_w.rstrip("%")) <= float(hi_w.rstrip("%")):
    fail("higher risk aversion did not reduce the risky weight (%s vs %s)"
         % (lo_w, hi_w))
else:
    ok("risk aversion sets the split (%s at A=1.5, %s at A=10)" % (lo_w, hi_w))

# The held-portfolio diagnosis must agree with the library, not just render.
# Three surfaces compute this now — library, Streamlit, static page — and the
# whole point of the diagnosis is the number.
at = frontier_at()
labels = {m.label: m.value for m in at.metric}
want = {"Return given up", "Volatility carried for nothing", "Below the CAL"}
if not want <= set(labels):
    fail("diagnosis metrics missing: %s" % (want - set(labels)))
else:
    import pandas as _pd  # noqa: E402
    import numpy as _np  # noqa: E402
    from finmodel.quant import MarkowitzOptimizer as _MO  # noqa: E402
    _names = ["Short bonds", "Broad equity", "Growth equity", "Real assets"]
    _mu = _pd.Series([.04, .08, .12, .10], index=_names)
    _sd = _np.array([.10, .18, .30, .22])
    _c = _np.full((4, 4), 0.35)
    _np.fill_diagonal(_c, 1.0)
    _cov = _pd.DataFrame(_np.outer(_sd, _sd) * _c, index=_names, columns=_names)
    _d = _MO(_mu, _cov, rf=0.02).diagnose(
        dict(zip(_names, [.10, .20, .60, .10])))
    pairs = [("Return given up", _d.return_gap),
             ("Volatility carried for nothing", _d.volatility_gap),
             ("Below the CAL", _d.cal_gap)]
    bad = [n for n, v in pairs
           if abs(float(labels[n].rstrip("%")) / 100 - v) > 5e-5]
    if bad:
        fail("diagnosis disagrees with the library: %s" % bad)
    else:
        ok("diagnosis matches the library (%s given up, %s excess vol)"
           % (labels["Return given up"],
              labels["Volatility carried for nothing"]))

    if not any("Inside the frontier" in w.value for w in at.warning):
        fail("an inefficient default holding was not flagged")
    else:
        ok("an inefficient holding is flagged as inside the frontier")

# --------------------------------------------------------------------------
print()
if failures:
    print("FAILED (%d)" % len(failures))
    for f in failures:
        print("  - %s" % f)
    sys.exit(1)
print("test_streamlit: all checks passed")
