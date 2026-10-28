"""Cross-check the static frontier page against finmodel/quant/portfolio.py.

build_frontier_page.py reimplements the frontier arithmetic in JavaScript so
the page runs with no server. Two implementations of one formula drift, and a
page that quietly disagrees with the library is worse than no page: it looks
authoritative and is wrong. So this loads the real page in headless Chromium
and compares its numbers against the Python the rest of the project uses.

Offline. Chromium and Playwright are preinstalled; if Playwright is missing
the suite skips rather than fails, so a bare checkout still runs green.
"""
from __future__ import annotations

import json
import sys
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from build_frontier_page import build, build_artifact
from finmodel.quant.portfolio import MarkowitzOptimizer, NoTangencyPortfolio

ok = lambda c: "OK" if c else "FAIL"
failures: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print("  [%s] %s%s" % (ok(cond), name, (" — " + detail) if detail else ""))
    if not cond:
        failures.append(name)


def close(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(b)))


try:
    from playwright.sync_api import sync_playwright
except ImportError:
    print("test_frontier_page: playwright not installed, skipping")
    sys.exit(0)


# --------------------------------------------------------------------------
# The same inputs on both sides.
# --------------------------------------------------------------------------
NAMES = ["Short bonds", "Broad equity", "Growth equity", "Real assets"]
MU_PCT = [4.0, 8.0, 12.0, 10.0]
SD_PCT = [10.0, 18.0, 30.0, 22.0]
RHO = 0.35


def python_model(rf: float, rho: float = RHO):
    mu = pd.Series([m / 100 for m in MU_PCT], index=NAMES)
    sd = np.array([s / 100 for s in SD_PCT])
    corr = np.full((len(sd), len(sd)), rho)
    np.fill_diagonal(corr, 1.0)
    cov = pd.DataFrame(np.outer(sd, sd) * corr, index=NAMES, columns=NAMES)
    return MarkowitzOptimizer(mu, cov, rf=rf)


JS = """
(args) => {
  const {mu, sd, rho, rf} = args;
  const S = sd.map((a,i)=>sd.map((b,j)=>a*b*(i===j?1:rho)));
  const g = window.__model.geometry(mu, S);
  if (!g) return {degenerate:true};
  const gmv = window.__model.gmvPortfolio(g, mu, S, rf);
  const tan = window.__model.tangency(g, mu, S, rf);
  const fr  = window.__model.frontier(g, mu, S, rf, 25, false);
  const full= window.__model.frontier(g, mu, S, rf, 25, true);
  return {
    a:g.a, b:g.b, c:g.c, d:g.d,
    gmvReturn:g.gmvReturn, gmvVol:g.gmvVol, asymptote:g.asymptote,
    maxSharpe: window.__model.maxSharpe(g, rf),
    gmv:{r:gmv.r, v:gmv.v, sharpe:gmv.sharpe, w:gmv.w},
    tangency: tan ? {r:tan.r, v:tan.v, sharpe:tan.sharpe, w:tan.w} : null,
    frontier: fr.map(p=>({r:p.r, v:p.v, sharpe:p.sharpe, eff:p.efficient})),
    fullBranches: {
      efficient: full.filter(p=>p.efficient).length,
      lower:     full.filter(p=>!p.efficient).length
    },
    volAt: [0.05,0.08,0.11,0.14].map(t=>window.__model.volAt(g,t)),
    cond: window.__model.conditionNumber(S)
  };
}
"""

tmp = Path(tempfile.mkdtemp(prefix="fm_front_")) / "frontier.html"
tmp.write_text(build("Efficient frontier and the CAL"), encoding="utf-8")

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    console_errors: list[str] = []
    page.on("console", lambda m: console_errors.append(m.text)
            if m.type == "error" else None)
    page.on("pageerror", lambda e: console_errors.append(str(e)))
    page.goto(tmp.as_uri())
    page.wait_for_function("() => window.__model !== undefined", timeout=15000)

    print("=" * 64)
    print("1) THE PAGE RENDERS")
    check("no console or page errors", not console_errors,
          "; ".join(console_errors[:2]))
    check("asset rows drawn",
          page.locator("#assets tbody tr").count() == 4,
          "%d rows" % page.locator("#assets tbody tr").count())
    check("headline metrics rendered",
          page.locator("#stats .stat").count() == 4)
    check("chart drawn as svg", page.locator("#chart svg").count() == 1)
    check("weights table has a row per asset",
          page.locator("#weights tbody tr").count() == 5,  # 4 + gross exposure
          "%d rows" % page.locator("#weights tbody tr").count())
    check("complete-portfolio ladder rendered",
          page.locator("#complete tbody tr").count() >= 5)

    args = {"mu": [m / 100 for m in MU_PCT], "sd": [s / 100 for s in SD_PCT],
            "rho": RHO, "rf": 0.02}

    print("\n2) GEOMETRY MATCHES THE PYTHON")
    js = page.evaluate(JS, args)
    opt = python_model(0.02)
    geo = opt.geometry()
    for k, want in (("a", geo.a), ("b", geo.b), ("c", geo.c), ("d", geo.d)):
        check("%s agrees" % k.upper(), close(js[k], want),
              "%.10f vs %.10f" % (js[k], want))
    check("GMV return agrees", close(js["gmvReturn"], geo.gmv_return),
          "%.10f vs %.10f" % (js["gmvReturn"], geo.gmv_return))
    check("GMV volatility agrees", close(js["gmvVol"], geo.gmv_volatility),
          "%.10f vs %.10f" % (js["gmvVol"], geo.gmv_volatility))
    check("asymptotic Sharpe agrees",
          close(js["asymptote"], geo.asymptotic_sharpe),
          "%.10f vs %.10f" % (js["asymptote"], geo.asymptotic_sharpe))
    check("max attainable Sharpe agrees",
          close(js["maxSharpe"], geo.max_sharpe(0.02)),
          "%.10f vs %.10f" % (js["maxSharpe"], geo.max_sharpe(0.02)))

    print("\n3) PORTFOLIOS MATCH THE PYTHON")
    gmv = opt.min_variance()
    check("GMV return", close(js["gmv"]["r"], gmv.expected_return))
    check("GMV volatility", close(js["gmv"]["v"], gmv.volatility))
    check("GMV weights",
          all(close(a, b, 1e-8) for a, b in
              zip(js["gmv"]["w"], gmv.weights.to_numpy())),
          "max diff %.2e" % max(abs(np.array(js["gmv"]["w"])
                                    - gmv.weights.to_numpy())))
    tan = opt.tangency()
    check("tangency return", close(js["tangency"]["r"], tan.expected_return),
          "%.10f vs %.10f" % (js["tangency"]["r"], tan.expected_return))
    check("tangency volatility", close(js["tangency"]["v"], tan.volatility))
    check("tangency Sharpe", close(js["tangency"]["sharpe"], tan.sharpe),
          "%.10f vs %.10f" % (js["tangency"]["sharpe"], tan.sharpe))
    check("tangency weights",
          all(close(a, b, 1e-8) for a, b in
              zip(js["tangency"]["w"], tan.weights.to_numpy())),
          "max diff %.2e" % max(abs(np.array(js["tangency"]["w"])
                                    - tan.weights.to_numpy())))

    print("\n4) THE FRONTIER MATCHES, AND IS EFFICIENT")
    py_fr = opt.efficient_frontier(points=25)
    check("same number of points", len(js["frontier"]) == len(py_fr),
          "%d vs %d" % (len(js["frontier"]), len(py_fr)))
    dv = max(abs(j["v"] - p) for j, p in
             zip(js["frontier"], py_fr.volatility.to_numpy()))
    dr = max(abs(j["r"] - p) for j, p in
             zip(js["frontier"], py_fr["return"].to_numpy()))
    check("volatilities agree", dv < 1e-9, "max diff %.2e" % dv)
    check("returns agree", dr < 1e-9, "max diff %.2e" % dr)
    check("every point is on the efficient branch",
          all(j["eff"] for j in js["frontier"]))
    check("no point below the GMV return",
          min(j["r"] for j in js["frontier"]) >= geo.gmv_return - 1e-12)
    v = [j["v"] for j in js["frontier"]]
    check("volatility is monotone",
          all(b >= a - 1e-12 for a, b in zip(v, v[1:])))
    check("include-lower returns both branches",
          js["fullBranches"]["efficient"] > 0 and js["fullBranches"]["lower"] > 0,
          "%d efficient / %d lower" % (js["fullBranches"]["efficient"],
                                       js["fullBranches"]["lower"]))
    dva = max(abs(a - geo.volatility_at(t)) for a, t in
              zip(js["volAt"], [0.05, 0.08, 0.11, 0.14]))
    check("closed-form sigma(r) agrees", dva < 1e-10, "max diff %.2e" % dva)

    print("\n5) THE NO-TANGENCY CASE BEHAVES THE SAME")
    hi = dict(args, rf=0.09)
    js_hi = page.evaluate(JS, hi)
    opt_hi = python_model(0.09)
    check("JavaScript returns null for the tangency portfolio",
          js_hi["tangency"] is None)
    raised = False
    try:
        opt_hi.tangency()
    except NoTangencyPortfolio:
        raised = True
    check("Python raises NoTangencyPortfolio", raised)
    check("both agree the rate is above the GMV return",
          js_hi["gmvReturn"] < 0.09)
    check("asymptotic Sharpe still agrees",
          close(js_hi["asymptote"], opt_hi.geometry().asymptotic_sharpe))

    print("\n6) THE PAGE EXPLAINS IT RATHER THAN SHOWING A NUMBER")
    page.locator("#rf").evaluate(
        "el => { el.value = 9; el.dispatchEvent(new Event('input')); }")
    page.wait_for_timeout(120)
    warn = page.locator("#alert .warn").inner_text() if \
        page.locator("#alert .warn").count() else ""
    check("a no-tangency explanation is shown",
          "No tangency portfolio exists" in warn,
          warn[:60])
    sharpe_tile = page.locator("#stats .stat").nth(3).inner_text()
    check("no Sharpe ratio is reported where none exists",
          "—" in sharpe_tile, sharpe_tile.replace("\n", " ")[:50])
    check("the complete-portfolio section is hidden",
          not page.locator("#completewrap").is_visible())

    print("\n7) CONDITION NUMBER AGREES WITH NUMPY")
    sd = np.array([s / 100 for s in SD_PCT])
    corr = np.full((4, 4), RHO)
    np.fill_diagonal(corr, 1.0)
    cov = np.outer(sd, sd) * corr
    want = float(np.linalg.cond(cov))
    check("Jacobi condition number matches numpy",
          abs(js["cond"] - want) / want < 1e-6,
          "%.6f vs %.6f" % (js["cond"], want))

    print("\n8) THE ARTIFACT BUILD IS A FRAGMENT, NOT A DOCUMENT")
    # The Artifact platform wraps the file in its own skeleton and sets the
    # theme on the root element. A second <html>/<body>, or a theme toggle
    # fighting the host's data-theme, both break it.
    frag = build_artifact()
    # "<head" alone also matches "<header>", which the fragment legitimately
    # has; match the real tags.
    for tag in ("<!doctype", "<html", "</html>", "<body", "</body>",
                "<head>", "</head>"):
        check("no %s in the artifact build" % tag, tag not in frag.lower())
    check("artifact keeps a title", frag.lstrip().startswith("<title>"))
    check("artifact declares the viewport-fit promise",
          'claude:viewport-fit-sides' in frag)
    check("artifact uses safe-area side gutters",
          "max(16px,env(safe-area-inset-left" in frag.replace(" ", "")
          or "max(16px, env(safe-area-inset-left" in frag)
    check("artifact drops the theme toggle", 'id="theme"' not in frag)
    check("site build keeps the theme toggle", 'id="theme"' in build())
    check("both builds carry the same model code",
          frag.count("function tangency(") == 1
          and build().count("function tangency(") == 1)

    art = Path(tempfile.mkdtemp(prefix="fm_art_")) / "a.html"
    art.write_text("<!doctype html><html><head><meta charset='utf-8'>"
                   "</head><body>" + frag + "</body></html>", encoding="utf-8")
    page2 = browser.new_page(viewport={"width": 390, "height": 800})
    errs2: list[str] = []
    page2.on("pageerror", lambda e: errs2.append(str(e)))
    page2.goto(art.as_uri())
    page2.wait_for_function("() => window.__model !== undefined", timeout=15000)
    check("artifact build renders without errors", not errs2,
          "; ".join(errs2[:2]))
    check("artifact build renders its metrics at phone width",
          page2.locator("#stats .stat").count() == 4)
    doc_w = page2.evaluate("() => document.documentElement.scrollWidth")
    check("no horizontal page scroll at 390px", doc_w <= 391,
          "scrollWidth %d" % doc_w)

    browser.close()

print("=" * 64)
if failures:
    print("FAILURES (%d): %s" % (len(failures), failures))
    sys.exit(1)
print("test_frontier_page: the page agrees with the library")
sys.exit(0)
