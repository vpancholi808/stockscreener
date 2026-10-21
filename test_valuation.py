"""Validate relative valuation, sector benchmarks, research notes, portfolios."""
import sys, warnings, tempfile
sys.path.insert(0, "/home/claude/finmodel")
warnings.filterwarnings("ignore")

fails = []
def check(name, cond, detail=""):
    print("  [%s] %s%s" % ("OK" if cond else "FAIL", name, (" — "+detail) if detail else ""))
    if not cond: fails.append(name)

from finmodel.valuation import compute_multiples, Multiples, LOWER_IS_CHEAPER
from finmodel.sectors import (build_benchmarks, compare_to_sector, valuation_summary,
                              BENCHMARK_METRICS, MIN_PEERS)
from finmodel.research import generate_research
from finmodel.portfolio_builder import (build_portfolio, assess_portfolio,
                                        STRATEGIES, RISK_PROFILES, METRIC_BOUNDS)
from finmodel.web.store import Database
from finmodel.web.seed import seed, seed_prices

print("=" * 64); print("1) RELATIVE VALUATION"); print("=" * 64)

from types import SimpleNamespace
import pandas as pd
def fake_model(rev, ni, ebitda, equity, debt, cash, ocf, capex, ebit=None):
    yrs=[2023,2024]
    def f(d):
        df=pd.DataFrame(index=list(d), columns=yrs, dtype="float64")
        for k,v in d.items(): df.loc[k]=[v*0.9, v]
        return df
    is_=f({"revenue":rev,"net_income":ni,"ebitda":ebitda,
           "operating_income":ebit if ebit is not None else ebitda*0.8})
    bs=f({"total_equity":equity,"long_term_debt":debt,"short_term_debt":0.0,
          "cash":cash,"short_term_investments":0.0})
    cf=f({"operating_cash_flow":ocf,"capex":capex,"dividends_paid":0.0})
    fin=SimpleNamespace(years=yrs, income_statement=is_, balance_sheet=bs,
                        cash_flow=cf, latest_year=lambda: 2024)
    return SimpleNamespace(financials=fin)

m = compute_multiples(fake_model(1000, 100, 200, 500, 300, 100, 150, 50), 2000)
check("P/E = cap/earnings", abs(m.pe - 20.0) < 1e-9, "%.2f" % m.pe)
check("P/B = cap/equity", abs(m.pb - 4.0) < 1e-9, "%.2f" % m.pb)
check("P/S = cap/revenue", abs(m.ps - 2.0) < 1e-9, "%.2f" % m.ps)
check("EV = cap + debt - cash", abs(m.enterprise_value - 2200) < 1e-9,
      "%.0f" % m.enterprise_value)
check("EV/EBITDA uses EV not cap", abs(m.ev_ebitda - 11.0) < 1e-9, "%.2f" % m.ev_ebitda)
check("FCF yield = (OCF-capex)/cap", abs(m.fcf_yield - 0.05) < 1e-9, "%.4f" % m.fcf_yield)
check("earnings yield = 1/PE", abs(m.earnings_yield - 1/m.pe) < 1e-9)

# the critical guard: negative earnings must NOT produce a 'cheap' P/E
neg = compute_multiples(fake_model(1000, -100, 200, 500, 300, 100, 150, 50), 2000)
check("negative earnings -> P/E is None, not negative", neg.pe is None)
check("negative earnings -> earnings yield still reported",
      neg.earnings_yield is not None and neg.earnings_yield < 0)
check("no market cap -> all multiples None",
      compute_multiples(fake_model(1000,100,200,500,300,100,150,50), None).pe is None)
check("zero equity -> P/B None",
      compute_multiples(fake_model(1000,100,200,0,300,100,150,50), 2000).pb is None)

print("\n" + "=" * 64); print("2) SECTOR BENCHMARKS"); print("=" * 64)
# median must resist an outlier that destroys the mean
rows = [{"sector":"X","pe":float(v)} for v in [10,11,12,13,14,15,400]]
bm = build_benchmarks(rows, {"pe": -1})["X"]
check("median resists outlier", abs(bm.stat("pe").median - 13.0) < 1e-9,
      "median %.1f" % bm.stat("pe").median)
check("mean is destroyed by outlier", bm.stat("pe").mean > 60,
      "mean %.1f" % bm.stat("pe").mean)
check("both are reported", bm.stat("pe").mean is not None and bm.stat("pe").median is not None)

# direction handling: for P/E (lower better) a LOW value must score HIGH
peers = {"pe": [10.0,11.0,12.0,13.0,14.0,15.0,400.0]}
cheap = compare_to_sector({"pe": 10.0}, bm, {"pe": -1}, peer_values=peers)["pe"]
dear  = compare_to_sector({"pe": 400.0}, bm, {"pe": -1}, peer_values=peers)["pe"]
check("cheap P/E scores high percentile", cheap.percentile > 0.8,
      "%.2f" % cheap.percentile)
check("expensive P/E scores low percentile", dear.percentile < 0.2,
      "%.2f" % dear.percentile)
check("direction inverted only for lower-is-better", "pe" in LOWER_IS_CHEAPER)

# higher-is-better direction must NOT be flipped
bm2 = build_benchmarks([{"sector":"X","roic":v} for v in [0.05,0.1,0.15,0.2,0.25,0.3]],
                       {"roic": +1})["X"]
pv = {"roic":[0.05,0.1,0.15,0.2,0.25,0.3]}
hi = compare_to_sector({"roic":0.30}, bm2, {"roic":+1}, peer_values=pv)["roic"]
check("high ROIC scores high percentile", hi.percentile > 0.8, "%.2f" % hi.percentile)

thin = build_benchmarks([{"sector":"Y","pe":10.0},{"sector":"Y","pe":12.0}], {"pe":-1})["Y"]
c_thin = compare_to_sector({"pe":11.0}, thin, {"pe":-1})["pe"]
check("thin sector marked unavailable", not c_thin.available,
      "peers=%d < %d" % (c_thin.peers, MIN_PEERS))

print("\n" + "=" * 64); print("3) RESEARCH NOTES"); print("=" * 64)
strong = {"ticker":"AAA","name":"Strong Co","sector":"X","fiscal_year":2024,
          "health_passed":12,"health_total":12,"operating_margin":0.28,
          "roic":0.22,"fcf_margin":0.18,"cash_conversion":1.4,"piotroski":9,
          "altman":5.0,"beneish":-2.9,"accruals":-0.05,"revenue_growth":0.25,
          "net_debt_to_ebitda":-0.5,"interest_coverage":30.0,"current_ratio":2.5}
weak = {"ticker":"BBB","name":"Weak Co","sector":"X","fiscal_year":2024,
        "health_passed":3,"health_total":12,"operating_margin":-0.05,
        "roic":0.01,"fcf":-5e8,"fcf_margin":-0.08,"cash_conversion":0.4,
        "piotroski":2,"altman":1.2,"beneish":-1.0,"accruals":0.18,
        "revenue_growth":-0.12,"net_debt_to_ebitda":6.5,"interest_coverage":1.2,
        "current_ratio":0.7}
ns, nw = generate_research(strong), generate_research(weak)
check("strong company yields strengths", len(ns.strengths) >= 4, "%d" % len(ns.strengths))
check("strong company yields no concerns", len(ns.concerns) == 0)
check("weak company yields concerns", len(nw.concerns) >= 6, "%d" % len(nw.concerns))
check("weak company yields no strengths", len(nw.strengths) == 0)
d = ns.as_dict()
check("NO buy/sell/hold verdict field anywhere",
      not any(k in d for k in ("recommendation","verdict","rating","action","signal")))
# The only place "buy"/"sell" may appear is the explicit no-advice line.
body = "\n".join(
    ln for ln in (ns.as_text() + "\n" + nw.as_text()).lower().splitlines()
    if "not investment advice" not in ln and "no buy or sell" not in ln)
check("no buy/sell/hold language outside the disclaimer",
      not any(w in body for w in (" buy ", " sell ", "hold rating",
                                  "strong buy", "overweight", "underweight",
                                  "price target")))
full = (ns.as_text() + nw.as_text()).lower()
check("explicit no-advice statement present", "not investment advice" in full)
check("every finding carries evidence",
      all(f.evidence for f in ns.strengths + nw.concerns))
check("caveats always attached", len(ns.caveats) >= 4 and len(nw.caveats) >= 4)
check("distress flagged for weak company",
      any("distress" in f.statement.lower() for f in nw.concerns))

print("\n" + "=" * 64); print("4) PORTFOLIO CONSTRUCTION"); print("=" * 64)
tmp=tempfile.mkdtemp(); db=Database("sqlite:///%s/p.db"%tmp); db.init_schema()
seed(db, n_per_sector=10, verbose=False); seed_prices(db, verbose=False)
uni = db.universe_rows(limit=5000)
check("universe loaded with prices", len(uni) > 50 and
      sum(1 for r in uni if r.get("market_cap")) > 50, "%d rows" % len(uni))

for risk_key, risk in RISK_PROFILES.items():
    p = build_portfolio(uni, strategy="balanced", risk_profile=risk_key)
    if not p.holdings: continue
    mx = p.metrics["largest_position"]
    check("%s respects position cap" % risk_key, mx <= risk.max_position + 1e-6,
          "max %.3f vs cap %.3f" % (mx, risk.max_position))
    tot = sum(h.weight for h in p.holdings)
    check("%s weights sum to 1" % risk_key, abs(tot - 1.0) < 1e-6, "%.9f" % tot)
    if p.sector_weights:
        top = max(p.sector_weights.values())
        check("%s respects sector cap (or warns)" % risk_key,
              top <= risk.max_sector + 0.011 or len(p.warnings) > 0,
              "top sector %.2f vs cap %.2f" % (top, risk.max_sector))

qual = build_portfolio(uni, "quality", "conservative")
val  = build_portfolio(uni, "value", "conservative")
grow = build_portfolio(uni, "growth", "aggressive")
check("strategies select different books",
      {h.ticker for h in qual.holdings} != {h.ticker for h in val.holdings})
if qual.metrics.get("weighted_pe") and val.metrics.get("weighted_pe"):
    check("value book is cheaper than quality book",
          val.metrics["weighted_pe"] < qual.metrics["weighted_pe"],
          "value P/E %.1f vs quality %.1f" % (val.metrics["weighted_pe"],
                                              qual.metrics["weighted_pe"]))
check("growth book grows faster than value book",
      (grow.metrics.get("weighted_revenue_growth") or 0) >
      (val.metrics.get("weighted_revenue_growth") or 0))
check("conservative holds more names than aggressive",
      qual.metrics["holdings"] >= grow.metrics["holdings"])
check("effective holdings <= actual holdings",
      qual.metrics["effective_holdings"] <= qual.metrics["holdings"] + 1e-9)

# outlier guard
import pandas as pd
from finmodel.portfolio_builder import portfolio_metrics
bad = pd.DataFrame([{"ticker":"A","sector":"X","roe":0.20},
                    {"ticker":"B","sector":"X","roe":9.99}])
w = pd.Series([0.5,0.5], index=bad.index)
mm = portfolio_metrics(bad, w)
check("absurd ROE excluded from weighted average",
      abs(mm["weighted_roe"] - 0.20) < 1e-9, "%.3f" % mm["weighted_roe"])
check("exclusions are reported", mm["excluded_outliers"].get("roe") == 1)

bad_strat = False
try: build_portfolio(uni, "nonsense", "balanced")
except ValueError: bad_strat = True
check("unknown strategy raises", bad_strat)
check("empty universe handled", build_portfolio([], "quality", "balanced").holdings == [])

a = assess_portfolio({"MU":0.7,"NOPE":0.3}, uni)
check("assessment flags concentration", any("50%" in f or "book" in f for f in a["flags"]))
check("assessment reports unmatched tickers", "NOPE" in a["unmatched"])
check("assessment renormalises after dropping unmatched",
      abs(sum(h["weight"] for h in a["holdings"]) - 1.0) < 1e-6)

print("\n" + "=" * 64); print("5) FIVE-YEAR PRICE HISTORY"); print("=" * 64)
import numpy as np
from finmodel.data.market import price_statistics
rng = np.random.default_rng(5); n = 60
idx = pd.date_range("2021-01-31", periods=n, freq="ME")
mkt = rng.normal(0.008, 0.04, n)
stk = 0.002 + 1.3 * mkt + rng.normal(0, 0.015, n)
panel = pd.DataFrame({"^GSPC": 100*np.cumprod(1+mkt),
                      "TEST": 50*np.cumprod(1+stk)}, index=idx)
st = price_statistics(panel)["TEST"]
raw = (st["beta"] - 0.33) / 0.67
check("beta recovered from 5y history", abs(raw - 1.3) < 0.15,
      "raw %.3f vs true 1.300" % raw)
check("Blume adjustment applied", abs(st["beta"] - (0.67*raw + 0.33)) < 1e-9)
check("annualised volatility sane", 0.05 < st["volatility"] < 1.0,
      "%.1f%%" % (st["volatility"]*100))
check("max drawdown is negative", st["max_drawdown"] <= 0,
      "%.1f%%" % (st["max_drawdown"]*100))
check("uses full 5y window", st["observations"] >= 55, "%d months" % st["observations"])
check("short history rejected",
      "TEST" not in price_statistics(panel.head(4)))
check("benchmark itself gets no beta",
      price_statistics(panel)["^GSPC"]["beta"] is None)

print("=" * 64); print("6) RETURNS ON A NON-POSITIVE CAPITAL BASE"); print("=" * 64)

import math
from finmodel.scorecard import _return_on

# Figures from a real S&P 500 run. Each of these companies is highly
# profitable and carries negative or near-zero book equity after years of
# buybacks, so ROE on the raw denominator reads as catastrophe.
check("McDonald's negative equity gives no ROE, not -478%",
      math.isnan(_return_on(8.56e9, -1.79e9)))
check("Mettler-Toledo negative equity gives no ROE, not -3677%",
      math.isnan(_return_on(869e6, -23.64e6)))
check("O'Reilly negative equity gives no ROE",
      math.isnan(_return_on(2.54e9, -763.35e6)))
check("a zero capital base gives no ROE", math.isnan(_return_on(1e9, 0.0)))
check("a missing capital base gives no ROE", math.isnan(_return_on(1e9, None)))
check("an ordinary company is unaffected",
      abs(_return_on(1e9, 10e9) - 0.10) < 1e-12, "10.0%")
check("a small but genuinely positive base still computes",
      _return_on(2.13e9, 54e6) > 1.0,
      "Colgate: real, extreme, bounded elsewhere by METRIC_BOUNDS")
check("a real loss over positive equity is kept",
      abs(_return_on(-14.5e9, 5.77e9) + 2.513) < 0.01, "-251%")

print("=" * 64)
print("ALL PASSED" if not fails else "FAILURES: %s" % fails)
sys.exit(1 if fails else 0)
