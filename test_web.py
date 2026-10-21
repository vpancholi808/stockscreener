"""Validate the web data layer: store round-trip, filters, sector ranking, API gating."""
import json
import sys, os, tempfile, warnings
sys.path.insert(0, "/home/claude/finmodel")
warnings.filterwarnings("ignore")

fails = []
def check(name, cond, detail=""):
    print("  [%s] %s%s" % ("OK" if cond else "FAIL", name, (" — " + detail) if detail else ""))
    if not cond: fails.append(name)

from finmodel.web.store import Database, SORTABLE, NUMERIC_FILTERS
from finmodel.web.seed import seed

tmp = tempfile.mkdtemp()
db = Database("sqlite:///%s/t.db" % tmp)
db.init_schema()
res = seed(db, n_per_sector=4, verbose=False)

print("=" * 62); print("  WEB DATA LAYER"); print("=" * 62)
check("seed loaded companies", res["loaded"] > 20, "%d loaded" % res["loaded"])
check("no seed failures", res["skipped"] == 0, "%d skipped" % res["skipped"])

s = db.stats()
check("stats returns counts", s["companies"] == res["loaded"])
check("multiple sectors present", s["sectors"] >= 5, "%d sectors" % s["sectors"])

# Micron is real data — verify it survived the round trip intact
mu = db.company("MU")
check("MU round-trips", mu is not None)
check("MU health 11/12", mu["health_passed"] == 11 and mu["health_total"] == 12,
      "%s/%s" % (mu["health_passed"], mu["health_total"]))
check("MU revenue ~37.4B", abs(mu["revenue"] - 37380e6) < 1e6,
      "%.1fB" % (mu["revenue"] / 1e9))
check("MU piotroski 9", mu["piotroski"] == 9, str(mu["piotroski"]))
check("MU history has 3 years", len(mu["history"]) == 3,
      str([h["fiscal_year"] for h in mu["history"]]))
check("MU fails on FCF margin only", mu["fcf_margin"] < 0.05)

# filters
allr = db.screen(limit=500)
hi = db.screen(filters={"roic__min": 0.20}, limit=500)
check("roic filter narrows set", 0 < hi["total"] < allr["total"],
      "%d of %d" % (hi["total"], allr["total"]))
check("roic filter is correct", all(r["roic"] >= 0.20 for r in hi["rows"]))

lev = db.screen(filters={"net_debt_to_ebitda__max": 1.0}, limit=500)
check("max filter direction correct",
      all(r["net_debt_to_ebitda"] <= 1.0 for r in lev["rows"]))

# unknown filter keys must be ignored, not crash or inject
safe = db.screen(filters={"bogus__min": 1, "revenue); DROP TABLE companies;--__min": 1},
                 limit=5)
check("unknown filter keys ignored safely", safe["total"] == allr["total"])
check("table still exists after injection attempt", db.stats()["companies"] == res["loaded"])

# sorting
asc = db.screen(sort_by="revenue", desc=False, limit=10)
revs = [r["revenue"] for r in asc["rows"] if r["revenue"] is not None]
check("ascending sort works", revs == sorted(revs))
bad = db.screen(sort_by="'; DROP TABLE companies;--", limit=3)
check("invalid sort column falls back", len(bad["rows"]) == 3)

# search + sector
check("search finds Micron", db.screen(search="micron")["total"] >= 1)
sec = db.screen(sector="Semiconductors", limit=100)
check("sector filter works", all(r["sector"] == "Semiconductors" for r in sec["rows"]))

# pagination
p1 = db.screen(limit=5, offset=0)
p2 = db.screen(limit=5, offset=5)
check("pagination returns distinct rows",
      not ({r["cik"] for r in p1["rows"]} & {r["cik"] for r in p2["rows"]}))
check("total stable across pages", p1["total"] == p2["total"])

# sector-relative ranking
pcts = [r["sector_percentile"] for r in allr["rows"] if r["sector_percentile"] is not None]
check("percentiles computed", len(pcts) > 20, "%d ranked" % len(pcts))
check("percentiles within [0,1]", all(0 <= p <= 1 for p in pcts))
by_sector = {}
for r in allr["rows"]:
    if r["sector_percentile"] is not None:
        by_sector.setdefault(r["sector"], []).append(r["sector_percentile"])
# A true percentile puts the sector's best at 1.0 — unless companies tie at
# the top, where rank(method="average") correctly gives them a shared value.
def top_ok(vals):
    if len(vals) < 2:
        return True
    if abs(max(vals) - 1.0) < 1e-6:
        return True
    return sum(1 for x in vals if abs(x - max(vals)) < 1e-9) > 1   # tie at top
check("each sector's best reaches 1.0 (or ties)",
      all(top_ok(v) for v in by_sector.values()))
check("percentiles spread across range (not compressed to mid)",
      all(min(v) < 0.5 for v in by_sector.values() if len(v) > 2))

print("\n" + "=" * 62); print("  API LAYER"); print("=" * 62)
os.environ["SERVE_MARKET_DATA"] = "false"
os.environ["DATABASE_URL"] = "sqlite:///%s/t.db" % tmp
for m in [k for k in list(sys.modules) if k.startswith("finmodel.web.api")]:
    del sys.modules[m]
from fastapi.testclient import TestClient
import finmodel.web.api as apimod
client = TestClient(apimod.app)

r = client.get("/api/health")
check("health endpoint", r.status_code == 200 and r.json()["status"] == "ok")
r = client.get("/api/companies?limit=3")
check("companies endpoint", r.status_code == 200 and len(r.json()["rows"]) == 3)
check("disclaimer in list response", "not investment advice" in r.json()["disclaimer"].lower())
check("market fields stripped when unlicensed",
      not any(k in r.json()["rows"][0] for k in ("price", "pe", "market_cap")))
r = client.get("/api/companies/MU")
check("detail endpoint", r.status_code == 200 and r.json()["ticker"] == "MU")
check("market block withheld", r.json()["market"] is None)
check("404 on unknown ticker", client.get("/api/companies/ZZZZ").status_code == 404)
r = client.get("/api/meta")
check("meta exposes thresholds", "scorecard_thresholds" in r.json()
      and len(r.json()["scorecard_thresholds"]) >= 10)
check("sectors endpoint", len(client.get("/api/sectors").json()["sectors"]) >= 5)
check("limit is capped", client.get("/api/companies?limit=9999").status_code == 422)



# --------------------------------------------------------------------------
print()
print("=" * 64)
print("STATIC SITE BUILDER (build_site.py)")
print("=" * 64)

import importlib
import os as _os
import re as _re

_os.environ.pop("SERVE_MARKET_DATA", None)
import build_site as _bs
importlib.reload(_bs)

_html = _bs.build(db, "Test Screener")
check("site renders a full HTML document",
      _html.startswith("<!doctype html>") and _html.rstrip().endswith("</html>"))
check("the title is substituted", "Test Screener" in _html)
check("no placeholder survives",
      "__DATA__" not in _html and "__TITLE__" not in _html)

_m = _re.search(r'<script id="data" type="application/json">(.*?)</script>',
                _html, _re.S)
check("the data block is parseable", _m is not None)
_data = json.loads(_m.group(1).replace("<\\/", "</"))
check("every company is embedded",
      len(_data["companies"]) == db.stats()["companies"],
      "%d rows" % len(_data["companies"]))
check("history is keyed by cik as a string",
      all(isinstance(k, str) for k in _data["history"]))

# The licensing gate is the whole reason this file can be published at all.
_leaked = sorted({f for r in _data["companies"] for f in r
                  if f in _bs.PRICE_DERIVED})
check("no price-derived field is embedded by default", not _leaked,
      "leaked: %s" % _leaked if _leaked else "clean")
check("price-derived benchmarks are excluded by default",
      not any(b.get("price_derived")
              for s in _data["benchmarks"].values() for b in s.values()))
# Match contiguous markers: the licence sentence is assembled from
# concatenated JS string literals, so "public domain" is split across them.
check("the page states why they are missing",
      "US government work" in _html and "redistributable" in _html)
check("the disclaimer is present", "Not investment advice" in _html)
check("no buy/sell verdict appears",
      "no buy, sell or hold" in _html.lower())

# A </script> inside the JSON would end the block early and break the page.
check("closing tags inside the payload are escaped",
      "<\\/" in _html or "</" not in _m.group(1))

_os.environ["SERVE_MARKET_DATA"] = "true"
importlib.reload(_bs)
_html2 = _bs.build(db, "Test")
_m2 = _re.search(r'<script id="data" type="application/json">(.*?)</script>',
                 _html2, _re.S)
_data2 = json.loads(_m2.group(1).replace("<\\/", "</"))
check("the gate opens when explicitly enabled",
      _data2["serves_market_data"] is True)
_os.environ.pop("SERVE_MARKET_DATA", None)


# --------------------------------------------------------------------------
print()
print("=" * 62)
print("  PUBLISHABLE DATASET EXPORT (export_dataset.py)")
print("=" * 62)

import shutil as _shutil
import sqlite3 as _sq
from pathlib import Path as _P
import export_dataset as _ed

# Benchmarks and market rows must exist before the export can be shown to
# remove them; the base fixture has neither.
from finmodel.web.loader import compute_benchmarks as _cb
db.upsert_market([{"cik": r["cik"], "price": 100.0, "market_cap": 1e9,
                   "pe": 20.0, "source": "test"}
                  for r in db.screen(limit=500)["rows"]])
_cb(db, verbose=False)

_t2 = _P(tempfile.mkdtemp(prefix="finmodel_export_"))
_src = _t2 / "src.db"
_shutil.copyfile(_P(tmp) / "t.db", _src)
_res = _ed.export(_src, _t2 / "pub.db")

_c = _sq.connect(_t2 / "pub.db")
check("companies survive the export",
      _res["after"]["companies"] > 0, "%d rows" % _res["after"]["companies"])
check("filed history survives", _res["after"]["financial_history"] > 0,
      "%d rows" % _res["after"]["financial_history"])

# The point of the whole exercise: a public repository is redistribution,
# and scraped price data is not ours to redistribute.
check("market_data is emptied", _res["after"]["market_data"] == 0,
      "was %d" % _res["before"]["market_data"])
check("price-derived benchmarks are removed",
      _c.execute("SELECT COUNT(*) FROM sector_benchmarks "
                 "WHERE price_derived = 1").fetchone()[0] == 0)
check("non-price benchmarks are kept",
      _c.execute("SELECT COUNT(*) FROM sector_benchmarks").fetchone()[0] > 0,
      "%d rows" % _res["after"]["sector_benchmarks"])
_c.close()

# And the stripped copy must still drive the app.
_pub = Database("sqlite:///%s" % (_t2 / "pub.db"))
_s = _pub.stats()
check("the exported copy still answers stats", _s["companies"] > 0)
check("it reports nothing priced", (_s["priced"] or 0) == 0)
check("the screener still works on it", _pub.screen(limit=5)["total"] > 0)
_tk = _pub.screen(limit=1)["rows"][0]["ticker"]
check("a company page still resolves",
      bool(_pub.company(_tk) and _pub.company(_tk)["history"]), _tk)
_pub.close()
_shutil.rmtree(_t2, ignore_errors=True)

print("=" * 62)
print("ALL PASSED" if not fails else "FAILURES: %s" % fails)
sys.exit(1 if fails else 0)
