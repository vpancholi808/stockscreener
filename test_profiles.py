"""Yahoo profiles and partial market-data writes.

Two real defects are pinned here.

The first is a unit trap. Yahoo reports `dividendYield` as a percent in
current versions (0.32 meaning 0.32%) and reported it as a fraction in older
ones. Picking either convention and hard-coding it means the app eventually
tells someone Apple yields 32%.

The second is silent data loss. Prices and profile fields arrive in separate
passes, and the market-data write used to store every column every time —
so whichever pass ran second blanked the other's work with None. The row
still existed and the screener still rendered; the numbers were just gone.
"""
import os
import sys
import tempfile
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

from finmodel.data.profiles import (FRACTION_BOUNDS, MAX_REASONABLE, Profile,
                                    _dividend_yield, compare_sectors,
                                    fetch_profiles, profile_from_info)

fails = []


def check(name, cond, detail=""):
    print("  [%s] %s%s" % ("OK" if cond else "FAIL", name,
                           (" — " + detail) if detail else ""))
    if not cond:
        fails.append(name)


print("=" * 66)
print("1) DIVIDEND YIELD: THE SCALE IS DERIVED, NOT ASSUMED")
print("=" * 66)

# Apple as the probe actually returned it: 0.32 means 0.32%, and the
# dividendRate over the price proves it.
apple = {"dividendYield": 0.32, "dividendRate": 1.04, "currentPrice": 328.0}
got = _dividend_yield(apple)
check("a percent-scaled yield is converted to a fraction",
      abs(got - 0.0032) < 1e-4, "%.5f (0.32%% not 32%%)" % got)

# The older convention, same company.
old = {"dividendYield": 0.0032, "dividendRate": 1.04, "currentPrice": 328.0}
got = _dividend_yield(old)
check("a fraction-scaled yield is left alone",
      abs(got - 0.0032) < 1e-4, "%.5f" % got)

# A genuinely high yielder must not be mangled either way.
high = {"dividendYield": 7.5, "dividendRate": 3.0, "currentPrice": 40.0}
got = _dividend_yield(high)
check("a 7.5% yielder resolves to 0.075", abs(got - 0.075) < 1e-3,
      "%.4f" % got)

check("with no reported yield, rate/price is used",
      abs(_dividend_yield({"dividendRate": 2.0, "currentPrice": 50.0})
          - 0.04) < 1e-9)
check("with no price, an implausible value is treated as a percent",
      abs(_dividend_yield({"dividendYield": 3.1}) - 0.031) < 1e-9)
check("with no price, a plausible fraction is kept",
      abs(_dividend_yield({"dividendYield": 0.031}) - 0.031) < 1e-9)
check("a non-payer yields None", _dividend_yield({}) is None)
check("previousClose is accepted when currentPrice is absent",
      abs(_dividend_yield({"dividendYield": 2.0, "dividendRate": 1.0,
                           "previousClose": 50.0}) - 0.02) < 1e-9)

print()
print("=" * 66)
print("2) IMPLAUSIBLE VALUES ARE DROPPED, NOT STORED")
print("=" * 66)

info = {
    "sector": "Technology", "industry": "Consumer Electronics",
    "sharesOutstanding": 14_594_180_000, "marketCap": 4.975e12,
    "trailingPE": 39.09, "forwardPE": 35.62, "pegRatio": 2.1,
    "trailingEps": 8.4, "forwardEps": 9.2, "payoutRatio": 0.124,
    "beta": 1.069, "targetMeanPrice": 328.35, "numberOfAnalystOpinions": 39,
    "fiftyTwoWeekHigh": 348.15, "fiftyTwoWeekLow": 243.42,
    "profitMargins": 0.247, "returnOnEquity": 1.5,
    "revenueGrowth": 0.08, "earningsGrowth": 0.12,
    "dividendYield": 0.32, "dividendRate": 1.04, "currentPrice": 328.0,
}
p = profile_from_info("AAPL", info)
check("sector and industry are carried through",
      p.yahoo_sector == "Technology" and p.yahoo_industry == "Consumer Electronics")
check("forward P/E is populated — SEC cannot supply this",
      abs(p.forward_pe - 35.62) < 1e-9, "%.2f" % p.forward_pe)
check("analyst count is an int", isinstance(p.analyst_count, int))
check("the 52-week range is captured",
      p.week52_high > p.week52_low > 0)
check("dividend yield is a fraction",
      abs(p.dividend_yield - 0.0032) < 1e-4, "%.5f" % p.dividend_yield)

negative = dict(info, forwardPE=-12.0, trailingPE=-8.0, pegRatio=-1.0)
q = profile_from_info("X", negative)
check("a negative forward P/E becomes None, not 'cheap'", q.forward_pe is None)
check("a negative trailing P/E becomes None", q.yahoo_trailing_pe is None)
check("a negative PEG becomes None", q.peg is None)

absurd = dict(info, profitMargins=45.0, payoutRatio=900.0)
r = profile_from_info("Y", absurd)
check("an out-of-bounds margin is dropped", r.yahoo_profit_margin is None)
check("an out-of-bounds payout ratio is dropped", r.payout_ratio is None)
check("bounds exist for every fraction field",
      all(k in FRACTION_BOUNDS for k in
          ("payout_ratio", "yahoo_profit_margin", "dividend_yield")))

empty = profile_from_info("Z", {})
check("an empty payload yields a Profile of Nones",
      isinstance(empty, Profile) and empty.forward_pe is None)
check("string values do not become floats",
      profile_from_info("Z", {"marketCap": "N/A"}).yahoo_market_cap is None)

print()
print("=" * 66)
print("3) THE PER-TICKER CRAWL REFUSES AN UNBOUNDED UNIVERSE")
print("=" * 66)

try:
    fetch_profiles(["X%d" % i for i in range(MAX_REASONABLE + 1)])
    check("too many tickers raises", False)
except ValueError as exc:
    check("too many tickers raises with the time cost named",
          "minutes" in str(exc), str(exc)[:70])
except ImportError:
    check("too many tickers raises", False, "yfinance missing")

try:
    fetch_profiles([])
    check("an empty list raises", False)
except ValueError:
    check("an empty list raises", True)

print()
print("=" * 66)
print("4) PARTIAL WRITES MUST NOT BLANK THE OTHER PASS")
print("=" * 66)

from finmodel.web.store import Database, MARKET_COLS

tmp = tempfile.mkdtemp(prefix="finmodel_profiles_")
db = Database("sqlite:///%s/t.db" % tmp)
db.init_schema()
db.upsert_companies([{"cik": 1, "ticker": "AAA", "name": "Alpha",
                      "sector": "Technology", "fiscal_year": 2025,
                      "shares_diluted": 1000.0}])

# Pass one: prices and multiples.
db.upsert_market([{"cik": 1, "price": 100.0, "market_cap": 100000.0,
                   "pe": 20.0, "beta": 1.1, "source": "yfinance"}])
# Pass two: profile fields only. This used to erase pass one.
db.upsert_market([{"cik": 1, "forward_pe": 18.0, "analyst_count": 39,
                   "yahoo_sector": "Technology"}])

conn = db.connect()
cur = conn.cursor()
cur.execute("SELECT %s FROM market_data WHERE cik=1" % ",".join(MARKET_COLS))
row = dict(zip(MARKET_COLS, cur.fetchone()))

check("price from the first pass survives the second",
      row["price"] == 100.0, repr(row["price"]))
check("P/E from the first pass survives", row["pe"] == 20.0, repr(row["pe"]))
check("beta from the first pass survives", row["beta"] == 1.1)
check("market cap from the first pass survives", row["market_cap"] == 100000.0)
check("forward P/E from the second pass is stored", row["forward_pe"] == 18.0)
check("analyst count from the second pass is stored", row["analyst_count"] == 39)
check("yahoo sector from the second pass is stored",
      row["yahoo_sector"] == "Technology")

# And the reverse order must behave the same way.
db.upsert_market([{"cik": 1, "price": 111.0}])
cur.execute("SELECT price, forward_pe FROM market_data WHERE cik=1")
price, fpe = cur.fetchone()
check("a later price update preserves the profile fields",
      price == 111.0 and fpe == 18.0, "price=%s forward_pe=%s" % (price, fpe))

check("a row with only cik writes nothing",
      db.upsert_market([{"cik": 1}]) == 0)
check("an empty list writes nothing", db.upsert_market([]) == 0)
db.close()

print()
print("=" * 66)
print("5) SECTOR CROSS-CHECK")
print("=" * 66)

profiles = {"AAA": Profile(ticker="AAA", yahoo_sector="Technology"),
            "BBB": Profile(ticker="BBB", yahoo_sector="Financials"),
            "CCC": Profile(ticker="CCC", yahoo_sector=None)}
diff = compare_sectors(profiles, {"AAA": "Technology", "BBB": "Industrials",
                                  "CCC": "Utilities"})
check("agreement is not reported", "AAA" not in list(diff.get("ticker", [])))
check("disagreement is reported", "BBB" in list(diff["ticker"]))
check("a missing Yahoo sector is not a disagreement",
      "CCC" not in list(diff.get("ticker", [])))

print()
if fails:
    print("FAILED (%d): %s" % (len(fails), "; ".join(fails)))
    sys.exit(1)
print("test_profiles: all checks passed")
