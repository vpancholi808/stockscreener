"""Yahoo price import — parsing, alignment, caching and the audit screen.

Every check runs offline. `download_prices` is the only function that touches
the network and it is deliberately thin; all the logic that can be wrong
lives in pure functions over DataFrames, which is what is exercised here.
"""
import json
import os
import shutil
import sys
import tempfile
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from finmodel.data.yahoo import (_cache_key, _read_cache, _write_cache,
                                 align_prices, audit_prices, coverage,
                                 drop_empty, normalize_download)

fails = []


def check(name, cond, detail=""):
    print("  [%s] %s%s" % ("OK" if cond else "FAIL", name,
                           (" — " + detail) if detail else ""))
    if not cond:
        fails.append(name)


IDX = pd.date_range("2024-01-01", periods=5, freq="B")

print("=" * 64)
print("1) EVERY yfinance RESPONSE SHAPE REDUCES TO ONE COLUMN PER TICKER")
print("=" * 64)

# Multi-ticker, group_by='column': MultiIndex (field, ticker)
cols = pd.MultiIndex.from_product([["Close", "Volume"], ["AAA", "BBB"]])
raw = pd.DataFrame(np.arange(20.0).reshape(5, 4), index=IDX, columns=cols)
check("MultiIndex (field, ticker)",
      list(normalize_download(raw, ["AAA", "BBB"]).columns) == ["AAA", "BBB"])

# Multi-ticker, group_by='ticker': MultiIndex (ticker, field)
cols = pd.MultiIndex.from_product([["AAA", "BBB"], ["Close", "Volume"]])
raw2 = pd.DataFrame(np.arange(20.0).reshape(5, 4), index=IDX, columns=cols)
check("MultiIndex (ticker, field)",
      list(normalize_download(raw2, ["AAA", "BBB"]).columns) == ["AAA", "BBB"])

# Single ticker: flat columns labelled by field, not by symbol.
flat = pd.DataFrame({"Close": [1.0, 2, 3, 4, 5], "Volume": [1] * 5}, index=IDX)
out = normalize_download(flat, ["AAA"])
check("flat single-ticker frame is relabelled to the symbol",
      list(out.columns) == ["AAA"])
check("flat single-ticker values are preserved",
      out["AAA"].tolist() == [1.0, 2, 3, 4, 5])

check("an empty response still returns the requested columns",
      list(normalize_download(pd.DataFrame(), ["AAA", "BBB"]).columns)
      == ["AAA", "BBB"])

missing = normalize_download(raw, ["AAA", "ZZZ"])
check("a ticker absent from the response becomes an all-NaN column",
      bool(missing["ZZZ"].isna().all()) and bool(missing["AAA"].notna().any()))
check("column order follows the request, not the response",
      list(normalize_download(raw, ["BBB", "AAA"]).columns) == ["BBB", "AAA"])
check("the index is datetime and named Date",
      isinstance(normalize_download(raw, ["AAA"]).index, pd.DatetimeIndex))

print()
print("=" * 64)
print("2) FAILED SYMBOLS ARE REPORTED, NOT SILENTLY DROPPED")
print("=" * 64)

prices = pd.DataFrame({
    "GOOD": [1.0, 2, 3, 4, 5],
    "DEAD": [np.nan] * 5,
    "LATE": [np.nan, np.nan, 3.0, 4, 5],
}, index=IDX)

kept, empty = drop_empty(prices)
check("an all-NaN column is reported as missing", empty == ["DEAD"])
check("partial history is kept, not discarded",
      list(kept.columns) == ["GOOD", "LATE"])

cov = coverage(kept)
check("coverage reports a late listing", cov["observations"]["LATE"] == 3,
      "first %s" % cov["first"]["LATE"].date())
check("coverage reports full history", cov["observations"]["GOOD"] == 5)
check("coverage reports the missing fraction",
      abs(cov["missing_pct"]["LATE"] - 0.4) < 1e-9)

print()
print("=" * 64)
print("3) ALIGNMENT")
print("=" * 64)

common, notes = align_prices(kept, how="common")
check("common alignment keeps only dates every ticker has",
      len(common) == 3, "%d rows" % len(common))
check("alignment says how many dates it dropped",
      any("dropped 2" in n for n in notes), "; ".join(notes))

gappy = pd.DataFrame({"A": [1.0, np.nan, 3.0, 4, 5],
                      "B": [1.0, 2, 3, 4, 5]}, index=IDX)
filled, notes = align_prices(gappy, how="ffill", max_ffill=5)
check("ffill bridges a one-day hole", len(filled) == 5)
check("ffill reports how many prices it invented",
      any("forward-filled 1" in n for n in notes), "; ".join(notes))

long_gap = pd.DataFrame({"A": [1.0] + [np.nan] * 3 + [5.0],
                         "B": [1.0, 2, 3, 4, 5]}, index=IDX)
limited, _ = align_prices(long_gap, how="ffill", max_ffill=2)
# limit=2 carries the price two days and then stops; the still-missing day is
# dropped rather than invented. The gap is therefore never fully bridged.
check("ffill stops at the limit and drops the rest of a long gap",
      len(limited) == 4 and pd.Timestamp("2024-01-04") not in limited.index,
      "%d of 5 rows kept" % len(limited))
check("how='none' leaves the frame untouched",
      len(align_prices(gappy, how="none")[0]) == 5)

try:
    align_prices(gappy, how="nonsense")
    check("an unknown alignment mode raises", False)
except ValueError:
    check("an unknown alignment mode raises", True)

print()
print("=" * 64)
print("4) THE AUDIT FINDS PLANTED ERRORS AND LEAVES CLEAN DATA ALONE")
print("=" * 64)

rng = np.random.default_rng(0)
n = 300
dates = pd.date_range("2024-01-01", periods=n, freq="B")
clean = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.010, n)))

bad = clean.copy()
bad[150] *= 1.40                       # a spike undone the next day
stale = clean.copy()
stale[60:70] = stale[60]               # ten identical closes
crash = clean.copy()
crash[200:] *= 0.70                    # a real 30% fall that does NOT revert

panel = pd.DataFrame({"CLEAN": clean, "BADPRINT": bad, "STALE": stale,
                      "CRASH": crash}, index=dates)
flags = audit_prices(panel)
by_ticker = flags.groupby("ticker")["flags"].apply(lambda s: ",".join(s))

check("a clean series is not flagged at all",
      "CLEAN" not in by_ticker.index,
      "%d flags" % int((flags["ticker"] == "CLEAN").sum()))
check("a spike-and-revert is flagged as a reversal",
      "reversal" in by_ticker.get("BADPRINT", ""))
check("the spike is also flagged on size", "jump" in by_ticker.get("BADPRINT", ""))
check("a stale run is flagged", "stale" in by_ticker.get("STALE", ""))
check("a genuine one-way fall is NOT called a reversal",
      "reversal" not in by_ticker.get("CRASH", ""),
      "flags: %s" % by_ticker.get("CRASH", "none"))

negative = pd.DataFrame({"X": [10.0, 5.0, -1.0, 4.0, 5.0]}, index=IDX)
check("a non-positive price is flagged",
      "nonpositive" in ",".join(audit_prices(negative)["flags"]))

check("an empty frame audits to an empty report",
      audit_prices(pd.DataFrame()).empty)
check("a two-row frame audits without raising",
      audit_prices(panel.iloc[:2]).empty)

print()
print("=" * 64)
print("5) THE CACHE RECORDS ITS VINTAGE AND EXPIRES")
print("=" * 64)

tmp = Path(tempfile.mkdtemp(prefix="finmodel_cache_"))
try:
    frame = pd.DataFrame({"AAA": [1.0, 2, 3, 4, 5]}, index=IDX)
    now = datetime.now(timezone.utc)
    path = tmp / "x.csv"
    _write_cache(path, frame, {"vintage": now.isoformat(timespec="seconds")})

    got, meta = _read_cache(path, max_age_hours=24)
    check("a fresh cache entry is returned", got is not None)
    check("the cached frame round-trips",
          got is not None and np.allclose(got["AAA"], frame["AAA"]))
    check("the vintage is recorded", "vintage" in (meta or {}))

    got, _ = _read_cache(path, max_age_hours=0)
    check("a cache older than max_age is refused", got is None)

    stale_meta = {"vintage": (now - timedelta(days=30)).isoformat(
        timespec="seconds")}
    _write_cache(path, frame, stale_meta)
    got, _ = _read_cache(path, max_age_hours=24)
    check("a month-old entry is refused at a 24-hour limit", got is None)
    got, _ = _read_cache(path, max_age_hours=None)
    check("max_age_hours=None accepts any vintage", got is not None)

    path.with_suffix(".json").write_text("not json")
    got, _ = _read_cache(path, max_age_hours=24)
    check("a corrupt sidecar is treated as a cache miss, not an error",
          got is None)

    check("a missing cache entry is a miss",
          _read_cache(tmp / "nope.csv", 24)[0] is None)

    a = _cache_key(["AAA", "BBB"], "2024-01-01", None, "1d", True, "Close")
    b = _cache_key(["BBB", "AAA"], "2024-01-01", None, "1d", True, "Close")
    c = _cache_key(["AAA", "BBB"], "2024-01-01", None, "1d", False, "Close")
    check("the cache key ignores ticker order", a == b)
    check("the cache key separates auto_adjust settings", a != c,
          "adjusted and unadjusted prices must not share an entry")
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print()
if fails:
    print("FAILED (%d): %s" % (len(fails), "; ".join(fails)))
    sys.exit(1)
print("test_yahoo: all checks passed")
