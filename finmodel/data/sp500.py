"""S&P 500 constituents.

The index membership list is not published by the SEC and is not available
through yfinance. The practical public source is the Wikipedia constituents
table, which carries ticker, company name, GICS sector and CIK.

No hardcoded 500-name list ships with this module. A stale membership list is
worse than no list: it silently drops recent additions and keeps companies that
were removed, which quietly biases every backtest and sector benchmark built on
top of it. The list is fetched and cached instead, and a fetch failure is
reported rather than papered over with an out-of-date copy.
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Optional

import requests

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
CACHE = Path.home() / ".finmodel_cache" / "sp500.json"
MAX_AGE_DAYS = 7


class ConstituentError(RuntimeError):
    pass


def _parse_wikipedia(html: str) -> list[dict]:
    """Extract the constituents table. Uses pandas.read_html when lxml is
    available, falling back to a narrow regex parse."""
    try:
        import pandas as pd
        from io import StringIO
        tables = pd.read_html(StringIO(html))
        for t in tables:
            cols = {str(c).strip().lower() for c in t.columns}
            if {"symbol", "security"} <= cols:
                t.columns = [str(c).strip().lower() for c in t.columns]
                sector_col = next(
                    (c for c in t.columns if "sector" in c and "sub" not in c),
                    None)
                sub_col = next((c for c in t.columns if "sub" in c), None)
                cik_col = next((c for c in t.columns if "cik" in c), None)
                rows = []
                for _, r in t.iterrows():
                    sym = str(r["symbol"]).strip().upper().replace(".", "-")
                    if not sym or sym == "NAN":
                        continue
                    rows.append({
                        "ticker": sym,
                        "name": str(r["security"]).strip(),
                        "gics_sector": (str(r[sector_col]).strip()
                                        if sector_col else None),
                        "gics_sub_industry": (str(r[sub_col]).strip()
                                              if sub_col else None),
                        "cik": (int(r[cik_col])
                                if cik_col and str(r[cik_col]).strip().isdigit()
                                else None),
                    })
                if rows:
                    return rows
    except Exception:
        pass

    # Fallback: pull <tr> blocks out of the first wikitable.
    import re
    m = re.search(r'<table[^>]*class="[^"]*wikitable[^"]*"[^>]*>(.*?)</table>',
                  html, re.S)
    if not m:
        raise ConstituentError("could not locate the constituents table")
    rows = []
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", m.group(1), re.S):
        cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", tr, re.S)
        if len(cells) < 4:
            continue
        clean = [re.sub(r"<[^>]+>", "", c).strip() for c in cells]
        sym = clean[0].upper().replace(".", "-")
        if not sym or not re.fullmatch(r"[A-Z\-\.]{1,8}", sym):
            continue
        rows.append({"ticker": sym, "name": clean[1],
                     "gics_sector": clean[2] if len(clean) > 2 else None,
                     "gics_sub_industry": clean[3] if len(clean) > 3 else None,
                     "cik": None})
    if not rows:
        raise ConstituentError("constituents table parsed but produced no rows")
    return rows


def fetch_sp500(user_agent: Optional[str] = None,
                use_cache: bool = True,
                max_age_days: int = MAX_AGE_DAYS) -> list[dict]:
    """Current S&P 500 constituents. Cached locally for `max_age_days`."""
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    if use_cache and CACHE.exists():
        age_days = (time.time() - CACHE.stat().st_mtime) / 86400
        if age_days < max_age_days:
            try:
                data = json.loads(CACHE.read_text())
                if data.get("constituents"):
                    return data["constituents"]
            except Exception:
                pass

    headers = {"User-Agent": user_agent or
               "finmodel/0.1 (contact: set EDGAR_USER_AGENT)"}
    try:
        resp = requests.get(WIKI_URL, headers=headers, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        if CACHE.exists():
            try:
                data = json.loads(CACHE.read_text())
                if data.get("constituents"):
                    print("  WARNING: live fetch failed (%s); using cached list "
                          "from %s" % (e, data.get("fetched_at", "unknown")))
                    return data["constituents"]
            except Exception:
                pass
        raise ConstituentError(
            "could not fetch the S&P 500 list (%s). Supply one with "
            "load_constituents_csv(path) instead." % e)

    rows = _parse_wikipedia(resp.text)
    CACHE.write_text(json.dumps(
        {"fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
         "source": WIKI_URL, "constituents": rows}, indent=1))
    return rows


def load_constituents_csv(path: str) -> list[dict]:
    """Load an index membership list from CSV.

    Requires a `ticker` column; `name`, `gics_sector` and `cik` are optional.
    Use this for any index other than the S&P 500, or to pin membership to a
    specific date for a backtest."""
    out = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            keys = {k.lower().strip(): v for k, v in row.items() if k}
            sym = (keys.get("ticker") or keys.get("symbol") or "").strip().upper()
            if not sym:
                continue
            cik = keys.get("cik", "")
            out.append({
                "ticker": sym.replace(".", "-"),
                "name": (keys.get("name") or keys.get("security") or sym).strip(),
                "gics_sector": (keys.get("gics_sector")
                                or keys.get("sector") or None),
                "gics_sub_industry": keys.get("gics_sub_industry") or None,
                "cik": int(cik) if str(cik).strip().isdigit() else None,
            })
    if not out:
        raise ConstituentError(f"no tickers found in {path}")
    return out


def tickers(constituents: Optional[list[dict]] = None, **kw) -> list[str]:
    return [c["ticker"] for c in (constituents or fetch_sp500(**kw))]
