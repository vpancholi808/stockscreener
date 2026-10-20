"""Bulk SEC data access — the whole market, not one ticker at a time.

Fetching 8,000 companies through the per-company API is slow and gets you
throttled. SEC publishes the entire XBRL dataset as one file instead:

  companyfacts.zip   ~1.5 GB compressed, one JSON per filer (all fundamentals)
  submissions.zip    company metadata: ticker, name, SIC industry code

Extracted, companyfacts is well over 15 GB of JSON, so this module NEVER
extracts it. It streams one company out of the zip at a time, which keeps
memory flat regardless of universe size.
"""
from __future__ import annotations

import json
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import requests

BULK_FACTS_URL = "https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip"
TICKERS_URL = "https://www.sec.gov/files/company_tickers_exchange.json"
DEFAULT_DIR = Path.home() / ".finmodel_bulk"

# SIC code ranges -> broad sector. Screening thresholds that make sense for
# software are meaningless for a grocery distributor, so sector is carried
# through and used for peer-relative ranking rather than one global cutoff.
SIC_SECTORS = [
    (100, 999, "Agriculture"),
    (1000, 1499, "Mining"),
    (1500, 1799, "Construction"),
    (2000, 2199, "Food & Beverage"),
    (2200, 2399, "Textiles & Apparel"),
    (2400, 2599, "Wood & Furniture"),
    (2600, 2699, "Paper"),
    (2700, 2799, "Publishing"),
    (2800, 2829, "Chemicals"),
    (2830, 2836, "Pharma & Biotech"),
    (2840, 2899, "Chemicals"),
    (2900, 2999, "Oil Refining"),
    (3000, 3299, "Rubber, Plastics & Stone"),
    (3300, 3399, "Metals"),
    (3400, 3499, "Fabricated Metal"),
    (3500, 3569, "Industrial Machinery"),
    (3570, 3579, "Computer Hardware"),
    (3580, 3599, "Industrial Machinery"),
    (3600, 3659, "Electrical Equipment"),
    (3660, 3669, "Communications Equipment"),
    (3670, 3679, "Semiconductors"),
    (3680, 3699, "Computer Hardware"),
    (3700, 3799, "Transportation Equipment"),
    (3800, 3879, "Instruments"),
    (3840, 3851, "Medical Devices"),
    (3900, 3999, "Misc Manufacturing"),
    (4000, 4499, "Transportation"),
    (4500, 4599, "Airlines"),
    (4600, 4799, "Transportation"),
    (4800, 4899, "Telecom"),
    (4900, 4999, "Utilities"),
    (5000, 5199, "Wholesale"),
    (5200, 5999, "Retail"),
    (6000, 6199, "Banking"),
    (6200, 6299, "Capital Markets"),
    (6300, 6499, "Insurance"),
    (6500, 6599, "Real Estate"),
    (6700, 6799, "Holding & Investment"),
    (7200, 7299, "Consumer Services"),
    (7370, 7379, "Software & IT Services"),
    (7380, 7399, "Business Services"),
    (7800, 7999, "Entertainment"),
    (8000, 8099, "Healthcare Services"),
    (8200, 8299, "Education"),
    (8700, 8799, "Professional Services"),
]


def sector_for_sic(sic) -> str:
    try:
        s = int(sic)
    except (TypeError, ValueError):
        return "Unknown"
    for lo, hi, name in SIC_SECTORS:
        if lo <= s <= hi:
            return name
    return "Other"


@dataclass
class BulkPaths:
    directory: Path

    @property
    def facts(self) -> Path:
        return self.directory / "companyfacts.zip"

    @property
    def tickers(self) -> Path:
        return self.directory / "company_tickers_exchange.json"


class BulkData:
    """Downloads and streams SEC's bulk XBRL dataset."""

    def __init__(self, directory: Optional[str] = None, user_agent: Optional[str] = None):
        self.paths = BulkPaths(Path(directory) if directory else DEFAULT_DIR)
        self.paths.directory.mkdir(parents=True, exist_ok=True)
        ua = user_agent or os.environ.get("EDGAR_USER_AGENT", "")
        if "@" not in ua:
            raise ValueError(
                "SEC requires a real User-Agent with an email. Pass user_agent="
                "'Your Name you@email.com' or set EDGAR_USER_AGENT.")
        self.headers = {"User-Agent": ua, "Accept-Encoding": "gzip, deflate"}

    # ---- download -------------------------------------------------------
    def _download(self, url: str, dest: Path, force: bool = False) -> Path:
        if dest.exists() and not force and dest.stat().st_size > 0:
            mb = dest.stat().st_size / 1e6
            print("  using cached %s (%.0f MB)" % (dest.name, mb))
            return dest
        print("  downloading %s ..." % dest.name)
        tmp = dest.with_suffix(dest.suffix + ".part")
        with requests.get(url, headers=self.headers, stream=True, timeout=120) as r:
            r.raise_for_status()
            total = int(r.headers.get("Content-Length", 0))
            done = 0
            with open(tmp, "wb") as fh:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
                    done += len(chunk)
                    if total:
                        pct = 100.0 * done / total
                        print("\r    %.0f%%  (%.0f / %.0f MB)"
                              % (pct, done / 1e6, total / 1e6), end="", flush=True)
            print()
        tmp.replace(dest)
        return dest

    def ensure(self, force: bool = False) -> None:
        """Download the bulk files if not already present (~1.5 GB, once)."""
        self._download(TICKERS_URL, self.paths.tickers, force)
        self._download(BULK_FACTS_URL, self.paths.facts, force)

    # ---- metadata -------------------------------------------------------
    def ticker_map(self) -> dict:
        """CIK -> {ticker, name, exchange}. Only listed companies."""
        with open(self.paths.tickers) as fh:
            raw = json.load(fh)
        fields = raw["data"]["fields"] if "data" in raw else raw["fields"]
        rows = raw["data"]["data"] if "data" in raw else raw["data"]
        ix = {f: i for i, f in enumerate(fields)}
        out = {}
        for row in rows:
            cik = int(row[ix["cik"]])
            # keep the first (primary) listing per CIK
            if cik not in out:
                out[cik] = {"ticker": row[ix["ticker"]],
                            "name": row[ix["name"]],
                            "exchange": row[ix.get("exchange", ix["ticker"])]}
        return out

    # ---- streaming ------------------------------------------------------
    def iter_companies(self, limit: Optional[int] = None,
                       only_ciks: Optional[set] = None) -> Iterator[tuple]:
        """Yield (cik, facts_dict) one at a time. Memory stays flat.

        The zip is never extracted; each member is decompressed, parsed, and
        released before the next is read."""
        with zipfile.ZipFile(self.paths.facts) as zf:
            names = [n for n in zf.namelist() if n.endswith(".json")]
            count = 0
            for name in names:
                base = Path(name).stem            # "CIK0000320193"
                if not base.upper().startswith("CIK"):
                    continue
                try:
                    cik = int(base[3:])
                except ValueError:
                    continue
                if only_ciks is not None and cik not in only_ciks:
                    continue
                try:
                    with zf.open(name) as fh:
                        facts = json.load(fh)
                except (json.JSONDecodeError, KeyError, OSError):
                    continue
                yield cik, facts
                count += 1
                if limit and count >= limit:
                    return

    def company_count(self) -> int:
        with zipfile.ZipFile(self.paths.facts) as zf:
            return sum(1 for n in zf.namelist() if n.endswith(".json"))
