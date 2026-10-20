"""SEC EDGAR data client.

Pulls fundamentals from the free SEC EDGAR XBRL API:
  - ticker -> CIK map:   https://www.sec.gov/files/company_tickers.json
  - company facts:       https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json

SEC requires a descriptive User-Agent (your name + email). Requests without one
are throttled or blocked. Set it via EdgarClient(user_agent="You you@email.com")
or the EDGAR_USER_AGENT environment variable. SEC asks for <=10 requests/second.
"""
from __future__ import annotations

import json
import os
import time
from typing import Optional

import requests

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
DEFAULT_UA = os.environ.get("EDGAR_USER_AGENT", "finmodel-toolkit example@example.com")


class EdgarError(RuntimeError):
    pass


class EdgarClient:
    """Thin, polite wrapper over the SEC EDGAR JSON API."""

    def __init__(self, user_agent: str = DEFAULT_UA, min_interval: float = 0.15):
        if "@" not in user_agent:
            # Not fatal, but SEC strongly prefers a contact email.
            print("WARNING: set a real User-Agent with your email "
                  "(EdgarClient(user_agent='Name you@email.com')).")
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent,
                                     "Accept-Encoding": "gzip, deflate"})
        self.min_interval = min_interval
        self._last_call = 0.0
        self._ticker_map: Optional[dict] = None

    def _throttle(self) -> None:
        elapsed = time.time() - self._last_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_call = time.time()

    def _get(self, url: str) -> dict:
        self._throttle()
        resp = self.session.get(url, timeout=30)
        if resp.status_code != 200:
            raise EdgarError(f"GET {url} -> HTTP {resp.status_code}")
        return resp.json()

    def ticker_to_cik(self, ticker: str) -> int:
        """Resolve a ticker symbol to its 10-digit CIK (returned as int)."""
        if self._ticker_map is None:
            data = self._get(TICKERS_URL)
            # Payload is {"0": {"cik_str":..,"ticker":"AAPL",...}, ...}
            self._ticker_map = {row["ticker"].upper(): int(row["cik_str"])
                                for row in data.values()}
        cik = self._ticker_map.get(ticker.upper())
        if cik is None:
            raise EdgarError(f"Ticker '{ticker}' not found in SEC ticker map.")
        return cik

    def company_facts(self, ticker_or_cik: str | int) -> dict:
        """Fetch the raw companyfacts JSON for a ticker or CIK."""
        cik = (ticker_or_cik if isinstance(ticker_or_cik, int)
               else self.ticker_to_cik(ticker_or_cik))
        return self._get(FACTS_URL.format(cik=cik))

    @staticmethod
    def load_facts_from_file(path: str) -> dict:
        """Load a previously saved companyfacts JSON (useful offline)."""
        with open(path) as fh:
            return json.load(fh)
