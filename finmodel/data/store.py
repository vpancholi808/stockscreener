"""Local caching layer for SEC data.

Every companyfacts pull is cached on disk keyed by CIK, so repeat runs are
instant, reproducible, and work offline. A pluggable client means the store can
be tested without network access (inject a fake fetcher) and later pointed at a
different data source without touching anything downstream.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from .edgar import EdgarClient

DEFAULT_CACHE = Path.home() / ".finmodel_cache"


class DataStore:
    def __init__(self, cache_dir: Optional[str] = None, client=None,
                 ttl_days: float = 1.0):
        self.cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.client = client or EdgarClient()
        self.ttl_seconds = ttl_days * 86400

    def _path(self, cik: int) -> Path:
        return self.cache_dir / f"CIK{int(cik):010d}.json"

    def _resolve_cik(self, ticker_or_cik) -> int:
        return (ticker_or_cik if isinstance(ticker_or_cik, int)
                else self.client.ticker_to_cik(ticker_or_cik))

    def get_facts(self, ticker_or_cik, force_refresh: bool = False) -> dict:
        """Return companyfacts, using the disk cache when fresh."""
        cik = self._resolve_cik(ticker_or_cik)
        path = self._path(cik)
        if not force_refresh and path.exists():
            if (time.time() - path.stat().st_mtime) < self.ttl_seconds:
                with open(path) as fh:
                    return json.load(fh)
        facts = self.client.company_facts(cik)
        with open(path, "w") as fh:
            json.dump(facts, fh)
        return facts

    def is_cached(self, ticker_or_cik) -> bool:
        return self._path(self._resolve_cik(ticker_or_cik)).exists()

    def cached_ciks(self) -> list[int]:
        return sorted(int(p.stem.replace("CIK", ""))
                      for p in self.cache_dir.glob("CIK*.json"))

    def clear(self) -> int:
        n = 0
        for p in self.cache_dir.glob("CIK*.json"):
            p.unlink(); n += 1
        return n
