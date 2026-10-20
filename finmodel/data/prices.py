"""Price feeds for market-based calculations (beta, returns, market cap).

The rest of the toolkit talks to a `PriceFeed` interface, never to a specific
vendor. `YFinancePriceFeed` is the default (free); `FramePriceFeed` lets you
inject prices from a DataFrame/CSV for testing or to swap in a paid vendor
without touching downstream code.

All feeds return a total-return-adjusted close price Series indexed by date.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Optional, Protocol

import pandas as pd


class PriceFeed(Protocol):
    def history(self, ticker: str, start: Optional[str] = None,
                end: Optional[str] = None) -> pd.Series:
        """Adjusted close prices indexed by a DatetimeIndex."""
        ...


class YFinancePriceFeed:
    """Default free feed. yfinance is imported lazily so the package imports
    fine without it installed and offline tests never touch the network."""

    def history(self, ticker: str, start: Optional[str] = None,
                end: Optional[str] = None) -> pd.Series:
        import yfinance as yf
        df = yf.download(ticker, start=start, end=end, auto_adjust=True,
                         progress=False)
        if df is None or df.empty:
            raise ValueError(f"No price data returned for '{ticker}'.")
        close = df["Close"]
        # Recent yfinance can return a single-column DataFrame for one ticker.
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close.name = ticker
        return close.dropna()


class FramePriceFeed:
    """Serve prices from in-memory Series/DataFrames (testing or custom vendor)."""

    def __init__(self, frames: dict[str, pd.Series]):
        self._frames = {k: v.dropna() for k, v in frames.items()}

    def history(self, ticker: str, start: Optional[str] = None,
                end: Optional[str] = None) -> pd.Series:
        if ticker not in self._frames:
            raise ValueError(f"No frame loaded for '{ticker}'.")
        s = self._frames[ticker]
        if start:
            s = s[s.index >= pd.Timestamp(start)]
        if end:
            s = s[s.index <= pd.Timestamp(end)]
        return s


_FREQ = {"D": ("D", 252), "W": ("W-FRI", 52), "M": ("ME", 12)}


def periods_per_year(frequency: str) -> int:
    return _FREQ[frequency][1]


def to_returns(prices: pd.Series, frequency: str = "M") -> pd.Series:
    """Resample to the chosen frequency and return simple periodic returns."""
    if frequency not in _FREQ:
        raise ValueError(f"frequency must be one of {list(_FREQ)}")
    rule = _FREQ[frequency][0]
    px = prices.sort_index()
    if frequency != "D":
        px = px.resample(rule).last()
    return px.pct_change().dropna()


def default_window(lookback_years: int) -> tuple[str, str]:
    end = date.today()
    start = end - timedelta(days=int(lookback_years * 365.25) + 10)
    return start.isoformat(), end.isoformat()
