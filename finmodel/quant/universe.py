"""Multi-asset return panel — the shared data layer for portfolio, risk, and
factor models.

Wraps the existing PriceFeed to pull a universe of tickers, align them on common
dates, and expose a returns matrix plus convenience statistics. Everything in
finmodel.quant consumes a Universe rather than touching a vendor directly, so
the same code works against yfinance, a CSV, or a paid feed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from ..data.prices import (PriceFeed, YFinancePriceFeed, default_window,
                           periods_per_year, to_returns)


@dataclass
class Universe:
    prices: pd.DataFrame          # columns = tickers, index = dates (aligned)
    returns: pd.DataFrame         # periodic simple returns
    frequency: str

    @property
    def tickers(self) -> list[str]:
        return list(self.returns.columns)

    @property
    def periods_per_year(self) -> int:
        return periods_per_year(self.frequency)

    # ---- constructors -----------------------------------------------------
    @classmethod
    def from_tickers(cls, tickers: Sequence[str], feed: Optional[PriceFeed] = None,
                     frequency: str = "M", lookback_years: int = 5,
                     start: Optional[str] = None, end: Optional[str] = None) -> "Universe":
        feed = feed or YFinancePriceFeed()
        if start is None or end is None:
            start, end = default_window(lookback_years)
        series = {}
        for t in tickers:
            try:
                series[t] = feed.history(t, start, end)
            except Exception as e:  # keep going; report missing names
                print(f"WARNING: skipping {t}: {e}")
        if not series:
            raise ValueError("No price series could be loaded.")
        prices = pd.DataFrame(series).sort_index()
        rets = pd.concat({t: to_returns(prices[t].dropna(), frequency)
                          for t in prices.columns}, axis=1)
        rets = rets.dropna(how="any")
        return cls(prices=prices, returns=rets, frequency=frequency)

    @classmethod
    def from_prices(cls, prices: pd.DataFrame, frequency: str = "M") -> "Universe":
        prices = prices.sort_index()
        rets = pd.concat({t: to_returns(prices[t].dropna(), frequency)
                          for t in prices.columns}, axis=1).dropna(how="any")
        return cls(prices=prices, returns=rets, frequency=frequency)

    @classmethod
    def from_returns(cls, returns: pd.DataFrame, frequency: str = "M") -> "Universe":
        return cls(prices=pd.DataFrame(), returns=returns.dropna(how="any"),
                   frequency=frequency)

    # ---- statistics -------------------------------------------------------
    def mean_returns(self, annualized: bool = True) -> pd.Series:
        mu = self.returns.mean()
        return mu * self.periods_per_year if annualized else mu

    def cov_matrix(self, annualized: bool = True) -> pd.DataFrame:
        cov = self.returns.cov()
        return cov * self.periods_per_year if annualized else cov

    def corr_matrix(self) -> pd.DataFrame:
        return self.returns.corr()

    def volatility(self, annualized: bool = True) -> pd.Series:
        vol = self.returns.std()
        return vol * np.sqrt(self.periods_per_year) if annualized else vol
