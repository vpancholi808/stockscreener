"""Batched market data for thousands of tickers.

Two things make naive price fetching unusable at market scale:

  1. yfinance's Ticker.info is one HTTP round trip per ticker (~1-2s).
     For 8,000 names that is several hours. This module never calls it.
  2. Fetching prices one ticker at a time has the same problem.

Instead, prices come from yf.download() in batches of a few hundred symbols
(one request per batch), and market cap is computed as

    market cap = latest close x diluted shares outstanding

where the share count comes from the SEC filings already parsed. That removes
the slow path entirely and keeps everything consistent with the fundamentals.
"""
from __future__ import annotations

import time
from typing import Iterable, Optional

import pandas as pd


def _chunks(seq, n):
    seq = list(seq)
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def batch_last_prices(tickers: Iterable[str], batch_size: int = 200,
                      period: str = "5d", pause: float = 0.6,
                      verbose: bool = True) -> dict:
    """Latest close for many tickers. One request per batch, not per ticker."""
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("pip install yfinance")

    out, tickers = {}, [t for t in tickers if t]
    batches = list(_chunks(tickers, batch_size))
    for i, batch in enumerate(batches, 1):
        if verbose:
            print("\r  prices: batch %d/%d (%d symbols)"
                  % (i, len(batches), len(batch)), end="", flush=True)
        try:
            df = yf.download(" ".join(batch), period=period, interval="1d",
                             auto_adjust=True, progress=False,
                             group_by="column", threads=True)
        except Exception:
            time.sleep(pause * 3)
            continue
        if df is None or len(df) == 0:
            continue
        try:
            close = df["Close"] if "Close" in df else df
            if isinstance(close, pd.Series):          # single ticker came back flat
                close = close.to_frame(batch[0])
            for sym in close.columns:
                s = close[sym].dropna()
                if len(s):
                    out[str(sym)] = float(s.iloc[-1])
        except (KeyError, IndexError, TypeError):
            continue
        time.sleep(pause)
    if verbose:
        print("\r  prices: %d/%d symbols resolved%s"
              % (len(out), len(tickers), " " * 20))
    return out


def batch_returns(tickers: Iterable[str], period: str = "2y",
                  interval: str = "1mo", batch_size: int = 200,
                  pause: float = 0.6, verbose: bool = True) -> pd.DataFrame:
    """Return history for many tickers (for beta / momentum / risk work)."""
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("pip install yfinance")

    frames, batches = [], list(_chunks([t for t in tickers if t], batch_size))
    for i, batch in enumerate(batches, 1):
        if verbose:
            print("\r  returns: batch %d/%d" % (i, len(batches)), end="", flush=True)
        try:
            df = yf.download(" ".join(batch), period=period, interval=interval,
                             auto_adjust=True, progress=False, threads=True)
        except Exception:
            continue
        if df is None or len(df) == 0:
            continue
        close = df["Close"] if "Close" in df else df
        if isinstance(close, pd.Series):
            close = close.to_frame(batch[0])
        frames.append(close)
        time.sleep(pause)
    if verbose:
        print()
    if not frames:
        return pd.DataFrame()
    prices = pd.concat(frames, axis=1)
    return prices.pct_change().dropna(how="all")


def market_caps(prices: dict, shares: dict) -> dict:
    """market cap = price x diluted shares (shares come from SEC filings).

    `shares` values are share COUNTS as filed (already absolute, not millions)."""
    caps = {}
    for t, px in prices.items():
        sh = shares.get(t)
        if sh and px and sh > 0:
            caps[t] = px * sh
    return caps


# ---------------------------------------------------------------------------
# Multi-year price history -> risk and momentum statistics
# ---------------------------------------------------------------------------

def batch_price_history(tickers: Iterable[str], period: str = "5y",
                        interval: str = "1mo", batch_size: int = 150,
                        pause: float = 0.8, verbose: bool = True):
    """Adjusted close history for many tickers, in batches.

    `auto_adjust=True` gives total-return series (dividends and splits folded
    in), which is what any return or volatility figure should be built on —
    raw close prices understate the return of anything that pays a dividend."""
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("pip install yfinance")
    import pandas as pd

    frames, batches = [], list(_chunks([t for t in tickers if t], batch_size))
    for i, batch in enumerate(batches, 1):
        if verbose:
            print("\r  history: batch %d/%d (%d symbols)"
                  % (i, len(batches), len(batch)), end="", flush=True)
        try:
            df = yf.download(" ".join(batch), period=period, interval=interval,
                             auto_adjust=True, progress=False, threads=True)
        except Exception:
            time.sleep(pause * 3)
            continue
        if df is None or len(df) == 0:
            continue
        close = df["Close"] if "Close" in df else df
        if isinstance(close, pd.Series):
            close = close.to_frame(batch[0])
        frames.append(close)
        time.sleep(pause)
    if verbose:
        print()
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1)


def price_statistics(prices, benchmark: str = "^GSPC",
                     periods_per_year: int = 12) -> dict:
    """Per-ticker risk and momentum figures from a price history panel.

    Returns ticker -> {volatility, beta, return_1y, return_total,
    max_drawdown, observations}. Beta is the OLS slope against the benchmark
    column when it is present in the panel."""
    import numpy as np
    import pandas as pd

    if prices is None or len(prices) == 0:
        return {}
    prices = prices.sort_index()
    rets = prices.pct_change().dropna(how="all")
    bench = rets[benchmark] if benchmark in rets.columns else None

    out: dict[str, dict] = {}
    for col in prices.columns:
        s = prices[col].dropna()
        r = rets[col].dropna()
        if len(r) < 6:
            continue
        vol = float(r.std(ddof=1) * np.sqrt(periods_per_year))

        beta = None
        if bench is not None and col != benchmark:
            pair = pd.concat([r, bench], axis=1, join="inner").dropna()
            if len(pair) >= 12:
                x = pair.iloc[:, 1].to_numpy()
                y = pair.iloc[:, 0].to_numpy()
                var = float(((x - x.mean()) ** 2).sum())
                if var > 0:
                    raw = float(((x - x.mean()) * (y - y.mean())).sum() / var)
                    # Blume adjustment, matching the CAPM layer.
                    beta = 0.67 * raw + 0.33

        one_year = None
        if len(s) > periods_per_year:
            prev = float(s.iloc[-(periods_per_year + 1)])
            if prev > 0:
                one_year = float(s.iloc[-1]) / prev - 1.0

        total = None
        if float(s.iloc[0]) > 0:
            total = float(s.iloc[-1]) / float(s.iloc[0]) - 1.0

        curve = (1 + r).cumprod()
        dd = float((curve / curve.cummax() - 1).min())

        out[str(col)] = {
            "volatility": vol, "beta": beta, "return_1y": one_year,
            "return_total": total, "max_drawdown": dd,
            "observations": int(len(r)),
            "last_price": float(s.iloc[-1]),
        }
    return out
