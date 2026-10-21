"""Hardened price import from Yahoo Finance.

`market.py` fetches prices at market scale (thousands of tickers, latest close
and summary statistics). This module is for the other job: pulling a clean,
auditable price *history* for a named set of tickers, which is what return
statistics and performance measurement need.

Four things go wrong with a naive `yf.download()` and all four are handled here:

  1. **The shape of the result changes with the request.** One ticker comes
     back with flat columns, several come back with a MultiIndex, and the
     nesting order depends on `group_by`. Code that indexes `raw["Close"]`
     works until the day someone passes a single ticker.

  2. **Failures are silent.** A delisted or misspelled symbol does not raise;
     the column is simply absent, or present and entirely NaN. A screen then
     runs on four tickers while reporting five.

  3. **Caches go stale invisibly.** Yahoo restates history — splits and
     dividend adjustments are applied retroactively — so the same request on
     two dates returns different numbers. A cache keyed only on the date range
     will serve last month's vintage forever without saying so. Every cache
     entry here records when it was downloaded, and `max_age_hours` controls
     when it is refetched.

  4. **Bad prints survive into the statistics.** A missed dividend adjustment
     looks exactly like a 15% single-day move. It barely moves the mean, but
     it lands in the fourth moment with its value raised to the fourth power
     and can dominate the kurtosis of the whole sample. `audit_prices` finds
     these before they reach a model; `kurtosis_concentration` in returns.py
     measures what they did if they got through.

Only `download_prices` touches the network. Everything else is a pure function
over a DataFrame, so the parsing, alignment and audit logic is tested offline.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

DEFAULT_CACHE = Path(".finmodel_prices")

# A daily move this large is not impossible, but it is rare enough that it
# should be looked at rather than averaged in. March 2020 produced genuine
# moves above 15% in large caps; so does a missed dividend adjustment.
JUMP_THRESHOLD = 0.25
SIGMA_THRESHOLD = 10.0
STALE_RUN = 5


# --------------------------------------------------------------------------
# results
# --------------------------------------------------------------------------
@dataclass
class PriceDownload:
    """Prices plus everything needed to judge whether to trust them."""
    prices: pd.DataFrame
    requested: list[str]
    missing: list[str] = field(default_factory=list)
    vintage: str = ""
    source: str = "yahoo"
    from_cache: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def resolved(self) -> list[str]:
        return list(self.prices.columns)

    def summary(self) -> str:
        lines = [
            "Prices: %d rows x %d tickers" % self.prices.shape,
            "  range     %s to %s" % (
                self.prices.index.min().date() if len(self.prices) else "-",
                self.prices.index.max().date() if len(self.prices) else "-"),
            "  vintage   %s%s" % (self.vintage,
                                  " (from cache)" if self.from_cache else ""),
        ]
        if self.missing:
            lines.append("  MISSING   %s" % ", ".join(self.missing))
        for note in self.notes:
            lines.append("  note      %s" % note)
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


# --------------------------------------------------------------------------
# pure: shape normalisation
# --------------------------------------------------------------------------
def normalize_download(raw: pd.DataFrame, tickers: Iterable[str],
                       field_name: str = "Close") -> pd.DataFrame:
    """Reduce any yfinance response shape to one column per ticker.

    yfinance returns flat columns for a single ticker and a MultiIndex for
    several, and either level of that MultiIndex may hold the field name
    depending on `group_by`. Rather than assume a shape, find the one that
    actually came back.
    """
    tickers = [str(t) for t in tickers]
    if raw is None or len(raw) == 0:
        return pd.DataFrame(index=pd.DatetimeIndex([], name="Date"),
                            columns=tickers, dtype=float)

    if isinstance(raw.columns, pd.MultiIndex):
        frame = None
        for level in range(raw.columns.nlevels):
            values = raw.columns.get_level_values(level)
            if field_name in set(values):
                frame = raw.xs(field_name, axis=1, level=level)
                break
        if frame is None:                      # no such field at any level
            return pd.DataFrame(index=raw.index, columns=tickers, dtype=float)
    elif field_name in raw.columns:
        frame = raw[[field_name]]
        # A single-ticker request labels the column by field, not by symbol.
        if len(tickers) == 1:
            frame = frame.set_axis(tickers, axis="columns")
    else:
        frame = raw

    if isinstance(frame, pd.Series):
        frame = frame.to_frame(tickers[0] if len(tickers) == 1 else frame.name)

    frame = frame.reindex(columns=tickers)
    frame.index = pd.to_datetime(frame.index)
    frame.index.name = "Date"
    return frame.astype(float)


def drop_empty(prices: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Separate tickers that returned nothing from those that returned data.

    An all-NaN column is a failed symbol, not a quiet stock. Returning it
    silently is how a five-ticker screen becomes a four-ticker screen without
    anyone noticing.
    """
    empty = [c for c in prices.columns if prices[c].notna().sum() == 0]
    return prices.drop(columns=empty), empty


def align_prices(prices: pd.DataFrame, how: str = "common",
                 max_ffill: int = 5) -> tuple[pd.DataFrame, list[str]]:
    """Put every ticker on one calendar.

    `common` keeps only dates where every ticker has a price — the right
    default for cross-sectional work, where a covariance estimated off
    mismatched dates is meaningless.

    `ffill` carries the last price forward over short gaps (a holiday on one
    exchange, a halt) but refuses to bridge long ones: a forward-filled run
    reads as a string of exactly-zero returns, which deflates volatility and
    inflates kurtosis. `max_ffill` caps how far it will reach.
    """
    notes: list[str] = []
    if prices.empty:
        return prices, notes

    if how == "ffill":
        out = prices.ffill(limit=max_ffill)
        filled = int(prices.isna().sum().sum() - out.isna().sum().sum())
        if filled:
            notes.append("forward-filled %d missing prices (limit %d days)"
                         % (filled, max_ffill))
        out = out.dropna()
    elif how == "common":
        out = prices.dropna()
    elif how == "none":
        out = prices
    else:
        raise ValueError("how must be 'common', 'ffill' or 'none'")

    dropped = len(prices) - len(out)
    if dropped:
        notes.append("dropped %d of %d dates to align tickers"
                     % (dropped, len(prices)))
    return out, notes


def coverage(prices: pd.DataFrame) -> pd.DataFrame:
    """First and last real price per ticker, and how much is missing between.

    A ticker that listed part-way through the sample, or stopped trading
    before the end, is not an error — but it silently shortens any common
    sample built from it, so it should be visible rather than inferred.
    """
    rows = []
    for col in prices.columns:
        s = prices[col]
        valid = s.dropna()
        if valid.empty:
            rows.append({"ticker": col, "observations": 0, "first": pd.NaT,
                         "last": pd.NaT, "missing_within": 0,
                         "missing_pct": 1.0})
            continue
        span = s.loc[valid.index[0]:valid.index[-1]]
        rows.append({
            "ticker": col,
            "observations": int(valid.size),
            "first": valid.index[0],
            "last": valid.index[-1],
            "missing_within": int(span.isna().sum()),
            "missing_pct": float(1 - valid.size / max(len(s), 1)),
        })
    return pd.DataFrame(rows).set_index("ticker")


# --------------------------------------------------------------------------
# pure: the audit
# --------------------------------------------------------------------------
def audit_prices(prices: pd.DataFrame, jump: float = JUMP_THRESHOLD,
                 sigma: float = SIGMA_THRESHOLD,
                 stale_run: int = STALE_RUN) -> pd.DataFrame:
    """Flag observations that are more likely data errors than market moves.

    None of these flags proves an error — March 2020 produced genuine moves
    past every threshold here. They mark the handful of points worth opening
    before they are allowed to set the shape of a distribution.

    reversal is the strongest signal. A real crash is not undone the next day;
    a mis-adjusted price is, almost exactly, because the adjustment lands on
    one observation and the next return divides it straight back out.
    """
    if prices.empty or len(prices) < 3:
        return pd.DataFrame(columns=["ticker", "date", "price", "return",
                                     "z", "flags"])

    returns = prices.pct_change()
    rows = []
    for col in prices.columns:
        r = returns[col].dropna()
        if r.empty:
            continue
        mu, sd = r.mean(), r.std()
        z = (r - mu) / sd if sd > 0 else r * 0.0

        flagged: dict[pd.Timestamp, set] = {}

        def mark(idx, label):
            flagged.setdefault(idx, set()).add(label)

        for idx in r.index[r.abs() > jump]:
            mark(idx, "jump")
        for idx in z.index[z.abs() > sigma]:
            mark(idx, "sigma")

        # Spike-and-revert: a large move immediately undone.
        values = r.to_numpy()
        for i in range(len(values) - 1):
            a, b = values[i], values[i + 1]
            if abs(a) > jump and a * b < 0 and abs(b) > 0.5 * abs(a):
                mark(r.index[i], "reversal")
                mark(r.index[i + 1], "reversal")

        # A repeated price is a stale quote, and reads as zero volatility.
        p = prices[col].dropna()
        if len(p) > stale_run:
            same = (p.diff() == 0).astype(int)
            run = same.groupby((same == 0).cumsum()).cumsum()
            for idx in run.index[run >= stale_run]:
                mark(idx, "stale")

        for idx in p.index[p <= 0]:
            mark(idx, "nonpositive")

        for idx in sorted(flagged):
            rows.append({
                "ticker": col,
                "date": idx,
                "price": float(prices[col].get(idx, np.nan)),
                "return": float(r.get(idx, np.nan)),
                "z": float(z.get(idx, np.nan)),
                "flags": ",".join(sorted(flagged[idx])),
            })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["ticker", "date"]).reset_index(drop=True)
    return out


# --------------------------------------------------------------------------
# cache
# --------------------------------------------------------------------------
def _cache_key(tickers: list[str], start, end, interval: str,
               auto_adjust: bool, field_name: str) -> str:
    payload = json.dumps({
        "tickers": sorted(tickers), "start": str(start), "end": str(end),
        "interval": interval, "auto_adjust": auto_adjust, "field": field_name,
    }, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _read_cache(path: Path, max_age_hours: Optional[float]):
    meta_path = path.with_suffix(".json")
    if not (path.exists() and meta_path.exists()):
        return None, None
    try:
        meta = json.loads(meta_path.read_text())
        downloaded = datetime.fromisoformat(meta["vintage"])
    except (ValueError, KeyError, json.JSONDecodeError, OSError):
        return None, None

    if max_age_hours is not None:
        age = (datetime.now(timezone.utc) - downloaded).total_seconds() / 3600
        if age > max_age_hours:
            return None, None
    try:
        frame = pd.read_csv(path, index_col=0, parse_dates=True)
    except (OSError, ValueError):
        return None, None
    frame.index.name = "Date"
    return frame, meta


def _write_cache(path: Path, frame: pd.DataFrame, meta: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index_label="Date")
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    except OSError:
        pass          # a cache that cannot be written is not a failed download


# --------------------------------------------------------------------------
# network
# --------------------------------------------------------------------------
def download_prices(tickers, start=None, end=None, interval: str = "1d",
                    auto_adjust: bool = True, field_name: str = "Close",
                    align: str = "common", batch_size: int = 50,
                    retries: int = 3, pause: float = 0.5,
                    cache_dir=DEFAULT_CACHE, use_cache: bool = True,
                    max_age_hours: Optional[float] = 24.0,
                    audit: bool = True, verbose: bool = False
                    ) -> PriceDownload:
    """Download an auditable price history.

    `end` is exclusive, matching yfinance, so the last bar is the trading day
    before it. `auto_adjust=True` gives total-return prices (splits and
    dividends), which is what return statistics want; set it False only when
    replicating something that uses quoted prices, such as a price-weighted
    index.
    """
    tickers = [str(t).upper().strip() for t in tickers if str(t).strip()]
    if not tickers:
        raise ValueError("no tickers given")

    cache_dir = Path(cache_dir)
    key = _cache_key(tickers, start, end, interval, auto_adjust, field_name)
    cache_path = cache_dir / ("%s.csv" % key)
    notes: list[str] = []

    if use_cache:
        cached, meta = _read_cache(cache_path, max_age_hours)
        if cached is not None:
            frame, empty = drop_empty(cached.reindex(columns=tickers))
            frame, align_notes = align_prices(frame, how=align)
            return PriceDownload(
                prices=frame, requested=tickers, missing=empty,
                vintage=meta.get("vintage", ""), from_cache=True,
                notes=notes + align_notes + _audit_notes(frame, audit))

    try:
        import yfinance as yf
    except ImportError:
        raise ImportError(
            "yfinance is required for live prices: pip install yfinance")

    frames = []
    for batch in [tickers[i:i + batch_size]
                  for i in range(0, len(tickers), batch_size)]:
        raw, last_error = None, None
        for attempt in range(retries):
            try:
                raw = yf.download(
                    batch, start=start, end=end, interval=interval,
                    auto_adjust=auto_adjust, progress=False,
                    group_by="column", threads=True)
                break
            except Exception as exc:            # transient: rate limit, DNS, 5xx
                last_error = exc
                if attempt < retries - 1:
                    # Back off rather than hammer a service that is refusing.
                    time.sleep(pause * (2 ** attempt))
        if raw is None:
            notes.append("batch %s failed after %d attempts: %s"
                         % (",".join(batch), retries, last_error))
            continue
        frames.append(normalize_download(raw, batch, field_name))
        if verbose:
            print("  downloaded %d/%d" % (len(frames) * batch_size,
                                          len(tickers)))
        if len(tickers) > batch_size:
            time.sleep(pause)

    if not frames:
        raise RuntimeError(
            "no data returned for any ticker. Check the symbols, the date "
            "range, and network access to Yahoo Finance.")

    combined = pd.concat(frames, axis=1).reindex(columns=tickers)
    vintage = datetime.now(timezone.utc).isoformat(timespec="seconds")

    if use_cache:
        _write_cache(cache_path, combined, {
            "vintage": vintage, "tickers": tickers, "start": str(start),
            "end": str(end), "interval": interval,
            "auto_adjust": auto_adjust, "field": field_name,
        })

    frame, empty = drop_empty(combined)
    if empty:
        notes.append("no data for %s" % ", ".join(empty))
    frame, align_notes = align_prices(frame, how=align)

    return PriceDownload(
        prices=frame, requested=tickers, missing=empty, vintage=vintage,
        from_cache=False, notes=notes + align_notes + _audit_notes(frame, audit))


def _audit_notes(prices: pd.DataFrame, run: bool) -> list[str]:
    if not run or prices.empty:
        return []
    flags = audit_prices(prices)
    if flags.empty:
        return []
    per_ticker = flags.groupby("ticker").size().sort_values(ascending=False)
    worst = ", ".join("%s:%d" % (t, n) for t, n in per_ticker.head(5).items())
    return ["audit flagged %d observation(s) across %d ticker(s) [%s] — "
            "call audit_prices(prices) to inspect before trusting moments"
            % (len(flags), per_ticker.size, worst)]
