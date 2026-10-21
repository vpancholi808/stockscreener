"""Company profiles from Yahoo Finance, for an S&P 500-sized universe.

`market.py` deliberately never calls `yfinance`'s per-ticker `.info`, because
across a whole-market universe of ~8,000 names it is one HTTP round trip each
and takes hours. That reasoning does not survive a change of scale. Measured
on a GitHub runner, `.info` returns in roughly 0.2-0.5s, so 500 constituents
cost about four minutes — perfectly affordable once, on a schedule.

So this module exists only for a bounded universe, and says so: `fetch_profiles`
refuses a list long enough to turn into an hours-long crawl unless explicitly
told otherwise. For the full market, keep using `market.py`.

What it adds that filings cannot:

  * forward P/E, forward EPS and PEG, which need analyst consensus estimates.
    SEC does not publish those, which is why `valuation.py` leaves forward_pe
    empty — this is where it comes from.
  * analyst price targets and the number of opinions behind them.
  * the 52-week range.
  * Yahoo's own sector and industry, useful as a cross-check on the SIC-derived
    classification rather than as a replacement.

Everything here is scraped vendor data, not SEC filings, so all of it sits
behind SERVE_MARKET_DATA for the same licensing reason the price fields do.

UNITS. Yahoo is not internally consistent about percentages, and has changed
its mind between versions. `profitMargins`, `returnOnEquity`, `payoutRatio`
and the growth fields come back as fractions (0.24 = 24%), but `dividendYield`
in current versions comes back as a *percent* (0.32 = 0.32%, not 32%). Storing
that unconverted would show Apple yielding 32%. Rather than hard-code either
convention and break on the next release, `_dividend_yield` recomputes the
yield from `dividendRate / price` and uses it to decide the scale.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Iterable, Optional

import pandas as pd

# Above this, a per-ticker crawl stops being a few minutes and becomes an
# outage waiting to happen. Pass allow_large=True if you really mean it.
MAX_REASONABLE = 1200

FIELDS = {
    # Yahoo key                 -> our column
    "sector": "yahoo_sector",
    "industry": "yahoo_industry",
    "sharesOutstanding": "shares_outstanding",
    "marketCap": "yahoo_market_cap",
    "enterpriseValue": "yahoo_enterprise_value",
    "trailingPE": "yahoo_trailing_pe",
    "forwardPE": "forward_pe",
    "pegRatio": "peg",
    "trailingEps": "trailing_eps",
    "forwardEps": "forward_eps",
    "payoutRatio": "payout_ratio",
    "beta": "yahoo_beta",
    "targetMeanPrice": "target_mean_price",
    "numberOfAnalystOpinions": "analyst_count",
    "fiftyTwoWeekHigh": "week52_high",
    "fiftyTwoWeekLow": "week52_low",
    "profitMargins": "yahoo_profit_margin",
    "returnOnEquity": "yahoo_roe",
    "revenueGrowth": "yahoo_revenue_growth",
    "earningsGrowth": "yahoo_earnings_growth",
    "freeCashflow": "yahoo_free_cashflow",
    "totalDebt": "yahoo_total_debt",
    "totalCash": "yahoo_total_cash",
}

# Fractions in Yahoo's own convention; bounded to catch a unit change.
FRACTION_BOUNDS = {
    "payout_ratio": (-1.0, 5.0),
    "yahoo_profit_margin": (-10.0, 1.0),
    "yahoo_roe": (-10.0, 10.0),
    "yahoo_revenue_growth": (-1.0, 20.0),
    "yahoo_earnings_growth": (-10.0, 50.0),
    "dividend_yield": (0.0, 0.25),
}


@dataclass
class Profile:
    ticker: str
    yahoo_sector: Optional[str] = None
    yahoo_industry: Optional[str] = None
    shares_outstanding: Optional[float] = None
    yahoo_market_cap: Optional[float] = None
    yahoo_enterprise_value: Optional[float] = None
    yahoo_trailing_pe: Optional[float] = None
    forward_pe: Optional[float] = None
    peg: Optional[float] = None
    trailing_eps: Optional[float] = None
    forward_eps: Optional[float] = None
    dividend_yield: Optional[float] = None
    payout_ratio: Optional[float] = None
    yahoo_beta: Optional[float] = None
    target_mean_price: Optional[float] = None
    analyst_count: Optional[int] = None
    week52_high: Optional[float] = None
    week52_low: Optional[float] = None
    yahoo_profit_margin: Optional[float] = None
    yahoo_roe: Optional[float] = None
    yahoo_revenue_growth: Optional[float] = None
    yahoo_earnings_growth: Optional[float] = None
    yahoo_free_cashflow: Optional[float] = None
    yahoo_total_debt: Optional[float] = None
    yahoo_total_cash: Optional[float] = None

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class ProfileDownload:
    profiles: dict
    requested: list
    missing: list = field(default_factory=list)
    vintage: str = ""
    notes: list = field(default_factory=list)

    def frame(self) -> pd.DataFrame:
        if not self.profiles:
            return pd.DataFrame()
        return pd.DataFrame([p.as_dict() for p in self.profiles.values()]
                            ).set_index("ticker")

    def summary(self) -> str:
        lines = ["Profiles: %d of %d tickers" % (len(self.profiles),
                                                 len(self.requested)),
                 "  vintage  %s" % self.vintage]
        if self.missing:
            shown = ", ".join(self.missing[:12])
            more = "" if len(self.missing) <= 12 else " (+%d more)" % (
                len(self.missing) - 12)
            lines.append("  missing  %s%s" % (shown, more))
        for n in self.notes:
            lines.append("  note     %s" % n)
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


def _num(value) -> Optional[float]:
    if value is None or isinstance(value, str):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out and abs(out) != float("inf") else None


def _dividend_yield(info: dict) -> Optional[float]:
    """Return the dividend yield as a fraction, whatever scale Yahoo used.

    Current versions report a percent (0.32 meaning 0.32%); older ones
    reported a fraction (0.0032). Both appear in the wild depending on the
    installed yfinance, so the scale is derived rather than assumed:
    `dividendRate` is an annual per-share amount and the price is known, so
    the true yield is computable and tells us which convention is in play.
    """
    reported = _num(info.get("dividendYield"))
    rate = _num(info.get("dividendRate"))
    price = (_num(info.get("currentPrice"))
             or _num(info.get("regularMarketPrice"))
             or _num(info.get("previousClose")))

    if rate is not None and price and price > 0:
        true_yield = rate / price
        if reported is None:
            return true_yield
        # Whichever interpretation of the reported figure is closer to the
        # computed one wins. No guessing about versions.
        as_fraction = abs(reported - true_yield)
        as_percent = abs(reported / 100.0 - true_yield)
        return reported if as_fraction <= as_percent else reported / 100.0

    if reported is None:
        return None
    # No price to check against: fall back on plausibility. No S&P 500
    # constituent yields above 25%, so anything larger is a percent.
    return reported / 100.0 if reported > 0.25 else reported


def profile_from_info(ticker: str, info: dict) -> Profile:
    """Map a Yahoo info payload onto a Profile, dropping implausible values."""
    data = {"ticker": ticker}
    for key, column in FIELDS.items():
        value = info.get(key)
        data[column] = value if column in ("yahoo_sector", "yahoo_industry") \
            else _num(value)
    data["analyst_count"] = (int(data["analyst_count"])
                             if data.get("analyst_count") else None)
    data["dividend_yield"] = _dividend_yield(info)

    # A multiple on a non-positive denominator is undefined, not cheap — the
    # same rule valuation.py applies. Yahoo sometimes returns negatives here.
    for column in ("forward_pe", "yahoo_trailing_pe", "peg"):
        if data.get(column) is not None and data[column] <= 0:
            data[column] = None

    for column, (lo, hi) in FRACTION_BOUNDS.items():
        value = data.get(column)
        if value is not None and not (lo <= value <= hi):
            data[column] = None

    return Profile(**data)


def fetch_profiles(tickers: Iterable[str], pause: float = 0.05,
                   retries: int = 2, allow_large: bool = False,
                   verbose: bool = True) -> ProfileDownload:
    """Pull Yahoo profiles one ticker at a time, politely.

    Per-ticker by necessity: Yahoo exposes no batch endpoint for this payload.
    Affordable only because the universe is bounded — see the module docstring.
    """
    tickers = [str(t).upper().strip() for t in tickers if str(t).strip()]
    if not tickers:
        raise ValueError("no tickers given")
    if len(tickers) > MAX_REASONABLE and not allow_large:
        raise ValueError(
            "%d tickers would mean roughly %.0f minutes of sequential "
            "requests. Use market.py for whole-market work, or pass "
            "allow_large=True." % (len(tickers), len(tickers) * 0.35 / 60))

    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance is required: pip install yfinance")

    profiles: dict = {}
    missing: list = []
    notes: list = []
    started = time.time()

    for i, ticker in enumerate(tickers, 1):
        info = None
        for attempt in range(retries):
            try:
                info = yf.Ticker(ticker).info
                if info and len(info) > 5:
                    break
                info = None
            except Exception:
                if attempt < retries - 1:
                    time.sleep(pause * (2 ** attempt) + 0.2)
        if not info:
            missing.append(ticker)
        else:
            profiles[ticker] = profile_from_info(ticker, info)
        if verbose and (i % 25 == 0 or i == len(tickers)):
            rate = i / max(time.time() - started, 1e-9)
            print("\r  profiles: %d/%d (%.1f/s, %d missing)"
                  % (i, len(tickers), rate, len(missing)), end="", flush=True)
        if pause:
            time.sleep(pause)

    if verbose:
        print("\r  profiles: %d/%d resolved in %.0fs%s"
              % (len(profiles), len(tickers), time.time() - started, " " * 20))

    if missing:
        notes.append("no profile for %d ticker(s)" % len(missing))
    forward = sum(1 for p in profiles.values() if p.forward_pe is not None)
    notes.append("forward P/E available for %d of %d" % (forward, len(profiles)))

    return ProfileDownload(
        profiles=profiles, requested=tickers, missing=missing,
        vintage=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        notes=notes)


def compare_sectors(profiles: dict, sec_sectors: dict) -> pd.DataFrame:
    """Where Yahoo's sector disagrees with the SIC-derived one.

    Neither is authoritative. Disagreement is worth seeing because sector
    benchmarks are computed per sector, so a misclassified company is compared
    against the wrong peers — and that is invisible in the output.
    """
    rows = []
    for ticker, profile in profiles.items():
        ours = sec_sectors.get(ticker)
        theirs = profile.yahoo_sector
        if ours and theirs and ours.lower() != theirs.lower():
            rows.append({"ticker": ticker, "ours": ours, "yahoo": theirs})
    return pd.DataFrame(rows)
