#!/usr/bin/env python3
"""Monte Carlo simulation and VaR on live Yahoo data.

    python example_montecarlo.py                 # NVDA alone
    python example_montecarlo.py NVDA MSFT JNJ   # a portfolio
    python example_montecarlo.py --years 10 --paths 100000

Needs network (Yahoo). The development sandbox reaches neither Yahoo nor
SEC, so this runs either on a machine with access or through
`.github/workflows/montecarlo.yml`, which reports every number as a
workflow annotation.

What it does, and why in this order:

  1. Download and AUDIT the prices. Every moment below is a function of
     this series, and a mis-adjusted print is indistinguishable from a
     crash until you look for the reversal. `audit_prices` looks.
  2. Report the return distribution, including the share of excess
     kurtosis held by the largest few observations. A fat-tail estimate
     resting on two days is not an estimate.
  3. Simulate forward under four models and compare their horizon
     distributions.
  4. Compute VaR four ways and then BACKTEST each. The comparison table
     on its own cannot tell you which number is right; Kupiec and
     Christoffersen can tell you which one breaches at the rate it
     promises and whether its breaches cluster.

The point of (4) is that a simulation is not evidence. It reproduces
whatever you assumed. The backtest is the only part of this script that
can say a model is wrong.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from finmodel.data.yahoo import audit_prices, coverage, download_prices
from finmodel.quant.correlation import correlation_matrix, diversification_ratio
from finmodel.quant.montecarlo import monte_carlo_var, simulate_portfolio
from finmodel.quant.risk import backtest_var, risk_report
from finmodel.returns import (annualize_mean, annualize_volatility,
                              compute_returns, infer_periods_per_year,
                              kurtosis_concentration)

# In CI the log archive is often unreachable, so every line is also an
# annotation. Locally the prefix is stripped and it reads as plain output.
CI = os.environ.get("GITHUB_ACTIONS") == "true"


def say(line: str = "") -> None:
    if CI and line.strip():
        print("::notice::%s" % line[:420])
    else:
        print(line)


def head(title: str) -> None:
    say()
    say(title)
    say("-" * len(title))


def pct(x, nd=2):
    """Format a fraction as a percentage.

    Coerces here rather than at each call site: several of the returns.py
    helpers accept a Series and hand back a one-element Series, so a bare
    `np.isfinite(x)` raises "truth value of a Series is ambiguous" on about
    half the lines below.
    """
    if isinstance(x, (pd.Series, pd.DataFrame)):
        x = x.squeeze()
        if isinstance(x, (pd.Series, pd.DataFrame)):
            return "n/a"
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if not np.isfinite(x) else "%.*f%%" % (nd, 100 * x)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tickers", nargs="*", default=["NVDA"])
    ap.add_argument("--years", type=int, default=5)
    ap.add_argument("--paths", type=int, default=50_000)
    ap.add_argument("--horizon-days", type=int, default=10)
    ap.add_argument("--confidence", type=float, default=0.99)
    ap.add_argument("--risk-free", type=float, default=0.042)
    args = ap.parse_args()

    tickers = [t.upper() for t in (args.tickers or ["NVDA"])]
    end = pd.Timestamp.today().normalize()
    start = end - pd.DateOffset(years=args.years)

    # ---------------------------------------------------------------- 1
    head("1. Prices")
    dl = download_prices(tickers, start=start, end=end, interval="1d",
                         auto_adjust=True, use_cache=False, audit=True)
    prices = dl.prices
    if prices.empty:
        say("FAILED: no price data returned for %s" % ", ".join(tickers))
        return 1

    say("%s | %d rows, %s to %s"
        % (", ".join(prices.columns), len(prices),
           prices.index[0].date(), prices.index[-1].date()))
    for n in dl.notes[:6]:
        say("note: %s" % n)

    cov_rep = coverage(prices)
    for t in cov_rep.index:
        r = cov_rep.loc[t]
        say("coverage %-6s %d obs, %s to %s, %s missing within"
            % (t, int(r["observations"]), str(r["first"])[:10],
               str(r["last"])[:10], pct(r["missing_pct"] / 100.0, 2)))

    # The strongest flag is `reversal`: a real crash is not undone the
    # next day, a mis-adjusted price is, almost exactly.
    aud = audit_prices(prices)
    if aud.empty:
        say("audit: clean — no jumps, reversals or sigma flags")
    else:
        say("audit: %d flagged observation(s)" % len(aud))
        for _, r in aud.head(8).iterrows():
            say("  %s %s  ret %s  flags %s"
                % (r.get("ticker", "?"),
                   str(r.get("date", ""))[:10],
                   pct(r.get("return")), r.get("flags", "")))
        if (aud.get("flags", pd.Series(dtype=str))
                .astype(str).str.contains("reversal").any()):
            say("  WARNING: a reversal flag usually means a bad "
                "adjustment, not a market event. Moments below inherit it.")

    # ---------------------------------------------------------------- 2
    head("2. Return distribution")
    rets = compute_returns(prices, kind="simple")
    k = infer_periods_per_year(rets.index)
    say("inferred %d periods per year" % k)

    ann_mu = annualize_mean(rets, k).astype(float)
    ann_vol = annualize_volatility(rets, k).astype(float)
    for t in rets.columns:
        r = rets[t].dropna()
        say("%-6s mean %s  vol %s  skew %+.2f  exkurt %+.2f  n=%d"
            % (t, pct(ann_mu[t]), pct(ann_vol[t]),
               float(r.skew()), float(r.kurtosis()), len(r)))

    # Excess kurtosis is a fourth moment, so one 12-sigma day contributes
    # 20,736 where an average day contributes 1. Above ~85% the statistic
    # is describing a couple of observations, not a distribution.
    conc = kurtosis_concentration(rets, k=2)
    for t in conc.index:
        share = float(conc.loc[t, "top_k_share"])
        verdict = ("two observations carry it — suspect a bad print"
                   if share > 0.85 else
                   "genuinely fat-tailed" if share > 0.10 else
                   "near-normal tails")
        say("%-6s exkurt %6.1f | top-2 share %s (%s, max|z| %.1f) — %s"
            % (t, float(conc.loc[t, "excess_kurtosis"]), pct(share, 1),
               conc.loc[t, "dates"], float(conc.loc[t, "max_abs_z"]), verdict))

    if len(rets.columns) > 1:
        cm = correlation_matrix(rets, repair=True)
        say("average pairwise correlation %.3f"
            % float(cm.to_numpy()[np.triu_indices(len(cm), 1)].mean()))

    # ---------------------------------------------------------------- 3
    head("3. Simulation")
    # Passed the whole frame, which returns a Series indexed by ticker. A
    # per-column dict comprehension instead stores one-element Series as
    # the VALUES, giving a Series of Series that only fails much later,
    # inside the simulator's .to_numpy().
    mu = annualize_mean(rets, k).astype(float)
    cov = rets.cov() * k
    w = pd.Series(1.0 / len(rets.columns), index=list(rets.columns))

    if len(w) > 1:
        say("equal weights, diversification ratio %.3f"
            % diversification_ratio(w, cov))
    say("mu  %s" % ", ".join("%s %s" % (t, pct(mu[t], 1)) for t in mu.index))
    say("vol %s" % ", ".join("%s %s" % (t, pct(math.sqrt(cov.loc[t, t]), 1))
                             for t in mu.index))

    hy = 1.0
    say("%d paths, %.0fy horizon, %d models" % (args.paths, hy, 4))
    for model in ("normal", "t", "bootstrap", "block"):
        kw = {"history": rets} if model in ("bootstrap", "block") else {}
        sim = simulate_portfolio(mu, cov, w, horizon_years=hy,
                                 periods_per_year=k, n_paths=args.paths,
                                 model=model, df=5.0, risk_free=args.risk_free,
                                 seed=20260101, **kw)
        term = sim.terminal
        say("%-9s median %+6s | 5th %+7s | 95th %+7s | P(loss) %5s | "
            "med maxDD %s"
            % (model, pct(np.median(term) - 1, 1),
               pct(np.quantile(term, 0.05) - 1, 1),
               pct(np.quantile(term, 0.95) - 1, 1),
               pct(sim.probability_below(1.0), 1),
               pct(float(np.median(sim.max_drawdown())), 1)))

    # ---------------------------------------------------------------- 4
    head("4. VaR at %.1f%% over %d day(s)" % (100 * args.confidence,
                                              args.horizon_days))
    hz = args.horizon_days / k
    port_hist = (rets * w).sum(axis=1).dropna()

    rr = risk_report(port_hist, confidence=args.confidence,
                     horizon_days=args.horizon_days)
    say("historical    %s" % pct(rr.var_historical))
    say("parametric    %s" % pct(rr.var_parametric))
    say("cornish-fisher %s" % pct(rr.var_cornish_fisher))

    mc_rows = []
    for model in ("normal", "t", "bootstrap", "block"):
        kw = {"history": rets} if model in ("bootstrap", "block") else {}
        m = monte_carlo_var(mu, cov, w, confidence=args.confidence,
                            horizon_years=hz, periods_per_year=k,
                            n_paths=max(args.paths, 100_000), model=model,
                            df=5.0, seed=20260101, **kw)
        mc_rows.append((model, m))
        lo, hi = m.var_ci
        say("MC %-9s %s +/- %s (95%% CI %s to %s) | ES %s"
            % (model, pct(m.var), pct(m.var_stderr, 3),
               pct(lo), pct(hi), pct(m.expected_shortfall)))

    # A VaR number is worthless without a backtest, and there are two
    # separate ways it fails. Kupiec asks whether the breach COUNT is
    # right; Christoffersen asks whether the breaches are independent.
    # A model can pass the first and fail the second, which means it has
    # the right average and the wrong timing — and no comparison of VaR
    # magnitudes above can reveal that.
    head("5. Backtest (out of sample, rolling 250-day window)")
    if len(port_hist) <= 260:
        say("not enough history to backtest (%d rows)" % len(port_hist))
    else:
        for method in ("historical", "parametric", "cornish_fisher"):
            bt = backtest_var(port_hist, confidence=args.confidence,
                           window=250, method=method)
            say("%-15s breaches %3d of %4d (%s, expected %s) | "
                "Kupiec p=%.3f %s | Christoffersen p=%.3f %s"
                % (method, bt.breaches, bt.observations,
                   pct(bt.breach_rate, 2), pct(1 - args.confidence, 2),
                   bt.kupiec_p, "OK" if bt.kupiec_p > 0.05 else "REJECT",
                   bt.christoffersen_p,
                   "OK" if bt.christoffersen_p > 0.05 else "CLUSTERED"))
        say("A REJECT on coverage means the level is wrong. CLUSTERED means "
            "the level is right and the timing is not — breaches arrive "
            "together, which is when losing twice in a week matters.")

    head("Reading this")
    say("MC-normal must agree with parametric: it is the wiring check, "
        "not a finding. Where the bootstrap and block numbers exceed both, "
        "the tail weight is in the data and not in the normal assumption.")
    say("Nothing above is a forecast. Every path is a draw from assumptions "
        "fitted to %d days of history; the backtest is the only section "
        "that can falsify them." % len(port_hist))
    return 0


if __name__ == "__main__":
    sys.exit(main())
