#!/usr/bin/env python3
"""Validate the frontier, the tangency portfolio and the CAL on real prices.

    python validate_frontier_real.py --limit 40 --rf 0.042

test_frontier.py proves the mathematics on a well-behaved synthetic
covariance. That is necessary and not sufficient: real covariance matrices
are near-singular, real expected returns estimated from five years of monthly
data are mostly noise, and the pathological cases — a risk-free rate above
the minimum-variance return, a frontier that barely bends — are common rather
than exotic. This runs the same identities against live S&P 500 prices.

Everything it checks is an identity, so it needs no stored expectations:

  * GMV return and volatility against B/A and sqrt(1/A)
  * tangency Sharpe against sqrt((mu-rf)'S^-1(mu-rf))
  * the frontier against its closed-form hyperbola
  * no returned frontier point dominated by another
  * the CAL at or above the frontier everywhere, touching at the tangency
  * two-fund separation reconstructed from the holdings

It also reports what the theory predicts will be ugly and the synthetic case
cannot show: the condition number with and without shrinkage, the gross
exposure of unconstrained weights, and whether a tangency portfolio exists at
a realistic risk-free rate at all.

Failures are emitted as GitHub Actions annotations as well as text, because
log archives are not always reachable from where this gets read.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

from finmodel.quant.portfolio import (MarkowitzOptimizer, NoTangencyPortfolio,
                                      ledoit_wolf_cov)
from finmodel.quant.universe import Universe

IN_CI = os.environ.get("GITHUB_ACTIONS") == "true"
failures: list[str] = []
notes: list[str] = []


def annotate(level: str, msg: str) -> None:
    if IN_CI:
        print("::%s::%s" % (level, msg.replace("\n", " ")))


def check(name: str, cond: bool, detail: str = "") -> bool:
    print("  [%s] %s%s" % ("OK" if cond else "FAIL", name,
                           (" — " + detail) if detail else ""))
    if not cond:
        failures.append(name)
        annotate("error", "%s: %s" % (name, detail or "identity violated"))
    return cond


def note(msg: str) -> None:
    """An observation rather than a pass/fail.

    These are emitted as workflow notices as well as printed, because the
    log archive is not reachable from every environment this gets read from
    while the annotations API is. The observations are the whole point of a
    real-data run — whether a tangency portfolio exists at all, how badly
    conditioned the covariance is, how levered the unconstrained weights
    get — so losing them to an unreachable log defeats the exercise.
    """
    print("  .    %s" % msg)
    notes.append(msg)
    annotate("notice", msg)


def summary(lines: list[str]) -> None:
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not path:
        return
    with open(path, "a") as fh:
        fh.write("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=40,
                    help="how many S&P 500 constituents to pull")
    ap.add_argument("--rf", type=float, default=0.042,
                    help="risk-free rate, annual")
    ap.add_argument("--years", type=int, default=5)
    args = ap.parse_args()

    print("=" * 66)
    print("1) UNIVERSE")
    from finmodel.data.sp500 import fetch_sp500
    constituents = fetch_sp500()
    tickers = [c["ticker"] for c in constituents][:args.limit]
    print("  pulling %d tickers, %dy monthly" % (len(tickers), args.years))
    u = Universe.from_tickers(tickers, frequency="M",
                              lookback_years=args.years)
    print("  got %d assets x %d periods" % (u.returns.shape[1],
                                            u.returns.shape[0]))
    if u.returns.shape[1] < 5:
        check("enough assets loaded", False,
              "only %d usable series" % u.returns.shape[1])
        return 1
    n_obs, n_assets = u.returns.shape
    note("observations per asset: %d; assets: %d; ratio p/n = %.2f"
         % (n_obs, n_assets, n_assets / n_obs))

    print("\n2) COVARIANCE CONDITIONING")
    sample = u.cov_matrix(annualized=True)
    shrunk = ledoit_wolf_cov(u.returns, u.periods_per_year)
    c_sample = float(np.linalg.cond(sample.to_numpy()))
    c_shrunk = float(np.linalg.cond(shrunk.to_numpy()))
    note("condition number: sample %.3g, Ledoit-Wolf %.3g (%.0fx better)"
         % (c_sample, c_shrunk, c_sample / max(c_shrunk, 1e-12)))
    check("shrinkage improves conditioning", c_shrunk <= c_sample,
          "%.3g vs %.3g" % (c_shrunk, c_sample))
    eig = np.linalg.eigvalsh(shrunk.to_numpy())
    check("shrunk covariance is positive definite", eig.min() > 0,
          "min eigenvalue %.3e" % eig.min())
    if n_assets >= n_obs:
        note("p >= n, so the SAMPLE covariance is singular by construction; "
             "this is exactly the regime shrinkage exists for")

    mu = u.mean_returns(annualized=True)
    opt = MarkowitzOptimizer(mu, shrunk, rf=args.rf)
    geo = opt.geometry()
    inv = np.linalg.pinv(shrunk.to_numpy())

    print("\n3) FRONTIER GEOMETRY ON REAL DATA")
    note("A=%.4g B=%.4g C=%.4g D=%.4g" % (geo.a, geo.b, geo.c, geo.d))
    gmv = opt.min_variance()
    check("GMV return = B/A",
          abs(gmv.expected_return - geo.gmv_return) < 1e-8,
          "%.6f vs %.6f" % (gmv.expected_return, geo.gmv_return))
    check("GMV volatility = sqrt(1/A)",
          abs(gmv.volatility - geo.gmv_volatility) < 1e-8,
          "%.6f vs %.6f" % (gmv.volatility, geo.gmv_volatility))
    check("GMV weights sum to 1", abs(gmv.weights.sum() - 1) < 1e-8)
    note("GMV: %.2f%% return at %.2f%% volatility"
         % (gmv.expected_return * 100, gmv.volatility * 100))
    note("GMV gross exposure %.1fx, largest short %.1f%%, largest long %.1f%%"
         % (gmv.weights.abs().sum(), gmv.weights.min() * 100,
            gmv.weights.max() * 100))

    print("\n4) THE EFFICIENT BRANCH")
    fr = opt.efficient_frontier(points=60)
    check("frontier returned points", len(fr) > 10, "%d points" % len(fr))
    check("no point below the GMV return",
          bool(fr["return"].min() >= geo.gmv_return - 1e-9),
          "lowest %.6f vs GMV %.6f" % (fr["return"].min(), geo.gmv_return))
    check("every point flagged efficient", bool(fr.efficient.all()))
    v, r = fr.volatility.to_numpy(), fr["return"].to_numpy()
    check("volatility is monotone", bool((np.diff(v) >= -1e-9).all()))
    dominated = sum(bool(((v <= vi + 1e-9) & (r >= ri + 1e-6)).any())
                    for vi, ri in zip(v, r))
    check("no point dominated by another", dominated == 0,
          "%d dominated" % dominated)
    closed = max(abs(geo.volatility_at(ri) - vi) for ri, vi in zip(r, v))
    check("matches the closed-form hyperbola", closed < 1e-8,
          "max error %.3e" % closed)

    print("\n5) TANGENCY AND THE CAL AT rf=%.2f%%" % (args.rf * 100))
    if geo.gmv_return <= args.rf:
        note("the GMV return (%.2f%%) is at or below rf (%.2f%%), so NO "
             "tangency portfolio exists — this is the case the old "
             "normalisation silently sign-flipped"
             % (geo.gmv_return * 100, args.rf * 100))
        raised = False
        try:
            opt.tangency()
        except NoTangencyPortfolio as exc:
            raised = True
            note("asymptotic Sharpe (never attained): %.4f"
                 % exc.asymptotic_sharpe)
            check("best frontier Sharpe is below the asymptote",
                  fr.sharpe.max() < exc.asymptotic_sharpe + 1e-9,
                  "%.4f < %.4f" % (fr.sharpe.max(), exc.asymptotic_sharpe))
        check("tangency() raises rather than returning a portfolio", raised)
        lo = opt.tangency(long_only=True)
        check("long-only fallback still returns a maximum",
              np.isfinite(lo.sharpe), "Sharpe %.4f" % lo.sharpe)
    else:
        cal = opt.capital_allocation_line()
        tan = cal.tangency
        excess = mu.to_numpy() - args.rf
        ident = float(np.sqrt(max(excess @ inv @ excess, 0.0)))
        check("tangency Sharpe = sqrt((mu-rf)'S^-1(mu-rf))",
              abs(tan.sharpe - ident) < 1e-6,
              "%.8f vs %.8f" % (tan.sharpe, ident))
        check("tangency weights sum to 1", abs(tan.weights.sum() - 1) < 1e-8)
        check("no frontier point beats the tangency Sharpe",
              fr.sharpe.max() <= tan.sharpe + 1e-8,
              "%.6f vs %.6f" % (fr.sharpe.max(), tan.sharpe))
        # On estimated moments the tangency portfolio is often far outside
        # the range of individual asset returns — a heavily levered point the
        # default grid (which stops at mu.max()) never reaches. The CAL
        # touches the FRONTIER, not an arbitrary truncation of it, so extend
        # the grid past the tangency return before testing the contact.
        span = opt.efficient_frontier(
            points=200,
            max_return=max(float(mu.max()), tan.expected_return * 1.25))
        sv = span.volatility.to_numpy()
        sr = span["return"].to_numpy()
        gap = np.array([cal.return_at(vi) - ri for vi, ri in zip(sv, sr)])
        check("CAL never below the frontier", gap.min() >= -1e-7,
              "min gap %.3e over %d points" % (gap.min(), len(span)))
        touch = float(sv[int(np.argmin(gap))])
        tol = max(0.02, 0.05 * tan.volatility)
        check("CAL touches at the tangency volatility",
              abs(touch - tan.volatility) <= tol,
              "contact at %.4f vs tangency %.4f (tol %.4f)"
              % (touch, tan.volatility, tol))
        note("tangency: %.2f%% return at %.2f%% volatility, Sharpe %.3f "
             "(GMV Sharpe %.3f)"
             % (tan.expected_return * 100, tan.volatility * 100,
                tan.sharpe, gmv.sharpe))
        note("tangency gross exposure %.1fx, largest position %.1f%%, "
             "largest short %.1f%%"
             % (tan.weights.abs().sum(), tan.weights.max() * 100,
                tan.weights.min() * 100))
        if tan.weights.abs().sum() > 4:
            note("gross exposure above 4x is the error-maximiser effect: "
                 "unconstrained weights on estimated moments are not "
                 "implementable as they stand")

        cp = cal.complete_portfolio(risk_aversion=3.0)
        w = cp.asset_weights.to_numpy()
        direct_r = float(w @ mu.to_numpy()) + cp.risk_free_weight * args.rf
        direct_v = float(np.sqrt(max(w @ shrunk.to_numpy() @ w, 0.0)))
        check("complete portfolio E[r] matches its holdings",
              abs(cp.expected_return - direct_r) < 1e-10,
              "%.10f vs %.10f" % (cp.expected_return, direct_r))
        check("complete portfolio volatility matches its holdings",
              abs(cp.volatility - direct_v) < 1e-10,
              "%.10f vs %.10f" % (cp.volatility, direct_v))
        check("Sharpe preserved along the CAL",
              abs(cp.sharpe - tan.sharpe) < 1e-8)
        note("at risk aversion 3: %.1f%% in the tangency portfolio, "
             "%.1f%% at rf" % (cp.risky_weight * 100,
                               cp.risk_free_weight * 100))

    print("\n6) LONG-ONLY, WHICH IS WHAT A REAL BOOK CAN HOLD")
    lo_tan = opt.optimize("max_sharpe", long_only=True)
    lo_gmv = opt.optimize("min_variance", long_only=True)
    check("long-only weights are non-negative",
          lo_tan.weights.min() >= -1e-7,
          "min %.3e" % lo_tan.weights.min())
    check("long-only GMV volatility >= unconstrained",
          lo_gmv.volatility >= gmv.volatility - 1e-7,
          "%.6f vs %.6f" % (lo_gmv.volatility, gmv.volatility))
    note("long-only max-Sharpe holds %d of %d names, largest %.1f%%"
         % (int((lo_tan.weights > 1e-4).sum()), n_assets,
            lo_tan.weights.max() * 100))
    note("equal-weight Sharpe %.3f vs long-only optimised %.3f"
         % ((float(np.ones(n_assets) / n_assets @ mu.to_numpy()) - args.rf)
            / float(np.sqrt(np.ones(n_assets) / n_assets
                            @ shrunk.to_numpy() @ (np.ones(n_assets) / n_assets))),
            lo_tan.sharpe))

    print("=" * 66)
    lines = ["## Frontier validation on real prices", "",
             "| | |", "|---|---|",
             "| Assets | **%d** |" % n_assets,
             "| Monthly observations | **%d** |" % n_obs,
             "| Risk-free rate | **%.2f%%** |" % (args.rf * 100),
             "| Condition number (sample) | **%.3g** |" % c_sample,
             "| Condition number (Ledoit-Wolf) | **%.3g** |" % c_shrunk,
             "| Checks failed | **%d** |" % len(failures), "", "### Observed", ""]
    lines += ["- %s" % n for n in notes]
    if failures:
        lines += ["", "### Failed", ""] + ["- %s" % f for f in failures]
    summary(lines)

    if failures:
        print("FAILURES (%d): %s" % (len(failures), failures))
        return 1
    print("ALL IDENTITIES HOLD ON REAL DATA")
    return 0


if __name__ == "__main__":
    sys.exit(main())
