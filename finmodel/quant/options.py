"""Black-Scholes-Merton option pricing, Greeks, and implied volatility.

European options on a dividend-paying underlying (continuous yield q). Greeks
are returned in conventional units: delta per $1 of spot, gamma per $1², vega
per 1.00 (100 vol points) of vol, theta per year, rho per 1.00 of rate. Divide
vega/100 and theta/365 for the common per-1%-and-per-day quotes.

No SciPy dependency for the core: the normal CDF uses math.erf so pricing is
exact and self-contained. Implied vol uses a safeguarded Newton/bisection.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

OptionType = Literal["call", "put"]
SQRT_2PI = math.sqrt(2 * math.pi)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / SQRT_2PI


def _d1_d2(S, K, r, sigma, T, q):
    st = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / st
    return d1, d1 - st


def bs_price(S: float, K: float, r: float, sigma: float, T: float,
             option: OptionType = "call", q: float = 0.0) -> float:
    if T <= 0 or sigma <= 0:
        intrinsic = max(S - K, 0.0) if option == "call" else max(K - S, 0.0)
        return intrinsic
    d1, d2 = _d1_d2(S, K, r, sigma, T, q)
    df_r, df_q = math.exp(-r * T), math.exp(-q * T)
    if option == "call":
        return S * df_q * _norm_cdf(d1) - K * df_r * _norm_cdf(d2)
    return K * df_r * _norm_cdf(-d2) - S * df_q * _norm_cdf(-d1)


@dataclass
class Greeks:
    price: float
    delta: float
    gamma: float
    vega: float       # per 1.00 vol (divide by 100 for per 1%)
    theta: float      # per year (divide by 365 for per day)
    rho: float        # per 1.00 rate (divide by 100 for per 1%)


def bs_greeks(S: float, K: float, r: float, sigma: float, T: float,
              option: OptionType = "call", q: float = 0.0) -> Greeks:
    d1, d2 = _d1_d2(S, K, r, sigma, T, q)
    df_r, df_q = math.exp(-r * T), math.exp(-q * T)
    pdf = _norm_pdf(d1)
    price = bs_price(S, K, r, sigma, T, option, q)
    gamma = df_q * pdf / (S * sigma * math.sqrt(T))
    vega = S * df_q * pdf * math.sqrt(T)
    if option == "call":
        delta = df_q * _norm_cdf(d1)
        theta = (-S * df_q * pdf * sigma / (2 * math.sqrt(T))
                 - r * K * df_r * _norm_cdf(d2) + q * S * df_q * _norm_cdf(d1))
        rho = K * T * df_r * _norm_cdf(d2)
    else:
        delta = -df_q * _norm_cdf(-d1)
        theta = (-S * df_q * pdf * sigma / (2 * math.sqrt(T))
                 + r * K * df_r * _norm_cdf(-d2) - q * S * df_q * _norm_cdf(-d1))
        rho = -K * T * df_r * _norm_cdf(-d2)
    return Greeks(price=price, delta=delta, gamma=gamma, vega=vega,
                  theta=theta, rho=rho)


def implied_vol(price: float, S: float, K: float, r: float, T: float,
                option: OptionType = "call", q: float = 0.0,
                tol: float = 1e-8, max_iter: int = 100) -> float:
    """Recover sigma from a market price (Newton with bisection fallback)."""
    intrinsic = (max(S - K, 0.0) if option == "call" else max(K - S, 0.0))
    if price <= intrinsic + 1e-12:
        return 0.0
    lo, hi = 1e-6, 5.0
    sigma = 0.2
    for _ in range(max_iter):
        diff = bs_price(S, K, r, sigma, T, option, q) - price
        if abs(diff) < tol:
            return sigma
        v = bs_greeks(S, K, r, sigma, T, option, q).vega
        step = diff / v if v > 1e-12 else 0.0
        new = sigma - step
        if not (lo < new < hi):  # fall back to bisection
            if diff > 0:
                hi = sigma
            else:
                lo = sigma
            new = 0.5 * (lo + hi)
        sigma = new
    return sigma
