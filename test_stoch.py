"""Known-answer validation for the stochastic / rates / credit cluster."""
import sys
sys.path.insert(0, "/home/claude/finmodel")

import numpy as np
from scipy.stats import norm

from finmodel.quant import (gbm, ornstein_uhlenbeck, brownian_motion,
                            Vasicek, CIR, solve_merton, heston_mc_price, bs_price)
from finmodel.quant.credit import MertonCredit

fails = []
def check(name, cond, detail=""):
    print(f"  [{'OK' if cond else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}")
    if not cond: fails.append(name)

print("=" * 64)
print("1) STOCHASTIC FOUNDATIONS")
# GBM: E[S_T] = S0 e^{mu T}
S = gbm(100, 0.08, 0.20, 1.0, 250, 100_000, seed=1)
analytic_mean = 100 * np.exp(0.08)
check("GBM E[S_T] ~ S0 e^{muT}", abs(S[:, -1].mean() - analytic_mean) < 0.4,
      f"{S[:,-1].mean():.2f} vs {analytic_mean:.2f}")
# OU: long-run mean -> theta
X = ornstein_uhlenbeck(0.0, kappa=2.0, theta=0.05, sigma=0.02, T=20, n_steps=2000,
                       n_paths=8_000, seed=2)
check("OU terminal mean ~ theta", abs(X[:, -1].mean() - 0.05) < 2e-3,
      f"{X[:,-1].mean():.4f} vs 0.05")
# Brownian: Var(X_T) ~ sigma^2 T
B = brownian_motion(4.0, 500, 20_000, mu=0.0, sigma=0.5, seed=3)
check("BM Var(X_T) ~ sigma^2 T", abs(B[:, -1].var() - 0.25 * 4.0) < 0.05,
      f"{B[:,-1].var():.3f} vs {0.25*4:.3f}")

print("\n2) INTEREST-RATE MODELS")
vas = Vasicek(kappa=0.5, theta=0.04, sigma=0.02, r0=0.03)
p_an = vas.zcb_price(5.0)
p_mc = vas.zcb_price_mc(5.0, n_steps=250, n_paths=60_000, seed=4)
check("Vasicek analytic ~ MC ZCB(5y)", abs(p_an - p_mc) < 4e-3,
      f"analytic {p_an:.4f} vs MC {p_mc:.4f}")
check("Vasicek yield positive & sensible", 0 < vas.yield_to(5.0) < 0.10,
      f"y(5y)={vas.yield_to(5.0):.4f}")
cir = CIR(kappa=0.5, theta=0.04, sigma=0.08, r0=0.03)
pc_an = cir.zcb_price(5.0)
pc_mc = cir.zcb_price_mc(5.0, n_steps=300, n_paths=60_000, seed=5)
check("CIR analytic ~ MC ZCB(5y)", abs(pc_an - pc_mc) < 5e-3,
      f"analytic {pc_an:.4f} vs MC {pc_mc:.4f}")
paths = cir.simulate(5.0, 300, 5000, seed=6)
check("CIR paths stay non-negative", paths.min() >= 0.0)
check("CIR Feller flag correct", cir.feller_satisfied == (2*0.5*0.04 >= 0.08**2))

print("\n3) MERTON STRUCTURAL CREDIT")
mc = solve_merton(equity_value=30.0, equity_vol=0.40, debt_face=70.0, r=0.03, T=1.0)
# Plug solved (V, sigma_V) back into the two defining equations -> residuals ~ 0.
V, sV, D, r, T, E, sE = (mc.asset_value, mc.asset_volatility, 70.0, 0.03, 1.0, 30.0, 0.40)
d1 = (np.log(V/D) + (r + 0.5*sV**2)*T)/(sV*np.sqrt(T)); d2 = d1 - sV*np.sqrt(T)
res1 = V*norm.cdf(d1) - D*np.exp(-r*T)*norm.cdf(d2) - E
res2 = norm.cdf(d1)*sV*V - sE*E
check("Merton equations solved (residuals ~ 0)", abs(res1) < 1e-6 and abs(res2) < 1e-6,
      f"res=({res1:.2e},{res2:.2e})")
check("Merton PD in (0,1) and spread > 0",
      0 < mc.default_probability < 1 and mc.credit_spread > 0,
      f"PD={mc.default_probability:.2%}, spread={mc.credit_spread*1e4:.0f}bps")
# Monotonic: higher equity vol -> higher PD
pd_lo = solve_merton(30, 0.25, 70, 0.03).default_probability
pd_hi = solve_merton(30, 0.55, 70, 0.03).default_probability
check("PD rises with equity vol", pd_hi > pd_lo, f"{pd_lo:.2%} -> {pd_hi:.2%}")
# Low leverage -> near-zero PD
pd_safe = solve_merton(equity_value=900, equity_vol=0.20, debt_face=100, r=0.03).default_probability
check("Low-leverage firm PD ~ 0", pd_safe < 1e-3, f"PD={pd_safe:.2e}")

print("\n4) HESTON -> BLACK-SCHOLES LIMIT")
# With tiny vol-of-vol and v0=theta=sigma^2, rho=0, Heston ~ BS(sigma).
sig = 0.20
bs = bs_price(100, 100, 0.03, sig, 1.0, "call")
hes = heston_mc_price(100, 100, 0.03, v0=sig**2, kappa=2.0, theta=sig**2,
                      xi=0.001, rho=0.0, T=1.0, option="call",
                      n_steps=150, n_paths=150_000, seed=7)
check("Heston MC ~ BS in zero-vol-of-vol limit", abs(hes - bs) < 0.4,
      f"Heston {hes:.3f} vs BS {bs:.3f}")

print("=" * 64)
print("ALL PASSED" if not fails else f"FAILURES: {fails}")
sys.exit(1 if fails else 0)
