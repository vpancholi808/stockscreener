"""The efficient frontier and the capital allocation line.

This page is deliberately assumption-driven rather than data-driven, and the
inputs are editable on the page. Markowitz is a map from (mu, Sigma) to
weights, and almost everything that goes wrong with it in practice goes wrong
in mu: Merton (1980) showed that estimating an expected return to useful
precision takes decades of data, while a covariance converges in months. So
the expected returns here are *yours*, entered explicitly, and the page shows
what the mathematics does with them.

The page needs no database and no price feed, so it works on a fresh
deployment. Where price history is available it can be loaded instead, which
sets the covariance from data and leaves the expected returns to you — the
split the estimation-error literature recommends.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from finmodel.quant.portfolio import (MarkowitzOptimizer, NoTangencyPortfolio,
                                      ledoit_wolf_cov)
from finmodel.web import ui

DEFAULT = pd.DataFrame({
    "Asset": ["Short bonds", "Broad equity", "Growth equity", "Real assets"],
    "Expected return %": [4.0, 8.0, 12.0, 10.0],
    "Volatility %": [10.0, 18.0, 30.0, 22.0],
})

st.title("Efficient frontier and the CAL")
st.caption(
    "Mean-variance optimisation over your own assumptions. Every number on "
    "this page follows from the table below — change it and watch the "
    "frontier move."
)


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------
def uniform_cov(sd: np.ndarray, rho: float) -> np.ndarray:
    n = len(sd)
    corr = np.full((n, n), float(rho))
    np.fill_diagonal(corr, 1.0)
    return np.outer(sd, sd) * corr


@st.cache_data(ttl=300, show_spinner=False)
def cov_from_prices(csv_bytes: bytes) -> tuple[pd.Series, pd.DataFrame]:
    """Annualised mu and a Ledoit-Wolf covariance from an uploaded price CSV."""
    from io import BytesIO
    px = pd.read_csv(BytesIO(csv_bytes), index_col=0, parse_dates=True)
    px = px.sort_index().apply(pd.to_numeric, errors="coerce").dropna(how="all")
    rets = px.pct_change().dropna(how="any")
    if len(rets) < 24:
        raise ValueError(f"only {len(rets)} usable return rows; need at least 24")
    # Infer the sampling frequency from the median spacing of the index.
    days = np.median(np.diff(rets.index.values).astype("timedelta64[D]")
                     .astype(float))
    per_year = int(round(365.0 / max(days, 1.0)))
    per_year = min(max(per_year, 1), 252)
    return rets.mean() * per_year, ledoit_wolf_cov(rets, per_year)


source = st.radio(
    "Inputs",
    ["My assumptions", "Upload price history (CSV)"],
    horizontal=True,
    help="A CSV sets the covariance from data. Expected returns are still "
         "yours to set — they are the hard part to estimate.",
)

mu = cov = None
labels: list[str] = []

if source == "My assumptions":
    left, right = st.columns([3, 2])
    with left:
        edited = st.data_editor(
            DEFAULT, num_rows="dynamic", width="stretch",
            column_config={
                "Asset": st.column_config.TextColumn(required=True),
                "Expected return %": st.column_config.NumberColumn(
                    format="%.2f", step=0.5),
                "Volatility %": st.column_config.NumberColumn(
                    format="%.2f", step=0.5, min_value=0.01),
            },
            key="assumptions")
    tbl = edited.dropna(subset=["Asset"])
    tbl = tbl[tbl["Asset"].astype(str).str.strip() != ""]
    n = len(tbl)
    with right:
        # A uniform correlation matrix is positive semi-definite only for
        # rho >= -1/(n-1); below that the "covariance" describes no possible
        # set of assets and the optimiser would happily return weights for it.
        floor = -1.0 / (n - 1) if n > 1 else 0.0
        rho = st.slider("Correlation between every pair", float(round(floor + 0.01, 2)),
                        0.95, 0.35, 0.05,
                        help=f"With {n} assets, anything below "
                             f"{floor:.2f} is not a valid correlation matrix.")
        st.caption(
            "One number for every pair keeps the table small. Correlation is "
            "what diversification is made of: at 1.0 the frontier is a "
            "straight line and there is nothing to gain."
        )
    if n >= 2:
        labels = [str(a) for a in tbl["Asset"]]
        mu = pd.Series((tbl["Expected return %"].astype(float) / 100).to_numpy(),
                       index=labels)
        sd = (tbl["Volatility %"].astype(float) / 100).to_numpy()
        cov = pd.DataFrame(uniform_cov(sd, rho), index=labels, columns=labels)
else:
    st.caption(
        "A CSV with a date column first and one price column per asset. "
        "Monthly or daily both work — the frequency is inferred from the "
        "dates. The covariance uses Ledoit-Wolf shrinkage."
    )
    up = st.file_uploader("Price history", type=["csv"])
    if up is not None:
        try:
            mu_data, cov = cov_from_prices(up.getvalue())
        except Exception as exc:
            st.error(f"Could not read that file: {exc}")
            cov = None
        else:
            labels = list(cov.index)
            st.success(f"Loaded {len(labels)} assets.")
            hist = pd.DataFrame({
                "Asset": labels,
                "Historical return %": (mu_data * 100).round(2).to_numpy(),
                "Expected return %": (mu_data * 100).round(2).to_numpy(),
                "Volatility %": (np.sqrt(np.diag(cov.to_numpy())) * 100).round(2),
            })
            st.caption(
                "Historical mean returns are prefilled, but they are a poor "
                "forecast — this is where you overwrite them with a view."
            )
            ed = st.data_editor(
                hist, width="stretch", key="from_prices",
                disabled=["Asset", "Historical return %", "Volatility %"],
                column_config={"Expected return %":
                               st.column_config.NumberColumn(format="%.2f",
                                                             step=0.5)})
            mu = pd.Series((ed["Expected return %"].astype(float) / 100).to_numpy(),
                           index=labels)

c1, c2 = st.columns(2)
rf = c1.slider("Risk-free rate %", 0.0, 10.0, 2.0, 0.1,
               help="The CAL starts here. Above the minimum-variance "
                    "portfolio's return there is no tangency portfolio at "
                    "all — the page will say so.") / 100
aversion = c2.slider("Risk aversion (A)", 1.0, 10.0, 3.0, 0.5,
                     help="The A in U = E[r] - (A/2)sigma^2. Two to four is "
                          "the conventional range. It sets how much of the "
                          "tangency portfolio you hold and nothing else.")

if mu is None or cov is None or len(mu) < 2:
    st.info("Add at least two assets to trace a frontier.")
    ui.footer()
    st.stop()

eig = np.linalg.eigvalsh(cov.to_numpy())
if eig.min() < -1e-10:
    st.error(
        f"That covariance matrix is not positive semi-definite (smallest "
        f"eigenvalue {eig.min():.2e}), which means no set of assets could "
        f"produce it. Lower the correlation."
    )
    ui.footer()
    st.stop()


# --------------------------------------------------------------------------
# the model
# --------------------------------------------------------------------------
opt = MarkowitzOptimizer(mu, cov, rf=rf)
try:
    geo = opt.geometry()
except ValueError as exc:
    st.error(str(exc))
    ui.footer()
    st.stop()

gmv = opt.min_variance()
curve = opt.efficient_frontier(points=120, include_inefficient=True)

cal = None
try:
    cal = opt.capital_allocation_line()
except NoTangencyPortfolio as exc:
    st.warning(
        f"**No tangency portfolio exists at a {rf:.2%} risk-free rate.**\n\n"
        f"The minimum-variance portfolio returns {geo.gmv_return:.2%}, which "
        f"is below the risk-free rate, so no line from {rf:.2%} ever touches "
        f"the efficient branch. The Sharpe ratio rises along the frontier "
        f"toward {exc.asymptotic_sharpe:.3f} — the slope of the hyperbola's "
        f"asymptote — without ever reaching it, so the maximum is a limit "
        f"rather than a portfolio.\n\n"
        f"This is not a numerical failure. It is the honest answer when the "
        f"assumed returns do not compensate for the assumed risk: lending at "
        f"{rf:.2%} dominates. Lower the risk-free rate or raise an expected "
        f"return to see the CAL.",
        icon="⚠️")

m = st.columns(4)
m[0].metric("Min-variance return", f"{geo.gmv_return:.2%}",
            help="B/A. Needs no expected returns at all.")
m[1].metric("Min-variance volatility", f"{geo.gmv_volatility:.2%}",
            help="sqrt(1/A), the lowest risk any mix of these assets can reach.")
if cal is not None:
    m[2].metric("Tangency return", f"{cal.tangency.expected_return:.2%}",
                f"{cal.tangency.volatility:.2%} vol", delta_color="off")
    m[3].metric("Best Sharpe ratio", f"{cal.slope:.3f}",
                f"{cal.slope - gmv.sharpe:+.3f} vs min-variance")
else:
    m[2].metric("Tangency return", "—")
    m[3].metric("Best Sharpe ratio", "—",
                f"supremum {geo.asymptotic_sharpe:.3f}", delta_color="off")


# --------------------------------------------------------------------------
# the picture
# --------------------------------------------------------------------------
eff = curve[curve.efficient]
ineff = curve[~curve.efficient]
vmax = float(max(curve.volatility.max(),
                 np.sqrt(np.diag(cov.to_numpy())).max())) * 1.12

layers = []
if not ineff.empty:
    layers.append(
        alt.Chart(ineff).mark_line(strokeDash=[5, 4], color="#b4b4b4",
                                   strokeWidth=2)
        .encode(x="volatility:Q", y="return:Q"))
layers.append(
    alt.Chart(eff).mark_line(color="#8a8a8a", strokeWidth=3)
    .encode(x=alt.X("volatility:Q", title="Volatility (annual σ)",
                    axis=alt.Axis(format="%"),
                    scale=alt.Scale(domain=[0, vmax])),
            y=alt.Y("return:Q", title="Expected return",
                    axis=alt.Axis(format="%")),
            tooltip=[alt.Tooltip("return:Q", format=".2%", title="E[r]"),
                     alt.Tooltip("volatility:Q", format=".2%", title="σ"),
                     alt.Tooltip("sharpe:Q", format=".3f")]))

if cal is not None:
    line = pd.DataFrame({"volatility": [0.0, vmax],
                         "return": [rf, cal.return_at(vmax)]})
    layers.append(alt.Chart(line).mark_line(color="#2a6fdb", strokeWidth=3)
                  .encode(x="volatility:Q", y="return:Q"))
    pts = pd.DataFrame([
        {"volatility": 0.0, "return": rf, "label": "Risk-free"},
        {"volatility": cal.tangency.volatility,
         "return": cal.tangency.expected_return, "label": "Tangency"},
    ])
    layers.append(alt.Chart(pts).mark_point(size=160, filled=True,
                                            color="#2a6fdb")
                  .encode(x="volatility:Q", y="return:Q",
                          tooltip=["label", alt.Tooltip("return:Q", format=".2%"),
                                   alt.Tooltip("volatility:Q", format=".2%")]))

gmv_pt = pd.DataFrame([{"volatility": gmv.volatility,
                        "return": gmv.expected_return,
                        "label": "Minimum variance"}])
layers.append(alt.Chart(gmv_pt).mark_point(size=160, filled=True,
                                           color="#6b6b6b")
              .encode(x="volatility:Q", y="return:Q",
                      tooltip=["label", alt.Tooltip("return:Q", format=".2%"),
                               alt.Tooltip("volatility:Q", format=".2%")]))

assets = pd.DataFrame({"label": list(mu.index),
                       "return": mu.to_numpy(),
                       "volatility": np.sqrt(np.diag(cov.to_numpy()))})
layers.append(alt.Chart(assets).mark_point(size=90, filled=True,
                                           color="#c8c8c8", opacity=0.9)
              .encode(x="volatility:Q", y="return:Q",
                      tooltip=["label", alt.Tooltip("return:Q", format=".2%"),
                               alt.Tooltip("volatility:Q", format=".2%")]))
layers.append(alt.Chart(assets).mark_text(align="left", dx=9, dy=2,
                                          fontSize=11, color="#8a8a8a")
              .encode(x="volatility:Q", y="return:Q", text="label:N"))

st.altair_chart(alt.layer(*layers).properties(height=430)
                .configure_view(strokeWidth=0), width="stretch")

note = (
    "Solid grey is the efficient branch; dashed grey is the lower branch of "
    "the same hyperbola, where every portfolio is beaten by one directly "
    "above it at identical risk. "
)
if cal is not None:
    note += (
        f"The blue capital allocation line runs from the risk-free rate "
        f"through the tangency portfolio and lies above the frontier "
        f"everywhere except that one point of contact — which is the whole "
        f"argument for holding cash or leverage alongside one risky mix "
        f"rather than picking a frontier portfolio directly."
    )
st.caption(note)


# --------------------------------------------------------------------------
# weights and the complete portfolio
# --------------------------------------------------------------------------
st.subheader("Weights")
w = pd.DataFrame({"Asset": list(mu.index),
                  "Minimum variance": gmv.weights.to_numpy()})
if cal is not None:
    w["Tangency"] = cal.tangency.weights.to_numpy()
long_only = opt.optimize("max_sharpe", long_only=True)
w["Tangency, long-only"] = long_only.weights.to_numpy()
st.dataframe(w.style.format({c: "{:.1%}" for c in w.columns if c != "Asset"}),
             width="stretch", hide_index=True)

if (w.select_dtypes("number") < -1e-6).any().any():
    st.caption(
        "Negative weights are short positions. Unconstrained mean-variance "
        "optimisation produces them freely, and they are the main reason "
        "textbook Markowitz weights are rarely implementable as they stand — "
        "the long-only column is what a constrained book would hold."
    )

if cal is not None:
    st.subheader("Your complete portfolio")
    cp = cal.complete_portfolio(risk_aversion=aversion)
    a, b, c = st.columns(3)
    a.metric("In the tangency portfolio", f"{cp.risky_weight:.1%}")
    b.metric("At the risk-free rate", f"{cp.risk_free_weight:.1%}")
    c.metric("Expected return", f"{cp.expected_return:.2%}",
             f"{cp.volatility:.2%} vol", delta_color="off")
    if cp.is_levered:
        st.caption(
            f"Above 100% means borrowing at {rf:.2%} to lever the tangency "
            f"portfolio. The arithmetic assumes you can borrow at the "
            f"risk-free rate, which nobody can; a realistic borrowing rate "
            f"kinks the CAL downward beyond the tangency point."
        )

    ladder = pd.DataFrame([
        {"Risk aversion": a_,
         "In tangency": cal.complete_portfolio(risk_aversion=a_).risky_weight,
         "Expected return":
             cal.complete_portfolio(risk_aversion=a_).expected_return,
         "Volatility": cal.complete_portfolio(risk_aversion=a_).volatility,
         "Sharpe": cal.complete_portfolio(risk_aversion=a_).sharpe}
        for a_ in (1.5, 2.0, 3.0, 5.0, 8.0)])
    st.dataframe(ladder.style.format({"Risk aversion": "{:.1f}",
                                      "In tangency": "{:.1%}",
                                      "Expected return": "{:.2%}",
                                      "Volatility": "{:.2%}",
                                      "Sharpe": "{:.3f}"}),
                 width="stretch", hide_index=True)
    st.caption(
        "The Sharpe column is constant: moving along the CAL changes how much "
        "risk you carry, never how well you are paid for carrying it. That is "
        "Tobin's two-fund separation — the risky mix is the same for a "
        "cautious investor and an aggressive one, and only the cash weight "
        "differs."
    )

with st.expander("The geometry behind it"):
    st.markdown(
        f"""
The whole frontier reduces to four numbers computed from the expected returns
μ and the covariance Σ:

| | | value |
|---|---|---|
| `A` | 1′Σ⁻¹1 | {geo.a:,.3f} |
| `B` | 1′Σ⁻¹μ | {geo.b:,.3f} |
| `C` | μ′Σ⁻¹μ | {geo.c:,.3f} |
| `D` | AC − B² | {geo.d:,.3f} |

Everything else follows. The frontier is the hyperbola
σ²(r) = (A r² − 2B r + C) / D. Its vertex — the minimum-variance portfolio —
sits at return B/A = {geo.gmv_return:.2%} and volatility √(1/A) =
{geo.gmv_volatility:.2%}. Returns above B/A are efficient; returns below it
are the dominated branch drawn dashed.

The highest Sharpe ratio available from these assets is
√(C − 2·rf·B + rf²·A) = {geo.max_sharpe(rf):.3f}, attained at the tangency
portfolio provided rf < B/A. As expected return grows the frontier flattens
toward an asymptote of slope √(D/A) = {geo.asymptotic_sharpe:.3f}, which is
why a risk-free rate at or above {geo.gmv_return:.2%} leaves the problem with
a supremum and no maximiser.

Σ is inverted to get any of this, so its condition number —
**{opt.condition_number:,.0f}** here — is roughly the factor by which errors
in μ are amplified in the weights. That is the practical case for shrinkage
estimators and for long-only constraints, which act as shrinkage by another
route (Jagannathan and Ma, 2003).
        """
    )

ui.footer()
