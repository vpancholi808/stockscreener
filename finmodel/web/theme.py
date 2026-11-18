"""Glass theme for the Streamlit app.

The static pages (build_site.py, build_frontier_page.py) own their own CSS.
Streamlit renders its own chrome, so matching them means injecting CSS over
that chrome rather than writing a stylesheet from scratch.

Three rules shape everything here:

  * **Target data-testid, not generated class names.** Streamlit's emotion
    classes change between releases; the testids are part of its public
    surface and survive upgrades. Where a testid does not exist the selector
    is written defensively so a miss degrades to default Streamlit rather
    than to a broken layout.
  * **The mesh has to sit behind the app, not inside it.** Streamlit owns
    <body>, so the backdrop is a fixed pseudo-element on the app container at
    a negative z-index, with the container itself made transparent. Painting
    the body instead leaves Streamlit's own opaque background on top of it.
  * **The palette follows Streamlit's configured base, never the viewer's
    prefers-color-scheme.** Streamlit resolves its text, widget and chart
    colours from that server-side setting, so a media query recolours the
    backdrop while Streamlit keeps painting the other theme's text over it.

Flip `base` in .streamlit/config.toml and both halves move together.
"""
from __future__ import annotations

import streamlit as st

LIGHT = """
  --base:#e9edf2;
  --m1:rgba(120,170,210,.55); --m2:rgba(190,150,215,.40);
  --m3:rgba(240,185,140,.45); --m4:rgba(130,205,195,.38);
  --glass:rgba(255,255,255,.58); --glass-2:rgba(255,255,255,.80);
  --edge:rgba(255,255,255,.80); --border:rgba(20,34,44,.12);
  --ink:#152430; --muted:#5a7180; --faint:#8699a6;
  --accent:#1c6a86; --accent-soft:rgba(28,106,134,.12);
  --shadow:0 10px 34px rgba(14,26,36,.12);
"""

DARK = """
  --base:#0a1014;
  --m1:rgba(40,95,125,.55); --m2:rgba(95,60,120,.42);
  --m3:rgba(165,95,45,.45); --m4:rgba(30,110,100,.38);
  --glass:rgba(26,36,44,.56); --glass-2:rgba(22,31,38,.86);
  --edge:rgba(255,255,255,.12); --border:rgba(255,255,255,.11);
  --ink:#e7eff4; --muted:#93a9b6; --faint:#6f8593;
  --accent:#6ec2d8; --accent-soft:rgba(110,194,216,.14);
  --shadow:0 10px 34px rgba(0,0,0,.42);
"""

CSS = """
<style>
:root{__PALETTE__}

[data-testid="stAppViewContainer"]{
  background:var(--base);
  color:var(--ink);
  isolation:isolate;
}
[data-testid="stAppViewContainer"]::before{
  content:"";position:fixed;inset:-20%;z-index:-1;pointer-events:none;
  filter:blur(70px) saturate(130%);
  background:
    radial-gradient(38% 42% at 12% 18%, var(--m1) 0%, transparent 70%),
    radial-gradient(34% 38% at 86% 12%, var(--m2) 0%, transparent 70%),
    radial-gradient(42% 46% at 78% 82%, var(--m3) 0%, transparent 72%),
    radial-gradient(36% 40% at 22% 88%, var(--m4) 0%, transparent 70%);
}
[data-testid="stHeader"]{background:transparent}
[data-testid="stMain"],[data-testid="stAppViewBlockContainer"]{
  background:transparent}

/* Sidebar as a frosted panel. */
[data-testid="stSidebar"]{
  background:var(--glass);
  -webkit-backdrop-filter:blur(22px) saturate(150%);
  backdrop-filter:blur(22px) saturate(150%);
  border-right:1px solid var(--border);
}
[data-testid="stSidebarContent"]{background:transparent}

/* Metrics become the stat tiles from the static pages. */
[data-testid="stMetric"]{
  background:var(--glass);
  -webkit-backdrop-filter:blur(22px) saturate(150%);
  backdrop-filter:blur(22px) saturate(150%);
  border:1px solid var(--border);border-radius:16px;
  box-shadow:var(--shadow),inset 0 1px 0 var(--edge);
  padding:14px 16px;
}
[data-testid="stMetricLabel"]{
  color:var(--faint)!important;text-transform:uppercase;
  letter-spacing:.07em;font-size:10.5px!important;font-weight:600!important}
[data-testid="stMetricValue"]{
  font-variant-numeric:tabular-nums;letter-spacing:-.02em}
[data-testid="stMetricDelta"]{font-size:12px}

/* Any st.container(border=True), expander, dataframe, table or alert gets
   the same treatment, so a view written with plain Streamlit primitives
   comes out looking designed. */
[data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid]),
[data-testid="stExpander"],
[data-testid="stDataFrame"],
[data-testid="stTable"],
[data-testid="stAlertContainer"],
[data-testid="stNotification"],
div[data-baseweb="notification"]{
  background:var(--glass);
  -webkit-backdrop-filter:blur(22px) saturate(150%);
  backdrop-filter:blur(22px) saturate(150%);
  border:1px solid var(--border)!important;border-radius:16px!important;
  box-shadow:var(--shadow),inset 0 1px 0 var(--edge);
}
[data-testid="stExpander"] details{background:transparent;border:0}
[data-testid="stExpander"] summary{color:var(--accent)}

/* Inputs and controls: rounded, translucent, accent focus. */
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stSelectbox"] div[data-baseweb="select"]>div,
[data-testid="stMultiSelect"] div[data-baseweb="select"]>div,
[data-testid="stTextArea"] textarea{
  background:var(--glass-2)!important;
  border:1px solid var(--border)!important;border-radius:10px!important;
  color:var(--ink)!important;
}
.stButton>button,.stDownloadButton>button,.stFormSubmitButton>button{
  background:var(--glass-2);border:1px solid var(--border);
  border-radius:10px;color:var(--ink);
}
.stButton>button:hover,.stDownloadButton>button:hover{
  border-color:var(--accent);color:var(--accent)}
[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"]{
  border-color:var(--accent)}

/* Radio and tabs read as segmented pills, which is how the profile chooser
   on the static page is shaped. */
[data-testid="stRadio"] [role="radiogroup"]{gap:6px;flex-wrap:wrap}
[data-testid="stRadio"] label{
  background:var(--glass-2);border:1px solid var(--border);
  border-radius:999px;padding:5px 12px;margin:0}
[data-testid="stRadio"] label:hover{border-color:var(--accent)}
[data-testid="stTabs"] button[role="tab"]{border-radius:10px 10px 0 0}

h1,h2,h3,h4{letter-spacing:-.015em}
hr{border-color:var(--border)}
a{color:var(--accent)}
@media (prefers-reduced-motion:reduce){*{animation:none!important;
  transition:none!important}}
</style>
"""


def inject() -> None:
    """Paint the glass theme over Streamlit's chrome.

    The palette follows `theme.base` from .streamlit/config.toml rather than
    the viewer's `prefers-color-scheme`. That is not a shortcut, it is the
    only correct source: Streamlit resolves its own text, widget and chart
    colours from that server-side setting and writes them into the DOM, so a
    media query here recolours the backdrop while Streamlit keeps painting
    the other theme's text over it. The first version did exactly that and
    rendered dark text on a dark ground.

    Flip `base` to "dark" in config.toml and both halves move together.

    Call once, from the entry point, after `st.set_page_config` and before
    navigation runs. Calling it per view would re-inject the same block on
    every rerun of every page for no benefit.
    """
    try:
        base = str(st.get_option("theme.base") or "light").strip().lower()
    except Exception:
        base = "light"
    palette = DARK if base == "dark" else LIGHT
    st.markdown(CSS.replace("__PALETTE__", palette), unsafe_allow_html=True)
