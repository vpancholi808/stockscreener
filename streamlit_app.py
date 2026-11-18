"""stockscreener — Streamlit entry point.

    streamlit run streamlit_app.py

This replaces both halves of the FastAPI + Next.js stack: one Python process
that reads the same database the worker fills. The worker is still the only
component that talks to SEC; nothing is fetched while a page renders.

Navigation is declared here rather than relying on a pages/ directory, so the
sidebar labels are chosen rather than derived from filenames. The views
themselves live in views/ and each one is independently runnable.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import streamlit as st

st.set_page_config(
    page_title="stockscreener",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# The app is the surface most people actually open, so it carries the same
# glass treatment as the published static pages. Injected once here rather
# than per view: Streamlit reruns the whole script on every interaction, so a
# per-view call would re-emit the identical style block on every click.
from finmodel.web import theme  # noqa: E402

theme.inject()

# url_path is set explicitly rather than left to default off the filename, so
# the links are stable if a view is ever renamed. A company page is shareable
# as /company?ticker=MU.
nav = st.navigation([
    st.Page("views/screener.py", title="Screener", icon="📊",
            url_path="screener", default=True),
    st.Page("views/company.py", title="Company", icon="🏢",
            url_path="company"),
    st.Page("views/portfolio.py", title="Portfolio", icon="📁",
            url_path="portfolio"),
    st.Page("views/frontier.py", title="Frontier", icon="📈",
            url_path="frontier"),
    st.Page("views/methodology.py", title="Methodology", icon="📐",
            url_path="methodology"),
])
nav.run()
