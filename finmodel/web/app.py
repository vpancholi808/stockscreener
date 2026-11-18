"""The application itself, so every entry point runs the same thing.

`main.py` and `streamlit_app.py` both call `run()`. Keeping the real setup
here rather than in either file means the two can never drift apart — a
second copy of st.navigation is exactly the kind of thing that gets updated
in one place and not the other.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]

PAGES = [
    ("views/screener.py", "Screener", "📊", "screener", True),
    ("views/company.py", "Company", "🏢", "company", False),
    ("views/portfolio.py", "Portfolio", "📁", "portfolio", False),
    ("views/frontier.py", "Portfolio builder", "📈", "frontier", False),
    ("views/methodology.py", "Methodology", "📐", "methodology", False),
]


def run() -> None:
    """Configure, theme and start the app."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    st.set_page_config(
        page_title="stockscreener",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Injected once here rather than per view: Streamlit reruns the whole
    # script on every interaction, so a per-view call would re-emit the
    # identical style block on every click.
    from . import theme
    theme.inject()

    # url_path is set explicitly rather than derived from the filename, so
    # links stay stable if a view is ever renamed. A company page is
    # shareable as /company?ticker=MU.
    st.navigation([
        st.Page(str(ROOT / path), title=title, icon=icon,
                url_path=url, default=default)
        for path, title, icon, url, default in PAGES
    ]).run()
