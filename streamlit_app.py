"""stockscreener — Streamlit entry point.

    streamlit run streamlit_app.py

Identical to `main.py`; both call `finmodel.web.app.run`, so there is one
copy of the application and the two cannot drift. This filename is kept
because existing Streamlit Community Cloud deployments point at it.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from finmodel.web.app import run  # noqa: E402

run()
