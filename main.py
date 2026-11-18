#!/usr/bin/env python3
"""stockscreener — run the website.

    streamlit run main.py

Everything the project does is reachable from here: the S&P 500 screener,
company statements and peer comparison, the portfolio builder (risk profile
to allocation, on the efficient frontier and the capital allocation line),
and the methodology.

`streamlit_app.py` is the same application under its original filename,
because Streamlit Community Cloud deployments point at it. Both call
`finmodel.web.app.run`, so there is one copy of the app and no way for the
two entry points to drift.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from finmodel.web.app import run  # noqa: E402

run()
