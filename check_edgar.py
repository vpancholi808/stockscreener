#!/usr/bin/env python3
"""Confirm this machine can pull SEC EDGAR data, and say what to fix if not.

    python check_edgar.py
    python check_edgar.py MU AAPL

There is nothing to sign up for. EDGAR is free and open, and SEC issues no API
keys. The one rule is that every request must identify who is making it:

    User-Agent: Your Name you@email.com

Requests without a real contact are throttled and then blocked, which is why
this project refuses to send them. SEC also asks for no more than 10 requests
a second; EdgarClient paces itself at about 6.7.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
CONFIG = HERE / ".finmodel_config"


def clean(value: str) -> str:
    """Collapse whitespace. A trailing newline makes requests reject the
    header outright, which looks like an SEC block but never reaches SEC."""
    return " ".join(str(value or "").split())


def resolve_user_agent() -> tuple[str, str]:
    """Return (user_agent, where_it_came_from)."""
    env = clean(os.environ.get("EDGAR_USER_AGENT", ""))
    if "@" in env:
        return env, "EDGAR_USER_AGENT environment variable"
    if CONFIG.exists():
        try:
            ua = clean(json.loads(CONFIG.read_text()).get("user_agent", ""))
            if "@" in ua:
                return ua, str(CONFIG.name)
        except (json.JSONDecodeError, OSError):
            pass
    return "", ""


def prompt_and_save() -> str:
    print("\n  SEC requires a real name and email on every request.")
    print("  This is sent to SEC as the User-Agent header. It is stored in")
    print("  .finmodel_config, which is gitignored and never committed.\n")
    try:
        name = input("  Your name:  ").strip()
        email = input("  Your email: ").strip()
    except (EOFError, KeyboardInterrupt):
        return ""
    if "@" not in email:
        print("\n  That does not look like an email address.")
        return ""
    ua = clean(f"{name} {email}")
    CONFIG.write_text(json.dumps({"user_agent": ua}))
    print(f"\n  Saved to {CONFIG.name}")
    return ua


def main() -> int:
    tickers = [t.upper() for t in sys.argv[1:]] or ["MU"]

    print("=" * 66)
    print("SEC EDGAR ACCESS CHECK")
    print("=" * 66)

    user_agent, source = resolve_user_agent()
    if not user_agent:
        print("\n  No contact details found.")
        user_agent = prompt_and_save()
        source = str(CONFIG.name)
        if not user_agent:
            print("\n  Set one and re-run:")
            print('    export EDGAR_USER_AGENT="Your Name you@email.com"')
            return 1
    print(f"\n  Identifying as : {user_agent}")
    print(f"  Source         : {source}")

    try:
        import requests
    except ImportError:
        print("\n  requests is not installed:  pip install -r requirements.txt")
        return 1

    from finmodel.data.edgar import EdgarClient, EdgarError

    client = EdgarClient(user_agent=user_agent)

    print("\n" + "-" * 66)
    print("1. Ticker -> CIK map  (www.sec.gov)")
    print("-" * 66)
    start = time.time()
    try:
        cik = client.ticker_to_cik(tickers[0])
    except EdgarError as exc:
        return diagnose(exc)
    except requests.exceptions.RequestException as exc:
        return diagnose(exc)
    print(f"  OK  {tickers[0]} -> CIK {cik:010d}   ({time.time()-start:.1f}s)")

    print("\n" + "-" * 66)
    print("2. Company facts  (data.sec.gov)")
    print("-" * 66)
    ok = 0
    for ticker in tickers:
        start = time.time()
        try:
            facts = client.company_facts(ticker)
        except (EdgarError, requests.exceptions.RequestException) as exc:
            print(f"  FAIL  {ticker}: {exc}")
            continue
        name = facts.get("entityName", "?")
        concepts = len(facts.get("facts", {}).get("us-gaap", {}))
        print(f"  OK  {ticker:6s} {name[:38]:38s} {concepts:4d} XBRL concepts"
              f"  ({time.time()-start:.1f}s)")
        ok += 1

    print("\n" + "=" * 66)
    if ok == len(tickers):
        print("EDGAR ACCESS IS WORKING")
        print("=" * 66)
        print("\n  Pull real data into the screener:")
        print("    python worker.py %s" % " ".join(tickers))
        print("    python worker.py --sp500        # S&P 500, ~20 min")
        print("    python worker.py --universe     # whole market, ~1.5 GB")
        print("\n  Then:  streamlit run streamlit_app.py")
        return 0
    print("SOME REQUESTS FAILED — see the messages above")
    print("=" * 66)
    return 1


def diagnose(exc: Exception) -> int:
    """Turn a failure into the specific thing to do about it."""
    text = str(exc)
    low = text.lower()
    print(f"  FAILED: {text}")
    print("\n" + "=" * 66)

    # Connection-level problems are checked FIRST. A blocked proxy returns its
    # own "403 Forbidden" on the CONNECT tunnel, which has nothing to do with
    # SEC and is not an HTTP response from SEC at all. Matching on the status
    # code first would send you off chasing your User-Agent for a network fault.
    network = ("tunnel", "proxy", "max retries", "connection", "timed out",
               "timeout", "resolve", "ssl", "certificate", "getaddrinfo")
    if any(s in low for s in network):
        print("  The request never reached SEC — it failed at the network")
        print("  layer, so this is not an EDGAR or User-Agent problem.")
        print("\n  Check that this machine can open https://www.sec.gov in a")
        print("  browser. Corporate networks, VPNs, and sandboxed or cloud")
        print("  environments frequently block it.")
        if "tunnel" in low or "proxy" in low:
            print("\n  'Tunnel connection failed' specifically means a proxy")
            print("  refused the connection before SEC was ever contacted.")
    elif "403" in text:
        print("  403 from SEC means the request's identity was rejected.")
        print("  Set a real name and email:")
        print('    export EDGAR_USER_AGENT="Your Name you@email.com"')
    elif "429" in text:
        print("  429 means rate limited. SEC allows ~10 requests/second.")
        print("  Wait a few minutes. If it persists, slow the client down:")
        print("    EdgarClient(user_agent=..., min_interval=0.3)")
    elif "404" in text:
        print("  404 means that ticker has no EDGAR filings under that")
        print("  symbol. Foreign issuers and funds often do not.")
    else:
        print("  Unexpected failure. Check https://www.sec.gov/ is up.")
    print("=" * 66)
    return 1


if __name__ == "__main__":
    sys.exit(main())
