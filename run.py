#!/usr/bin/env python3
"""
finmodel — one command to analyze a company.

    python run.py MU
    python run.py AAPL MSFT NVDA      (several at once = screener + dashboard)

First run will ask for your name and email once (the SEC requires it) and
remember it in a file called .finmodel_config next to this script.

Everything it produces is written to the ./reports folder and opened in your
browser automatically.
"""
import os
import sys
import json
import webbrowser
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
CONFIG = HERE / ".finmodel_config"
REPORTS = HERE / "reports"


def die(msg, hint=""):
    print(f"\n  ERROR: {msg}")
    if hint:
        print(f"  -> {hint}")
    sys.exit(1)


def check_packages():
    missing = []
    for pkg in ("numpy", "pandas", "scipy", "requests"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        die(f"missing packages: {', '.join(missing)}",
            "run this first:  pip install -r requirements.txt")


def get_user_agent():
    """SEC requires a real name + email on every request."""
    if CONFIG.exists():
        try:
            ua = json.loads(CONFIG.read_text()).get("user_agent", "")
            if "@" in ua:
                return ua
        except Exception:
            pass
    env = os.environ.get("EDGAR_USER_AGENT", "")
    if "@" in env:
        return env
    print("\n  One-time setup. The SEC requires a name and email with each request")
    print("  (they block anonymous traffic). This is stored locally and sent only to SEC.\n")
    name = input("  Your name:  ").strip()
    email = input("  Your email: ").strip()
    if "@" not in email:
        die("that doesn't look like an email address")
    ua = f"{name} {email}"
    CONFIG.write_text(json.dumps({"user_agent": ua}))
    print(f"\n  Saved. You won't be asked again.\n")
    return ua


def money(v):
    if v is None or v != v:
        return "n/a"
    a = abs(v)
    if a >= 1e12: return f"${v/1e12:.2f}T"
    if a >= 1e9:  return f"${v/1e9:.1f}B"
    if a >= 1e6:  return f"${v/1e6:.0f}M"
    return f"${v:,.0f}"


def analyze_one(ticker, ua):
    from finmodel import (ThreeStatementModel, forensic_report, tearsheet_html,
                          CapmModel, CapmConfig)
    print(f"  [{ticker}] fetching filings from SEC EDGAR ...")
    model = ThreeStatementModel.from_ticker(ticker, user_agent=ua)

    market_cap, wacc = None, None
    try:
        print(f"  [{ticker}] fetching prices, computing beta ...")
        capm = CapmModel.from_ticker(
            ticker, config=CapmConfig(risk_free_rate=0.042, equity_risk_premium=0.05),
            three_statement=model)
        if capm.wacc:
            market_cap, wacc = capm.wacc.equity_value, capm.wacc.wacc
    except Exception as e:
        print(f"  [{ticker}] note: price data unavailable ({type(e).__name__}); "
              f"skipping beta/WACC")
        capm = None

    print(f"  [{ticker}] scoring ...")
    forensic = forensic_report(model, market_cap=market_cap, wacc=wacc)

    # ---- console summary ----
    sc = model.scorecard
    r = sc["raw"]
    print(f"\n  {'='*54}")
    print(f"  {model.financials.entity_name}  ({ticker})   FY{sc['fiscal_year']}")
    print(f"  {'='*54}")
    print(f"  Health score      {sc['passed']}/{sc['total']}  ({sc['score']*100:.0f}%)")
    print(f"  Revenue           {money(r.get('revenue'))}")
    print(f"  EBITDA            {money(r.get('ebitda'))}")
    print(f"  Free cash flow    {money(r.get('fcf'))}")
    print(f"  Net debt          {money(r.get('net_debt'))}")
    if capm:
        print(f"  Beta (adjusted)   {capm.beta.adjusted_beta:.2f}")
        if capm.wacc:
            print(f"  WACC              {capm.wacc.wacc:.2%}")
    print(f"\n  Failed checks:")
    fails = [m for m in sc["metrics"].values() if not m.passed]
    if fails:
        for m in fails:
            print(f"    - {m.name}: {m.value:.2f} (needs "
                  f"{'>=' if m.higher_is_better else '<='} {m.threshold})")
    else:
        print("    none - passes all checks")
    print(f"\n  Scores:")
    for s in forensic.all_scores():
        print(f"    {s}")
    print(f"\n  {model.quality_summary()}")

    # ---- html report ----
    REPORTS.mkdir(exist_ok=True)
    out = REPORTS / f"{ticker}_tearsheet.html"
    out.write_text(tearsheet_html(model, forensic,
                                  title=f"{model.financials.entity_name} — health tearsheet"))
    print(f"\n  Report saved: {out}")
    return model, out


def screen_many(tickers, ua):
    from finmodel import (ThreeStatementModel, Screener, ScreenConfig, dashboard_html)
    models = {}
    for t in tickers:
        try:
            print(f"  [{t}] fetching ...")
            models[t] = ThreeStatementModel.from_ticker(t, user_agent=ua)
        except Exception as e:
            print(f"  [{t}] skipped: {e}")
    if not models:
        die("no companies could be loaded")
    scr = Screener.from_models(models, config=ScreenConfig())
    print(f"\n  {scr.summary()}\n")
    print(scr.table()[["health_passed", "revenue_growth", "operating_margin",
                       "fcf_margin", "roic", "passes_screen"]].to_string())
    REPORTS.mkdir(exist_ok=True)
    out = REPORTS / "screen_dashboard.html"
    out.write_text(dashboard_html(scr, title="Watchlist screen"))
    print(f"\n  Report saved: {out}")
    return out


def main():
    check_packages()
    tickers = [t.upper() for t in sys.argv[1:]]
    if not tickers:
        print(__doc__)
        tickers = [input("  Ticker to analyze (e.g. MU): ").strip().upper()]
        if not tickers[0]:
            sys.exit(0)
    ua = get_user_agent()

    try:
        if len(tickers) == 1:
            _, out = analyze_one(tickers[0], ua)
        else:
            out = screen_many(tickers, ua)
    except Exception as e:
        name, msg = type(e).__name__, str(e)
        if "403" in msg:
            die("the SEC refused the request (HTTP 403)",
                "usually a bad User-Agent. Delete .finmodel_config and re-run to "
                "re-enter your real name and email. If you're on a work VPN or "
                "restricted network, try another connection.")
        if "404" in msg:
            die(f"no SEC filings found for that ticker",
                "check the symbol. Only US-listed companies that file with the "
                "SEC are covered (foreign issuers may have partial data).")
        if "429" in msg:
            die("rate-limited by the SEC", "wait a minute and try again")
        if "not found in SEC ticker map" in msg:
            die(msg, "check the ticker symbol is a US-listed company")
        if "Connection" in name or "Timeout" in name or "Resolve" in msg:
            die("could not reach the SEC", "check your internet connection")
        raise

    try:
        webbrowser.open(f"file://{out}")
        print("  Opening in your browser ...\n")
    except Exception:
        print(f"  Open this file in your browser: {out}\n")


if __name__ == "__main__":
    main()
