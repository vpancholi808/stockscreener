#!/usr/bin/env python3
"""
Screen the ENTIRE US public market, not a watchlist.

    python run_universe.py                  full market (first run downloads ~1.5 GB)
    python run_universe.py --limit 500      try it on 500 companies first
    python run_universe.py --sector Semiconductors
    python run_universe.py --report         rebuild the HTML dashboard only

How it works:
  1. Downloads SEC's bulk companyfacts.zip ONCE (~1.5 GB) - every filer's
     fundamentals in one file, instead of 8,000 separate API calls.
  2. Streams it one company at a time, so memory stays flat.
  3. Fetches prices from yfinance in BATCHES (never the slow per-ticker .info).
  4. Writes results to universe.db (SQLite) as it goes, so it is resumable -
     Ctrl-C and re-run and it picks up where it stopped.

First full run: expect ~10-20 minutes for the download plus a few minutes of
parsing. Re-runs reuse the cached file.
"""
import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
CONFIG = HERE / ".finmodel_config"


def get_user_agent():
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
    print("\n  One-time setup. SEC requires a name and email with each request.")
    print("  Stored locally, sent only to sec.gov.\n")
    name = input("  Your name:  ").strip()
    email = input("  Your email: ").strip()
    if "@" not in email:
        print("  That doesn't look like an email address.")
        sys.exit(1)
    ua = "%s %s" % (name, email)
    CONFIG.write_text(json.dumps({"user_agent": ua}))
    return ua


def main():
    ap = argparse.ArgumentParser(description="Market-wide equity screen")
    ap.add_argument("--limit", type=int, default=None,
                    help="only process N companies (use for a first trial)")
    ap.add_argument("--db", default="universe.db")
    ap.add_argument("--min-revenue", type=float, default=50e6,
                    help="skip companies below this revenue (default $50M)")
    ap.add_argument("--no-prices", action="store_true",
                    help="skip yfinance price fetching")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore previous results and start over")
    ap.add_argument("--sector", default=None, help="show one sector's table")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--report", action="store_true",
                    help="only rebuild the HTML report from existing results")
    args = ap.parse_args()

    from finmodel import UniverseScreen, ScreenConfig

    if args.report:
        screen = UniverseScreen(Path(args.db))
    else:
        ua = get_user_agent()
        print("\n  Screening the US public market ...")
        screen = UniverseScreen.run(
            user_agent=ua, db_path=args.db, config=ScreenConfig(),
            limit=args.limit, min_revenue=args.min_revenue,
            resume=not args.fresh, fetch_prices=not args.no_prices)

    s = screen.stats()
    print("\n" + "=" * 70)
    print("  UNIVERSE: %d companies across %d sectors | %d pass the screen"
          % (s["companies"], s["sectors"], s["passing"]))
    print("=" * 70)

    print("\n  BY SECTOR")
    print(screen.sector_summary().round(3).to_string(index=False))

    if args.sector:
        print("\n  %s — top %d" % (args.sector, args.top))
        df = screen.by_sector(args.sector, limit=args.top)
    else:
        print("\n  TOP %d PASSING THE SCREEN" % args.top)
        df = screen.passing(limit=args.top)

    cols = [c for c in ["ticker", "name", "sector", "health_passed",
                        "revenue_growth", "operating_margin", "fcf_margin",
                        "roic", "piotroski", "composite_pct"] if c in df.columns]
    if len(df):
        show = df[cols].copy()
        if "name" in show:
            show["name"] = show["name"].str.slice(0, 24)
        print(show.round(3).to_string(index=False))
    else:
        print("  (nothing matched - try --min-revenue 0 or a looser screen)")

    print("\n  Results database: %s" % args.db)
    print("  Query it directly, e.g.:")
    print("    sqlite3 %s \"SELECT ticker,roic FROM results "
          "WHERE sector='Semiconductors' ORDER BY roic DESC LIMIT 10\"" % args.db)
    print()


if __name__ == "__main__":
    main()
