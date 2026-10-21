#!/usr/bin/env python3
"""
Data worker — the only thing that talks to SEC. Run it on a schedule.

    python worker.py --universe              whole market via SEC bulk data
    python worker.py AAPL MSFT NVDA MU       specific tickers via the EDGAR API
    python worker.py --seed                  demo data, no network needed

Never call this from a web request. SEC rate-limits to roughly 10 requests per
second and the bulk file is ~1.5 GB; the site reads only from the database.

A sensible cron, filings being a slow-moving dataset:
    0 6 * * *   cd /srv/finmodel && ./venv/bin/python worker.py --universe
"""
import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
CONFIG = HERE / ".finmodel_config"


def get_user_agent() -> str:
    env = os.environ.get("EDGAR_USER_AGENT", "")
    if "@" in env:
        return env
    if CONFIG.exists():
        try:
            ua = json.loads(CONFIG.read_text()).get("user_agent", "")
            if "@" in ua:
                return ua
        except Exception:
            pass
    print("\n  SEC requires a name and email on every request.")
    name = input("  Your name:  ").strip()
    email = input("  Your email: ").strip()
    if "@" not in email:
        sys.exit("  That doesn't look like an email address.")
    ua = f"{name} {email}"
    CONFIG.write_text(json.dumps({"user_agent": ua}))
    return ua


def main():
    ap = argparse.ArgumentParser(description="finmodel data worker")
    ap.add_argument("tickers", nargs="*", help="specific tickers to load")
    ap.add_argument("--universe", action="store_true",
                    help="load the whole market from SEC bulk data")
    ap.add_argument("--seed", action="store_true",
                    help="load demo data (no network required)")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap companies processed (try --limit 500 first)")
    ap.add_argument("--min-revenue", type=float, default=50e6,
                    help="skip companies below this revenue (default $50M)")
    ap.add_argument("--database-url", default=None,
                    help="overrides DATABASE_URL")
    ap.add_argument("--no-prices", action="store_true",
                    help="skip the price fetch and valuation step")
    ap.add_argument("--sp500", action="store_true",
                    help="load the current S&P 500 constituents (the default "
                         "universe for this project)")
    ap.add_argument("--no-profiles", action="store_true",
                    help="skip the Yahoo profile pass (forward P/E, analyst "
                         "targets, 52-week range)")
    args = ap.parse_args()

    import warnings
    warnings.filterwarnings("ignore")

    from finmodel.web.store import Database

    db = Database(args.database_url) if args.database_url else Database()
    db.init_schema()
    print("  database: %s" % ("postgres" if db.is_postgres else "sqlite"))

    if args.seed:
        from finmodel.web.seed import seed, seed_prices
        result = seed(db)
        seed_prices(db)
    elif args.universe:
        from finmodel.web.loader import load_universe
        result = load_universe(get_user_agent(), db=db, limit=args.limit,
                               min_revenue=args.min_revenue)
    elif args.sp500:
        from finmodel.data.sp500 import fetch_sp500
        from finmodel.web.loader import load_tickers
        ua = get_user_agent()
        members = fetch_sp500(user_agent=ua)
        print("  S&P 500: %d constituents" % len(members))
        if args.limit:
            members = members[:args.limit]
            print("  limited to the first %d" % len(members))
        # GICS sectors come with the constituent list and are the standard
        # classification for this index, so they are used in preference to
        # deriving a sector from the filer's SIC code.
        result = load_tickers([m["ticker"] for m in members], ua, db=db,
                              sectors={m["ticker"]: m.get("gics_sector")
                                       for m in members})
    elif args.tickers:
        from finmodel.web.loader import load_tickers
        result = load_tickers([t.upper() for t in args.tickers],
                              get_user_agent(), db=db)
    else:
        ap.print_help()
        return

    # Valuation and sector benchmarks both depend on the company set, so they
    # are rebuilt after every load rather than drifting out of date.
    if not args.seed and not args.no_prices:
        from finmodel.web.loader import attach_valuation
        attach_valuation(db)
        # Profiles are per-ticker and only affordable for a bounded universe,
        # so they run for the S&P 500 or an explicit ticker list, never for
        # the whole market.
        if not args.no_profiles and not args.universe:
            from finmodel.web.loader import attach_profiles
            attach_profiles(db)
    from finmodel.web.loader import compute_benchmarks
    compute_benchmarks(db)
    db.compute_sector_percentiles()

    stats = db.stats()
    print("\n  loaded %d, skipped %d" % (result["loaded"], result["skipped"]))
    print("  database now holds %d companies across %d sectors"
          % (stats["companies"], stats["sectors"]))
    print("  %d pass the default screen" % (stats["passing"] or 0))


if __name__ == "__main__":
    main()
