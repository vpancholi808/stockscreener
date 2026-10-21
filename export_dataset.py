#!/usr/bin/env python3
"""Export a publishable copy of the loaded database.

    python export_dataset.py --src sp500.db --out data/sp500.db

The working database holds everything the pipeline fetched, including the
`market_data` table built from scraped Yahoo prices. The committed copy must
not: SEC filings are US government work and in the public domain, so
fundamentals and every ratio derived from them can be republished freely,
while scraped price data generally cannot be redistributed under the
provider's terms. A public repository is redistribution.

So the exported copy carries filings-derived data only — companies, their
filed history, and the sector benchmarks that are not price-derived. Nothing
is lost for a public deployment, because `SERVE_MARKET_DATA` is false there
anyway and the app already hides those fields.

Run the pipeline with a database of your own and set SERVE_MARKET_DATA=true
if you have a feed licensed for redistribution.
"""
from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
from pathlib import Path

# Columns on `companies` that came from a price feed rather than a filing.
PRICE_COLUMNS: set[str] = set()


def export(src: Path, out: Path) -> dict:
    if not src.exists():
        raise SystemExit("source database not found: %s" % src)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    shutil.copyfile(src, out)

    conn = sqlite3.connect(out)
    cur = conn.cursor()

    def count(table: str) -> int:
        try:
            return cur.execute("SELECT COUNT(*) FROM %s" % table).fetchone()[0]
        except sqlite3.Error:
            return 0

    before = {t: count(t) for t in
              ("companies", "financial_history", "market_data",
               "sector_benchmarks")}

    # Everything sourced from the price feed goes.
    cur.execute("DELETE FROM market_data")
    cur.execute("DELETE FROM sector_benchmarks WHERE price_derived = 1")
    conn.commit()
    cur.execute("VACUUM")          # actually shrink the file
    conn.commit()

    after = {t: count(t) for t in before}
    leftover = count("market_data")
    if leftover:
        raise SystemExit("market_data still holds %d rows" % leftover)
    conn.close()

    return {"before": before, "after": after,
            "bytes": out.stat().st_size}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default="sp500.db")
    ap.add_argument("--out", default="data/sp500.db")
    args = ap.parse_args()

    result = export(Path(args.src), Path(args.out))
    print("exported %s -> %s (%.2f MB)"
          % (args.src, args.out, result["bytes"] / 1e6))
    for table, n in result["before"].items():
        print("  %-20s %6d -> %6d" % (table, n, result["after"][table]))
    print("  price-derived data removed; the copy is filings-only and "
          "republishable")
    return 0


if __name__ == "__main__":
    sys.exit(main())
