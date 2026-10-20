"""Seed the web database so the stack runs with no network access.

Loads a small set of real companies (figures taken from published annual
reports) plus a spread of generated companies across sectors, so the screener,
filters, sector ranking and detail pages all have something to exercise.

    python -m finmodel.web.seed

Replace with the real worker once you have network access:

    python worker.py --universe          # whole market via SEC bulk data
    python worker.py AAPL MSFT NVDA      # specific tickers via the EDGAR API
"""
from __future__ import annotations

import random
from datetime import date, timedelta

from ..model import ThreeStatementModel
from .loader import model_to_rows
from .store import Database

M = 1e6
FILED = {2020: "2021-02-15", 2021: "2022-02-15", 2022: "2023-02-15",
         2023: "2024-02-15", 2024: "2025-02-15", 2025: "2026-02-15"}


def _period_start(year: int, fy_end: str) -> str:
    """Start date one full year before the fiscal year end.

    The statement parser requires a 350-380 day span for duration concepts
    (that's how it tells an annual figure from a quarterly one), so the start
    must be derived from the actual year end, not hardcoded to January."""
    end = date(year, int(fy_end[:2]), int(fy_end[3:]))
    return (end - timedelta(days=364)).isoformat()


def _dur(vals, fy_end="12-31"):
    return {"units": {"USD": [
        {"val": v, "fy": y, "fp": "FY", "form": "10-K",
         "start": _period_start(y, fy_end), "end": "%d-%s" % (y, fy_end),
         "filed": FILED.get(y, "%d-02-15" % (y + 1))}
        for y, v in vals.items()]}}


def _inst(vals, fy_end="12-31"):
    return {"units": {"USD": [
        {"val": v, "fy": y, "fp": "FY", "form": "10-K",
         "end": "%d-%s" % (y, fy_end), "filed": FILED.get(y, "%d-02-15" % (y + 1))}
        for y, v in vals.items()]}}


def _shr(vals, fy_end="12-31"):
    return {"units": {"shares": [
        {"val": v, "fy": y, "fp": "FY", "form": "10-K",
         "start": _period_start(y, fy_end), "end": "%d-%s" % (y, fy_end),
         "filed": FILED.get(y, "%d-02-15" % (y + 1))}
        for y, v in vals.items()]}}


def _facts(name, years, rev, cogs, oi, ni, interest, tax, cash, sti, recv, inv,
           ca, ppe, ta, cl, std, ltd, tl, eq, re_, ocf, capex, da, shares,
           fy_end="12-31"):
    z = lambda v: dict(zip(years, v))
    return {"entityName": name, "facts": {"us-gaap": {
        "Revenues": _dur(z(rev), fy_end),
        "CostOfGoodsAndServicesSold": _dur(z(cogs), fy_end),
        "OperatingIncomeLoss": _dur(z(oi), fy_end),
        "NetIncomeLoss": _dur(z(ni), fy_end),
        "InterestExpense": _dur(z(interest), fy_end),
        "IncomeTaxExpenseBenefit": _dur(z(tax), fy_end),
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest":
            _dur(z([n + t for n, t in zip(ni, tax)]), fy_end),
        "CashAndCashEquivalentsAtCarryingValue": _inst(z(cash), fy_end),
        "ShortTermInvestments": _inst(z(sti), fy_end),
        "AccountsReceivableNetCurrent": _inst(z(recv), fy_end),
        "InventoryNet": _inst(z(inv), fy_end),
        "AssetsCurrent": _inst(z(ca), fy_end),
        "PropertyPlantAndEquipmentNet": _inst(z(ppe), fy_end),
        "Assets": _inst(z(ta), fy_end),
        "LiabilitiesCurrent": _inst(z(cl), fy_end),
        "DebtCurrent": _inst(z(std), fy_end),
        "LongTermDebtNoncurrent": _inst(z(ltd), fy_end),
        "Liabilities": _inst(z(tl), fy_end),
        "StockholdersEquity": _inst(z(eq), fy_end),
        "RetainedEarningsAccumulatedDeficit": _inst(z(re_), fy_end),
        "NetCashProvidedByUsedInOperatingActivities": _dur(z(ocf), fy_end),
        "PaymentsToAcquirePropertyPlantAndEquipment": _dur(z(capex), fy_end),
        "DepreciationDepletionAndAmortization": _dur(z(da), fy_end),
        "WeightedAverageNumberOfDilutedSharesOutstanding": _shr(z(shares), fy_end),
    }}}


# --- real companies, figures from published annual reports (USD millions) ----

REAL = [
    # Micron FY2023-FY2025 (fiscal year ends late August)
    (723125, "MU", "Micron Technology, Inc.", "Semiconductors", _facts(
        "Micron Technology, Inc.", [2023, 2024, 2025],
        rev=[15540 * M, 25111 * M, 37380 * M],
        cogs=[16956 * M, 19545 * M, 22428 * M],
        oi=[-5745 * M, 1300 * M, 9000 * M],
        ni=[-5833 * M, 778 * M, 8539 * M],
        interest=[533 * M, 516 * M, 490 * M],
        tax=[-37 * M, 388 * M, 1063 * M],
        cash=[8577 * M, 7041 * M, 8734 * M],
        sti=[1400 * M, 1073 * M, 3206 * M],
        recv=[3308 * M, 6615 * M, 7436 * M],
        inv=[8387 * M, 8875 * M, 8355 * M],
        ca=[21289 * M, 25287 * M, 28887 * M],
        ppe=[38284 * M, 39943 * M, 46900 * M],
        ta=[64254 * M, 69416 * M, 82798 * M],
        cl=[4810 * M, 10163 * M, 11500 * M],
        std=[324 * M, 1000 * M, 600 * M],
        ltd=[13056 * M, 11237 * M, 11533 * M],
        tl=[20134 * M, 24285 * M, 28633 * M],
        eq=[44120 * M, 45131 * M, 54165 * M],
        re_=[32000 * M, 32265 * M, 40282 * M],
        ocf=[1559 * M, 8507 * M, 17525 * M],
        capex=[7676 * M, 8386 * M, 15857 * M],
        da=[7756 * M, 7780 * M, 8352 * M],
        shares=[1093 * M, 1118 * M, 1131 * M], fy_end="08-28")),
]


# --- generated companies across sectors, for a realistic screening surface ---

SECTOR_PROFILES = {
    "Software & IT Services": dict(gm=(0.68, 0.85), om=(0.10, 0.34), growth=(0.05, 0.35),
                                   capex=(0.02, 0.05), lev=(0.0, 0.5)),
    "Semiconductors":         dict(gm=(0.35, 0.68), om=(0.08, 0.40), growth=(-0.10, 0.45),
                                   capex=(0.08, 0.30), lev=(0.1, 0.8)),
    "Pharma & Biotech":       dict(gm=(0.60, 0.85), om=(0.05, 0.35), growth=(-0.05, 0.25),
                                   capex=(0.03, 0.09), lev=(0.1, 1.0)),
    "Retail":                 dict(gm=(0.20, 0.42), om=(0.02, 0.11), growth=(-0.02, 0.14),
                                   capex=(0.02, 0.07), lev=(0.3, 1.6)),
    "Banking":                dict(gm=(0.55, 0.80), om=(0.20, 0.42), growth=(0.0, 0.14),
                                   capex=(0.01, 0.03), lev=(0.2, 1.2)),
    "Utilities":              dict(gm=(0.30, 0.55), om=(0.14, 0.28), growth=(-0.01, 0.07),
                                   capex=(0.12, 0.28), lev=(0.9, 2.2)),
    "Oil Refining":           dict(gm=(0.10, 0.35), om=(0.04, 0.26), growth=(-0.18, 0.30),
                                   capex=(0.07, 0.20), lev=(0.2, 1.3)),
    "Medical Devices":        dict(gm=(0.50, 0.74), om=(0.10, 0.28), growth=(0.02, 0.18),
                                   capex=(0.03, 0.08), lev=(0.1, 0.9)),
    "Transportation":         dict(gm=(0.18, 0.38), om=(0.05, 0.18), growth=(-0.04, 0.12),
                                   capex=(0.06, 0.18), lev=(0.4, 1.8)),
    "Food & Beverage":        dict(gm=(0.28, 0.50), om=(0.07, 0.19), growth=(0.0, 0.09),
                                   capex=(0.03, 0.08), lev=(0.3, 1.5)),
}

FIRST = ["Vertex", "Northwind", "Clearwater", "Aldridge", "Brightline", "Kestrel",
         "Summit", "Ironwood", "Lakeshore", "Meridian", "Calder", "Oakfield",
         "Harborview", "Stonebridge", "Westvale", "Pinnacle", "Fairmount",
         "Granite", "Linden", "Ashford", "Beaumont", "Crestwood", "Dunmore",
         "Eastgate", "Foxglove", "Gladstone", "Hollis", "Inverness", "Juniper",
         "Kingsley", "Langford", "Marbury", "Norwood", "Oakmont", "Pemberton"]
SUFFIX = {"Software & IT Services": ["Systems", "Software", "Labs", "Technologies"],
          "Semiconductors": ["Micro", "Semiconductor", "Silicon", "Devices"],
          "Pharma & Biotech": ["Therapeutics", "Biosciences", "Pharma", "Bio"],
          "Retail": ["Retail Group", "Stores", "Markets", "Brands"],
          "Banking": ["Financial", "Bancorp", "Capital", "Trust"],
          "Utilities": ["Energy", "Power", "Utilities", "Electric"],
          "Oil Refining": ["Petroleum", "Resources", "Energy Partners", "Refining"],
          "Medical Devices": ["Medical", "Health Systems", "Surgical", "Diagnostics"],
          "Transportation": ["Logistics", "Freight", "Transport", "Carriers"],
          "Food & Beverage": ["Foods", "Beverage Co", "Provisions", "Mills"]}


def _generate(n_per_sector: int, seed: int = 7):
    rng = random.Random(seed)
    years = [2020, 2021, 2022, 2023, 2024]   # five fiscal years
    out, cik, used = [], 900000, set()
    for sector, p in SECTOR_PROFILES.items():
        for _ in range(n_per_sector):
            cik += 1
            while True:
                nm = "%s %s" % (rng.choice(FIRST), rng.choice(SUFFIX[sector]))
                if nm not in used:
                    used.add(nm)
                    break
            parts = nm.split()
            ticker = (parts[0][:3] + parts[1][:1]).upper()
            while ticker in {t for _, t, *_ in out}:
                ticker = parts[0][:2].upper() + rng.choice("ABCDEFGHJKLMNP") + \
                    rng.choice("ABCDEFGHJKLMNP")

            g = rng.uniform(*p["growth"])
            gm = rng.uniform(*p["gm"])
            om = min(rng.uniform(*p["om"]), gm - 0.02)
            cx = rng.uniform(*p["capex"])
            lev = rng.uniform(*p["lev"])
            r0 = rng.uniform(80, 14000) * M

            rev = [r0 * (1 + g) ** i for i in range(len(years))]
            cogs = [r * (1 - gm) for r in rev]
            oi = [r * om for r in rev]
            ni = [o * rng.uniform(0.62, 0.80) for o in oi]
            tax = [max(0.0, o - n) for o, n in zip(oi, ni)]
            ltd = [r * lev * 0.85 for r in rev]
            std = [r * lev * 0.15 for r in rev]
            interest = [(l + s) * 0.055 for l, s in zip(ltd, std)]
            ocf = [n * rng.uniform(0.85, 1.65) for n in ni]
            capex = [r * cx for r in rev]
            da = [r * rng.uniform(0.03, 0.09) for r in rev]
            cash = [r * rng.uniform(0.04, 0.35) for r in rev]
            sti = [r * rng.uniform(0.0, 0.10) for r in rev]
            recv = [r * rng.uniform(0.08, 0.20) for r in rev]
            invv = [r * (0.02 if sector in ("Software & IT Services", "Banking")
                         else rng.uniform(0.08, 0.22)) for r in rev]
            ca = [c + s + rc + iv + r * 0.04
                  for c, s, rc, iv, r in zip(cash, sti, recv, invv, rev)]
            ppe = [r * rng.uniform(0.15, 0.95) for r in rev]
            ta = [c + p_ + r * rng.uniform(0.15, 0.45)
                  for c, p_, r in zip(ca, ppe, rev)]
            cl = [r * rng.uniform(0.12, 0.30) + s for r, s in zip(rev, std)]
            tl = [c + l + r * 0.05 for c, l, r in zip(cl, ltd, rev)]
            # Equity as a realistic share of assets. An earlier version used
            # max(assets - liabilities, assets*0.08), whose floor produced
            # near-zero book values and ROEs in the hundreds of percent.
            equity_ratio = rng.uniform(0.25, 0.60)
            eq = [a * equity_ratio for a in ta]
            tl = [a - e for a, e in zip(ta, eq)]
            re_ = [e * rng.uniform(0.35, 0.80) for e in eq]
            shares = [rng.uniform(20, 900) * M] * len(years)

            out.append((cik, ticker, nm, sector, _facts(
                nm, years, rev, cogs, oi, ni, interest, tax, cash, sti, recv,
                invv, ca, ppe, ta, cl, std, ltd, tl, eq, re_, ocf, capex, da,
                shares)))
    return out


def seed(db: Database | None = None, n_per_sector: int = 14,
         verbose: bool = True) -> dict:
    db = db or Database()
    db.init_schema()
    rows, hist, failed = [], [], 0
    for cik, ticker, name, sector, facts in REAL + _generate(n_per_sector):
        try:
            model = ThreeStatementModel.from_facts(facts, ticker=ticker, n_years=5)
            c, h = model_to_rows(model, cik, ticker, name, sector)
            rows.append(c)
            hist.extend(h)
        except Exception as e:
            failed += 1
            if verbose:
                print("  skipped %s: %s" % (ticker, e))
    db.upsert_companies(rows)
    db.upsert_history(hist)
    db.compute_sector_percentiles()
    db.record_run(len(rows), failed, "seed", "demo data — real + generated")
    if verbose:
        print("  seeded %d companies (%d history rows, %d failed)"
              % (len(rows), len(hist), failed))
    return {"loaded": len(rows), "skipped": failed}


if __name__ == "__main__":
    import sys
    import warnings
    warnings.filterwarnings("ignore")
    seed()
    print("  done. start the API with:  uvicorn finmodel.web.api:app --reload")


def seed_prices(db: Database | None = None, verbose: bool = True) -> int:
    """Assign plausible demo prices so the valuation layer has something to work
    on without a live feed.

    Prices are derived from each company's own earnings and a sector-typical
    multiple plus noise, so the resulting P/E and EV/EBITDA distributions look
    like a real market rather than uniform noise. These are NOT real prices and
    are flagged as source='demo' in the database."""
    import random
    from ..web.loader import attach_valuation

    db = db or Database()
    conn = db.connect()
    cur = conn.cursor()
    cur.execute("SELECT cik, ticker, sector, revenue, net_margin, "
                "shares_diluted FROM companies")
    rows = db._rows_to_dicts(cur)
    if not rows:
        return 0

    # Sector-typical earnings multiples, roughly in line with long-run norms.
    typical_pe = {
        "Software & IT Services": 34, "Semiconductors": 26,
        "Pharma & Biotech": 19, "Medical Devices": 25, "Retail": 17,
        "Banking": 12, "Utilities": 18, "Oil Refining": 11,
        "Transportation": 16, "Food & Beverage": 19,
    }
    rng = random.Random(21)
    prices = {}
    for r in rows:
        shares = r.get("shares_diluted")
        revenue = r.get("revenue")
        margin = r.get("net_margin")
        if not shares or not revenue or margin is None:
            continue
        earnings = revenue * margin
        if earnings <= 0:
            # Loss-makers get priced off sales instead, as the market does.
            cap = revenue * rng.uniform(0.8, 3.5)
        else:
            base = typical_pe.get(r.get("sector"), 18)
            cap = earnings * base * rng.uniform(0.55, 1.75)
        prices[r["ticker"]] = cap / shares

    n = attach_valuation(db, prices=prices, source="demo", verbose=False)
    if verbose:
        print("  demo prices + valuation for %d companies" % n)
    return n
