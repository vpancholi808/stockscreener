"""XBRL tag resolution — especially revenue, which is not first-wins.

This suite exists because of a specific real-data failure. A full S&P 500
run produced an operating margin of 9,587% for Essex Property Trust:
revenue came back as $9.38M against $899M of operating income. Every REIT
and insurer in the index was distorted the same way.

The cause is accounting, not parsing. `RevenueFromContractWithCustomer...`
is ASC 606 revenue. A property trust books rental income under ASC 842 and
an insurer books premiums under ASC 944, so for them that tag holds a sliver
of the total while `Revenues` holds all of it. Resolving revenue by taking
the first tag that carries data picks whichever the chain happens to list
first; taking the largest picks the most complete one, because every
alternative concept is a subset of total revenue.
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

from finmodel.data.concepts import (ALL_MAPS, INCOME_STATEMENT, SUBSET_MAX)
from finmodel.statements import _merge_by_max, _merge_by_priority

fails = []


def check(name, cond, detail=""):
    print("  [%s] %s%s" % ("OK" if cond else "FAIL", name,
                           (" — " + detail) if detail else ""))
    if not cond:
        fails.append(name)


def dur(val, fy, days=365):
    from datetime import date, timedelta
    end = date(fy, 12, 31)
    return {"val": val, "fy": fy, "fp": "FY", "form": "10-K",
            "end": end.isoformat(),
            "start": (end - timedelta(days=days)).isoformat()}


def facts_for(**tags):
    return {"facts": {"us-gaap": {
        tag: {"units": {"USD": rows}} for tag, rows in tags.items()}}}


REV = INCOME_STATEMENT["revenue"]

print("=" * 66)
print("1) THE ESSEX CASE")
print("=" * 66)

# Numbers from the real run: the ASC 606 slice, and the true total.
reit = facts_for(
    Revenues=[dur(1_800_000_000, 2025)],
    RevenueFromContractWithCustomerExcludingAssessedTax=[dur(9_380_000, 2025)])

got = _merge_by_max(reit, REV, False)[2025]
check("a REIT resolves to total revenue, not the ASC 606 slice",
      got == 1_800_000_000, "$%.2fB" % (got / 1e9))

operating_income = 899_320_000
margin = operating_income / got
check("the resulting operating margin is sane",
      0 < margin < 1, "%.1f%% (was 9,587%%)" % (margin * 100))

# The defect was order-dependent; the fix must not be.
reversed_chain = list(reversed(REV))
check("max-merge is independent of chain order",
      _merge_by_max(reit, reversed_chain, False)[2025] == 1_800_000_000)
check("first-wins on the old order reproduces the bug",
      _merge_by_priority(
          reit, ["RevenueFromContractWithCustomerExcludingAssessedTax",
                 "Revenues"], False)[2025] == 9_380_000,
      "which is why revenue is not resolved that way")

print()
print("=" * 66)
print("2) THE SAME SHAPE FOR INSURERS AND BANKS")
print("=" * 66)

insurer = facts_for(
    Revenues=[dur(70_000_000_000, 2025)],
    RevenueFromContractWithCustomerExcludingAssessedTax=[dur(2_440_000_000, 2025)])
check("an insurer resolves to total revenue",
      _merge_by_max(insurer, REV, False)[2025] == 70_000_000_000)

bank = facts_for(
    RevenuesNetOfInterestExpense=[dur(50_000_000_000, 2025)],
    RevenueFromContractWithCustomerExcludingAssessedTax=[dur(8_000_000_000, 2025)])
check("a bank's net-of-interest revenue is in the chain",
      _merge_by_max(bank, REV, False)[2025] == 50_000_000_000)

for tag in ("RealEstateRevenueNet", "OperatingLeaseLeaseIncome",
            "RevenuesNetOfInterestExpense"):
    check("chain includes %s" % tag, tag in REV)

print()
print("=" * 66)
print("3) ORDINARY COMPANIES ARE UNAFFECTED")
print("=" * 66)

# Micron: tags only ASC 606 revenue, and that figure is correct.
micron = facts_for(
    RevenueFromContractWithCustomerExcludingAssessedTax=[dur(37_380_000_000, 2025)])
check("a single-tag filer is unchanged",
      _merge_by_max(micron, REV, False)[2025] == 37_380_000_000,
      "$37.38B, as Micron files it")

# Several years, several tags, no contradiction.
multi = facts_for(
    Revenues=[dur(100.0, 2023), dur(110.0, 2024)],
    SalesRevenueNet=[dur(90.0, 2023)])
resolved = _merge_by_max(multi, REV, False)
check("every year is resolved", set(resolved) == {2023, 2024})
check("the larger candidate wins per year", resolved[2023] == 100.0)

check("an empty payload yields nothing",
      _merge_by_max(facts_for(), REV, False) == {})

print()
print("=" * 66)
print("4) MAX-SELECTION IS SCOPED TO REVENUE ONLY")
print("=" * 66)

check("revenue is the only max-selected item", SUBSET_MAX == {"revenue"},
      str(SUBSET_MAX))
check("every max-selected item exists in the maps",
      all(any(item in m for m in ALL_MAPS.values()) for item in SUBSET_MAX))

# Costs and expenses must stay first-wins: there is no subset argument for
# them, and taking the largest would silently prefer the broadest expense
# definition available.
for item in ("cogs", "operating_income", "net_income"):
    check("%s is not max-selected" % item, item not in SUBSET_MAX)

# A multi-year duration must still be excluded by the period filter, so
# "largest" cannot reach for a cumulative figure.
cumulative = facts_for(Revenues=[dur(1_000.0, 2025),
                                 dur(5_000.0, 2025, days=1825)])
got = _merge_by_max(cumulative, REV, False)
check("a five-year duration is not eligible to win on size",
      got.get(2025) == 1_000.0, str(got))

print()
if fails:
    print("FAILED (%d): %s" % (len(fails), "; ".join(fails)))
    sys.exit(1)
print("test_concepts: all checks passed")
