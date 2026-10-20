"""Validate the rebuilt foundation: point-in-time correctness, caching, quality."""
import sys, tempfile
sys.path.insert(0, "/home/claude/finmodel")
from finmodel import ThreeStatementModel, DataStore
from finmodel.statements import build_financials, derive_missing

fails = []
def check(name, cond, detail=""):
    print(f"  [{'OK' if cond else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}")
    if not cond: fails.append(name)

def dur(entries):  # entries: list of (val, fy, filed)
    return {"units": {"USD": [{"val": v, "fy": fy, "fp": "FY", "form": "10-K",
            "end": f"{fy}-12-31", "start": f"{fy}-01-01", "filed": filed}
            for (v, fy, filed) in entries]}}
def inst(entries):
    return {"units": {"USD": [{"val": v, "fy": fy, "fp": "FY", "form": "10-K",
            "end": f"{fy}-12-31", "filed": filed} for (v, fy, filed) in entries]}}

print("=" * 60)
print("1) POINT-IN-TIME (restatement handling)")
# FY2023 revenue originally filed 2024-02-01 = 100, RESTATED 2024-08-01 = 110.
facts = {"entityName": "PIT Corp", "facts": {"us-gaap": {
    "Revenues": dur([(100e9, 2023, "2024-02-01"), (110e9, 2023, "2024-08-01"),
                     (90e9, 2022, "2023-02-01")]),
    "NetIncomeLoss": dur([(20e9, 2023, "2024-02-01"), (18e9, 2022, "2023-02-01")]),
    "OperatingIncomeLoss": dur([(25e9, 2023, "2024-02-01"), (22e9, 2022, "2023-02-01")]),
    "Assets": inst([(200e9, 2023, "2024-02-01"), (180e9, 2022, "2023-02-01")]),
    "StockholdersEquity": inst([(120e9, 2023, "2024-02-01"), (110e9, 2022, "2023-02-01")]),
    "CashAndCashEquivalentsAtCarryingValue": inst([(30e9, 2023, "2024-02-01")]),
    "NetCashProvidedByUsedInOperatingActivities": dur([(28e9, 2023, "2024-02-01")]),
    "PaymentsToAcquirePropertyPlantAndEquipment": dur([(6e9, 2023, "2024-02-01")]),
}}}
before = derive_missing(build_financials(facts, "PIT", 3, as_of="2024-03-01"))
after  = derive_missing(build_financials(facts, "PIT", 3, as_of="2024-09-01"))
now    = derive_missing(build_financials(facts, "PIT", 3))  # as_of=None -> latest known
check("as_of BEFORE restatement sees original 100",
      before.income_statement.at["revenue", 2023] == 100e9,
      f"{before.income_statement.at['revenue',2023]/1e9:.0f}B")
check("as_of AFTER restatement sees revised 110",
      after.income_statement.at["revenue", 2023] == 110e9,
      f"{after.income_statement.at['revenue',2023]/1e9:.0f}B")
check("no future leak: pre-restatement view != revised value",
      before.income_statement.at["revenue", 2023] != after.income_statement.at["revenue", 2023])

print("\n2) CACHING (no redundant network calls)")
class FakeClient:
    def __init__(self): self.calls = 0
    def ticker_to_cik(self, t): return 111
    def company_facts(self, cik): self.calls += 1; return facts
with tempfile.TemporaryDirectory() as d:
    fc = FakeClient()
    store = DataStore(cache_dir=d, client=fc, ttl_days=1)
    store.get_facts("PIT"); store.get_facts("PIT"); store.get_facts("PIT")
    check("3 reads -> 1 fetch (cache hit)", fc.calls == 1, f"calls={fc.calls}")
    check("cache reports entry present", store.is_cached("PIT"))
    check("force_refresh re-fetches", (store.get_facts("PIT", force_refresh=True), fc.calls==2)[1])

print("\n3) DATA-QUALITY REPORT")
m = ThreeStatementModel.from_facts(facts, "PIT", 3)
print(m.quality_summary())
# Build a company with NO debt tags -> should flag zero-debt assumption
q = m.quality
check("quality report produced", "grade" in q)
check("flags zero-debt assumption (no debt tags present)",
      any("debt" in f.lower() for f in q["flags"]))

print("=" * 60)
print("ALL PASSED" if not fails else f"FAILURES: {fails}")
sys.exit(1 if fails else 0)
