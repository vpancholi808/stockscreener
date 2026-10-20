import sys; sys.path.insert(0,'/home/claude/finmodel')
import run_micron as R
from finmodel import ThreeStatementModel, forensic_report
M=1e6
# add retained earnings (Micron 10-K equity statement, approx: FY23 32.0 / FY24 32.3 / FY25 40.3 $B)
R.facts["facts"]["us-gaap"]["RetainedEarningsAccumulatedDeficit"] = R.inst(
    {2023: 32000*M, 2024: 32265*M, 2025: 40282*M})
R.facts["facts"]["us-gaap"]["SellingGeneralAndAdministrativeExpense"] = R.dur(
    {2023: 1000*M, 2024: 1116*M, 2025: 1300*M})
m = ThreeStatementModel.from_facts(R.facts, "MU", 3)
rep = forensic_report(m, market_cap=1.0e12, wacc=0.1086)

print("="*66); print("  MICRON — FORENSIC & VALUE-CREATION SCORES (FY2025)"); print("="*66)
for s in rep.all_scores(): print(f"  {s}")
print("\n  Piotroski detail (9 binary tests):")
for k,v in rep.piotroski.detail.items():
    print(f"    [{'✓' if v else '·'}] {k}")
print("\n  Altman components:")
for k,v in rep.altman.detail.items():
    print(f"    {k:14s} {v:.3f}" if isinstance(v,float) else f"    {k:14s} {v}")
print("\n  Beneish components (>1.0 = deteriorating vs prior year):")
for k,v in rep.beneish.detail.items(): print(f"    {k:6s} {v:.3f}")
print("="*66)
import json
json.dump({"piotroski":rep.piotroski.value,"piotroski_detail":rep.piotroski.detail,
 "altman":rep.altman.value,"altman_verdict":rep.altman.verdict,
 "beneish":rep.beneish.value,"beneish_verdict":rep.beneish.verdict,
 "beneish_detail":rep.beneish.detail,
 "accruals":rep.accruals.value,"accruals_verdict":rep.accruals.verdict,
 "spread":rep.spread.value,"spread_verdict":rep.spread.verdict,
 "scorecard":{k:{"value":float(v.value),"threshold":v.threshold,"passed":v.passed,
   "hib":v.higher_is_better} for k,v in m.scorecard["metrics"].items()},
 "raw":{k:(float(v) if v==v else None) for k,v in m.scorecard["raw"].items()}},
 open("/home/claude/mu_scores.json","w"), indent=1)
