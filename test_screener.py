"""Validate screener pass/fail logic and dashboard generation."""
import sys
sys.path.insert(0, "/home/claude/finmodel")
from finmodel import ThreeStatementModel, Screener, ScreenConfig, dashboard_html

def dur(s): return {"units":{"USD":[{"val":v,"fy":y,"fp":"FY","form":"10-K",
    "end":f"{y}-12-31","start":f"{y}-01-01"} for y,v in s.items()]}}
def inst(s): return {"units":{"USD":[{"val":v,"fy":y,"fp":"FY","form":"10-K",
    "end":f"{y}-12-31"} for y,v in s.items()]}}
ys=[2022,2023,2024]
def co(name, growth, margin, debt):
    rev={y:30e9*(1+growth)**i for i,y in enumerate(ys)}
    oi={y:rev[y]*margin for y in ys}; ni={y:rev[y]*margin*0.75 for y in ys}
    return {"entityName":name,"facts":{"us-gaap":{
        "Revenues":dur(rev),"CostOfRevenue":dur({y:rev[y]*0.55 for y in ys}),
        "OperatingIncomeLoss":dur(oi),"NetIncomeLoss":dur(ni),
        "InterestExpense":dur({y:debt*0.05 for y in ys}),
        "NetCashProvidedByUsedInOperatingActivities":dur({y:ni[y]*1.3 for y in ys}),
        "PaymentsToAcquirePropertyPlantAndEquipment":dur({y:rev[y]*0.05 for y in ys}),
        "DepreciationDepletionAndAmortization":dur({y:rev[y]*0.05 for y in ys}),
        "CashAndCashEquivalentsAtCarryingValue":inst({y:rev[y]*0.3 for y in ys}),
        "AssetsCurrent":inst({y:rev[y]*0.6 for y in ys}),
        "LiabilitiesCurrent":inst({y:rev[y]*0.3 for y in ys}),
        "LongTermDebtNoncurrent":inst({y:debt for y in ys}),
        "StockholdersEquity":inst({y:rev[y]*0.7 for y in ys}),
        "Assets":inst({y:rev[y]*1.4 for y in ys}),
        "PropertyPlantAndEquipmentNet":inst({y:rev[y]*0.5 for y in ys}),
    }}}
fails=[]
def check(n,c):
    print(f"  [{'OK' if c else 'FAIL'}] {n}");
    if not c: fails.append(n)

models={"GOOD":ThreeStatementModel.from_facts(co("Good Co",0.20,0.30,4e9),"GOOD",3),
        "BAD": ThreeStatementModel.from_facts(co("Bad Co",-0.05,0.06,25e9),"BAD",3)}
scr=Screener.from_models(models, config=ScreenConfig())
tbl=scr.table()
check("GOOD passes screen", bool(tbl.loc["GOOD","passes_screen"]))
check("BAD fails screen", not bool(tbl.loc["BAD","passes_screen"]))
check("GOOD ranks above BAD", list(tbl.index)==["GOOD","BAD"])
check("summary counts 1 pass", "1 pass" in scr.summary())
h=dashboard_html(scr,"Test")
check("dashboard is valid html", h.startswith("<!doctype html>") and h.rstrip().endswith("</html>"))
check("dashboard has 2 data rows", h.count('<tr class="on"')+h.count('<tr class="off"')==2)
check("stricter screen rejects GOOD too",
      not bool(Screener.from_models(models,config=ScreenConfig(min_roic=0.50)).table().loc["GOOD","passes_screen"]))
print("ALL PASSED" if not fails else f"FAILURES: {fails}")
sys.exit(1 if fails else 0)
