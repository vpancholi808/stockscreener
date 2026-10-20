"""Build a sample screening dashboard from synthetic companies (offline demo)."""
import sys
sys.path.insert(0, "/home/claude/finmodel")
from finmodel import ThreeStatementModel, Screener, ScreenConfig, dashboard_html

def dur(s): return {"units":{"USD":[{"val":v,"fy":y,"fp":"FY","form":"10-K",
    "end":f"{y}-12-31","start":f"{y}-01-01","filed":f"{y+1}-02-15"} for y,v in s.items()]}}
def inst(s): return {"units":{"USD":[{"val":v,"fy":y,"fp":"FY","form":"10-K",
    "end":f"{y}-12-31","filed":f"{y+1}-02-15"} for y,v in s.items()]}}
def shr(s): return {"units":{"shares":[{"val":v,"fy":y,"fp":"FY","form":"10-K",
    "end":f"{y}-12-31","start":f"{y}-01-01","filed":f"{y+1}-02-15"} for y,v in s.items()]}}
ys=[2021,2022,2023,2024]

def company(name, rev0, growth, op_margin, debt, capex_pct=0.06, shares=2e9):
    rev={y:rev0*(1+growth)**i for i,y in enumerate(ys)}
    oi={y:rev[y]*op_margin for y in ys}
    ni={y:rev[y]*op_margin*0.75 for y in ys}
    return {"entityName":name,"facts":{"us-gaap":{
        "Revenues":dur(rev),"CostOfRevenue":dur({y:rev[y]*0.55 for y in ys}),
        "OperatingIncomeLoss":dur(oi),"NetIncomeLoss":dur(ni),
        "IncomeTaxExpenseBenefit":dur({y:ni[y]*0.25 for y in ys}),
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest":dur({y:ni[y]*1.33 for y in ys}),
        "InterestExpense":dur({y:debt*0.05 for y in ys}),
        "NetCashProvidedByUsedInOperatingActivities":dur({y:ni[y]*1.25 for y in ys}),
        "PaymentsToAcquirePropertyPlantAndEquipment":dur({y:rev[y]*capex_pct for y in ys}),
        "DepreciationDepletionAndAmortization":dur({y:rev[y]*0.05 for y in ys}),
        "WeightedAverageNumberOfDilutedSharesOutstanding":shr({y:shares for y in ys}),
        "CashAndCashEquivalentsAtCarryingValue":inst({y:rev[y]*0.3 for y in ys}),
        "ShortTermInvestments":inst({y:rev[y]*0.1 for y in ys}),
        "AssetsCurrent":inst({y:rev[y]*0.6 for y in ys}),
        "LiabilitiesCurrent":inst({y:rev[y]*0.35 for y in ys}),
        "LongTermDebtNoncurrent":inst({y:debt for y in ys}),
        "StockholdersEquity":inst({y:rev[y]*0.7 for y in ys}),
        "Assets":inst({y:rev[y]*1.4 for y in ys}),
        "PropertyPlantAndEquipmentNet":inst({y:rev[y]*0.5 for y in ys}),
    }}}

specs = {
    "VLTA": ("Volta Systems",      40e9, 0.22, 0.30,  8e9, 0.06, 2.4e9),
    "MERD": ("Meridian Health",    28e9, 0.11, 0.18, 20e9, 0.05, 1.8e9),
    "CASC": ("Cascade Foods",      15e9, 0.04, 0.09, 12e9, 0.07, 1.2e9),
    "NORT": ("Northwind Energy",   52e9, 0.02, 0.14, 34e9, 0.12, 3.1e9),
    "AXIS": ("Axis Software",      12e9, 0.31, 0.34,  2e9, 0.04, 0.9e9),
    "PBLT": ("Pinnacle Builders",   9e9, -0.03, 0.06, 15e9, 0.03, 0.6e9),
}
models = {t: ThreeStatementModel.from_facts(company(name, *rest), t, 4)
          for t,(name,*rest) in specs.items()}
# rough market caps (price x shares) for valuation columns
caps = {"VLTA":260e9,"MERD":95e9,"CASC":42e9,"NORT":140e9,"AXIS":180e9,"PBLT":18e9}

scr = Screener.from_models(models, market_caps=caps, config=ScreenConfig())
print(scr.summary())
print(scr.table().to_string())

html_out = dashboard_html(scr, title="My Watchlist Screen")
open("/mnt/user-data/outputs/screen_dashboard.html","w").write(html_out)
print("\nHTML bytes:", len(html_out))
print("contains meter svg:", "meter" in html_out, "| sparkline:", "spark" in html_out,
      "| sortable:", "sortable" in html_out)
