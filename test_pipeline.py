"""Validate the pipeline offline with a synthetic companyfacts payload that
mirrors SEC's real JSON structure (form/fp/start/end/val fields)."""
import sys
sys.path.insert(0, "/home/claude/finmodel")

from finmodel import ThreeStatementModel, Assumptions, ScorecardConfig


def fact(val, fy, end, start=None):
    f = {"val": val, "fy": fy, "fp": "FY", "form": "10-K", "end": end}
    if start:
        f["start"] = start
    return f


def duration(tag, series):  # series: {fy: val}
    return {"units": {"USD": [fact(v, fy, f"{fy}-12-31", f"{fy}-01-01")
                              for fy, v in series.items()]}}


def instant(tag, series):
    return {"units": {"USD": [fact(v, fy, f"{fy}-12-31") for fy, v in series.items()]}}


# Four years of a healthy, growing, cash-generative company (USD).
yrs = [2021, 2022, 2023, 2024]
rev = dict(zip(yrs, [80e9, 92e9, 105e9, 120e9]))
cogs = dict(zip(yrs, [40e9, 45e9, 50e9, 56e9]))
oi = dict(zip(yrs, [20e9, 24e9, 28e9, 33e9]))
ni = dict(zip(yrs, [15e9, 18e9, 21e9, 25e9]))
tax = dict(zip(yrs, [4e9, 5e9, 6e9, 7e9]))
pre = dict(zip(yrs, [19e9, 23e9, 27e9, 32e9]))
intr = dict(zip(yrs, [1e9, 1e9, 1e9, 1e9]))
da = dict(zip(yrs, [5e9, 5.5e9, 6e9, 6.5e9]))
ocf = dict(zip(yrs, [22e9, 26e9, 30e9, 35e9]))
capex = dict(zip(yrs, [6e9, 6.5e9, 7e9, 8e9]))
div = dict(zip(yrs, [-3e9, -3.5e9, -4e9, -4.5e9]))

facts = {
    "entityName": "Synthetic Test Corp",
    "facts": {"us-gaap": {
        "Revenues": duration("Revenues", rev),
        "CostOfRevenue": duration("CostOfRevenue", cogs),
        "OperatingIncomeLoss": duration("OperatingIncomeLoss", oi),
        "NetIncomeLoss": duration("NetIncomeLoss", ni),
        "IncomeTaxExpenseBenefit": duration("IncomeTaxExpenseBenefit", tax),
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest":
            duration("pretax", pre),
        "InterestExpense": duration("InterestExpense", intr),
        "NetCashProvidedByUsedInOperatingActivities": duration("ocf", ocf),
        "PaymentsToAcquirePropertyPlantAndEquipment": duration("capex", capex),
        "DepreciationDepletionAndAmortization": duration("da", da),
        "PaymentsOfDividendsCommonStock": duration("div", div),
        # Balance sheet (instant)
        "CashAndCashEquivalentsAtCarryingValue":
            instant("cash", dict(zip(yrs, [20e9, 24e9, 30e9, 38e9]))),
        "AssetsCurrent": instant("ca", dict(zip(yrs, [45e9, 50e9, 58e9, 68e9]))),
        "InventoryNet": instant("inv", dict(zip(yrs, [8e9, 9e9, 10e9, 11e9]))),
        "PropertyPlantAndEquipmentNet":
            instant("ppe", dict(zip(yrs, [30e9, 31e9, 32e9, 33.5e9]))),
        "Assets": instant("ta", dict(zip(yrs, [90e9, 100e9, 115e9, 132e9]))),
        "LiabilitiesCurrent": instant("cl", dict(zip(yrs, [25e9, 27e9, 30e9, 33e9]))),
        "LongTermDebtNoncurrent": instant("ltd", dict(zip(yrs, [20e9, 20e9, 18e9, 16e9]))),
        "Liabilities": instant("tl", dict(zip(yrs, [50e9, 52e9, 53e9, 54e9]))),
        "StockholdersEquity": instant("eq", dict(zip(yrs, [40e9, 48e9, 62e9, 78e9]))),
    }}
}

print("=" * 64)
m = ThreeStatementModel.from_facts(facts, ticker="TEST", n_years=4)
print(m.summary())
print("\n--- SCORECARD ---")
print(m.scorecard_frame().to_string(index=False))
print("\n--- HISTORICAL INCOME STATEMENT (USD bn) ---")
print((m.financials.income_statement / 1e9).round(2).to_string())
print("\n--- PROJECTED FCF (USD bn) ---")
print((m.projection.cash_flow.loc[["operating_cash_flow", "capex", "free_cash_flow"]] / 1e9).round(2).to_string())
print("\n--- INFERRED ASSUMPTIONS ---")
a = m.projection.assumptions
for k in ["revenue_growth", "gross_margin", "opex_pct_revenue", "da_pct_revenue",
          "capex_pct_revenue", "nwc_pct_revenue", "tax_rate", "dividend_payout"]:
    print(f"  {k:18s} {getattr(a, k):.4f}")
print("\n--- BALANCE CHECK (assets - L&E, should be ~0) ---")
print((m.projection.balance_check / 1e9).round(4).to_string())
print("=" * 64)
