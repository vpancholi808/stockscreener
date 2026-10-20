"""TEST RUN: Micron Technology (MU) through the full finmodel chain.

Data source: Micron's FY2025 Form 10-K (fiscal years ended Aug 28 2025,
Aug 29 2024, Aug 31 2023) plus Q4/FY25 press release. Entered manually here
because this sandbox has no network route to data.sec.gov; in your environment
this is exactly what ThreeStatementModel.from_ticker("MU") fetches.

Figures marked DERIVED are computed from disclosed roll-forwards rather than
lifted directly off a statement line.
"""
import sys
sys.path.insert(0, "/home/claude/finmodel")

import pandas as pd
from finmodel import (ThreeStatementModel, ScorecardConfig, Assumptions,
                      DcfConfig, DcfModel, Screener, ScreenConfig, dashboard_html)
from finmodel.capm import compute_wacc, cost_of_equity, CapmConfig

M = 1e6  # statements are reported in millions

# fiscal year -> (value, filed date)  FY2025 10-K filed ~2025-10-03
FILED = {2023: "2023-10-06", 2024: "2024-10-04", 2025: "2025-10-03"}


def dur(vals):  # duration/flow concept
    return {"units": {"USD": [
        {"val": v, "fy": fy, "fp": "FY", "form": "10-K",
         "start": f"{fy-1}-08-30", "end": f"{fy}-08-28", "filed": FILED[fy]}
        for fy, v in vals.items()]}}


def inst(vals):  # instant/balance-sheet concept
    return {"units": {"USD": [
        {"val": v, "fy": fy, "fp": "FY", "form": "10-K",
         "end": f"{fy}-08-28", "filed": FILED[fy]}
        for fy, v in vals.items()]}}


def shr(vals):
    return {"units": {"shares": [
        {"val": v, "fy": fy, "fp": "FY", "form": "10-K",
         "start": f"{fy-1}-08-30", "end": f"{fy}-08-28", "filed": FILED[fy]}
        for fy, v in vals.items()]}}


# ---------------------------------------------------------------- income stmt
revenue      = {2023: 15540*M, 2024: 25111*M, 2025: 37380*M}
gross_profit = {2023: -1416*M, 2024:  5566*M, 2025: 14952*M}   # GM -9% / 22% / 40%
cogs         = {y: revenue[y] - gross_profit[y] for y in revenue}
op_income    = {2023: -5745*M, 2024:  1300*M, 2025:  9000*M}   # DERIVED (GP - opex)
net_income   = {2023: -5833*M, 2024:   778*M, 2025:  8539*M}   # 10-K cash flow stmt
income_tax   = {2023:    -37*M, 2024:  388*M, 2025:  1063*M}   # DERIVED
pretax       = {y: net_income[y] + income_tax[y] for y in net_income}
interest_exp = {2023:   533*M, 2024:   516*M, 2025:   490*M}
shares_dil   = {2023:  1093*M, 2024:  1118*M, 2025:  1131*M}

# --------------------------------------------------------------- balance sheet
cash         = {2023:  8577*M, 2024:  7041*M, 2025:  8734*M}
st_invest    = {2023:  1400*M, 2024:  1073*M, 2025:  3206*M}
receivables  = {2023:  3308*M, 2024:  6615*M, 2025:  7436*M}
inventory    = {2023:  8387*M, 2024:  8875*M, 2025:  8355*M}
curr_assets  = {2023: 21289*M, 2024: 25287*M, 2025: 28887*M}
ppe_net      = {2023: 38284*M, 2024: 39943*M, 2025: 46900*M}   # DERIVED roll-forward
goodwill     = {2023:  1150*M, 2024:  1150*M, 2025:  1147*M}
total_assets = {2023: 64254*M, 2024: 69416*M, 2025: 82798*M}
curr_liab    = {2023:  4810*M, 2024: 10163*M, 2025: 11500*M}
st_debt      = {2023:   324*M, 2024:  1000*M, 2025:   600*M}
lt_debt      = {2023: 13056*M, 2024: 11237*M, 2025: 11533*M}
total_liab   = {2023: 20134*M, 2024: 24285*M, 2025: 28633*M}
total_equity = {2023: 44120*M, 2024: 45131*M, 2025: 54165*M}

# ------------------------------------------------------------------ cash flow
ocf          = {2023:  1559*M, 2024:  8507*M, 2025: 17525*M}
capex        = {2023:  7676*M, 2024:  8386*M, 2025: 15857*M}
d_and_a      = {2023:  7756*M, 2024:  7780*M, 2025:  8352*M}
sbc          = {2023:   596*M, 2024:   833*M, 2025:   972*M}
dividends    = {2023:  -504*M, 2024:  -513*M, 2025:  -522*M}
icf          = {2023: -6191*M, 2024: -8309*M, 2025: -14087*M}
fcf_fin      = {2023:  -900*M, 2024: -2900*M, 2025:  -5141*M}

facts = {"entityName": "Micron Technology, Inc.", "cik": 723125,
         "facts": {"us-gaap": {
    "Revenues": dur(revenue),
    "CostOfGoodsAndServicesSold": dur(cogs),
    "GrossProfit": dur(gross_profit),
    "OperatingIncomeLoss": dur(op_income),
    "NetIncomeLoss": dur(net_income),
    "IncomeTaxExpenseBenefit": dur(income_tax),
    "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest": dur(pretax),
    "InterestExpense": dur(interest_exp),
    "WeightedAverageNumberOfDilutedSharesOutstanding": shr(shares_dil),
    "CashAndCashEquivalentsAtCarryingValue": inst(cash),
    "ShortTermInvestments": inst(st_invest),
    "AccountsReceivableNetCurrent": inst(receivables),
    "InventoryNet": inst(inventory),
    "AssetsCurrent": inst(curr_assets),
    "PropertyPlantAndEquipmentNet": inst(ppe_net),
    "Goodwill": inst(goodwill),
    "Assets": inst(total_assets),
    "LiabilitiesCurrent": inst(curr_liab),
    "DebtCurrent": inst(st_debt),
    "LongTermDebtNoncurrent": inst(lt_debt),
    "Liabilities": inst(total_liab),
    "StockholdersEquity": inst(total_equity),
    "NetCashProvidedByUsedInOperatingActivities": dur(ocf),
    "NetCashProvidedByUsedInInvestingActivities": dur(icf),
    "NetCashProvidedByUsedInFinancingActivities": dur(fcf_fin),
    "PaymentsToAcquirePropertyPlantAndEquipment": dur(capex),
    "DepreciationDepletionAndAmortization": dur(d_and_a),
    "ShareBasedCompensation": dur(sbc),
    "PaymentsOfDividendsCommonStock": dur(dividends),
}}}

B = 1e9
print("=" * 70)
print("  finmodel TEST RUN — MICRON TECHNOLOGY (MU)")
print("=" * 70)

m = ThreeStatementModel.from_ticker.__func__ if False else None
model = ThreeStatementModel.from_facts(facts, ticker="MU", n_years=3)

print("\n[1] DATA QUALITY")
print(model.quality_summary())

print("\n[2] HISTORICAL STATEMENTS (USD bn)")
is_ = model.financials.income_statement
show = ["revenue", "gross_profit", "operating_income", "ebitda", "net_income"]
print((is_.loc[[r for r in show if r in is_.index]] / B).round(2).to_string())
bs = model.financials.balance_sheet
print((bs.loc[["cash", "total_assets", "long_term_debt", "total_equity"]] / B).round(2).to_string())

print("\n[3] HEALTH SCORECARD")
print(model.summary())
print()
print(model.scorecard_frame().to_string(index=False))

print("\n[4] PROJECTION DRIVERS (inferred from FY23-FY25)")
a = model.projection.assumptions
for k in ["revenue_growth", "gross_margin", "opex_pct_revenue", "da_pct_revenue",
          "capex_pct_revenue", "nwc_pct_revenue", "tax_rate", "dividend_payout"]:
    v = getattr(a, k)
    print(f"   {k:20s} {v:>8.2%}" if v is not None else f"   {k:20s}      n/a")
print("\n   Projected FCFF-relevant lines (USD bn):")
pcf = model.projection.cash_flow
print((pcf.loc[["operating_cash_flow", "capex", "free_cash_flow"]] / B).round(2).to_string())
print(f"\n   Balance check (should be ~0): "
      f"{(model.projection.balance_check/B).abs().max():.6f}")

print("\n[5] COST OF CAPITAL")
cfg = CapmConfig(risk_free_rate=0.042, equity_risk_premium=0.05)
BETA = 1.35          # Micron's long-run beta (price feed unavailable here)
ke = cost_of_equity(BETA, cfg)
y = model.financials.latest_year()
debt = float(bs.at["long_term_debt", y] + bs.at["short_term_debt", y])
kd = float(is_.at["interest_expense", y]) / debt
MKT_CAP = 1.0e12     # ~$1T market cap (Aug 2026)
w = compute_wacc(ke, kd, a.tax_rate, MKT_CAP, debt)
print(f"   beta (manual) {BETA}  ->  ke = {ke:.2%}")
print(f"   {w}")

print("\n[6] DCF VALUATION")
dcf = DcfModel.from_models(model, wacc=w.wacc,
                           config=DcfConfig(terminal_growth=0.025, mid_year=True),
                           current_price=MKT_CAP / shares_dil[2025])
print(dcf.result.summary())
print("\n   Sensitivity (intrinsic value/share):")
print(dcf.sensitivity_table.to_string())

print("\n[7] SCREENER ROW")
scr = Screener.from_models({"MU": model}, market_caps={"MU": MKT_CAP},
                           config=ScreenConfig())
print(scr.summary())
print(scr.table().T.to_string())

open("/mnt/user-data/outputs/micron_dashboard.html", "w").write(
    dashboard_html(scr, title="Micron Technology — finmodel test run"))
print("\n   dashboard written -> micron_dashboard.html")
print("=" * 70)
