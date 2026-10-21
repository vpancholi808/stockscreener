"""US-GAAP concept mappings.

SEC companies tag the same economic line item with different XBRL concepts.
Each line item below maps to an ordered list of candidate tags; the first tag
that yields data for a given period wins. This is the single place to extend
coverage when a company uses an unusual tag.

Concepts are split by statement and by period type:
  - DURATION concepts (income statement, cash flow) report a value over a span
    (start -> end). We want the full fiscal year.
  - INSTANT concepts (balance sheet) report a value at a point in time (end).
"""

# ---------------------------------------------------------------------------
# Income statement (duration concepts)
# ---------------------------------------------------------------------------
INCOME_STATEMENT = {
    # Order matters less here than SUBSET_MAX below: revenue is resolved by
    # taking the LARGEST candidate, not the first. See the note on SUBSET_MAX.
    "revenue": [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
        # Property trusts book most income as lease revenue under ASC 842,
        # which is not revenue from contracts with customers.
        "RealEstateRevenueNet",
        "OperatingLeaseLeaseIncome",
        # Banks and insurers: premiums and net investment income sit outside
        # ASC 606 as well.
        "RevenuesNetOfInterestExpense",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
    ],
    "cogs": [
        "CostOfGoodsAndServicesSold",
        "CostOfRevenue",
        "CostOfGoodsSold",
    ],
    "gross_profit": ["GrossProfit"],
    "rd_expense": ["ResearchAndDevelopmentExpense"],
    "sga_expense": [
        "SellingGeneralAndAdministrativeExpense",
        "GeneralAndAdministrativeExpense",
    ],
    "operating_expenses": ["OperatingExpenses", "CostsAndExpenses"],
    "operating_income": ["OperatingIncomeLoss"],
    "interest_expense": [
        "InterestExpense",
        "InterestExpenseNonoperating",
        "InterestIncomeExpenseNet",
    ],
    "pretax_income": [
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesDomestic",
    ],
    "income_tax": ["IncomeTaxExpenseBenefit"],
    "net_income": [
        "NetIncomeLoss",
        "ProfitLoss",
        "NetIncomeLossAvailableToCommonStockholdersBasic",
    ],
    "eps_diluted": ["EarningsPerShareDiluted"],
    "shares_diluted": [
        "WeightedAverageNumberOfDilutedSharesOutstanding",
        "WeightedAverageNumberOfSharesOutstandingBasic",
    ],
}

# ---------------------------------------------------------------------------
# Balance sheet (instant concepts)
# ---------------------------------------------------------------------------
BALANCE_SHEET = {
    "cash": [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
    ],
    "short_term_investments": [
        "ShortTermInvestments",
        "MarketableSecuritiesCurrent",
    ],
    "receivables": [
        "AccountsReceivableNetCurrent",
        "ReceivablesNetCurrent",
    ],
    "inventory": ["InventoryNet"],
    "current_assets": ["AssetsCurrent"],
    "ppe_net": ["PropertyPlantAndEquipmentNet"],
    "goodwill": ["Goodwill"],
    "total_assets": ["Assets"],
    "payables": ["AccountsPayableCurrent", "AccountsPayableAndAccruedLiabilitiesCurrent"],
    "short_term_debt": [
        "LongTermDebtCurrent",
        "DebtCurrent",
        "ShortTermBorrowings",
    ],
    "current_liabilities": ["LiabilitiesCurrent"],
    "long_term_debt": [
        "LongTermDebtNoncurrent",
        "LongTermDebt",
    ],
    "total_liabilities": ["Liabilities"],
    "retained_earnings": [
        "RetainedEarningsAccumulatedDeficit",
    ],
    "total_equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
}

# ---------------------------------------------------------------------------
# Cash flow statement (duration concepts)
# ---------------------------------------------------------------------------
CASH_FLOW = {
    "operating_cash_flow": [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ],
    "investing_cash_flow": ["NetCashProvidedByUsedInInvestingActivities"],
    "financing_cash_flow": ["NetCashProvidedByUsedInFinancingActivities"],
    "capex": [
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsToAcquireProductiveAssets",
    ],
    "depreciation_amortization": [
        "DepreciationDepletionAndAmortization",
        "DepreciationAmortizationAndAccretionNet",
        "DepreciationAndAmortization",
        "Depreciation",
    ],
    "stock_based_comp": ["ShareBasedCompensation"],
    "dividends_paid": [
        "PaymentsOfDividendsCommonStock",
        "PaymentsOfDividends",
    ],
}

# Which line items are instant (balance-sheet) vs duration (flow) concepts.
INSTANT_ITEMS = set(BALANCE_SHEET.keys())

ALL_MAPS = {
    "income_statement": INCOME_STATEMENT,
    "balance_sheet": BALANCE_SHEET,
    "cash_flow": CASH_FLOW,
}


# ---------------------------------------------------------------------------
# Line items resolved by taking the LARGEST candidate, not the first
# ---------------------------------------------------------------------------
# For most line items the first tag that carries data is the right answer.
# Revenue is different, because several of the candidate concepts are
# deliberately *partial* by accounting standard:
#
#   RevenueFromContractWithCustomerExcludingAssessedTax is ASC 606 revenue.
#   A property trust books rental income under ASC 842 and an insurer books
#   premiums under ASC 944, so for them this tag captures only a sliver —
#   management fees, say — while `Revenues` carries the total.
#
# A real S&P 500 run showed what first-wins ordering does with that: Essex
# Property Trust came back with $9.38M of revenue against $899M of operating
# income, a 9,587% operating margin, and every REIT and insurer in the index
# was distorted the same way. Reordering alone is fragile, because it only
# helps when the fuller tag happens to sit earlier.
#
# Taking the maximum is principled rather than lucky: each alternative is a
# subset of total revenue, so the largest candidate is the most complete one.
# The period filter (350-380 days) already rules out multi-year figures, so
# "largest" cannot pick up a cumulative value.
SUBSET_MAX = {"revenue"}
