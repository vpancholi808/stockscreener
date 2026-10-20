import sys; sys.path.insert(0,'/home/claude/finmodel')
import importlib, run_micron as R   # reuse the facts + model built there
from finmodel import ThreeStatementModel, Assumptions, DcfModel, DcfConfig
B=1e9
facts=R.facts; MKT=1.0e12; SH=R.shares_dil[2025]; WACC=0.1086
print("="*70); print("  MICRON — DCF UNDER DELIBERATE ASSUMPTION SETS"); print("="*70)

scen = {
 "Naive (auto-inferred)": None,
 "Cycle fades, capex normalizes": Assumptions(
     years=5, revenue_growth=0.15, gross_margin=0.50, opex_pct_revenue=0.14,
     da_pct_revenue=0.16, capex_pct_revenue=0.22, nwc_pct_revenue=0.20,
     tax_rate=0.15, dividend_payout=0.05),
 "Bear: memory normalizes hard": Assumptions(
     years=5, revenue_growth=0.02, gross_margin=0.32, opex_pct_revenue=0.15,
     da_pct_revenue=0.20, capex_pct_revenue=0.25, nwc_pct_revenue=0.20,
     tax_rate=0.15, dividend_payout=0.05),
 "Bull: AI memory demand persists": Assumptions(
     years=5, revenue_growth=0.25, gross_margin=0.58, opex_pct_revenue=0.12,
     da_pct_revenue=0.14, capex_pct_revenue=0.20, nwc_pct_revenue=0.20,
     tax_rate=0.15, dividend_payout=0.05),
}
rows=[]
for name,a in scen.items():
    m = ThreeStatementModel.from_facts(facts, ticker="MU", n_years=3, assumptions=a)
    d = DcfModel.from_models(m, wacc=WACC,
            config=DcfConfig(terminal_growth=0.025), current_price=MKT/SH)
    r=d.result
    fcf5 = float(m.projection.cash_flow.loc["free_cash_flow"].iloc[-1])/B
    rows.append((name, r.value_per_share, fcf5, r.terminal_pct))
print(f"\n{'scenario':34s}{'value/share':>13s}{'yr5 FCF $B':>12s}{'TV % of EV':>12s}")
print("-"*71)
for n,v,f,t in rows:
    print(f"{n:34s}{v:>13,.0f}{f:>12,.1f}{t:>11.0%}")
print(f"\nmarket price ~${MKT/SH:,.0f}/share (~$1.0T cap)")
print("="*70)
