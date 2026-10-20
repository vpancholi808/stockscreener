"""Live example — runs against real SEC EDGAR. Set your User-Agent first.

    export EDGAR_USER_AGENT="Your Name you@email.com"
    python example_live.py AAPL
"""
import sys
from finmodel import ThreeStatementModel, ScorecardConfig

ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"

cfg = ScorecardConfig(max_net_debt_to_ebitda=2.5, min_fcf_margin=0.08)
m = ThreeStatementModel.from_ticker(ticker, config=cfg)

print(m.summary(), "\n")
print(m.scorecard_frame().to_string(index=False), "\n")
print("Projected free cash flow (USD bn):")
print((m.projection.cash_flow.loc["free_cash_flow"] / 1e9).round(2).to_string())
