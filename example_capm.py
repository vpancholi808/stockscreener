"""Live CAPM example. Pulls fundamentals from EDGAR and prices from yfinance.

    export EDGAR_USER_AGENT="Your Name you@email.com"
    python example_capm.py AAPL
"""
import sys
from finmodel import ThreeStatementModel, CapmModel, CapmConfig

ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"

tsm = ThreeStatementModel.from_ticker(ticker)
cfg = CapmConfig(risk_free_rate=0.042, equity_risk_premium=0.05,
                 frequency="M", lookback_years=5)
capm = CapmModel.from_ticker(ticker, config=cfg, three_statement=tsm)

print(f"{ticker}  last price {capm.last_price:.2f}")
print(capm.beta)
print(f"cost of equity (ke) = {capm.cost_of_equity:.2%}")
print(capm.wacc)
