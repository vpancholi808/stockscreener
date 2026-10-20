"""Live DCF example: EDGAR fundamentals + yfinance prices -> intrinsic value.

    export EDGAR_USER_AGENT="Your Name you@email.com"
    python example_dcf.py AAPL
"""
import sys
from finmodel import (ThreeStatementModel, CapmModel, CapmConfig,
                      DcfModel, DcfConfig)

ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"

tsm = ThreeStatementModel.from_ticker(ticker)
capm = CapmModel.from_ticker(
    ticker,
    config=CapmConfig(risk_free_rate=0.042, equity_risk_premium=0.05),
    three_statement=tsm,
)
print(capm.beta)
print(capm.wacc, "\n")

dcf = DcfModel.from_models(tsm, capm=capm, config=DcfConfig(terminal_growth=0.025))
print(dcf.result.summary())
print("\nSensitivity (value/share):")
print(dcf.sensitivity_table.to_string())
