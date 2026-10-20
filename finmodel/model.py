"""High-level entry point for the 3-statement foundation.

    from finmodel import ThreeStatementModel
    m = ThreeStatementModel.from_ticker("AAPL", user_agent="You you@email.com")
    print(m.scorecard_frame())
    print(m.projection.income_statement)

This is the foundation layer. DCF, comps, CAPM/beta, factor, risk, and the
long/short portfolio model are designed to consume `m.financials` and
`m.projection` (e.g. DCF discounts `projection.cash_flow.loc['free_cash_flow']`).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from .data.edgar import EdgarClient
from .data.store import DataStore
from .data.quality import quality_report, quality_summary
from .statements import Financials, build_financials, derive_missing
from .projections import Assumptions, Projection, project
from .scorecard import ScorecardConfig, compute_scorecard, scorecard_to_frame


@dataclass
class ThreeStatementModel:
    financials: Financials
    projection: Projection
    scorecard: dict
    quality: dict

    # ---- constructors -----------------------------------------------------
    @classmethod
    def from_facts(cls, facts: dict, ticker: str = "", n_years: int = 5,
                   as_of: Optional[str] = None,
                   assumptions: Optional[Assumptions] = None,
                   config: Optional[ScorecardConfig] = None) -> "ThreeStatementModel":
        fin = derive_missing(build_financials(facts, ticker=ticker,
                                              n_years=n_years, as_of=as_of))
        proj = project(fin, assumptions)
        score = compute_scorecard(fin, config)
        qual = quality_report(fin)
        return cls(fin, proj, score, qual)

    @classmethod
    def from_ticker(cls, ticker: str, user_agent: str = None, n_years: int = 5,
                    as_of: Optional[str] = None, use_cache: bool = True,
                    cache_dir: Optional[str] = None,
                    assumptions: Optional[Assumptions] = None,
                    config: Optional[ScorecardConfig] = None) -> "ThreeStatementModel":
        """Pull fundamentals (cached by default) and build the model.

        as_of: ISO date for a point-in-time, look-ahead-free view.
        use_cache: read/write a local disk cache instead of always hitting SEC."""
        client = EdgarClient(user_agent=user_agent) if user_agent else EdgarClient()
        if use_cache:
            facts = DataStore(cache_dir=cache_dir, client=client).get_facts(ticker)
        else:
            facts = client.company_facts(ticker)
        return cls.from_facts(facts, ticker=ticker, n_years=n_years, as_of=as_of,
                              assumptions=assumptions, config=config)

    @classmethod
    def from_file(cls, path: str, ticker: str = "", **kw) -> "ThreeStatementModel":
        facts = EdgarClient.load_facts_from_file(path)
        return cls.from_facts(facts, ticker=ticker, **kw)

    def quality_summary(self) -> str:
        return quality_summary(self.quality)

    # ---- views ------------------------------------------------------------
    def scorecard_frame(self) -> pd.DataFrame:
        return scorecard_to_frame(self.scorecard)

    def summary(self) -> str:
        s = self.scorecard
        lines = [f"{self.financials.entity_name} ({s['ticker']}) — FY{s['fiscal_year']}",
                 f"Health score: {s['passed']}/{s['total']} "
                 f"({s['score']*100:.0f}%)", ""]
        r = s["raw"]
        def fmt(v):
            return f"{v/1e9:,.2f}B" if pd.notna(v) else "n/a"
        lines += [f"Revenue:   {fmt(r['revenue'])}",
                  f"EBITDA:    {fmt(r['ebitda'])}",
                  f"Net income:{fmt(r['net_income'])}",
                  f"FCF:       {fmt(r['fcf'])}",
                  f"Net debt:  {fmt(r['net_debt'])}"]
        return "\n".join(lines)
