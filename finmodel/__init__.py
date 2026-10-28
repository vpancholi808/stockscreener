"""finmodel — equity analysis toolkit. Foundation: 3-statement model.

Pipeline roadmap (each layer consumes the one before):
  3-statement (this)  ->  DCF  ->  comps  ->  ratio analysis
  CAPM/beta  ->  factor model  ->  risk model  ->  long/short portfolio
"""
from .model import ThreeStatementModel
from .statements import Financials, build_financials, derive_missing
from .projections import Assumptions, Projection, project
from .scorecard import ScorecardConfig, compute_scorecard, scorecard_to_frame
from .capm import (CapmConfig, CapmModel, BetaResult, WaccResult,
                   compute_beta, cost_of_equity, compute_wacc)
from .dcf import DcfConfig, DcfModel, DcfResult, run_dcf, sensitivity
from .data.edgar import EdgarClient
from .data.store import DataStore
from .data.quality import quality_report, quality_summary
from .forensic import (forensic_report, ForensicReport, piotroski_f,
                       altman_z, beneish_m, sloan_accruals, economic_spread)
from .valuation import Multiples, compute_multiples
from .sectors import (build_benchmarks, compare_to_sector, valuation_summary,
                      BENCHMARK_METRICS, SectorBenchmark, Comparison)
from .research import generate_research, ResearchNote
from .portfolio_builder import (build_portfolio, assess_portfolio,
                                STRATEGIES, RISK_PROFILES, PortfolioResult)
from .screener import Screener, ScreenConfig
from .universe import UniverseScreen
from .data.bulk import BulkData, sector_for_sic
from .report import dashboard_html
from .tearsheet import tearsheet_html
from . import quant
from .quant import (Universe, MarkowitzOptimizer, Portfolio, risk_report,
                    stress_test, bs_price, bs_greeks, implied_vol, FactorModel,
                    CapitalAllocationLine, CompletePortfolio, FrontierGeometry,
                    NoTangencyPortfolio)
from .data.prices import (PriceFeed, YFinancePriceFeed, FramePriceFeed,
                          to_returns, periods_per_year)
from .data.yahoo import (PriceDownload, download_prices, audit_prices,
                         align_prices, coverage, normalize_download)
from .returns import (DistributionReport, return_statistics, compute_returns,
                      annualize_mean, annualize_volatility, variance_drag,
                      jarque_bera, tail_exceedances, kurtosis_concentration,
                      rolling_volatility, aggregation_kurtosis,
                      infer_periods_per_year)
from .performance import (PerformanceResult, performance_report,
                          simulate_portfolio, xirr, xnpv, weighted_returns,
                          contribution_to_return)

__version__ = "0.1.0"
