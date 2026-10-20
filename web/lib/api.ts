/** Server-side API client. Every call runs on the server, so the browser never
 *  talks to the Python service directly and the API host stays private. */

const BASE = process.env.API_BASE || "http://127.0.0.1:8000";

export type Company = {
  cik: number;
  ticker: string;
  name: string;
  sector: string;
  fiscal_year: number | null;
  health_passed: number | null;
  health_total: number | null;
  health_score: number | null;
  revenue: number | null;
  revenue_growth: number | null;
  ebitda: number | null;
  fcf: number | null;
  net_debt: number | null;
  gross_margin: number | null;
  operating_margin: number | null;
  net_margin: number | null;
  fcf_margin: number | null;
  roe: number | null;
  roic: number | null;
  net_debt_to_ebitda: number | null;
  interest_coverage: number | null;
  current_ratio: number | null;
  debt_to_equity: number | null;
  cash_conversion: number | null;
  piotroski: number | null;
  altman: number | null;
  beneish: number | null;
  accruals: number | null;
  data_quality: string | null;
  passes_screen: boolean | number;
  sector_percentile: number | null;
};

export type HistoryRow = {
  fiscal_year: number;
  revenue: number | null;
  gross_profit: number | null;
  operating_income: number | null;
  net_income: number | null;
  ebitda: number | null;
  operating_cash_flow: number | null;
  capex: number | null;
  fcf: number | null;
  total_assets: number | null;
  total_equity: number | null;
  total_debt: number | null;
};

export type CompanyDetail = Company & {
  history: HistoryRow[];
  market: Record<string, number | string | null> | null;
  disclaimer: string;
};

export type ScreenResponse = {
  rows: Company[];
  total: number;
  limit: number;
  offset: number;
  disclaimer: string;
};

export type SectorRow = {
  sector: string;
  companies: number;
  passing: number | null;
  avg_health: number | null;
  avg_op_margin: number | null;
  avg_roic: number | null;
};

export type Stats = {
  companies: number;
  passing: number | null;
  sectors: number;
  latest_year: number | null;
  last_run: string | null;
  priced: number;
  serves_market_data: boolean;
  disclaimer: string;
};

async function get<T>(path: string, revalidate = 300): Promise<T> {
  const res = await fetch(`${BASE}${path}`, { next: { revalidate } });
  if (!res.ok) {
    throw new Error(`API ${res.status} on ${path}`);
  }
  return res.json() as Promise<T>;
}

export async function fetchScreen(
  params: Record<string, string | undefined>
): Promise<ScreenResponse> {
  const qs = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (v !== undefined && v !== "") qs.set(k, v);
  }
  return get<ScreenResponse>(`/api/companies?${qs.toString()}`);
}

export async function fetchCompany(ticker: string): Promise<CompanyDetail | null> {
  try {
    return await get<CompanyDetail>(
      `/api/companies/${encodeURIComponent(ticker)}`
    );
  } catch {
    return null;
  }
}

export async function fetchSectors(): Promise<SectorRow[]> {
  const d = await get<{ sectors: SectorRow[] }>("/api/sectors", 3600);
  return d.sectors;
}

export async function fetchStats(): Promise<Stats> {
  return get<Stats>("/api/stats", 600);
}

export async function fetchMeta(): Promise<any> {
  return get<any>("/api/meta", 3600);
}

// --- research, sector comparison, portfolio -------------------------------

export type Finding = {
  kind: "strength" | "concern" | "watch" | "context";
  topic: string;
  statement: string;
  evidence: string;
  metric: string | null;
};

export type ResearchNote = {
  ticker: string;
  name: string;
  sector: string;
  fiscal_year: number | null;
  headline: string;
  summary: string;
  strengths: Finding[];
  concerns: Finding[];
  watch_items: Finding[];
  context: Finding[];
  valuation_label: string | null;
  valuation_percentile: number | null;
  health_passed: number | null;
  health_total: number | null;
  caveats: string[];
};

export type SectorComparison = {
  metric: string;
  label: string;
  value: number | null;
  sector_median: number | null;
  percentile: number | null;
  direction: number;
  peers: number;
  is_percent: boolean;
  verdict: string;
  premium: number | null;
};

export type ResearchResponse = {
  research: ResearchNote;
  comparisons: SectorComparison[];
  no_recommendation: string;
  disclaimer: string;
};

export type StrategyOption = {
  key: string;
  name: string;
  description: string;
  filters: Record<string, { op: string; value: number }>;
  weights: Record<string, number>;
};

export type RiskOption = {
  key: string;
  name: string;
  description: string;
  max_position: number;
  min_holdings: number;
  target_holdings: number;
  max_sector: number;
  min_health_score: number;
  max_net_debt_to_ebitda: number | null;
  require_positive_fcf: boolean;
};

export type PortfolioOptions = {
  strategies: StrategyOption[];
  risk_profiles: RiskOption[];
  allocation_methods: { key: string; name: string; description: string }[];
};

export type PortfolioHolding = {
  ticker: string;
  name: string;
  sector: string;
  weight: number;
  strategy_score: number;
  health_passed: number | null;
  health_total: number | null;
  roic: number | null;
  fcf_margin: number | null;
  revenue_growth: number | null;
  pe?: number | null;
  net_debt_to_ebitda: number | null;
};

export type PortfolioResult = {
  strategy: string;
  strategy_name: string;
  risk_profile: string;
  risk_profile_name: string;
  allocation_method: string;
  holdings: PortfolioHolding[];
  sector_weights: Record<string, number>;
  metrics: Record<string, number | null>;
  universe_size: number;
  eligible_size: number;
  rejected: Record<string, number>;
  warnings: string[];
  no_recommendation?: string;
};

export async function fetchResearch(
  ticker: string
): Promise<ResearchResponse | null> {
  try {
    return await get<ResearchResponse>(
      `/api/companies/${encodeURIComponent(ticker)}/research`,
      600
    );
  } catch {
    return null;
  }
}

export async function fetchPortfolioOptions(): Promise<PortfolioOptions | null> {
  try {
    return await get<PortfolioOptions>("/api/portfolio/options", 3600);
  } catch {
    return null;
  }
}

export async function buildPortfolio(body: {
  strategy: string;
  risk_profile: string;
  allocation?: string;
  max_holdings?: number;
  sector_limit?: number;
}): Promise<PortfolioResult | null> {
  try {
    const res = await fetch(`${BASE}/api/portfolio/build`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      cache: "no-store",
    });
    if (!res.ok) return null;
    return (await res.json()) as PortfolioResult;
  } catch {
    return null;
  }
}
