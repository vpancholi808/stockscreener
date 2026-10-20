import Link from "next/link";
import Filters from "@/components/Filters";
import ScreenTable from "@/components/ScreenTable";
import { fetchScreen, fetchSectors, fetchStats } from "@/lib/api";
import type { Company, SectorRow, Stats } from "@/lib/api";

const PAGE_SIZE = 50;

const PASSTHROUGH = [
  "sector", "search", "passing_only", "sort_by", "desc", "offset",
  "min_health_score", "min_revenue", "min_revenue_growth",
  "min_operating_margin", "min_gross_margin", "min_fcf_margin",
  "min_roe", "min_roic", "max_net_debt_to_ebitda", "min_interest_coverage",
  "min_current_ratio", "max_debt_to_equity", "min_piotroski", "min_altman",
];

function Unavailable({ message }: { message: string }) {
  return (
    <div className="rounded-lg border border-fail/30 bg-fail/5 px-6 py-10">
      <h2 className="font-mono text-base font-semibold text-ink">
        The screener data service is not responding
      </h2>
      <p className="mt-2 max-w-xl text-sm text-body">{message}</p>
      <p className="mt-3 max-w-xl text-sm text-muted">
        Start the API with{" "}
        <code className="rounded bg-paper px-1.5 py-0.5 font-mono text-2xs">
          uvicorn finmodel.web.api:app --port 8000
        </code>{" "}
        and load data with{" "}
        <code className="rounded bg-paper px-1.5 py-0.5 font-mono text-2xs">
          python -m finmodel.web.seed
        </code>
        .
      </p>
    </div>
  );
}

export default async function ScreenerPage({
  searchParams,
}: {
  // Next 15: searchParams and params are Promises.
  searchParams: Promise<Record<string, string | string[] | undefined>>;
}) {
  const sp = await searchParams;
  const params: Record<string, string | undefined> = {};
  for (const key of PASSTHROUGH) {
    const raw = sp[key];
    const v = Array.isArray(raw) ? raw[0] : raw;
    if (v !== undefined && v !== "") params[key] = v;
  }

  let rows: Company[] = [];
  let total = 0;
  let sectors: SectorRow[] = [];
  let stats: Stats | null = null;
  let error: string | null = null;

  try {
    const [screen, sectorRows, statRow] = await Promise.all([
      fetchScreen({ ...params, limit: String(PAGE_SIZE) }),
      fetchSectors(),
      fetchStats(),
    ]);
    rows = screen.rows;
    total = screen.total;
    sectors = sectorRows;
    stats = statRow;
  } catch (e) {
    error = e instanceof Error ? e.message : String(e);
  }

  if (error) return <Unavailable message={error} />;

  const offset = Number(params.offset ?? 0);
  const showingTo = Math.min(offset + PAGE_SIZE, total);
  const nextParams = new URLSearchParams(
    Object.entries(params).filter(([k]) => k !== "offset") as [string, string][]
  );
  const prevParams = new URLSearchParams(nextParams.toString());
  nextParams.set("offset", String(offset + PAGE_SIZE));
  prevParams.set("offset", String(Math.max(0, offset - PAGE_SIZE)));

  return (
    <div className="space-y-5">
      <section className="flex flex-wrap items-end justify-between gap-5 border-b border-rule pb-5">
        <div>
          <h1 className="max-w-xl text-2xl font-semibold leading-tight tracking-tight text-ink sm:text-3xl">
            Screen the market on financial health, not price momentum
          </h1>
          <p className="mt-2 max-w-2xl text-sm leading-relaxed text-body">
            Twelve balance-sheet and cash-flow checks, four forensic accounting
            scores, and a rank against each company&rsquo;s own sector peers
            &mdash; computed from filed statements.
          </p>
        </div>
        {stats ? (
          <dl className="flex gap-6">
            <div>
              <dd className="font-mono text-2xl font-bold tabular-nums text-ink">
                {stats.companies.toLocaleString()}
              </dd>
              <dt className="text-2xs uppercase tracking-wider text-muted">
                companies
              </dt>
            </div>
            <div>
              <dd className="font-mono text-2xl font-bold tabular-nums text-pass">
                {(stats.passing ?? 0).toLocaleString()}
              </dd>
              <dt className="text-2xs uppercase tracking-wider text-muted">
                pass default screen
              </dt>
            </div>
            <div>
              <dd className="font-mono text-2xl font-bold tabular-nums text-ink">
                {stats.sectors}
              </dd>
              <dt className="text-2xs uppercase tracking-wider text-muted">
                sectors
              </dt>
            </div>
          </dl>
        ) : null}
      </section>

      <Filters sectors={sectors.map((s) => s.sector)} />

      <div className="flex items-baseline justify-between">
        <p className="font-mono text-2xs tabular-nums text-muted">
          {total === 0
            ? "no matches"
            : `showing ${offset + 1}–${showingTo} of ${total.toLocaleString()}`}
        </p>
        {!stats?.serves_market_data ? (
          <p className="text-2xs text-muted">
            Valuation multiples withheld &mdash; price data not licensed for
            redistribution
          </p>
        ) : null}
      </div>

      <ScreenTable rows={rows} params={params} />

      {total > PAGE_SIZE ? (
        <nav className="flex items-center justify-between" aria-label="Pagination">
          {offset > 0 ? (
            <Link
              href={`/?${prevParams.toString()}`}
              className="rounded border border-rule bg-panel px-3 py-1.5 text-sm text-body hover:border-accent/40"
            >
              ← Previous
            </Link>
          ) : (
            <span />
          )}
          {showingTo < total ? (
            <Link
              href={`/?${nextParams.toString()}`}
              className="rounded border border-rule bg-panel px-3 py-1.5 text-sm text-body hover:border-accent/40"
            >
              Next →
            </Link>
          ) : (
            <span />
          )}
        </nav>
      ) : null}

      {sectors.length > 0 ? (
        <section className="pt-4">
          <h2 className="mb-3 font-mono text-2xs uppercase tracking-[0.16em] text-muted">
            Pass rate by sector &mdash; why ranking is peer-relative
          </h2>
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-5">
            {sectors.map((s) => {
              const rate = s.companies ? (s.passing ?? 0) / s.companies : 0;
              return (
                <Link
                  key={s.sector}
                  href={`/?sector=${encodeURIComponent(s.sector)}`}
                  className="rounded border border-rule bg-panel px-3 py-2.5 hover:border-accent/40"
                >
                  <div className="truncate text-2xs font-medium text-ink">
                    {s.sector}
                  </div>
                  <div className="mt-1 font-mono text-sm tabular-nums text-body">
                    {s.passing ?? 0}
                    <span className="text-muted">/{s.companies}</span>
                  </div>
                  <div
                    className="mt-1.5 h-1 rounded-full bg-rule"
                    role="img"
                    aria-label={`${Math.round(rate * 100)} percent pass`}
                  >
                    <div
                      className="h-1 rounded-full bg-pass"
                      style={{ width: `${Math.max(2, rate * 100)}%` }}
                    />
                  </div>
                </Link>
              );
            })}
          </div>
          <p className="mt-3 max-w-2xl text-2xs leading-relaxed text-muted">
            Absolute thresholds favour asset-light sectors: a 30% gross-margin
            rule is trivial for software and unreachable for most retail. That is
            why the headline rank compares each company against its own sector
            rather than the whole market.
          </p>
        </section>
      ) : null}
    </div>
  );
}
