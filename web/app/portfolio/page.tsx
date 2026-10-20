import Link from "next/link";
import PortfolioControls from "@/components/PortfolioControls";
import { buildPortfolio, fetchPortfolioOptions } from "@/lib/api";
import type { PortfolioResult } from "@/lib/api";
import { DASH, mult, pct } from "@/lib/format";

export const metadata = {
  title: "Portfolio builder — finmodel",
  description:
    "Construct a portfolio from fundamental criteria, risk limits and allocation method.",
};

function Stat({
  label,
  value,
  hint,
}: {
  label: string;
  value: string;
  hint?: string;
}) {
  return (
    <div className="bg-panel px-3.5 py-3">
      <div className="text-2xs uppercase tracking-wide text-muted">{label}</div>
      <div className="mt-0.5 font-mono text-lg font-semibold tabular-nums text-ink">
        {value}
      </div>
      {hint ? <div className="mt-0.5 text-2xs text-muted">{hint}</div> : null}
    </div>
  );
}

function SectorBars({ weights }: { weights: Record<string, number> }) {
  const entries = Object.entries(weights).sort((a, b) => b[1] - a[1]);
  if (entries.length === 0) return null;
  return (
    <div className="space-y-1.5">
      {entries.map(([sector, w]) => (
        <div key={sector} className="grid grid-cols-[1fr_auto] items-center gap-3">
          <div>
            <div className="flex items-baseline justify-between">
              <span className="text-sm text-body">{sector}</span>
              <span className="font-mono text-2xs tabular-nums text-muted">
                {pct(w, 0)}
              </span>
            </div>
            <div className="mt-1 h-1.5 rounded-full bg-rule">
              <div
                className="h-1.5 rounded-full bg-accent"
                style={{ width: `${Math.max(2, w * 100)}%` }}
              />
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

export default async function PortfolioPage({
  searchParams,
}: {
  searchParams: Promise<Record<string, string | string[] | undefined>>;
}) {
  const sp = await searchParams;
  const options = await fetchPortfolioOptions();
  if (!options) {
    return (
      <div className="rounded-lg border border-fail/30 bg-fail/5 px-6 py-10">
        <h1 className="font-mono text-base font-semibold text-ink">
          Portfolio service unavailable
        </h1>
        <p className="mt-2 text-sm text-body">
          Start the API and load data, then reload this page.
        </p>
      </div>
    );
  }

  const one = (k: string, fallback: string) => {
    const raw = sp[k];
    const v = Array.isArray(raw) ? raw[0] : raw;
    return v && v !== "" ? v : fallback;
  };

  const strategy = one("strategy", "quality");
  const risk = one("risk", "balanced");
  const allocation = one("allocation", "score");
  const maxHoldingsRaw = one("max_holdings", "");
  const maxHoldings = maxHoldingsRaw ? Number(maxHoldingsRaw) : undefined;

  const result: PortfolioResult | null = await buildPortfolio({
    strategy,
    risk_profile: risk,
    allocation,
    max_holdings:
      maxHoldings && Number.isFinite(maxHoldings) ? maxHoldings : undefined,
  });

  const strat = options.strategies.find((s) => s.key === strategy);
  const riskOpt = options.risk_profiles.find((r) => r.key === risk);
  const m = result?.metrics ?? {};

  return (
    <div className="space-y-5">
      <header className="border-b border-rule pb-5">
        <h1 className="text-2xl font-semibold tracking-tight text-ink sm:text-3xl">
          Portfolio builder
        </h1>
        <p className="mt-2 max-w-3xl text-sm leading-relaxed text-body">
          Screen the universe by investment strategy, then allocate under risk
          limits. Risk here is enforced as <strong>constraints</strong> &mdash;
          position caps, sector caps, solvency floors &mdash; rather than as a
          score, because a limit you cannot breach is worth more than a number
          you can argue with.
        </p>
      </header>

      <PortfolioControls
        options={options}
        current={{ strategy, risk, allocation, max_holdings: maxHoldingsRaw }}
      />

      {strat && riskOpt ? (
        <div className="grid gap-3 sm:grid-cols-2">
          <div className="rounded-lg border border-rule bg-panel px-4 py-3">
            <div className="font-mono text-2xs uppercase tracking-wider text-muted">
              {strat.name} strategy
            </div>
            <p className="mt-1 text-sm leading-relaxed text-body">
              {strat.description}
            </p>
          </div>
          <div className="rounded-lg border border-rule bg-panel px-4 py-3">
            <div className="font-mono text-2xs uppercase tracking-wider text-muted">
              {riskOpt.name} risk profile
            </div>
            <p className="mt-1 text-sm leading-relaxed text-body">
              {riskOpt.description}
            </p>
            <p className="mt-1.5 font-mono text-2xs text-muted">
              max {pct(riskOpt.max_position, 0)} per holding &middot; max{" "}
              {pct(riskOpt.max_sector, 0)} per sector &middot; target{" "}
              {riskOpt.target_holdings} holdings
            </p>
          </div>
        </div>
      ) : null}

      {!result || result.holdings.length === 0 ? (
        <div className="rounded-lg border border-rule bg-panel px-6 py-12 text-center">
          <p className="font-mono text-sm text-ink">
            No companies met both the strategy filters and the risk limits.
          </p>
          <p className="mt-2 text-sm text-muted">
            Loosen the risk profile, pick a broader strategy, or load more
            companies into the database.
          </p>
          {result?.warnings?.length ? (
            <ul className="mx-auto mt-3 max-w-lg space-y-1 text-2xs text-muted">
              {result.warnings.map((w) => (
                <li key={w}>{w}</li>
              ))}
            </ul>
          ) : null}
        </div>
      ) : (
        <>
          <section>
            <h2 className="mb-3 font-mono text-2xs uppercase tracking-[0.16em] text-muted">
              Resulting book
            </h2>
            <div className="grid grid-cols-2 gap-px overflow-hidden rounded-lg border border-rule bg-rule sm:grid-cols-3 lg:grid-cols-6">
              <Stat label="Holdings" value={String(m.holdings ?? DASH)} />
              <Stat
                label="Largest position"
                value={pct(m.largest_position as number, 1)}
              />
              <Stat
                label="Effective holdings"
                value={
                  m.effective_holdings
                    ? (m.effective_holdings as number).toFixed(1)
                    : DASH
                }
                hint="after weighting"
              />
              <Stat label="Sectors" value={String(m.sectors ?? DASH)} />
              <Stat
                label="Weighted ROIC"
                value={pct(m.weighted_roic as number)}
              />
              <Stat
                label="Weighted growth"
                value={pct(m.weighted_revenue_growth as number)}
              />
            </div>
            <p className="mt-2 text-2xs text-muted">
              Screened {result.universe_size} companies &rarr;{" "}
              {result.eligible_size} passed the filters &rarr;{" "}
              {result.holdings.length} selected.
            </p>
          </section>

          {result.warnings.length > 0 ? (
            <div className="rounded-lg border border-warn/30 bg-warn/5 px-4 py-3">
              <h3 className="font-mono text-2xs uppercase tracking-wider text-warn">
                Constraints worth knowing about
              </h3>
              <ul className="mt-1.5 space-y-1 text-sm text-body">
                {result.warnings.map((w) => (
                  <li key={w}>{w}</li>
                ))}
              </ul>
            </div>
          ) : null}

          <section className="grid gap-5 lg:grid-cols-[1.7fr_1fr]">
            <div>
              <h2 className="mb-3 font-mono text-2xs uppercase tracking-[0.16em] text-muted">
                Holdings
              </h2>
              <div className="overflow-x-auto rounded-lg border border-rule bg-panel">
                <table className="w-full border-collapse text-sm">
                  <thead>
                    <tr className="border-b border-rule">
                      <th className="px-3 py-2 text-left text-2xs font-semibold uppercase tracking-wider text-muted">
                        Company
                      </th>
                      <th className="px-3 py-2 text-right text-2xs font-semibold uppercase tracking-wider text-muted">
                        Weight
                      </th>
                      <th className="hidden px-3 py-2 text-left text-2xs font-semibold uppercase tracking-wider text-muted sm:table-cell">
                        Sector
                      </th>
                      <th className="px-3 py-2 text-right text-2xs font-semibold uppercase tracking-wider text-muted">
                        Health
                      </th>
                      <th className="px-3 py-2 text-right text-2xs font-semibold uppercase tracking-wider text-muted">
                        ROIC
                      </th>
                      <th className="hidden px-3 py-2 text-right text-2xs font-semibold uppercase tracking-wider text-muted md:table-cell">
                        Growth
                      </th>
                      <th className="hidden px-3 py-2 text-right text-2xs font-semibold uppercase tracking-wider text-muted lg:table-cell">
                        ND/EBITDA
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.holdings.map((h) => (
                      <tr
                        key={h.ticker}
                        className="border-b border-rule/40 last:border-0 hover:bg-paper"
                      >
                        <td className="px-3 py-2">
                          <Link
                            href={`/company/${h.ticker}`}
                            className="group block max-w-[12rem]"
                          >
                            <span className="font-mono text-sm font-semibold text-ink group-hover:underline">
                              {h.ticker}
                            </span>
                            <span className="block truncate text-2xs text-muted">
                              {h.name}
                            </span>
                          </Link>
                        </td>
                        <td className="px-3 py-2 text-right">
                          <span className="flex items-center justify-end gap-2">
                            <span className="h-1.5 w-12 rounded-full bg-rule">
                              <span
                                className="block h-1.5 rounded-full bg-accent"
                                style={{
                                  width: `${Math.max(
                                    4,
                                    (h.weight /
                                      (result.metrics
                                        .largest_position as number)) *
                                      100
                                  )}%`,
                                }}
                              />
                            </span>
                            <span className="font-mono tabular-nums text-ink">
                              {pct(h.weight, 1)}
                            </span>
                          </span>
                        </td>
                        <td className="hidden truncate px-3 py-2 text-2xs text-muted sm:table-cell">
                          {h.sector}
                        </td>
                        <td className="px-3 py-2 text-right font-mono text-2xs tabular-nums text-muted">
                          {h.health_passed ?? DASH}/{h.health_total ?? DASH}
                        </td>
                        <td className="px-3 py-2 text-right font-mono tabular-nums text-ink">
                          {pct(h.roic)}
                        </td>
                        <td className="hidden px-3 py-2 text-right font-mono tabular-nums text-ink md:table-cell">
                          {pct(h.revenue_growth)}
                        </td>
                        <td className="hidden px-3 py-2 text-right font-mono tabular-nums text-ink lg:table-cell">
                          {mult(h.net_debt_to_ebitda)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            <div className="space-y-5">
              <div>
                <h2 className="mb-3 font-mono text-2xs uppercase tracking-[0.16em] text-muted">
                  Sector exposure
                </h2>
                <div className="rounded-lg border border-rule bg-panel px-4 py-3">
                  <SectorBars weights={result.sector_weights} />
                </div>
              </div>

              <div>
                <h2 className="mb-3 font-mono text-2xs uppercase tracking-[0.16em] text-muted">
                  Weighted characteristics
                </h2>
                <div className="rounded-lg border border-rule bg-panel px-4 py-3">
                  {(
                    [
                      ["Operating margin", "weighted_operating_margin", "pct"],
                      ["FCF margin", "weighted_fcf_margin", "pct"],
                      ["Return on equity", "weighted_roe", "pct"],
                      ["Net debt / EBITDA", "weighted_net_debt_to_ebitda", "x"],
                      ["Health score", "weighted_health_score", "pct"],
                      ["Piotroski", "weighted_piotroski", "num"],
                    ] as const
                  ).map(([label, key, kind]) => {
                    const v = m[key] as number | null | undefined;
                    return (
                      <div
                        key={key}
                        className="flex items-center justify-between border-b border-rule/40 py-1.5 text-sm last:border-0"
                      >
                        <span className="text-body">{label}</span>
                        <span className="font-mono tabular-nums text-ink">
                          {v === null || v === undefined
                            ? DASH
                            : kind === "pct"
                            ? pct(v)
                            : kind === "x"
                            ? mult(v)
                            : v.toFixed(1)}
                        </span>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          </section>

          {result.no_recommendation ? (
            <aside className="rounded-lg border border-accent/25 bg-accent/5 px-4 py-3">
              <p className="font-mono text-2xs uppercase tracking-[0.14em] text-accent">
                No recommendation
              </p>
              <p className="mt-1 max-w-3xl text-sm leading-relaxed text-body">
                {result.no_recommendation} Past financial health does not
                predict future returns, and a screen cannot account for your
                circumstances, tax position, time horizon or anything outside
                the filed numbers.
              </p>
            </aside>
          ) : null}
        </>
      )}
    </div>
  );
}
