import Link from "next/link";
import { notFound } from "next/navigation";
import { HealthMeter, ScoreChip, SectorStrip, Sparkline } from "@/components/Indicators";
import { FindingList, NoRecommendationNotice, PeerComparison } from "@/components/Research";
import { fetchCompany, fetchMeta, fetchResearch } from "@/lib/api";
import {
  DASH, altmanZone, beneishTone, money, mult, num, pct, piotroskiTone, verdict,
} from "@/lib/format";

type Row = {
  key: string;
  label: string;
  value: number | null;
  threshold: number;
  higherIsBetter: boolean;
  fmt: (v: number | null) => string;
};

export default async function CompanyPage({
  params,
}: {
  params: Promise<{ ticker: string }>;
}) {
  const { ticker } = await params;
  const [c, meta, research] = await Promise.all([
    fetchCompany(ticker),
    fetchMeta().catch(() => null),
    fetchResearch(ticker),
  ]);
  if (!c) notFound();

  const t = meta?.scorecard_thresholds ?? {};
  const thr = (k: string, fallback: number) =>
    t[k]?.min ?? t[k]?.max ?? fallback;

  const groups: { title: string; rows: Row[] }[] = [
    {
      title: "Solvency",
      rows: [
        { key: "debt_to_equity", label: "Debt / equity", value: c.debt_to_equity, threshold: thr("debt_to_equity", 2), higherIsBetter: false, fmt: mult },
        { key: "net_debt_to_ebitda", label: "Net debt / EBITDA", value: c.net_debt_to_ebitda, threshold: thr("net_debt_to_ebitda", 3), higherIsBetter: false, fmt: mult },
        { key: "interest_coverage", label: "Interest coverage", value: c.interest_coverage, threshold: thr("interest_coverage", 3), higherIsBetter: true, fmt: mult },
      ],
    },
    {
      title: "Liquidity",
      rows: [
        { key: "current_ratio", label: "Current ratio", value: c.current_ratio, threshold: thr("current_ratio", 1.2), higherIsBetter: true, fmt: mult },
      ],
    },
    {
      title: "Cash-flow quality",
      rows: [
        { key: "fcf_margin", label: "FCF margin", value: c.fcf_margin, threshold: thr("fcf_margin", 0.05), higherIsBetter: true, fmt: (v) => pct(v) },
        { key: "cash_conversion", label: "Cash conversion (OCF/NI)", value: c.cash_conversion, threshold: thr("cash_conversion", 0.8), higherIsBetter: true, fmt: mult },
      ],
    },
    {
      title: "Profitability",
      rows: [
        { key: "gross_margin", label: "Gross margin", value: c.gross_margin, threshold: thr("gross_margin", 0.3), higherIsBetter: true, fmt: (v) => pct(v) },
        { key: "operating_margin", label: "Operating margin", value: c.operating_margin, threshold: thr("operating_margin", 0.1), higherIsBetter: true, fmt: (v) => pct(v) },
        { key: "roe", label: "Return on equity", value: c.roe, threshold: thr("roe", 0.12), higherIsBetter: true, fmt: (v) => pct(v) },
        { key: "roic", label: "Return on invested capital", value: c.roic, threshold: thr("roic", 0.08), higherIsBetter: true, fmt: (v) => pct(v) },
      ],
    },
  ];

  const hist = c.history ?? [];
  const series = (k: keyof (typeof hist)[number]) =>
    hist.map((h) => (h[k] as number | null) ?? null);
  const marginSeries = hist.map((h) =>
    h.operating_income !== null && h.revenue ? h.operating_income / h.revenue : null
  );

  const headline = [
    { label: "Revenue", value: money(c.revenue), series: series("revenue") },
    { label: "EBITDA", value: money(c.ebitda), series: series("ebitda") },
    { label: "Net income", value: money(hist.at(-1)?.net_income ?? null), series: series("net_income") },
    { label: "Free cash flow", value: money(c.fcf), series: series("fcf") },
    { label: "Net debt", value: money(c.net_debt), series: null },
    { label: "Operating margin", value: pct(c.operating_margin), series: marginSeries },
  ];

  const az = altmanZone(c.altman);
  const passing = Boolean(c.passes_screen);

  return (
    <div className="space-y-6">
      <nav className="font-mono text-2xs text-muted">
        <Link href="/" className="hover:text-ink">
          screener
        </Link>
        <span className="px-1.5">/</span>
        <span className="text-body">{c.ticker}</span>
      </nav>

      <header className="flex flex-wrap items-end justify-between gap-5 border-b-2 border-ink pb-5">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="font-mono text-3xl font-bold tracking-tight text-ink">
              {c.ticker}
            </h1>
            {passing ? (
              <span className="rounded bg-pass/10 px-2 py-1 font-mono text-2xs font-bold tracking-wide text-pass">
                PASSES SCREEN
              </span>
            ) : (
              <span className="rounded border border-rule px-2 py-1 font-mono text-2xs tracking-wide text-muted">
                DOES NOT PASS
              </span>
            )}
          </div>
          <p className="mt-1 text-base text-body">{c.name}</p>
          <p className="mt-1 font-mono text-2xs uppercase tracking-wider text-muted">
            {c.sector} &middot; FY{c.fiscal_year} as filed &middot; data quality:{" "}
            {c.data_quality ?? "unknown"}
          </p>
        </div>
        <div className="text-right">
          <div className="font-mono text-3xl font-bold tabular-nums text-ink">
            {c.health_passed ?? DASH}
            <span className="text-xl text-muted">/{c.health_total ?? DASH}</span>
          </div>
          <div className="text-2xs uppercase tracking-wider text-muted">
            checks passed
          </div>
          <div className="mt-2 flex justify-end">
            <HealthMeter
              passed={c.health_passed}
              total={c.health_total}
              width={6}
              gap={3}
              height={13}
            />
          </div>
        </div>
      </header>

      {research ? (
        <section className="space-y-4">
          <div className="rounded-lg border border-rule bg-panel px-5 py-4">
            <h2 className="font-mono text-2xs uppercase tracking-[0.16em] text-muted">
              Research summary
            </h2>
            <p className="mt-2 text-base font-medium leading-snug text-ink">
              {research.research.headline}
            </p>
            <p className="mt-2 max-w-3xl text-sm leading-relaxed text-body">
              {research.research.summary}
            </p>
          </div>

          <div className="grid gap-4 lg:grid-cols-2">
            <div className="space-y-4">
              <FindingList kind="strength" items={research.research.strengths} />
              <FindingList kind="context" items={research.research.context} />
            </div>
            <div className="space-y-4">
              <FindingList kind="concern" items={research.research.concerns} />
              <FindingList kind="watch" items={research.research.watch_items} />
            </div>
          </div>

          <NoRecommendationNotice text={research.no_recommendation} />
        </section>
      ) : null}

      {research && research.comparisons.length > 0 ? (
        <section>
          <h2 className="mb-1 font-mono text-2xs uppercase tracking-[0.16em] text-muted">
            Against {c.sector} peers
          </h2>
          <p className="mb-3 max-w-3xl text-2xs leading-relaxed text-muted">
            Compared against the sector <strong className="font-semibold">median</strong>,
            not the mean. Valuation multiples are right-skewed &mdash; a single
            company on a very high multiple pulls a sector mean far above what a
            typical constituent trades at.
          </p>
          <PeerComparison comparisons={research.comparisons} />
        </section>
      ) : null}

      <section>
        <h2 className="mb-3 font-mono text-2xs uppercase tracking-[0.16em] text-muted">
          Headline figures
        </h2>
        <div className="grid grid-cols-2 gap-px overflow-hidden rounded-lg border border-rule bg-rule sm:grid-cols-3 lg:grid-cols-6">
          {headline.map((h) => (
            <div key={h.label} className="bg-panel px-3.5 py-3">
              <div className="text-2xs uppercase tracking-wide text-muted">
                {h.label}
              </div>
              <div className="mt-0.5 font-mono text-lg font-semibold tabular-nums text-ink">
                {h.value}
              </div>
              <div className="mt-1 h-6">
                {h.series ? <Sparkline values={h.series} /> : null}
              </div>
            </div>
          ))}
        </div>
      </section>

      <section>
        <h2 className="mb-3 font-mono text-2xs uppercase tracking-[0.16em] text-muted">
          Forensic &amp; peer position
        </h2>
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-5">
          <ScoreChip
            label="Piotroski F"
            value={c.piotroski === null ? DASH : `${c.piotroski}/9`}
            tone={piotroskiTone(c.piotroski)}
            hint="Nine tests of improving fundamentals"
          />
          <ScoreChip
            label="Altman Z"
            value={num(c.altman)}
            tone={az.tone}
            hint={`Bankruptcy risk — ${az.label} zone`}
          />
          <ScoreChip
            label="Beneish M"
            value={num(c.beneish)}
            tone={beneishTone(c.beneish)}
            hint="Above −1.78 warrants scrutiny"
          />
          <ScoreChip
            label="Accruals"
            value={num(c.accruals, 3)}
            tone={
              c.accruals === null ? "none" : c.accruals > 0.1 ? "fail" : "pass"
            }
            hint="Gap between profit and cash"
          />
          <div className="rounded border border-rule bg-paper px-3 py-2.5">
            <div className="text-2xs uppercase tracking-wider text-muted">
              Rank in {c.sector}
            </div>
            <div className="mt-1 font-mono text-lg font-semibold tabular-nums text-ink">
              {c.sector_percentile === null
                ? DASH
                : `${Math.round(c.sector_percentile * 100)}th`}
            </div>
            <div className="mt-1">
              <SectorStrip
                percentile={c.sector_percentile}
                sector={c.sector}
                width={110}
              />
            </div>
          </div>
        </div>
      </section>

      <section className="grid gap-5 lg:grid-cols-[1.1fr_1fr]">
        <div>
          <h2 className="mb-3 font-mono text-2xs uppercase tracking-[0.16em] text-muted">
            Scorecard by dimension
          </h2>
          <div className="rounded-lg border border-rule bg-panel px-4 py-3">
            {groups.map((g) => (
              <div key={g.title} className="py-2">
                <h3 className="mb-1.5 text-2xs font-semibold uppercase tracking-wider text-muted">
                  {g.title}
                </h3>
                {g.rows.map((r) => {
                  const v = verdict(r.value, r.threshold, r.higherIsBetter);
                  return (
                    <div
                      key={r.key}
                      className="grid grid-cols-[1.2rem_1fr_auto_4.5rem] items-center gap-2 border-b border-rule/40 py-1.5 last:border-0 text-sm"
                    >
                      <span
                        className={`font-mono text-xs font-bold ${
                          v === "pass"
                            ? "text-pass"
                            : v === "fail"
                            ? "text-fail"
                            : "text-muted"
                        }`}
                        aria-hidden="true"
                      >
                        {v === "pass" ? "✓" : v === "fail" ? "✕" : "·"}
                      </span>
                      <span className="text-body">{r.label}</span>
                      <span
                        className={`font-mono tabular-nums ${
                          v === "fail" ? "text-fail" : "text-ink"
                        }`}
                      >
                        {r.fmt(r.value)}
                      </span>
                      <span className="text-right font-mono text-2xs text-muted">
                        {r.higherIsBetter ? "≥" : "≤"} {r.fmt(r.threshold)}
                      </span>
                    </div>
                  );
                })}
              </div>
            ))}
          </div>
        </div>

        <div>
          <h2 className="mb-3 font-mono text-2xs uppercase tracking-[0.16em] text-muted">
            Reported history
          </h2>
          <div className="overflow-x-auto rounded-lg border border-rule bg-panel">
            <table className="w-full border-collapse text-sm">
              <thead>
                <tr className="border-b border-rule">
                  <th className="px-3 py-2 text-left text-2xs font-semibold uppercase tracking-wider text-muted">
                    Fiscal year
                  </th>
                  {hist.map((h) => (
                    <th
                      key={h.fiscal_year}
                      className="px-3 py-2 text-right font-mono text-2xs font-semibold tabular-nums text-muted"
                    >
                      {h.fiscal_year}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {(
                  [
                    ["Revenue", "revenue"],
                    ["Gross profit", "gross_profit"],
                    ["Operating income", "operating_income"],
                    ["Net income", "net_income"],
                    ["Operating cash flow", "operating_cash_flow"],
                    ["Capex", "capex"],
                    ["Free cash flow", "fcf"],
                    ["Total assets", "total_assets"],
                    ["Total equity", "total_equity"],
                    ["Total debt", "total_debt"],
                  ] as const
                ).map(([label, key]) => (
                  <tr key={key} className="border-b border-rule/40 last:border-0">
                    <td className="px-3 py-1.5 text-body">{label}</td>
                    {hist.map((h) => (
                      <td
                        key={h.fiscal_year}
                        className="px-3 py-1.5 text-right font-mono tabular-nums text-ink"
                      >
                        {money((h[key] as number | null) ?? null)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="mt-2 text-2xs leading-relaxed text-muted">
            Figures as reported in annual filings. Annual data only &mdash; for a
            company mid-cycle the latest filed year can lag current trading
            materially.
          </p>
          {research?.research.caveats?.length ? (
            <div className="mt-4 rounded-lg border border-rule bg-panel px-4 py-3">
              <h3 className="font-mono text-2xs uppercase tracking-[0.14em] text-muted">
                What this analysis cannot tell you
              </h3>
              <ul className="mt-2 list-disc space-y-1 pl-4 text-2xs leading-relaxed text-muted">
                {research.research.caveats.map((cav) => (
                  <li key={cav}>{cav}</li>
                ))}
              </ul>
            </div>
          ) : null}
        </div>
      </section>
    </div>
  );
}
