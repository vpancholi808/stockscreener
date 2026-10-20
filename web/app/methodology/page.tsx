import Link from "next/link";
import { fetchMeta } from "@/lib/api";

export const metadata = {
  title: "Methodology — finmodel",
  description:
    "What the screener measures, where the data comes from, and what it cannot tell you.",
};

function Section({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <section className="border-t border-rule pt-6">
      <h2 className="font-mono text-2xs uppercase tracking-[0.16em] text-muted">
        {title}
      </h2>
      <div className="mt-3 space-y-3 text-sm leading-relaxed text-body">
        {children}
      </div>
    </section>
  );
}

export default async function MethodologyPage() {
  const meta = await fetchMeta().catch(() => null);
  const thresholds: Record<string, any> = meta?.scorecard_thresholds ?? {};

  const fmt = (v: number, key: string) =>
    key.includes("margin") || key.startsWith("ro")
      ? `${(v * 100).toFixed(0)}%`
      : `${v}×`;

  return (
    <article className="max-w-3xl space-y-6">
      <header>
        <h1 className="text-2xl font-semibold tracking-tight text-ink sm:text-3xl">
          Methodology
        </h1>
        <p className="mt-2 text-sm leading-relaxed text-body">
          Everything here is derived from statements companies filed with the SEC.
          No estimates, no analyst forecasts, no price targets.
        </p>
      </header>

      <Section title="Where the data comes from">
        <p>
          Fundamentals are extracted from XBRL-tagged annual filings (Form 10-K)
          published by the U.S. Securities and Exchange Commission. SEC filings
          are U.S. government work and in the public domain.
        </p>
        <p>
          Companies tag the same economic line item inconsistently, so each line
          maps to an ordered list of candidate tags and the first one that
          resolves is used. Where a figure cannot be resolved, the affected check
          is marked unavailable and excluded from the score rather than counted
          as a failure.
        </p>
      </Section>

      <Section title="The twelve checks">
        <p>
          Each check is a threshold test against one ratio. The health figure is
          simply how many passed, grouped into four dimensions, because a company
          can be profitable and still fail (a liquidity failure) or cash-rich and
          still be a poor business (a returns failure).
        </p>
        {Object.keys(thresholds).length > 0 ? (
          <div className="overflow-hidden rounded-lg border border-rule bg-panel">
            <table className="w-full border-collapse text-sm">
              <thead>
                <tr className="border-b border-rule">
                  <th className="px-3 py-2 text-left text-2xs font-semibold uppercase tracking-wider text-muted">
                    Ratio
                  </th>
                  <th className="px-3 py-2 text-left text-2xs font-semibold uppercase tracking-wider text-muted">
                    Dimension
                  </th>
                  <th className="px-3 py-2 text-right text-2xs font-semibold uppercase tracking-wider text-muted">
                    Threshold
                  </th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(thresholds).map(([key, t]) => (
                  <tr key={key} className="border-b border-rule/40 last:border-0">
                    <td className="px-3 py-1.5 font-mono text-2xs text-ink">
                      {key}
                    </td>
                    <td className="px-3 py-1.5 text-body">{t.group}</td>
                    <td className="px-3 py-1.5 text-right font-mono tabular-nums text-ink">
                      {t.min !== undefined
                        ? `≥ ${fmt(t.min, key)}`
                        : `≤ ${fmt(t.max, key)}`}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : null}
      </Section>

      <Section title="Forensic scores">
        <p>
          <strong className="font-semibold text-ink">Piotroski F-score</strong>{" "}
          runs nine binary tests covering profitability, leverage and operating
          efficiency; 8&ndash;9 indicates improving fundamentals.
        </p>
        <p>
          <strong className="font-semibold text-ink">Altman Z-score</strong>{" "}
          combines five weighted ratios into a bankruptcy-risk reading: above
          2.99 is the safe zone, below 1.81 the distress zone. For very large
          companies the market-value term dominates the formula, so treat it as a
          binary signal rather than a magnitude.
        </p>
        <p>
          <strong className="font-semibold text-ink">Beneish M-score</strong>{" "}
          weights eight variables that tend to move when earnings are being
          managed &mdash; receivables outrunning sales, deteriorating margins,
          rising accruals. A reading above &minus;1.78 means a company is worth a
          closer look. It is not an accusation.
        </p>
        <p>
          <strong className="font-semibold text-ink">Accruals ratio</strong>{" "}
          measures the gap between reported profit and cash actually generated.
          A persistent gap is the most reliable early warning in the set.
        </p>
      </Section>

      <Section title="Why ranking is sector-relative">
        <p>
          Absolute thresholds systematically favour asset-light businesses. A 30%
          gross-margin rule is trivial for software and effectively unreachable
          for grocery retail; a leverage limit that flags a utility as dangerous
          describes how utilities are normally financed. Screening the market on
          one fixed set of cutoffs mostly rediscovers which sectors have good
          margins.
        </p>
        <p>
          The headline rank therefore places each company in the percentile
          distribution of its own sector. The absolute checks are still shown,
          because they are informative &mdash; they just are not the ranking.
        </p>
      </Section>

      <Section title="What this cannot tell you">
        <ul className="list-disc space-y-2 pl-5">
          <li>
            <strong className="font-semibold text-ink">It is annual data.</strong>{" "}
            Figures come from the last filed 10-K. For a company mid-cycle, that
            can understate or overstate current trading by a wide margin.
          </li>
          <li>
            <strong className="font-semibold text-ink">
              No forward estimates.
            </strong>{" "}
            Forward P/E and growth forecasts require analyst consensus data,
            which is not published by the SEC and is not included here.
          </li>
          <li>
            <strong className="font-semibold text-ink">
              Financial health is not valuation.
            </strong>{" "}
            A company can pass every check and still be expensive. These measures
            describe the business, not the price.
          </li>
          <li>
            <strong className="font-semibold text-ink">US filers only.</strong>{" "}
            Foreign issuers filing on Form 20-F have partial coverage.
          </li>
        </ul>
      </Section>

      <Section title="Not investment advice">
        <p>
          This is analysis tooling. It applies published, decades-old accounting
          tests to public filings and shows the result. It does not know your
          circumstances, it does not forecast prices, and it is not a
          recommendation to buy or sell anything. Verify every figure against the
          original filing before acting on it.
        </p>
        <p>
          <Link href="/" className="text-accent underline decoration-dotted">
            Back to the screener
          </Link>
        </p>
      </Section>
    </article>
  );
}
