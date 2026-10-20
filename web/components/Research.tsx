import type { Finding, SectorComparison } from "@/lib/api";
import { DASH, mult, pct } from "@/lib/format";

/** Research findings. Deliberately no buy/sell verdict anywhere in this
 *  component — every item is a statement with the number it rests on, so a
 *  reader can check the claim rather than trust a label. */

const KIND_STYLE: Record<
  Finding["kind"],
  { ring: string; mark: string; markColor: string; title: string }
> = {
  strength: {
    ring: "border-l-pass",
    mark: "✓",
    markColor: "text-pass",
    title: "What the figures support",
  },
  concern: {
    ring: "border-l-fail",
    mark: "✕",
    markColor: "text-fail",
    title: "What the figures flag",
  },
  watch: {
    ring: "border-l-warn",
    mark: "!",
    markColor: "text-warn",
    title: "Worth watching",
  },
  context: {
    ring: "border-l-accent",
    mark: "·",
    markColor: "text-accent",
    title: "Context",
  },
};

export function FindingList({
  kind,
  items,
}: {
  kind: Finding["kind"];
  items: Finding[];
}) {
  if (items.length === 0) return null;
  const style = KIND_STYLE[kind];
  return (
    <div>
      <h3 className="mb-2 font-mono text-2xs uppercase tracking-[0.14em] text-muted">
        {style.title}
      </h3>
      <ul className="space-y-1.5">
        {items.map((f, i) => (
          <li
            key={`${f.metric ?? f.topic}-${i}`}
            className={`rounded-r border-l-2 bg-panel py-2 pl-3 pr-3 ${style.ring}`}
          >
            <div className="flex gap-2">
              <span
                className={`mt-0.5 font-mono text-xs font-bold ${style.markColor}`}
                aria-hidden="true"
              >
                {style.mark}
              </span>
              <div>
                <p className="text-sm leading-snug text-ink">{f.statement}</p>
                <p className="mt-0.5 font-mono text-2xs text-muted">
                  {f.evidence}
                </p>
              </div>
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}

/** Company value against the sector median, with the peer distribution behind
 *  it. Percentiles arrive already direction-adjusted, so a longer bar always
 *  means "better", whether the metric is a margin (higher) or a multiple
 *  (lower). */
export function PeerComparison({
  comparisons,
}: {
  comparisons: SectorComparison[];
}) {
  if (comparisons.length === 0) {
    return (
      <p className="text-sm text-muted">
        Not enough sector peers loaded to build a comparison.
      </p>
    );
  }

  const fmt = (c: SectorComparison, v: number | null) =>
    v === null ? DASH : c.is_percent ? pct(v) : mult(v);

  return (
    <div className="overflow-hidden rounded-lg border border-rule bg-panel">
      <table className="w-full border-collapse text-sm">
        <thead>
          <tr className="border-b border-rule">
            <th className="px-3 py-2 text-left text-2xs font-semibold uppercase tracking-wider text-muted">
              Metric
            </th>
            <th className="px-3 py-2 text-right text-2xs font-semibold uppercase tracking-wider text-muted">
              Company
            </th>
            <th className="px-3 py-2 text-right text-2xs font-semibold uppercase tracking-wider text-muted">
              Sector median
            </th>
            <th className="hidden px-3 py-2 text-left text-2xs font-semibold uppercase tracking-wider text-muted sm:table-cell">
              Position in sector
            </th>
          </tr>
        </thead>
        <tbody>
          {comparisons.map((c) => {
            const p = c.percentile ?? 0;
            const good = p >= 0.5;
            return (
              <tr key={c.metric} className="border-b border-rule/40 last:border-0">
                <td className="px-3 py-2 text-body">
                  {c.label}
                  <span className="ml-1.5 font-mono text-2xs text-muted">
                    n={c.peers}
                  </span>
                </td>
                <td className="px-3 py-2 text-right font-mono tabular-nums text-ink">
                  {fmt(c, c.value)}
                </td>
                <td className="px-3 py-2 text-right font-mono tabular-nums text-muted">
                  {fmt(c, c.sector_median)}
                </td>
                <td className="hidden px-3 py-2 sm:table-cell">
                  <span className="flex items-center gap-2">
                    <span
                      className="relative h-1.5 w-28 shrink-0 rounded-full bg-rule"
                      role="img"
                      aria-label={`${Math.round(p * 100)}th percentile, ${c.verdict}`}
                    >
                      <span
                        className={`absolute inset-y-0 left-0 rounded-full ${
                          good ? "bg-pass" : "bg-warn"
                        }`}
                        style={{ width: `${Math.max(3, p * 100)}%` }}
                      />
                    </span>
                    <span className="whitespace-nowrap text-2xs text-muted">
                      {c.verdict}
                    </span>
                  </span>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export function NoRecommendationNotice({ text }: { text: string }) {
  return (
    <aside className="rounded-lg border border-accent/25 bg-accent/5 px-4 py-3">
      <p className="font-mono text-2xs uppercase tracking-[0.14em] text-accent">
        No recommendation
      </p>
      <p className="mt-1 text-sm leading-relaxed text-body">{text}</p>
    </aside>
  );
}
