import Link from "next/link";
import type { Company } from "@/lib/api";
import { money, pct, mult, DASH } from "@/lib/format";
import { HealthMeter, SectorStrip, PassBadge } from "./Indicators";

type Col = {
  key: string;
  label: string;
  render: (c: Company) => React.ReactNode;
  align?: "left" | "right";
  hideBelow?: "sm" | "md" | "lg";
};

const COLUMNS: Col[] = [
  {
    key: "revenue",
    label: "Revenue",
    align: "right",
    render: (c) => money(c.revenue),
  },
  {
    key: "revenue_growth",
    label: "Growth",
    align: "right",
    render: (c) => pct(c.revenue_growth),
  },
  {
    key: "operating_margin",
    label: "Op margin",
    align: "right",
    render: (c) => pct(c.operating_margin),
  },
  {
    key: "fcf_margin",
    label: "FCF margin",
    align: "right",
    hideBelow: "md",
    render: (c) => pct(c.fcf_margin),
  },
  {
    key: "roic",
    label: "ROIC",
    align: "right",
    render: (c) => pct(c.roic),
  },
  {
    key: "net_debt_to_ebitda",
    label: "ND/EBITDA",
    align: "right",
    hideBelow: "lg",
    render: (c) => mult(c.net_debt_to_ebitda),
  },
  {
    key: "piotroski",
    label: "Piotroski",
    align: "right",
    hideBelow: "lg",
    render: (c) => (c.piotroski === null ? DASH : `${c.piotroski}/9`),
  },
];

const HIDE = {
  sm: "hidden sm:table-cell",
  md: "hidden md:table-cell",
  lg: "hidden lg:table-cell",
} as const;

function sortHref(
  params: Record<string, string | undefined>,
  key: string
): string {
  const next = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (v !== undefined && v !== "" && k !== "offset") next.set(k, v);
  }
  const currentlyOn = (params.sort_by ?? "sector_percentile") === key;
  const currentlyDesc = (params.desc ?? "true") !== "false";
  next.set("sort_by", key);
  next.set("desc", currentlyOn && currentlyDesc ? "false" : "true");
  return `/?${next.toString()}`;
}

export default function ScreenTable({
  rows,
  params,
}: {
  rows: Company[];
  params: Record<string, string | undefined>;
}) {
  const activeSort = params.sort_by ?? "sector_percentile";
  const descending = (params.desc ?? "true") !== "false";

  if (rows.length === 0) {
    return (
      <div className="rounded-lg border border-rule bg-panel px-6 py-14 text-center">
        <p className="font-mono text-sm text-ink">No companies match these filters.</p>
        <p className="mt-2 text-sm text-muted">
          Loosen a ratio threshold, or clear the sector to search the whole market.
        </p>
      </div>
    );
  }

  return (
    <div className="overflow-x-auto rounded-lg border border-rule bg-panel">
      <table className="w-full border-collapse text-sm">
        <thead>
          <tr className="border-b border-rule">
            <th
              scope="col"
              className="px-3 py-2.5 text-left text-2xs font-semibold uppercase tracking-wider text-muted"
            >
              Company
            </th>
            <th
              scope="col"
              className="px-3 py-2.5 text-left text-2xs font-semibold uppercase tracking-wider text-muted"
            >
              Health
            </th>
            <th
              scope="col"
              className="hidden px-3 py-2.5 text-left text-2xs font-semibold uppercase tracking-wider text-muted sm:table-cell"
            >
              <Link
                href={sortHref(params, "sector_percentile")}
                className="hover:text-ink"
              >
                Sector rank
                {activeSort === "sector_percentile" ? (descending ? " ↓" : " ↑") : ""}
              </Link>
            </th>
            {COLUMNS.map((col) => (
              <th
                key={col.key}
                scope="col"
                className={`px-3 py-2.5 text-right text-2xs font-semibold uppercase tracking-wider text-muted ${
                  col.hideBelow ? HIDE[col.hideBelow] : ""
                }`}
              >
                <Link href={sortHref(params, col.key)} className="hover:text-ink">
                  {col.label}
                  {activeSort === col.key ? (descending ? " ↓" : " ↑") : ""}
                </Link>
              </th>
            ))}
            <th scope="col" className="px-3 py-2.5">
              <span className="sr-only">Screen result</span>
            </th>
          </tr>
        </thead>
        <tbody>
          {rows.map((c) => {
            const passing = Boolean(c.passes_screen);
            return (
              <tr
                key={c.cik}
                className={`border-b border-rule/50 last:border-0 hover:bg-paper ${
                  passing ? "" : "opacity-[0.72]"
                }`}
              >
                <td className="px-3 py-2.5">
                  <Link
                    href={`/company/${c.ticker}`}
                    className="group block max-w-[13rem]"
                  >
                    <span className="font-mono text-sm font-semibold text-ink group-hover:underline">
                      {c.ticker}
                    </span>
                    <span className="block truncate text-2xs text-muted">
                      {c.name}
                    </span>
                  </Link>
                </td>
                <td className="whitespace-nowrap px-3 py-2.5">
                  <span className="flex items-center gap-2">
                    <HealthMeter passed={c.health_passed} total={c.health_total} />
                    <span className="font-mono text-2xs tabular-nums text-muted">
                      {c.health_passed ?? DASH}/{c.health_total ?? DASH}
                    </span>
                  </span>
                </td>
                <td className="hidden whitespace-nowrap px-3 py-2.5 sm:table-cell">
                  <span className="flex items-center gap-2">
                    <SectorStrip
                      percentile={c.sector_percentile}
                      sector={c.sector}
                    />
                    <span className="font-mono text-2xs tabular-nums text-muted">
                      {c.sector_percentile === null
                        ? DASH
                        : `${Math.round(c.sector_percentile * 100)}`}
                    </span>
                  </span>
                </td>
                {COLUMNS.map((col) => (
                  <td
                    key={col.key}
                    className={`whitespace-nowrap px-3 py-2.5 text-right font-mono tabular-nums text-ink ${
                      col.hideBelow ? HIDE[col.hideBelow] : ""
                    }`}
                  >
                    {col.render(c)}
                  </td>
                ))}
                <td className="px-3 py-2.5 text-right">
                  <PassBadge passing={passing} />
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
