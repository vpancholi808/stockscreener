"use client";

import { useRouter, useSearchParams } from "next/navigation";
import { useCallback, useState, useTransition } from "react";

/** Filters live entirely in the URL. That keeps the screener page a server
 *  component (no client data fetching, no state library) and makes every view
 *  a shareable link — which matters for a screener people will send around. */

const NUMERIC: { key: string; label: string; unit: "pct" | "x" | "n" | "$"; step: string }[] = [
  { key: "min_revenue_growth", label: "Revenue growth ≥", unit: "pct", step: "1" },
  { key: "min_operating_margin", label: "Operating margin ≥", unit: "pct", step: "1" },
  { key: "min_fcf_margin", label: "FCF margin ≥", unit: "pct", step: "1" },
  { key: "min_roic", label: "ROIC ≥", unit: "pct", step: "1" },
  { key: "min_roe", label: "ROE ≥", unit: "pct", step: "1" },
  { key: "max_net_debt_to_ebitda", label: "Net debt / EBITDA ≤", unit: "x", step: "0.5" },
  { key: "min_interest_coverage", label: "Interest coverage ≥", unit: "x", step: "0.5" },
  { key: "min_current_ratio", label: "Current ratio ≥", unit: "x", step: "0.1" },
  { key: "min_piotroski", label: "Piotroski ≥", unit: "n", step: "1" },
  { key: "min_revenue", label: "Revenue ≥ ($M)", unit: "$", step: "100" },
];

export default function Filters({ sectors }: { sectors: string[] }) {
  const router = useRouter();
  const params = useSearchParams();
  const [pending, startTransition] = useTransition();
  const [open, setOpen] = useState(false);

  const current = useCallback(
    (key: string) => {
      const raw = params.get(key);
      if (raw === null) return "";
      const f = NUMERIC.find((n) => n.key === key);
      if (!f) return raw;
      const v = Number(raw);
      if (Number.isNaN(v)) return "";
      if (f.unit === "pct") return String(Math.round(v * 1000) / 10);
      if (f.unit === "$") return String(Math.round(v / 1e6));
      return String(v);
    },
    [params]
  );

  const push = useCallback(
    (next: URLSearchParams) => {
      next.delete("offset");
      startTransition(() => router.push(`/?${next.toString()}`, { scroll: false }));
    },
    [router]
  );

  const setNumeric = (key: string, raw: string) => {
    const next = new URLSearchParams(params.toString());
    const f = NUMERIC.find((n) => n.key === key);
    if (raw === "" || Number.isNaN(Number(raw))) {
      next.delete(key);
    } else {
      const v = Number(raw);
      const scaled =
        f?.unit === "pct" ? v / 100 : f?.unit === "$" ? v * 1e6 : v;
      next.set(key, String(scaled));
    }
    push(next);
  };

  const setParam = (key: string, value: string) => {
    const next = new URLSearchParams(params.toString());
    if (value === "") next.delete(key);
    else next.set(key, value);
    push(next);
  };

  const activeCount = NUMERIC.filter((n) => params.get(n.key) !== null).length;
  const passingOnly = params.get("passing_only") === "true";

  return (
    <section
      className={`rounded-lg border border-rule bg-panel transition-opacity ${
        pending ? "opacity-60" : ""
      }`}
      aria-busy={pending}
    >
      <div className="flex flex-wrap items-center gap-3 px-4 py-3">
        <label className="relative flex-1 min-w-[180px]">
          <span className="sr-only">Search by ticker or company name</span>
          <input
            type="search"
            defaultValue={params.get("search") ?? ""}
            placeholder="Ticker or company name"
            onChange={(e) => {
              const v = e.target.value;
              window.clearTimeout(
                (window as any).__fmSearchTimer
              );
              (window as any).__fmSearchTimer = window.setTimeout(
                () => setParam("search", v),
                300
              );
            }}
            className="w-full rounded border border-rule bg-paper px-3 py-1.5 font-mono text-sm text-ink placeholder:text-muted focus:border-accent focus:outline-none focus:ring-1 focus:ring-accent/30"
          />
        </label>

        <label className="flex items-center gap-2">
          <span className="sr-only">Sector</span>
          <select
            value={params.get("sector") ?? ""}
            onChange={(e) => setParam("sector", e.target.value)}
            className="rounded border border-rule bg-paper px-2 py-1.5 text-sm text-ink focus:border-accent focus:outline-none"
          >
            <option value="">All sectors</option>
            {sectors.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
        </label>

        <button
          type="button"
          onClick={() => setParam("passing_only", passingOnly ? "" : "true")}
          aria-pressed={passingOnly}
          className={`rounded border px-3 py-1.5 text-sm transition-colors ${
            passingOnly
              ? "border-pass/40 bg-pass/10 text-pass"
              : "border-rule bg-paper text-body hover:border-accent/40"
          }`}
        >
          Passing only
        </button>

        <button
          type="button"
          onClick={() => setOpen((v) => !v)}
          aria-expanded={open}
          className="rounded border border-rule bg-paper px-3 py-1.5 text-sm text-body hover:border-accent/40"
        >
          Ratios{activeCount ? ` (${activeCount})` : ""}
        </button>

        {(activeCount > 0 || params.toString()) && (
          <button
            type="button"
            onClick={() => startTransition(() => router.push("/"))}
            className="text-sm text-muted underline decoration-dotted hover:text-ink"
          >
            Reset
          </button>
        )}
      </div>

      {open && (
        <div className="grid grid-cols-1 gap-x-6 gap-y-3 border-t border-rule px-4 py-4 sm:grid-cols-2 lg:grid-cols-3">
          {NUMERIC.map((f) => (
            <label key={f.key} className="flex items-center justify-between gap-3">
              <span className="text-sm text-body">{f.label}</span>
              <span className="flex items-center gap-1">
                <input
                  type="number"
                  step={f.step}
                  defaultValue={current(f.key)}
                  onBlur={(e) => setNumeric(f.key, e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") setNumeric(f.key, e.currentTarget.value);
                  }}
                  className="w-20 rounded border border-rule bg-paper px-2 py-1 text-right font-mono text-sm tabular-nums text-ink focus:border-accent focus:outline-none"
                />
                <span className="w-3 font-mono text-2xs text-muted">
                  {f.unit === "pct" ? "%" : f.unit === "x" ? "×" : ""}
                </span>
              </span>
            </label>
          ))}
        </div>
      )}
    </section>
  );
}
