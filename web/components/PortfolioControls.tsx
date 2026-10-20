"use client";

import { useRouter, useSearchParams } from "next/navigation";
import { useTransition } from "react";
import type { PortfolioOptions } from "@/lib/api";

/** Same pattern as the screener filters: state lives in the URL, so the page
 *  stays a server component and every configuration is a shareable link. */
export default function PortfolioControls({
  options,
  current,
}: {
  options: PortfolioOptions;
  current: {
    strategy: string;
    risk: string;
    allocation: string;
    max_holdings: string;
  };
}) {
  const router = useRouter();
  const params = useSearchParams();
  const [pending, startTransition] = useTransition();

  const set = (key: string, value: string) => {
    const next = new URLSearchParams(params.toString());
    if (value === "") next.delete(key);
    else next.set(key, value);
    startTransition(() =>
      router.push(`/portfolio?${next.toString()}`, { scroll: false })
    );
  };

  return (
    <section
      className={`rounded-lg border border-rule bg-panel transition-opacity ${
        pending ? "opacity-60" : ""
      }`}
      aria-busy={pending}
    >
      <div className="grid gap-4 px-4 py-4 sm:grid-cols-2 lg:grid-cols-4">
        <label className="block">
          <span className="mb-1 block font-mono text-2xs uppercase tracking-wider text-muted">
            Investment strategy
          </span>
          <select
            value={current.strategy}
            onChange={(e) => set("strategy", e.target.value)}
            className="w-full rounded border border-rule bg-paper px-2 py-1.5 text-sm text-ink focus:border-accent focus:outline-none"
          >
            {options.strategies.map((s) => (
              <option key={s.key} value={s.key}>
                {s.name}
              </option>
            ))}
          </select>
        </label>

        <label className="block">
          <span className="mb-1 block font-mono text-2xs uppercase tracking-wider text-muted">
            Risk profile
          </span>
          <select
            value={current.risk}
            onChange={(e) => set("risk", e.target.value)}
            className="w-full rounded border border-rule bg-paper px-2 py-1.5 text-sm text-ink focus:border-accent focus:outline-none"
          >
            {options.risk_profiles.map((r) => (
              <option key={r.key} value={r.key}>
                {r.name}
              </option>
            ))}
          </select>
        </label>

        <label className="block">
          <span className="mb-1 block font-mono text-2xs uppercase tracking-wider text-muted">
            Allocation method
          </span>
          <select
            value={current.allocation}
            onChange={(e) => set("allocation", e.target.value)}
            className="w-full rounded border border-rule bg-paper px-2 py-1.5 text-sm text-ink focus:border-accent focus:outline-none"
          >
            {options.allocation_methods.map((a) => (
              <option key={a.key} value={a.key}>
                {a.name}
              </option>
            ))}
          </select>
        </label>

        <label className="block">
          <span className="mb-1 block font-mono text-2xs uppercase tracking-wider text-muted">
            Max holdings
          </span>
          <input
            type="number"
            min={3}
            max={60}
            defaultValue={current.max_holdings}
            placeholder="profile default"
            onBlur={(e) => set("max_holdings", e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") set("max_holdings", e.currentTarget.value);
            }}
            className="w-full rounded border border-rule bg-paper px-2 py-1.5 font-mono text-sm tabular-nums text-ink placeholder:font-sans placeholder:text-muted focus:border-accent focus:outline-none"
          />
        </label>
      </div>
    </section>
  );
}
