/** Number formatting. Financial figures are useless if you can't scan a column,
 *  so everything numeric renders in tabular monospace with a fixed shape. */

export const DASH = "—";

export function money(v: number | null | undefined): string {
  if (v === null || v === undefined || Number.isNaN(v)) return DASH;
  const a = Math.abs(v);
  if (a >= 1e12) return `$${(v / 1e12).toFixed(2)}T`;
  if (a >= 1e9) return `$${(v / 1e9).toFixed(1)}B`;
  if (a >= 1e6) return `$${(v / 1e6).toFixed(0)}M`;
  return `$${v.toFixed(0)}`;
}

export function pct(v: number | null | undefined, digits = 1): string {
  if (v === null || v === undefined || Number.isNaN(v)) return DASH;
  return `${(v * 100).toFixed(digits)}%`;
}

export function mult(v: number | null | undefined): string {
  if (v === null || v === undefined || Number.isNaN(v)) return DASH;
  return `${v.toFixed(2)}×`;
}

export function num(v: number | null | undefined, digits = 2): string {
  if (v === null || v === undefined || Number.isNaN(v)) return DASH;
  return v.toFixed(digits);
}

export function signedPct(v: number | null | undefined): string {
  if (v === null || v === undefined || Number.isNaN(v)) return DASH;
  const s = (v * 100).toFixed(1);
  return `${v >= 0 ? "+" : ""}${s}%`;
}

/** Threshold verdicts. `higherIsBetter` decides the direction of the test. */
export function verdict(
  v: number | null | undefined,
  threshold: number,
  higherIsBetter: boolean
): "pass" | "fail" | "none" {
  if (v === null || v === undefined || Number.isNaN(v)) return "none";
  return (higherIsBetter ? v >= threshold : v <= threshold) ? "pass" : "fail";
}

export function altmanZone(v: number | null | undefined) {
  if (v === null || v === undefined) return { label: DASH, tone: "none" as const };
  if (v > 2.99) return { label: "safe", tone: "pass" as const };
  if (v >= 1.81) return { label: "grey", tone: "warn" as const };
  return { label: "distress", tone: "fail" as const };
}

export function piotroskiTone(v: number | null | undefined) {
  if (v === null || v === undefined) return "none" as const;
  if (v >= 8) return "pass" as const;
  if (v >= 5) return "warn" as const;
  return "fail" as const;
}

export function beneishTone(v: number | null | undefined) {
  if (v === null || v === undefined) return "none" as const;
  return v > -1.78 ? "fail" as const : "pass" as const;
}
