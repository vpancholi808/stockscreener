/** Visual primitives. All inline SVG — no chart library, so these stay fast
 *  and render identically on the server. */

type Tone = "pass" | "fail" | "warn" | "none";

const TONE_FILL: Record<Tone, string> = {
  pass: "#0B6E5F",
  fail: "#9E3B3B",
  warn: "#8A6318",
  none: "#9AA7B8",
};

/** Health is a COUNT of checks passed, not a continuous score — so it reads as
 *  discrete segments rather than a progress bar. 11 of 12 should look like
 *  eleven things that passed and one that didn't. */
export function HealthMeter({
  passed,
  total,
  width = 3,
  gap = 2,
  height = 11,
}: {
  passed: number | null;
  total: number | null;
  width?: number;
  gap?: number;
  height?: number;
}) {
  const t = total ?? 12;
  const p = passed ?? 0;
  const w = t * (width + gap);
  return (
    <svg
      width={w}
      height={height}
      viewBox={`0 0 ${w} ${height}`}
      role="img"
      aria-label={`${p} of ${t} checks passed`}
      className="shrink-0"
    >
      {Array.from({ length: t }).map((_, i) => (
        <rect
          key={i}
          x={i * (width + gap)}
          y={0}
          width={width}
          height={height}
          rx={1}
          fill={i < p ? TONE_FILL.pass : "#DCE3EC"}
        />
      ))}
    </svg>
  );
}

/** The signature element. A raw ratio means little on its own — 8% ROIC is
 *  weak for software and strong for a utility — so this shows where the
 *  company sits inside its OWN sector's distribution. */
export function SectorStrip({
  percentile,
  sector,
  width = 128,
  height = 20,
}: {
  percentile: number | null;
  sector: string;
  width?: number;
  height?: number;
}) {
  if (percentile === null || percentile === undefined) {
    return <span className="font-mono text-2xs text-muted">no rank</span>;
  }
  const x = Math.max(2, Math.min(width - 2, percentile * width));
  const mid = height / 2;
  return (
    <svg
      width={width}
      height={height}
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-label={`${Math.round(percentile * 100)}th percentile within ${sector}`}
    >
      {/* distribution track, quartile ticks */}
      <line x1={0} y1={mid} x2={width} y2={mid} stroke="#DCE3EC" strokeWidth={3} />
      {[0.25, 0.5, 0.75].map((q) => (
        <line
          key={q}
          x1={q * width}
          y1={mid - 4}
          x2={q * width}
          y2={mid + 4}
          stroke="#C3CEDC"
          strokeWidth={1}
        />
      ))}
      {/* filled portion up to this company's position */}
      <line
        x1={0}
        y1={mid}
        x2={x}
        y2={mid}
        stroke={percentile >= 0.5 ? TONE_FILL.pass : TONE_FILL.warn}
        strokeWidth={3}
      />
      <circle
        cx={x}
        cy={mid}
        r={4}
        fill="#FFFFFF"
        stroke={percentile >= 0.5 ? TONE_FILL.pass : TONE_FILL.warn}
        strokeWidth={2.5}
      />
    </svg>
  );
}

export function Sparkline({
  values,
  width = 92,
  height = 24,
}: {
  values: (number | null)[];
  width?: number;
  height?: number;
}) {
  const clean = values.filter((v): v is number => v !== null && !Number.isNaN(v));
  if (clean.length < 2) {
    return <svg width={width} height={height} aria-hidden="true" />;
  }
  const lo = Math.min(...clean);
  const hi = Math.max(...clean);
  const span = hi - lo || 1;
  const step = width / (values.length - 1);
  const pts: [number, number][] = [];
  values.forEach((v, i) => {
    if (v === null || Number.isNaN(v)) return;
    pts.push([i * step, height - 3 - ((v - lo) / span) * (height - 6)]);
  });
  const rising = clean[clean.length - 1] >= clean[0];
  const color = rising ? TONE_FILL.pass : TONE_FILL.fail;
  const [cx, cy] = pts[pts.length - 1];
  return (
    <svg
      width={width}
      height={height}
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-label={rising ? "trending up" : "trending down"}
    >
      <polyline
        fill="none"
        stroke={color}
        strokeWidth={1.5}
        strokeLinejoin="round"
        points={pts.map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(" ")}
      />
      <circle cx={cx} cy={cy} r={1.9} fill={color} />
    </svg>
  );
}

export function ScoreChip({
  label,
  value,
  tone = "none",
  hint,
}: {
  label: string;
  value: string;
  tone?: Tone;
  hint?: string;
}) {
  const ring = {
    pass: "border-pass/30 bg-pass/5",
    fail: "border-fail/30 bg-fail/5",
    warn: "border-warn/30 bg-warn/5",
    none: "border-rule bg-paper",
  }[tone];
  const text = {
    pass: "text-pass",
    fail: "text-fail",
    warn: "text-warn",
    none: "text-muted",
  }[tone];
  return (
    <div className={`rounded border px-3 py-2.5 ${ring}`}>
      <div className="text-2xs uppercase tracking-wider text-muted">{label}</div>
      <div className={`mt-0.5 font-mono text-lg font-semibold tabular-nums ${text}`}>
        {value}
      </div>
      {hint ? (
        <div className="mt-0.5 text-2xs leading-snug text-muted">{hint}</div>
      ) : null}
    </div>
  );
}

export function PassBadge({ passing }: { passing: boolean }) {
  return passing ? (
    <span className="rounded bg-pass/10 px-1.5 py-0.5 font-mono text-2xs font-bold tracking-wide text-pass">
      PASS
    </span>
  ) : (
    <span className="font-mono text-2xs text-muted">{"—"}</span>
  );
}
