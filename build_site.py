#!/usr/bin/env python3
"""Build a self-contained static site from a loaded database.

    python worker.py --sp500
    python build_site.py --out site/index.html

One HTML file with the data embedded. No server, no database at view time,
no third-party requests — which means it can be served from GitHub Pages for
nothing, and it still works if you open the file from disk.

LICENSING. Price-derived fields are omitted unless SERVE_MARKET_DATA=true,
for the same reason the API and the Streamlit app omit them: SEC filings are
US government work and in the public domain, so fundamentals and every ratio
built from them can be republished freely, while scraped price data generally
cannot. A public static site is the most exposed surface this project has, so
the default here matters more than anywhere else.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

SERVE_MARKET_DATA = os.environ.get("SERVE_MARKET_DATA", "false").lower() == "true"

PRICE_DERIVED = {
    "price", "market_cap", "enterprise_value", "pe", "forward_pe", "pb", "ps",
    "ev_ebitda", "ev_sales", "ev_ebit", "peg", "fcf_yield", "earnings_yield",
    "dividend_yield", "beta", "volatility", "return_1y", "return_total",
    "max_drawdown", "trailing_eps", "forward_eps", "payout_ratio",
    "target_mean_price", "analyst_count", "week52_high", "week52_low",
}

COLUMNS = [
    ("ticker", "Ticker", "text"),
    ("name", "Company", "text"),
    ("sector", "Sector", "text"),
    ("fiscal_year", "FY", "int"),
    ("sector_percentile", "Sector rank", "bar"),
    ("health_score", "Health", "bar"),
    ("revenue", "Revenue", "money"),
    ("revenue_growth", "Growth", "pct"),
    ("gross_margin", "Gross", "pct"),
    ("operating_margin", "Op margin", "pct"),
    ("fcf_margin", "FCF margin", "pct"),
    ("roic", "ROIC", "pct"),
    ("roe", "ROE", "pct"),
    ("net_debt_to_ebitda", "ND/EBITDA", "x"),
    ("current_ratio", "Current", "x"),
    ("piotroski", "Piotroski", "int"),
    ("altman", "Altman Z", "num"),
]

DISCLAIMER = (
    "Computed from public SEC filings for informational purposes only. "
    "Not investment advice. Figures are as last filed and may be out of date; "
    "verify against primary sources before acting.")


def collect(db) -> dict:
    import pandas as pd

    conn = db.connect()
    companies = pd.read_sql_query("SELECT * FROM companies", conn)
    history = pd.read_sql_query(
        "SELECT * FROM financial_history ORDER BY cik, fiscal_year", conn)
    benchmarks = pd.read_sql_query("SELECT * FROM sector_benchmarks", conn)
    try:
        market = pd.read_sql_query("SELECT * FROM market_data", conn)
    except Exception:
        market = pd.DataFrame()

    if not SERVE_MARKET_DATA:
        benchmarks = benchmarks[~benchmarks["price_derived"].astype(bool)]
        market = pd.DataFrame()
    elif not market.empty:
        companies = companies.merge(market, on="cik", how="left",
                                    suffixes=("", "_mkt"))

    def clean(frame):
        return json.loads(frame.where(frame.notna(), None).to_json(
            orient="records"))

    hist_by_cik: dict = {}
    for row in clean(history):
        hist_by_cik.setdefault(str(row["cik"]), []).append(row)

    bench_by_sector: dict = {}
    for row in clean(benchmarks):
        bench_by_sector.setdefault(row["sector"], {})[row["metric"]] = row

    rows = clean(companies)
    if not SERVE_MARKET_DATA:
        rows = [{k: v for k, v in r.items() if k not in PRICE_DERIVED}
                for r in rows]

    stats = db.stats()
    return {
        "companies": rows,
        "history": hist_by_cik,
        "benchmarks": bench_by_sector,
        "stats": {k: (v if not hasattr(v, "item") else v.item())
                  for k, v in stats.items()},
        "columns": COLUMNS,
        "serves_market_data": SERVE_MARKET_DATA,
        "built": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "disclaimer": DISCLAIMER,
    }


TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<meta name="description" content="Equity health screening computed from SEC filings.">
<style>
/* Same design system as frontier.html: frosted panels over a fixed gradient
   mesh, cool and bright in light mode, warm and deep in dark. Every colour is
   a token so both themes are designed rather than inverted. */
:root{
  --base:#e9edf2;
  --m1:rgba(120,170,210,.55); --m2:rgba(190,150,215,.40);
  --m3:rgba(240,185,140,.45); --m4:rgba(130,205,195,.38);
  --glass:rgba(255,255,255,.58); --glass-2:rgba(255,255,255,.88);
  --edge:rgba(255,255,255,.80); --border:rgba(20,34,44,.12);
  --ink:#152430; --muted:#5a7180; --faint:#8699a6;
  --accent:#1c6a86; --accent-soft:rgba(28,106,134,.12);
  --good:#2c7a5a; --bad:#a63f2e; --bar:rgba(28,106,134,.18);
  --shadow:0 10px 34px rgba(14,26,36,.12);
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    --base:#0a1014;
    --m1:rgba(40,95,125,.55); --m2:rgba(95,60,120,.42);
    --m3:rgba(165,95,45,.45); --m4:rgba(30,110,100,.38);
    --glass:rgba(26,36,44,.56); --glass-2:rgba(18,26,33,.92);
    --edge:rgba(255,255,255,.12); --border:rgba(255,255,255,.11);
    --ink:#e7eff4; --muted:#93a9b6; --faint:#6f8593;
    --accent:#6ec2d8; --accent-soft:rgba(110,194,216,.14);
    --good:#58bd8f; --bad:#e08d7b; --bar:rgba(110,194,216,.22);
    --shadow:0 10px 34px rgba(0,0,0,.42);
    color-scheme:dark;
  }
}
:root[data-theme="dark"]{
  --base:#0a1014;
  --m1:rgba(40,95,125,.55); --m2:rgba(95,60,120,.42);
  --m3:rgba(165,95,45,.45); --m4:rgba(30,110,100,.38);
  --glass:rgba(26,36,44,.56); --glass-2:rgba(18,26,33,.92);
  --edge:rgba(255,255,255,.12); --border:rgba(255,255,255,.11);
  --ink:#e7eff4; --muted:#93a9b6; --faint:#6f8593;
  --accent:#6ec2d8; --accent-soft:rgba(110,194,216,.14);
  --good:#58bd8f; --bad:#e08d7b; --bar:rgba(110,194,216,.22);
  --shadow:0 10px 34px rgba(0,0,0,.42);
  color-scheme:dark;
}
*{box-sizing:border-box}
body{margin:0;background:var(--base);color:var(--ink);
  font:15px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,
  "Helvetica Neue",Arial,sans-serif;-webkit-text-size-adjust:100%}
.mesh{position:fixed;inset:-20%;z-index:-1;pointer-events:none;
  filter:blur(70px) saturate(130%);
  background:
    radial-gradient(38% 42% at 12% 18%, var(--m1) 0%, transparent 70%),
    radial-gradient(34% 38% at 86% 12%, var(--m2) 0%, transparent 70%),
    radial-gradient(42% 46% at 78% 82%, var(--m3) 0%, transparent 72%),
    radial-gradient(36% 40% at 22% 88%, var(--m4) 0%, transparent 70%);}
.glass,.scroll,.tools,header,.stat,dialog,footer{
  background:var(--glass);
  -webkit-backdrop-filter:blur(22px) saturate(150%);
  backdrop-filter:blur(22px) saturate(150%);
  border:1px solid var(--border);border-radius:18px;
  box-shadow:var(--shadow),inset 0 1px 0 var(--edge)}
.wrap{max-width:1500px;margin:0 auto;
  padding-left:max(16px,env(safe-area-inset-left,0px));
  padding-right:max(16px,env(safe-area-inset-right,0px));
  padding-block:18px 40px}
header{padding:20px 22px 18px;margin-bottom:14px}
h1{margin:0 0 4px;font-size:26px;letter-spacing:-.02em}
.sub{color:var(--muted);margin:0 0 10px;max-width:70ch}
.nav{margin:0 0 14px}
.nav a{color:var(--accent);text-decoration:none;font-size:14px}
.nav a:hover{text-decoration:underline}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
  gap:12px;margin-top:14px}
.stat{padding:12px 15px}
.stat b{display:block;font-size:24px;font-variant-numeric:tabular-nums}
.stat span{color:var(--faint);font-size:11px;text-transform:uppercase;
  letter-spacing:.07em}
.tools{display:flex;flex-wrap:wrap;gap:10px;align-items:center;
  padding:12px 14px;position:sticky;top:10px;z-index:5;margin-bottom:12px}
input,select,button{font:inherit;color:inherit;background:var(--glass-2);
  border:1px solid var(--border);border-radius:10px;padding:7px 10px}
input[type=search]{min-width:220px;flex:1 1 220px}
button{cursor:pointer}
button:hover{border-color:var(--accent)}
:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
label.chk{display:flex;align-items:center;gap:6px;color:var(--muted);
  font-size:14px;cursor:pointer}
.count{color:var(--muted);font-size:14px;margin-left:auto}
/* A scrolling panel rather than a bare overflow wrapper. position:sticky
   inside an overflow-x container sticks to THAT scrollport, not the viewport,
   so a page-level sticky header there lands behind the first row. Giving the
   panel its own vertical scroll makes the header stick where it should, and
   a fixed header is what you want for 500 rows anyway. */
.scroll{overflow:auto;max-height:calc(100vh - 260px);padding:2px}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{padding:7px 10px;text-align:right;white-space:nowrap;
  border-bottom:1px solid var(--border);font-size:14px}
th:nth-child(-n+3),td:nth-child(-n+3){text-align:left}
/* The sticky header sits over scrolling rows, so it needs an opaque fill of
   its own: a translucent one lets the rows read straight through it. */
th{position:sticky;top:0;background:var(--glass-2);
  -webkit-backdrop-filter:blur(18px);backdrop-filter:blur(18px);
  cursor:pointer;font-weight:600;color:var(--muted);font-size:11.5px;
  text-transform:uppercase;letter-spacing:.06em;user-select:none;z-index:4}
th:hover{color:var(--ink)}
th.sorted::after{content:"\\00a0▾";font-size:10px}
th.sorted.asc::after{content:"\\00a0▴"}
tbody tr{cursor:pointer}
tbody tr:hover{background:var(--accent-soft)}
td.tick{font-weight:600}
.bar{display:block;height:5px;border-radius:3px;background:var(--bar);
  position:relative;min-width:54px}
.bar i{position:absolute;inset:0 auto 0 0;background:var(--accent);
  border-radius:3px}
.pos{color:var(--good)} .neg{color:var(--bad)}
dialog{color:var(--ink);max-width:880px;width:calc(100% - 32px);padding:0;
  background:var(--glass-2)}
dialog::backdrop{background:rgba(0,0,0,.45);
  -webkit-backdrop-filter:blur(3px);backdrop-filter:blur(3px)}
.dh{padding:20px 22px 14px;border-bottom:1px solid var(--border);
  display:flex;justify-content:space-between;gap:16px;align-items:flex-start}
.db{padding:18px 22px 24px;max-height:70vh;overflow:auto}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));
  gap:14px;margin:0 0 20px}
.grid div b{display:block;font-size:17px;font-variant-numeric:tabular-nums}
.grid div span{color:var(--muted);font-size:12px}
h3{font-size:13px;text-transform:uppercase;letter-spacing:.06em;
  color:var(--faint);margin:22px 0 8px}
.mini{width:100%;font-size:13px}
.mini td,.mini th{padding:5px 8px}
footer{margin-top:16px;padding:18px 22px 22px;color:var(--muted);
  font-size:13px}
footer p{max-width:85ch}
details{margin:10px 0}
summary{cursor:pointer;color:var(--accent)}
@media (prefers-reduced-motion:reduce){*{animation:none!important;
  transition:none!important}}
@media (max-width:700px){
  .stat b{font-size:19px}h1{font-size:21px}
  .scroll{max-height:calc(100vh - 330px)}
  input[type=search]{min-width:140px}
}
</style>
</head>
<body>
<div class="mesh" aria-hidden="true"></div>
<div class="wrap">
<header>
  <h1>__TITLE__</h1>
  <p class="sub">Financial health, forensic accounting checks and
  peer-relative ranking, computed from SEC filings. No estimates, no price
  targets, no buy or sell verdicts.</p>
  <p class="nav"><a href="./frontier.html">Efficient frontier &amp; capital
  allocation line &rarr;</a></p>
  <div class="stats" id="stats"></div>
</header>

<div class="tools">
  <input type="search" id="q" placeholder="Ticker or company" aria-label="Search">
  <select id="sector"><option value="">All sectors</option></select>
  <label class="chk"><input type="checkbox" id="passing"> Passing only</label>
  <select id="minhealth">
    <option value="0">Any health</option>
    <option value="0.5">Health 50%+</option>
    <option value="0.75">Health 75%+</option>
    <option value="1">Health 100%</option>
  </select>
  <button id="csv">Download CSV</button>
  <button id="theme" title="Toggle theme">◐</button>
  <span class="count" id="count"></span>
</div>

<div class="scroll"><table id="t">
  <thead><tr id="hrow"></tr></thead>
  <tbody id="rows"></tbody>
</table></div>

<footer>
  <p id="vintage"></p>
  <p id="licence"></p>
  <details>
    <summary>Methodology and limitations</summary>
    <p><b>Sector rank</b> is peer-relative, computed in two steps: percentile
    rank within sector on six metrics, then a rank of that composite. A mean
    of percentiles is not itself a percentile — it compresses toward 0.5 and
    never reaches 1 — so the second step is what makes the number mean what
    the label says.</p>
    <p><b>Sector benchmarks use the median, never the mean.</b> Valuation
    multiples are right-skewed; one company at 400x drags a mean far above
    anything a typical constituent trades at.</p>
    <p><b>Not every metric applies to every sector.</b> Banks, insurers and
    property trusts report no cost of revenue, so they have no gross margin.
    Altman Z was estimated on manufacturers. A missing value is excluded from
    the health score rather than counted as a failure — a company is never
    marked down for not reporting something its accounting framework does not
    contain.</p>
    <p><b>Annual data only.</b> Every figure is as of the last filed 10-K and
    can be up to twelve months old. For a company mid-cycle the gap to current
    trading can be large.</p>
    <p><b>There are no recommendations, by design.</b> A public site telling
    people whether to invest is giving investment advice, a regulated activity
    in most jurisdictions, and a verdict built on data this stale would be
    confidently wrong often enough to cause real losses.</p>
  </details>
  <p id="disc"></p>
</footer>
</div>

<dialog id="dlg"><div class="dh">
  <div><h2 id="dname" style="margin:0;font-size:20px"></h2>
       <div id="dmeta" style="color:var(--muted);font-size:13px"></div></div>
  <button onclick="dlg.close()">Close</button>
</div><div class="db" id="dbody"></div></dialog>

<script id="data" type="application/json">__DATA__</script>
<script>
const D = JSON.parse(document.getElementById('data').textContent);
const COLS = D.columns, rowsEl = document.getElementById('rows');
let sortKey = 'sector_percentile', sortAsc = false;

const fmt = {
  text: v => v == null ? '—' : String(v),
  int:  v => v == null ? '—' : String(Math.round(v)),
  num:  v => v == null ? '—' : v.toFixed(2),
  x:    v => v == null ? '—' : v.toFixed(1) + '\\u00d7',
  pct:  v => v == null ? '—' : (v * 100).toFixed(1) + '%',
  money: v => {
    if (v == null) return '—';
    const a = Math.abs(v);
    for (const [c, s] of [[1e12,'T'],[1e9,'B'],[1e6,'M'],[1e3,'K']])
      if (a >= c) return '$' + (v / c).toFixed(2) + s;
    return '$' + v.toFixed(0);
  },
  bar: v => v == null ? '—' :
    `<span class="bar" title="${(v*100).toFixed(0)}%"><i style="width:${
      Math.max(0, Math.min(1, v)) * 100}%"></i></span>`
};

function statCards() {
  const s = D.stats, el = document.getElementById('stats');
  const items = [['Companies', s.companies], ['Pass screen', s.passing || 0],
                 ['Sectors', s.sectors], ['Latest FY', s.latest_year]];
  el.innerHTML = items.map(([k, v]) =>
    `<div class="stat"><b>${v ?? '—'}</b><span>${k}</span></div>`).join('');
}

function header() {
  document.getElementById('hrow').innerHTML = COLS.map(([k, label]) =>
    `<th data-k="${k}">${label}</th>`).join('');
  document.querySelectorAll('th').forEach(th => th.onclick = () => {
    const k = th.dataset.k;
    if (sortKey === k) sortAsc = !sortAsc; else { sortKey = k; sortAsc = false; }
    render();
  });
}

function filtered() {
  const q = document.getElementById('q').value.trim().toUpperCase();
  const sec = document.getElementById('sector').value;
  const passing = document.getElementById('passing').checked;
  const minh = parseFloat(document.getElementById('minhealth').value);
  return D.companies.filter(r => {
    if (sec && r.sector !== sec) return false;
    if (passing && !r.passes_screen) return false;
    if (minh > 0 && !(r.health_score >= minh)) return false;
    if (q && !((r.ticker || '').toUpperCase().includes(q) ||
               (r.name || '').toUpperCase().includes(q))) return false;
    return true;
  });
}

function render() {
  const rows = filtered().sort((a, b) => {
    const x = a[sortKey], y = b[sortKey];
    if (x == null && y == null) return 0;
    if (x == null) return 1;            // missing always sorts last
    if (y == null) return -1;
    const c = typeof x === 'string' ? x.localeCompare(y) : x - y;
    return sortAsc ? c : -c;
  });
  document.getElementById('count').textContent =
    rows.length + ' of ' + D.companies.length;
  document.querySelectorAll('th').forEach(th => {
    th.classList.toggle('sorted', th.dataset.k === sortKey);
    th.classList.toggle('asc', th.dataset.k === sortKey && sortAsc);
  });
  rowsEl.innerHTML = rows.map(r => '<tr data-cik="' + r.cik + '">' +
    COLS.map(([k, , kind]) => {
      const v = r[k];
      let cls = kind === 'text' && k === 'ticker' ? 'tick' : '';
      if (kind === 'pct' && typeof v === 'number')
        cls += v < 0 ? ' neg' : '';
      return `<td class="${cls}">${fmt[kind](v)}</td>`;
    }).join('') + '</tr>').join('');
  rowsEl.querySelectorAll('tr').forEach(tr =>
    tr.onclick = () => detail(tr.dataset.cik));
}

function detail(cik) {
  const r = D.companies.find(c => String(c.cik) === String(cik));
  if (!r) return;
  document.getElementById('dname').textContent =
    (r.name || r.ticker) + ' (' + r.ticker + ')';
  document.getElementById('dmeta').textContent =
    [r.sector, 'FY' + r.fiscal_year + ' as filed',
     'data quality ' + (r.data_quality || '—')].join(' · ');

  const metrics = [
    ['Health', (r.health_passed ?? '?') + '/' + (r.health_total ?? '?')],
    ['Sector rank', r.sector_percentile == null ? '—' :
      Math.round(r.sector_percentile * 100) + 'th'],
    ['Revenue', fmt.money(r.revenue)], ['Growth', fmt.pct(r.revenue_growth)],
    ['Gross margin', fmt.pct(r.gross_margin)],
    ['Operating margin', fmt.pct(r.operating_margin)],
    ['Net margin', fmt.pct(r.net_margin)], ['FCF margin', fmt.pct(r.fcf_margin)],
    ['ROIC', fmt.pct(r.roic)], ['ROE', fmt.pct(r.roe)],
    ['Net debt/EBITDA', fmt.x(r.net_debt_to_ebitda)],
    ['Interest cover', fmt.x(r.interest_coverage)],
    ['Current ratio', fmt.x(r.current_ratio)],
    ['Piotroski', r.piotroski == null ? '—' : Math.round(r.piotroski) + '/9'],
    ['Altman Z', fmt.num(r.altman)], ['Beneish M', fmt.num(r.beneish)]
  ];
  let html = '<div class="grid">' + metrics.map(([k, v]) =>
    `<div><b>${v}</b><span>${k}</span></div>`).join('') + '</div>';

  const hist = D.history[String(cik)] || [];
  if (hist.length) {
    const keys = [['revenue','Revenue'],['gross_profit','Gross profit'],
      ['operating_income','Operating income'],['net_income','Net income'],
      ['operating_cash_flow','Operating cash flow'],['capex','Capex'],
      ['fcf','Free cash flow'],['total_assets','Total assets'],
      ['total_equity','Total equity'],['total_debt','Total debt']];
    html += '<h3>Filed history</h3><table class="mini"><tr><th></th>' +
      hist.map(h => '<th style="text-align:right">FY' + h.fiscal_year +
        '</th>').join('') + '</tr>' +
      keys.map(([k, label]) => '<tr><td>' + label + '</td>' +
        hist.map(h => '<td style="text-align:right">' + fmt.money(h[k]) +
          '</td>').join('') + '</tr>').join('') + '</table>';
  }

  const bench = D.benchmarks[r.sector] || {};
  const cmp = [['gross_margin','Gross margin'],['operating_margin','Operating margin'],
    ['net_margin','Net margin'],['fcf_margin','FCF margin'],['roe','ROE'],
    ['roic','ROIC'],['revenue_growth','Revenue growth'],
    ['current_ratio','Current ratio'],['net_debt_to_ebitda','Net debt/EBITDA']]
    .filter(([k]) => bench[k] && bench[k].median != null && r[k] != null);
  if (cmp.length) {
    html += '<h3>Against the ' + r.sector + ' median</h3>' +
      '<table class="mini"><tr><th>Metric</th><th style="text-align:right">This company</th>' +
      '<th style="text-align:right">Sector median</th><th style="text-align:right">Peers</th></tr>' +
      cmp.map(([k, label]) => {
        const ratio = ['current_ratio','net_debt_to_ebitda'].includes(k);
        const f = ratio ? fmt.x : fmt.pct;
        return '<tr><td>' + label + '</td><td style="text-align:right">' +
          f(r[k]) + '</td><td style="text-align:right">' +
          f(bench[k].median) + '</td><td style="text-align:right">' +
          (bench[k].count ?? '—') + '</td></tr>';
      }).join('') + '</table>' +
      '<p style="color:var(--muted);font-size:13px">Comparisons use the ' +
      'median. A sector needs at least five constituents reporting a metric ' +
      'before a benchmark is shown.</p>';
  }
  html += '<p style="color:var(--muted);font-size:13px">This page reports ' +
    'what the filed figures show. It contains no buy, sell or hold ' +
    'recommendation, and none should be inferred.</p>';
  document.getElementById('dbody').innerHTML = html;
  document.getElementById('dlg').showModal();
}

document.getElementById('csv').onclick = () => {
  const keys = COLS.map(c => c[0]);
  const esc = v => v == null ? '' :
    /[",\\n]/.test(String(v)) ? '"' + String(v).replace(/"/g, '""') + '"' : v;
  const csv = [keys.join(',')].concat(
    filtered().map(r => keys.map(k => esc(r[k])).join(','))).join('\\n');
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([csv], {type: 'text/csv'}));
  a.download = 'screen.csv'; a.click(); URL.revokeObjectURL(a.href);
};

document.getElementById('theme').onclick = () => {
  const now = document.documentElement.getAttribute('data-theme');
  const next = now === 'dark' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', next);
  try { localStorage.setItem('theme', next); } catch (e) {}
};
try {
  const saved = localStorage.getItem('theme');
  if (saved) document.documentElement.setAttribute('data-theme', saved);
} catch (e) {}

const sectors = [...new Set(D.companies.map(r => r.sector).filter(Boolean))].sort();
document.getElementById('sector').innerHTML +=
  sectors.map(s => `<option>${s}</option>`).join('');
['q','sector','passing','minhealth'].forEach(id =>
  document.getElementById(id).addEventListener('input', render));

document.getElementById('vintage').textContent =
  'Built ' + D.built.replace('T', ' ').replace('+00:00', ' UTC') +
  ' from SEC EDGAR filings.';
document.getElementById('licence').textContent = D.serves_market_data
  ? 'Includes price-derived fields.'
  : 'Price-derived fields (market cap, P/E, EV/EBITDA, yields, beta) are ' +
    'omitted. SEC filing data is US government work and in the public ' +
    'domain; scraped price data generally is not redistributable.';
document.getElementById('disc').textContent = D.disclaimer;

statCards(); header(); render();
</script>
</body>
</html>
"""


def build(db, title: str) -> str:
    data = collect(db)
    payload = json.dumps(data, separators=(",", ":"), allow_nan=False)
    # </script> inside the JSON would terminate the block early.
    payload = payload.replace("</", "<\\/")
    return (TEMPLATE
            .replace("__TITLE__", title)
            .replace("__DATA__", payload))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="site/index.html")
    ap.add_argument("--title", default="S&P 500 Equity Health Screener")
    args = ap.parse_args()

    from finmodel.web.store import Database

    db = Database()
    if (db.stats().get("companies") or 0) == 0:
        print("No data loaded. Run: python worker.py --sp500")
        return 1

    html = build(db, args.title)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print("wrote %s (%.1f MB)" % (out, out.stat().st_size / 1e6))
    if not SERVE_MARKET_DATA:
        print("price-derived fields omitted (SERVE_MARKET_DATA is not true)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
