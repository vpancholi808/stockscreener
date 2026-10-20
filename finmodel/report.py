"""Self-contained HTML screening dashboard.

Renders a Screener into a single standalone .html file (inline CSS/SVG/JS, no
external fonts or libraries, opens offline anywhere). Design: an instrument-panel
look — cool paper, ink text, semantic green/red used as data (pass/fail), tabular
figures. Signature elements are a segmented health meter and inline sparklines
that track each company's key values over time.
"""
from __future__ import annotations

import html
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

from .screener import Screener

# ---- formatting helpers ---------------------------------------------------

def _money(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"
    a = abs(v)
    if a >= 1e12: return f"${v/1e12:.2f}T"
    if a >= 1e9:  return f"${v/1e9:.1f}B"
    if a >= 1e6:  return f"${v/1e6:.0f}M"
    return f"${v:,.0f}"

def _pct(v):
    return "—" if v is None or pd.isna(v) else f"{v*100:.1f}%"

def _x(v):
    return "—" if v is None or pd.isna(v) else f"{v:.1f}×"

def _num(v, fmt):
    return "—" if v is None or pd.isna(v) else fmt(v)


def _sparkline(values, w=112, h=30, pos="#157a52", neg="#b23b3b", ink="#8a8f98"):
    xs = [v for v in values if v is not None and not pd.isna(v)]
    if len(xs) < 2:
        return f'<svg class="spark" width="{w}" height="{h}"></svg>'
    lo, hi = min(xs), max(xs)
    span = (hi - lo) or 1.0
    n = len(values)
    step = w / (n - 1)
    pts = []
    for i, v in enumerate(values):
        if v is None or pd.isna(v):
            continue
        x = i * step
        yv = h - 4 - (v - lo) / span * (h - 8)
        pts.append((x, yv))
    color = pos if xs[-1] >= xs[0] else neg
    poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    cx, cy = pts[-1]
    return (f'<svg class="spark" width="{w}" height="{h}" viewBox="0 0 {w} {h}">'
            f'<polyline fill="none" stroke="{color}" stroke-width="1.6" points="{poly}"/>'
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="2.1" fill="{color}"/></svg>')


def _health_meter(passed, total):
    seg_w, gap = 13, 3
    parts = []
    for i in range(total):
        fill = "#157a52" if i < passed else "#e6e6e0"
        parts.append(f'<rect x="{i*(seg_w+gap)}" y="0" width="{seg_w}" height="10" rx="2" fill="{fill}"/>')
    width = total * (seg_w + gap)
    return f'<svg class="meter" width="{width}" height="10">{"".join(parts)}</svg>'


# columns: (key, label, formatter, screen-direction) direction: '+' higher good, '-' lower good, None neutral
_COLS = [
    ("revenue_growth", "Rev CAGR", _pct, "+"),
    ("operating_margin", "Op margin", _pct, "+"),
    ("fcf_margin", "FCF margin", _pct, "+"),
    ("net_debt_to_ebitda", "NetDebt/EBITDA", _x, "-"),
    ("roic", "ROIC", _pct, "+"),
    ("roe", "ROE", _pct, "+"),
]

# thresholds used only for cell coloring (mirror ScreenConfig defaults)
def _cell_class(key, val, cfg):
    thr = {
        "revenue_growth": cfg.min_revenue_growth,
        "operating_margin": cfg.min_operating_margin,
        "fcf_margin": cfg.min_fcf_margin,
        "roic": cfg.min_roic,
        "net_debt_to_ebitda": cfg.max_net_debt_to_ebitda,
    }.get(key)
    if thr is None or val is None or pd.isna(val):
        return "n"
    if key == "net_debt_to_ebitda":
        return "g" if val <= thr else "b"
    return "g" if val >= thr else "b"


def dashboard_html(screener: Screener, title: str = "Equity Screen") -> str:
    cfg = screener.config
    rows = screener.rows
    n = len(rows)
    n_pass = sum(r["passes_screen"] for r in rows)
    asof = screener.as_of or date.today().isoformat()

    # active screen chips
    labelmap = {"health_score": "Health", "revenue_growth": "Rev CAGR",
                "fcf_margin": "FCF margin", "net_debt_to_ebitda": "NetDebt/EBITDA",
                "operating_margin": "Op margin", "roic": "ROIC"}
    chips = []
    for metric, op, thr in cfg.criteria():
        disp = f"{thr*100:.0f}%" if metric != "net_debt_to_ebitda" and metric != "health_score" \
            else (f"{thr:.1f}×" if metric == "net_debt_to_ebitda" else f"{thr*100:.0f}%")
        chips.append(f'<span class="chip">{labelmap.get(metric, metric)} {op} {disp}</span>')

    # table rows
    trs = []
    for i, r in enumerate(rows):
        pass_badge = ('<span class="badge pass">PASS</span>' if r["passes_screen"]
                      else '<span class="badge fail">—</span>')
        cells = []
        for key, label, fmt, _ in _COLS:
            v = r.get(key)
            cells.append(f'<td class="num {_cell_class(key, v, cfg)}" '
                         f'data-v="{"" if v is None or pd.isna(v) else v}">{_num(v, fmt)}</td>')
        meter = _health_meter(r["health_passed"], r["health_total"])
        spark = _sparkline(r.get("trend_revenue", []))
        trs.append(
            f'<tr class="{"on" if r["passes_screen"] else "off"}">'
            f'<td class="rank">{i+1}</td>'
            f'<td class="tk"><span class="sym">{html.escape(r["ticker"])}</span>'
            f'<span class="co">{html.escape(str(r["name"])[:28])}</span></td>'
            f'<td class="health" data-v="{r["health_score"]}">{meter}'
            f'<span class="hs">{r["health_passed"]}/{r["health_total"]}</span></td>'
            + "".join(cells)
            + f'<td class="sp">{spark}</td><td class="pv">{pass_badge}</td></tr>')

    header_cells = "".join(
        f'<th data-col="{k}" class="num sortable">{lab}</th>' for k, lab, _, _ in _COLS)

    # detail cards
    cards = []
    for r in rows:
        def trend_block(lbl, series, fmt):
            last = next((v for v in reversed(series) if v is not None and not pd.isna(v)), None)
            return (f'<div class="tr"><span class="tl">{lbl}</span>'
                    f'{_sparkline(series, w=140, h=34)}'
                    f'<span class="tv">{_num(last, fmt)}</span></div>')
        val_rows = [
            ("Revenue", r.get("revenue"), _money), ("EBITDA", r.get("ebitda"), _money),
            ("Free cash flow", r.get("fcf"), _money), ("Net debt", r.get("net_debt"), _money),
            ("Gross margin", r.get("gross_margin"), _pct),
            ("Net margin", r.get("net_margin"), _pct),
        ]
        if "pe" in r:
            val_rows += [("P/E", r.get("pe"), _x), ("EV/EBITDA", r.get("ev_ebitda"), _x),
                         ("FCF yield", r.get("fcf_yield"), _pct)]
        vhtml = "".join(f'<div class="kv"><span>{lbl}</span><b>{_num(v, fmt)}</b></div>'
                        for lbl, v, fmt in val_rows)
        cards.append(
            f'<section class="card {"pass" if r["passes_screen"] else ""}">'
            f'<header><div><span class="csym">{html.escape(r["ticker"])}</span>'
            f'<span class="cname">{html.escape(str(r["name"]))}</span></div>'
            f'<div class="cmeter">{_health_meter(r["health_passed"], r["health_total"])}'
            f'<span class="hs">{r["health_passed"]}/{r["health_total"]} checks</span></div></header>'
            f'<div class="kvs">{vhtml}</div>'
            f'<div class="trends">'
            + trend_block("Revenue", r.get("trend_revenue", []), _money)
            + trend_block("Net income", r.get("trend_net_income", []), _money)
            + trend_block("Free cash flow", r.get("trend_fcf", []), _money)
            + trend_block("Operating margin", r.get("trend_op_margin", []), _pct)
            + '</div></section>')

    return _TEMPLATE.format(
        title=html.escape(title), asof=asof, n=n, n_pass=n_pass,
        chips="".join(chips), header_cells=header_cells,
        rows="".join(trs), cards="".join(cards))


_TEMPLATE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>
:root{{--ink:#14181d;--paper:#fcfcfa;--panel:#fff;--muted:#6b7280;--rule:#e6e6e0;
--pos:#157a52;--neg:#b23b3b;--accent:#29406b;--warn:#9a7a12;}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:var(--paper);color:var(--ink);
font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
-webkit-font-smoothing:antialiased}}
.num,.sym,.csym,td.num,.tv,.hs,b{{font-variant-numeric:tabular-nums;
font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}}
.wrap{{max-width:1180px;margin:0 auto;padding:32px 24px 80px}}
header.top{{display:flex;justify-content:space-between;align-items:flex-end;
border-bottom:2px solid var(--ink);padding-bottom:18px;margin-bottom:22px;flex-wrap:wrap;gap:16px}}
.eyebrow{{text-transform:uppercase;letter-spacing:.14em;font-size:11px;color:var(--muted);
font-weight:600;margin-bottom:6px}}
h1{{font-size:30px;letter-spacing:-.02em;font-weight:800}}
.stats{{display:flex;gap:26px;text-align:right}}
.stat b{{display:block;font-size:26px;letter-spacing:-.02em}}
.stat span{{font-size:11px;text-transform:uppercase;letter-spacing:.1em;color:var(--muted)}}
.stat.pass b{{color:var(--pos)}}
.screenbar{{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:20px}}
.screenbar .lab{{font-size:11px;text-transform:uppercase;letter-spacing:.12em;color:var(--muted);font-weight:600}}
.chip{{font-size:12px;padding:4px 9px;border:1px solid var(--rule);border-radius:20px;
background:var(--panel);color:var(--ink);font-variant-numeric:tabular-nums}}
table{{width:100%;border-collapse:collapse;background:var(--panel);
border:1px solid var(--rule)}}
thead th{{text-align:right;font-size:11px;text-transform:uppercase;letter-spacing:.06em;
color:var(--muted);font-weight:600;padding:12px 12px;border-bottom:1px solid var(--rule);
white-space:nowrap}}
thead th:nth-child(-n+3){{text-align:left}}
th.sortable{{cursor:pointer;user-select:none}}
th.sortable:hover{{color:var(--ink)}}
tbody td{{padding:11px 12px;border-bottom:1px solid #f0f0ea;text-align:right}}
tbody tr:last-child td{{border-bottom:none}}
tbody tr.off{{opacity:.62}}
tbody tr:hover{{background:#f7f7f2}}
.rank{{color:var(--muted);font-variant-numeric:tabular-nums;width:26px;text-align:left!important}}
.tk{{text-align:left!important}}.tk .sym{{font-weight:700;font-size:14px}}
.tk .co{{display:block;font-size:11px;color:var(--muted);font-family:system-ui}}
.health{{text-align:left!important;white-space:nowrap}}
.health .hs{{font-size:11px;color:var(--muted);margin-left:8px;vertical-align:2px}}
.meter{{vertical-align:middle}}
td.num.g{{color:var(--pos)}}td.num.b{{color:var(--neg)}}td.num.n{{color:var(--ink)}}
.sp{{width:120px}}.spark{{vertical-align:middle}}
.badge{{font-size:10px;font-weight:700;letter-spacing:.06em;padding:3px 7px;border-radius:4px}}
.badge.pass{{background:#e4f1eb;color:var(--pos)}}.badge.fail{{color:var(--muted)}}
.pv{{width:52px}}
.section-label{{margin:38px 0 16px;font-size:11px;text-transform:uppercase;
letter-spacing:.14em;color:var(--muted);font-weight:600;
border-top:1px solid var(--rule);padding-top:20px}}
.cards{{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:16px}}
.card{{background:var(--panel);border:1px solid var(--rule);border-radius:8px;padding:18px;
border-top:3px solid var(--rule)}}
.card.pass{{border-top-color:var(--pos)}}
.card header{{display:flex;justify-content:space-between;align-items:flex-start;
margin-bottom:14px;gap:10px}}
.csym{{font-weight:800;font-size:17px}}
.cname{{display:block;font-size:11px;color:var(--muted);font-family:system-ui;margin-top:2px}}
.cmeter{{text-align:right}}.cmeter .hs{{display:block;font-size:10px;color:var(--muted);margin-top:5px}}
.kvs{{display:grid;grid-template-columns:1fr 1fr;gap:6px 16px;margin-bottom:14px}}
.kv{{display:flex;justify-content:space-between;font-size:13px;border-bottom:1px dotted #ececd8;padding:3px 0}}
.kv span{{color:var(--muted);font-family:system-ui}}
.trends{{border-top:1px solid var(--rule);padding-top:12px;display:grid;gap:8px}}
.tr{{display:grid;grid-template-columns:1fr auto auto;align-items:center;gap:10px}}
.tl{{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.05em}}
.tv{{font-size:12px;font-weight:600}}
.foot{{margin-top:40px;font-size:11px;color:var(--muted);line-height:1.7}}
@media (max-width:640px){{.sp,thead th.sortable:nth-child(n+7){{display:none}}
td.num:nth-child(n+7){{display:none}}}}
@media (prefers-reduced-motion:no-preference){{tbody tr{{transition:background .12s}}}}
</style></head><body><div class="wrap">
<header class="top">
<div><div class="eyebrow">Financial screen · as of {asof}</div><h1>{title}</h1></div>
<div class="stats"><div class="stat"><b>{n}</b><span>screened</span></div>
<div class="stat pass"><b>{n_pass}</b><span>pass</span></div></div>
</header>
<div class="screenbar"><span class="lab">Screen</span>{chips}</div>
<table id="screen"><thead><tr>
<th></th><th class="tk">Company</th><th class="health">Health</th>
{header_cells}<th>Trend</th><th></th></tr></thead>
<tbody>{rows}</tbody></table>
<div class="section-label">Company detail — key values &amp; multi-year trends</div>
<div class="cards">{cards}</div>
<div class="foot">Health = scorecard checks passed (leverage, liquidity, cash-flow quality,
profitability). Green/red cells show pass/fail against the active screen thresholds.
Sparklines track each value across the available fiscal years. Not investment advice.</div>
</div>
<script>
(function(){{
 var t=document.getElementById('screen'),tb=t.tBodies[0];
 t.querySelectorAll('th.sortable').forEach(function(th){{
  var asc=false;
  th.addEventListener('click',function(){{
   var col=th.cellIndex;asc=!asc;
   var rows=Array.prototype.slice.call(tb.rows);
   rows.sort(function(a,b){{
    var x=parseFloat(a.cells[col].dataset.v),y=parseFloat(b.cells[col].dataset.v);
    if(isNaN(x))x=asc?1e18:-1e18;if(isNaN(y))y=asc?1e18:-1e18;
    return asc?x-y:y-x;}});
   rows.forEach(function(r){{tb.appendChild(r);}});
  }});
 }});
}})();
</script></body></html>"""
