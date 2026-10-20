"""Single-company health tearsheet.

Renders a ThreeStatementModel plus its ForensicReport into one self-contained
HTML page: the 12-metric scorecard, the five composite/forensic scores, the
Piotroski test breakdown, and the headline financials.
"""
from __future__ import annotations

import html
from datetime import date
from typing import Optional

import pandas as pd

from .report import _money, _pct, _x, _health_meter, _sparkline


def _fmt(v, kind):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"
    return {"pct": _pct, "x": _x, "money": _money,
            "num": lambda z: f"{z:.2f}"}[kind](v)


_LABEL = {
    "debt_to_equity": ("Debt / Equity", "x"),
    "net_debt_to_ebitda": ("Net debt / EBITDA", "x"),
    "interest_coverage": ("Interest coverage", "x"),
    "current_ratio": ("Current ratio", "x"),
    "quick_ratio": ("Quick ratio", "x"),
    "fcf_margin": ("FCF margin", "pct"),
    "cash_conversion": ("Cash conversion (OCF/NI)", "x"),
    "gross_margin": ("Gross margin", "pct"),
    "operating_margin": ("Operating margin", "pct"),
    "roe": ("ROE", "pct"),
    "roic": ("ROIC", "pct"),
    "positive_fcf": ("Free cash flow positive", "money"),
}
_GROUP = {
    "Solvency": ["debt_to_equity", "net_debt_to_ebitda", "interest_coverage"],
    "Liquidity": ["current_ratio", "quick_ratio"],
    "Cash-flow quality": ["fcf_margin", "cash_conversion", "positive_fcf"],
    "Profitability": ["gross_margin", "operating_margin", "roe", "roic"],
}


def tearsheet_html(model, forensic, title: Optional[str] = None,
                   as_of: Optional[str] = None) -> str:
    sc = model.scorecard
    fin = model.financials
    tick = html.escape(sc["ticker"])
    name = html.escape(str(fin.entity_name))
    asof = as_of or date.today().isoformat()
    passed, total = sc["passed"], sc["total"]

    # scorecard grouped rows
    groups = []
    for gname, keys in _GROUP.items():
        rows = []
        for k in keys:
            m = sc["metrics"].get(k)
            if not m:
                continue
            lab, kind = _LABEL[k]
            rule = "≥" if m.higher_is_better else "≤"
            thr = _fmt(m.threshold, kind) if k != "positive_fcf" else "> 0"
            rows.append(
                f'<div class="mrow {"ok" if m.passed else "no"}">'
                f'<span class="mk">{"✓" if m.passed else "✕"}</span>'
                f'<span class="ml">{lab}</span>'
                f'<span class="mv">{_fmt(m.value, kind)}</span>'
                f'<span class="mt">{rule} {thr}</span></div>')
        groups.append(f'<div class="grp"><h4>{gname}</h4>{"".join(rows)}</div>')

    # forensic cards
    def card(s, hint=""):
        if not s.available:
            return (f'<div class="fc na"><div class="fn">{html.escape(s.name)}</div>'
                    f'<div class="fv">—</div><div class="fd">{html.escape(s.reason)}</div></div>')
        good = any(w in s.verdict for w in
                   ("strong", "safe", "no flag", "clean", "creating"))
        warnish = any(w in s.verdict for w in ("grey", "elevated", "mixed", "roughly"))
        cls = "good" if good else ("warn" if warnish else "bad")
        val = (f"{s.value:.0f}/9" if "Piotroski" in s.name else
               f"{s.value:+.2%}" if "spread" in s.name else f"{s.value:.2f}")
        return (f'<div class="fc {cls}"><div class="fn">{html.escape(s.name)}</div>'
                f'<div class="fv">{val}</div>'
                f'<div class="fd">{html.escape(s.verdict)}</div>'
                + (f'<div class="fh">{hint}</div>' if hint else "") + '</div>')

    cards = "".join([
        card(forensic.piotroski, "9 binary tests of improving fundamentals"),
        card(forensic.altman, "bankruptcy risk: &gt;2.99 safe, &lt;1.81 distress"),
        card(forensic.beneish, "earnings manipulation: &gt;−1.78 flags scrutiny"),
        card(forensic.accruals, "gap between profit and cash"),
        card(forensic.spread, "returns above the cost of capital"),
    ])

    # piotroski dots
    dots = ""
    if forensic.piotroski.available:
        for k, v in forensic.piotroski.detail.items():
            dots += (f'<div class="pt {"y" if v else "n"}">'
                     f'<span>{"✓" if v else "·"}</span>{html.escape(k)}</div>')

    # headline figures + trends
    yrs = fin.years
    is_, cf = fin.income_statement, fin.cash_flow
    def series(df, item):
        return [df.at[item, y] if item in df.index and y in df.columns else None
                for y in yrs]
    rev_s, ni_s = series(is_, "revenue"), series(is_, "net_income")
    gm_s = [(_g / _r) if _g is not None and _r not in (None, 0) else None
            for _g, _r in zip(series(is_, "gross_profit"), rev_s)]
    r = sc["raw"]
    heads = [("Revenue", r.get("revenue"), "money", rev_s),
             ("EBITDA", r.get("ebitda"), "money", None),
             ("Net income", r.get("net_income"), "money", ni_s),
             ("Free cash flow", r.get("fcf"), "money", None),
             ("Net debt", r.get("net_debt"), "money", None),
             ("Gross margin", None, "pct", gm_s)]
    hh = ""
    for lab, val, kind, ser in heads:
        v = (_fmt(val, kind) if val is not None else
             (_fmt(gm_s[-1], "pct") if lab == "Gross margin" else "—"))
        spark = _sparkline([x for x in ser], w=96, h=26) if ser else ""
        hh += (f'<div class="hd"><span class="hl">{lab}</span>'
               f'<span class="hv">{v}</span><span class="hs">{spark}</span></div>')

    pct = passed / total if total else 0
    return _T.format(
        title=html.escape(title or f"{name} — health tearsheet"),
        tick=tick, name=name, asof=asof, passed=passed, total=total,
        pctlab=f"{pct*100:.0f}%", meter=_health_meter(passed, total),
        fy=sc["fiscal_year"], groups="".join(groups), cards=cards, dots=dots,
        heads=hh, quality=html.escape(model.quality["grade"]))


_T = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title>
<style>
:root{{--paper:#F6F8FB;--panel:#fff;--ink:#0F1B2D;--body:#26364B;--muted:#6B7C93;
--rule:#DFE5EE;--good:#0D7A6F;--warn:#B5822A;--bad:#A63D40;}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:var(--paper);color:var(--body);
font:15px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;-webkit-font-smoothing:antialiased}}
.mono,.hv,.mv,.fv,.tick,.mt{{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
font-variant-numeric:tabular-nums}}
.wrap{{max-width:940px;margin:0 auto;padding:0 18px 70px}}
header.top{{padding:34px 0 20px;border-bottom:3px solid var(--ink);
display:flex;justify-content:space-between;align-items:flex-end;gap:18px;flex-wrap:wrap}}
.eyebrow{{font-family:ui-monospace,Menlo,monospace;font-size:10.5px;letter-spacing:.2em;
text-transform:uppercase;color:var(--muted);margin-bottom:9px}}
.tick{{font-size:34px;font-weight:800;color:var(--ink);letter-spacing:-.02em}}
.name{{font-size:14px;color:var(--muted);margin-top:2px}}
.hscore{{text-align:right}}
.hscore .big{{font-size:30px;font-weight:800;color:var(--good);letter-spacing:-.02em;
font-family:ui-monospace,Menlo,monospace}}
.hscore .lbl{{font-size:10.5px;text-transform:uppercase;letter-spacing:.12em;color:var(--muted);
margin-top:3px}}
.hscore .mt2{{margin-top:7px}}
.sec{{margin-top:30px}}
.slab{{font-family:ui-monospace,Menlo,monospace;font-size:10.5px;letter-spacing:.16em;
text-transform:uppercase;color:var(--muted);margin-bottom:12px;display:flex;
align-items:center;gap:9px}}
.slab::before{{content:"";width:18px;height:2px;background:var(--ink)}}
.heads{{display:grid;grid-template-columns:repeat(auto-fit,minmax(148px,1fr));gap:1px;
background:var(--rule);border:1px solid var(--rule);border-radius:9px;overflow:hidden}}
.hd{{background:var(--panel);padding:13px 14px;display:grid;gap:3px}}
.hl{{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.05em}}
.hv{{font-size:17px;font-weight:700;color:var(--ink)}}
.fcards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(168px,1fr));gap:11px}}
.fc{{background:var(--panel);border:1px solid var(--rule);border-radius:9px;padding:15px;
border-top:3px solid var(--muted)}}
.fc.good{{border-top-color:var(--good)}}.fc.warn{{border-top-color:var(--warn)}}
.fc.bad{{border-top-color:var(--bad)}}.fc.na{{opacity:.6}}
.fn{{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.05em}}
.fv{{font-size:25px;font-weight:800;color:var(--ink);margin:5px 0 3px;letter-spacing:-.02em}}
.fc.good .fv{{color:var(--good)}}.fc.warn .fv{{color:var(--warn)}}.fc.bad .fv{{color:var(--bad)}}
.fd{{font-size:12.5px;color:var(--body);font-weight:600}}
.fh{{font-size:11px;color:var(--muted);margin-top:6px;line-height:1.45}}
.cols{{display:grid;grid-template-columns:1.25fr 1fr;gap:16px}}
@media(max-width:700px){{.cols{{grid-template-columns:1fr}}}}
.panel{{background:var(--panel);border:1px solid var(--rule);border-radius:9px;padding:16px 17px}}
.grp{{margin-bottom:14px}}.grp:last-child{{margin-bottom:0}}
.grp h4{{font-size:11px;text-transform:uppercase;letter-spacing:.1em;color:var(--muted);
margin-bottom:7px;font-weight:700}}
.mrow{{display:grid;grid-template-columns:16px 1fr auto 64px;gap:9px;align-items:center;
padding:5px 0;border-bottom:1px solid #F3F6F9;font-size:13.5px}}
.mrow:last-child{{border-bottom:none}}
.mk{{font-size:12px;font-weight:700}}.mrow.ok .mk{{color:var(--good)}}.mrow.no .mk{{color:var(--bad)}}
.ml{{color:var(--body)}}.mv{{font-weight:700;color:var(--ink);font-size:13px}}
.mt{{font-size:11px;color:var(--muted);text-align:right}}
.mrow.no .mv{{color:var(--bad)}}
.pt{{display:grid;grid-template-columns:16px 1fr;gap:8px;font-size:12.5px;padding:4px 0;
border-bottom:1px solid #F3F6F9}}
.pt:last-child{{border-bottom:none}}
.pt.y span{{color:var(--good);font-weight:700}}.pt.n span{{color:var(--muted)}}
.pt.n{{color:var(--muted)}}
.foot{{margin-top:34px;padding-top:16px;border-top:1px solid var(--rule);
font-size:11.5px;color:var(--muted);line-height:1.65}}
</style></head><body><div class="wrap">
<header class="top">
<div><div class="eyebrow">Financial health tearsheet · FY{fy} · as of {asof}</div>
<div class="tick">{tick}</div><div class="name">{name}</div></div>
<div class="hscore"><div class="big">{passed}/{total}</div>
<div class="lbl">checks passed · {pctlab}</div><div class="mt2">{meter}</div></div>
</header>

<div class="sec"><div class="slab">Headline figures</div>
<div class="heads">{heads}</div></div>

<div class="sec"><div class="slab">Composite &amp; forensic scores</div>
<div class="fcards">{cards}</div></div>

<div class="sec cols">
<div><div class="slab">Scorecard by dimension</div>
<div class="panel">{groups}</div></div>
<div><div class="slab">Piotroski breakdown</div>
<div class="panel">{dots}</div></div>
</div>

<p class="foot">Data quality: {quality}. Scorecard thresholds are screening defaults and should be
calibrated by sector. Composite scores are computed from filed statements; Altman uses market
value of equity where available. Educational tooling, not investment advice.</p>
</div></body></html>"""
