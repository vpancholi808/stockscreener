import sys; sys.path.insert(0,'/home/claude/finmodel')
import run_micron as R
from finmodel import ThreeStatementModel, forensic_report, tearsheet_html
M=1e6
R.facts["facts"]["us-gaap"]["RetainedEarningsAccumulatedDeficit"]=R.inst({2023:32000*M,2024:32265*M,2025:40282*M})
R.facts["facts"]["us-gaap"]["SellingGeneralAndAdministrativeExpense"]=R.dur({2023:1000*M,2024:1116*M,2025:1300*M})
m=ThreeStatementModel.from_facts(R.facts,"MU",3)
rep=forensic_report(m, market_cap=1.0e12, wacc=0.1086)
h=tearsheet_html(m,rep,title="Micron Technology — health tearsheet",as_of="2026-08-14")
open("/mnt/user-data/outputs/micron_tearsheet.html","w").write(h)
print("bytes",len(h),"| doctype",h.startswith("<!doctype"),"| closes",h.rstrip().endswith("</html>"))
print("cards",h.count('class="fc'),"| metric rows",h.count('class="mrow'),"| piotroski dots",h.count('class="pt'))
