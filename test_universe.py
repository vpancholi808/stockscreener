import sys, os, resource, time; sys.path.insert(0,'.')
os.environ["EDGAR_USER_AGENT"]="Test User test@example.com"
from pathlib import Path
from finmodel import UniverseScreen, ScreenConfig
from finmodel.data.bulk import BulkData
# point BulkData at the synthetic bundle and skip downloading
BulkData.ensure = lambda self, force=False: None
db = Path("/tmp/universe_test.db"); db.unlink(missing_ok=True)
t0=time.time()
u = UniverseScreen.run(user_agent="Test User test@example.com",
                       db_path=str(db), bulk_dir=str(Path.home()/".finmodel_bulk_test"),
                       config=ScreenConfig(), min_revenue=50e6,
                       fetch_prices=False, verbose=True)
el=time.time()-t0
mem=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
print("\n  elapsed %.1fs | peak memory %.0f MB"%(el,mem))
print("  stats:", u.stats())
print("\n=== SECTOR SUMMARY ===")
print(u.sector_summary().round(3).to_string(index=False))
print("\n=== TOP 10 PASSING (sector-relative composite) ===")
df=u.passing(limit=10)
print(df[["ticker","sector","health_passed","revenue_growth","operating_margin",
          "roic","piotroski","composite_pct"]].round(3).to_string(index=False))
