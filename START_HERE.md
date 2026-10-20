# START HERE

## Fastest path: open this folder in Claude Code

1. Unzip this folder somewhere you can find it.
2. Open a terminal in the folder (Mac: right-click > New Terminal at Folder.
   Windows: type `cmd` in the address bar of the folder).
3. Type:  `claude`
4. Paste this prompt:

   > Read CLAUDE.md and QUICKSTART.md. Install the dependencies from
   > requirements.txt, set up my EDGAR user agent, then run `python run.py MU`
   > and show me the report. If anything fails, fix it and retry.

Claude Code handles Python setup, dependencies, and the SEC user-agent step.

## Or run it yourself, without Claude Code

```
pip install -r requirements.txt
python3 run.py MU
```

## What's in here

| File / folder | What it is |
|---|---|
| **run.py** | The only file you run. `python run.py MU` |
| **QUICKSTART.md** | Setup, troubleshooting, what the scores mean |
| **CLAUDE.md** | Orientation for Claude Code (architecture, design rules) |
| **README.md** | Full technical documentation |
| `finmodel/` | The engine. You never run these directly. |
| `test_*.py` | Validation suites. Run to confirm nothing's broken. |
| `example_*.py` | Focused examples (CAPM, DCF, quant) |
| `run_micron.py` | Real Micron FY23–FY25 data, used as a worked example |
| `mu_*.py` | Micron scenario / forensic / tearsheet demos |

## Commands worth knowing

```
python3 run.py MU                  one company -> tearsheet
python3 run.py AAPL MSFT NVDA MU   several -> ranked screen dashboard
python3 test_pipeline.py           confirm the engine still works
```

Reports are written to a `reports/` folder and open in your browser.

## Known limitation

The model reads **annual** 10-K filings only. For a company mid-upcycle the
latest annual figures can badly understate current reality (Micron is the
extreme case). Quarterly/TTM support is the next planned addition.
