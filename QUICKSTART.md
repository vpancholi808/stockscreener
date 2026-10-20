# QUICKSTART

You only ever run **one file**: `run.py`. Everything else is the engine it uses.

---

## Step 1 — Install Python (once)

Check if you already have it. Open Terminal (Mac) or Command Prompt (Windows) and type:

```
python3 --version
```

If you see a version number like `Python 3.12.0`, skip to Step 2.
If not, download it from **python.org/downloads** and install. On Windows, tick
the box that says **"Add Python to PATH"** during install.

---

## Step 2 — Install the libraries (once)

Navigate to this folder, then install:

```
cd path/to/finmodel
pip install -r requirements.txt
```

Replace `path/to/finmodel` with wherever you unzipped it. Tip: on Mac you can
type `cd ` then drag the folder onto the Terminal window.

---

## Step 3 — Run it

```
python3 run.py MU
```

That's the whole thing. The first time, it asks for your name and email — the
SEC requires this with every request and blocks anonymous traffic. It's saved
locally and only ever sent to sec.gov.

You'll get a summary printed in the terminal **and** an HTML report that opens
in your browser automatically.

---

## What else you can do

```
python3 run.py AAPL                 analyze one company
python3 run.py AAPL MSFT NVDA MU    screen several -> ranked dashboard
python3 run.py                      no ticker? it will just ask you
```

Reports are saved in the `reports/` folder so you can keep or share them.

---

## If something goes wrong

| Message | What to do |
|---|---|
| `command not found: python3` | Try `python` instead of `python3` |
| `missing packages` | Run Step 2 again |
| `HTTP 403` | Delete `.finmodel_config`, run again, enter a real email |
| `no SEC filings found` | Check the ticker; only US-listed SEC filers work |
| `could not reach the SEC` | Check your internet connection |
| Nothing opens in browser | Open the file in `reports/` manually |

---

## What the output means

- **Health score (e.g. 11/12)** — how many of the 12 checks passed
- **Failed checks** — exactly which ones and by how much
- **Piotroski F-score** — 0–9, fundamental strength (8–9 is strong)
- **Altman Z-score** — bankruptcy risk (>2.99 safe, <1.81 distress)
- **Beneish M-score** — earnings manipulation flag (>−1.78 = worth scrutiny)
- **Sloan accruals** — are earnings backed by real cash?
- **ROIC − WACC** — is the business earning more than its capital costs?

---

## One important limitation

The model reads **annual** 10-K filings. For a company in a fast-moving year,
the most recent annual figures can be badly out of date. Micron is the extreme
case: its last 10-K shows $37B revenue while its current run-rate is far higher.
Quarterly/TTM support is the next planned addition.
