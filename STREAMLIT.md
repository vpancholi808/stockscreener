# Running and deploying the Streamlit app

The Streamlit app replaces *both* halves of the FastAPI + Next.js stack. It is
one Python process that reads the same database the worker fills — no API
layer, no Node, no `npm install`.

```
worker.py          Python. Talks to SEC. Fills the database.
streamlit_app.py   Python. Reads the database. Renders everything.
```

The FastAPI + Next.js stack still works and is still documented in
[`WEB.md`](WEB.md). Use it if you want a JSON API other things can call. Use
Streamlit if you want a working site with the least moving parts.

---

## Run it locally — three commands

```bash
pip install -r requirements.txt
python worker.py --seed          # demo data, no network, no SEC key
streamlit run streamlit_app.py
```

That opens http://localhost:8501 with 141 companies across 10 sectors.

If you skip the seed step the app still starts — it shows an empty-state panel
with a **Load demo data** button that does the same thing from inside the
browser.

### With real filings instead of the demo set

```bash
export EDGAR_USER_AGENT="Your Name you@email.com"
python worker.py AAPL MSFT NVDA MU     # a few names, ~30 seconds
python worker.py --sp500               # S&P 500, ~20 minutes
python worker.py --universe            # whole US market, ~1.5 GB first run
```

SEC requires a real name and email on every request and throttles or blocks
requests without one. Re-run the worker whenever you want fresher data; the app
picks it up within five minutes, or immediately via **Refresh data** in the
sidebar.

---

## The pages

| Page | What it does |
|---|---|
| **Screener** | Filter and sort the universe on 11 numeric criteria; download CSV |
| **Company** | Statements, five-year history, peer comparison, research note |
| **Portfolio** | Build under risk constraints, or assess a book you already hold |
| **Methodology** | Every threshold, every formula, and the known limitations |

Company pages are shareable: `?ticker=MU` selects a company from the URL.

---

## Deploying to Streamlit Community Cloud (free)

1. Push to GitHub (already done — `vpancholi808/stockscreener`).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with
   GitHub.
3. **New app** → pick the repo, branch `main`, main file `streamlit_app.py`.
4. Deploy.

It installs from `requirements.txt` and serves at
`https://<name>.streamlit.app`.

### The one thing that will catch you out: storage is ephemeral

Streamlit Cloud restarts containers freely, and anything written to local disk
is lost when it does. A SQLite file seeded on first run **will disappear**, and
the app will show the empty-state panel again.

Two ways to deal with it, depending on what you want:

**Demo deployment** — do nothing. The empty-state panel has a *Load demo data*
button; the data is generated offline in a few seconds. Fine for showing the
app to someone. Not fine as a real site.

**Real deployment** — use managed Postgres. [Neon](https://neon.tech) and
[Supabase](https://supabase.com) both have free tiers large enough; this
dataset is a few hundred megabytes for the whole US market. Then in your app's
**Settings → Secrets** paste:

```toml
DATABASE_URL = "postgresql://user:password@host/dbname"
EDGAR_USER_AGENT = "Your Name you@email.com"
```

Streamlit exposes secrets as environment variables, which is exactly what
`finmodel.web.store.Database` reads — no code change. Run the worker against
the same `DATABASE_URL` from your own machine (or the `refresh-data.yml` GitHub
Action, which already takes both as repository secrets) and the deployed app
serves whatever the worker last wrote.

### Secrets never go in the repo

`.streamlit/config.toml` is committed — it is only theming. `.streamlit/secrets.toml`
is gitignored and must stay that way: `EDGAR_USER_AGENT` contains your real name
and email, and `DATABASE_URL` contains a password. Anyone who can see a public
repo can read every file in it.

---

## Before you make the URL public

Two of these are legal rather than technical, and both are already enforced in
code. Do not switch them off without reading why.

- [ ] **Leave `SERVE_MARKET_DATA` unset.** SEC filings are U.S. government work
      and public domain, so fundamentals and every ratio derived from them can
      be republished freely. Scraped price data generally cannot be
      redistributed under the provider's terms. With the flag off, the app
      hides market cap, P/E, EV/EBITDA, yields and beta, and says on each page
      that it is doing so. Set it to `true` only after moving to a feed whose
      licence permits redistribution — Tiingo and Polygon both do on paid tiers.
- [ ] **Add TTM/quarterly data first.** Every figure comes from the last filed
      10-K. A visitor who knows a company mid-cycle will see a revenue figure
      far below its current run-rate and conclude the site is broken. Micron
      currently reads $37.4B against a materially higher present run-rate. The
      numbers are right; they are just not current. Treat this as a
      prerequisite, not a follow-up.
- [ ] **Keep the disclaimer and the no-verdict rule.** Both are rendered by
      `finmodel/web/ui.py` on every page. The research notes have no
      buy/sell/hold field by design, and the test suite asserts that no verdict
      field exists and no buy/sell language appears outside the disclaimer. A
      public site issuing verdicts is giving investment advice, a regulated
      activity in most jurisdictions.
- [ ] **Add a LICENSE.** The repo is public but unlicensed, which under default
      copyright means nobody has permission to use or fork it.

---

## Other hosts

Streamlit Cloud is the least work, but the app is an ordinary Python process:

```bash
streamlit run streamlit_app.py --server.port $PORT --server.address 0.0.0.0
```

That line works as-is on Render, Railway, Fly.io, Hugging Face Spaces, or any
container host. Set `DATABASE_URL` in the host's environment.

---

## Troubleshooting

**"No data loaded yet" after deploying** — expected on Streamlit Cloud with no
`DATABASE_URL`. Click *Load demo data*, or attach Postgres (above).

**Page is slow on first load** — the first request computes sector benchmarks
and warms the caches. Subsequent loads read from cache for five minutes
(`UI_CACHE_TTL` to change it).

**Worker ran but the app shows old numbers** — caches have a five-minute TTL.
Click **Refresh data** in the sidebar.

**`ModuleNotFoundError: finmodel`** — run `streamlit run` from the repository
root, not from inside `pages/`.

**Everything shows `—` in valuation columns** — `SERVE_MARKET_DATA` is off.
That is the default and it is deliberate; see above.
