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

## Run it locally — two commands

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

That opens http://localhost:8501 with **497 real S&P 500 companies**. The
dataset ships in the repository at `data/sp500.db` and is used automatically
when `DATABASE_URL` is unset, so there is nothing to load and no key to set
before the app works.

That file is filings-only. `export_dataset.py` strips `market_data` and every
price-derived benchmark before it is committed, because a public repository
is redistribution and scraped price data is not ours to redistribute. The app
hides those fields anyway unless `SERVE_MARKET_DATA=true`, so nothing is lost.
To get them locally, run the worker against a database of your own:

```bash
export EDGAR_USER_AGENT="Your Name you@email.com"
export DATABASE_URL="sqlite:///my.db"
python worker.py --sp500
SERVE_MARKET_DATA=true DATABASE_URL="sqlite:///my.db" streamlit run streamlit_app.py
```

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
| **Frontier** | Efficient frontier, capital allocation line and the complete portfolio, over assumptions you set |
| **Methodology** | Every threshold, every formula, and the known limitations |

The Frontier page needs no database and no price feed — it works on a fresh
deployment with nothing loaded. That is deliberate: Markowitz is a map from
expected returns and a covariance to weights, and the expected returns are an
assumption no data source supplies. You can upload a price CSV to set the
covariance from history and still enter the returns yourself, which is the
split the estimation-error literature recommends.

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

### You do not need a database

Streamlit Cloud restarts containers freely and anything written to local disk
is lost, which is why this used to require managed Postgres before the app
could show anything. It does not any more: the dataset is committed to the
repository, so a fresh container has real data the moment it boots. Nothing to
provision, nothing to pay for, no secrets to set.

**When you do want Postgres** is when the data must change between deploys —
a nightly refresh updating a live site rather than a dataset that moves when
CI rebuilds it. [Neon](https://neon.tech) and [Supabase](https://supabase.com)
both have free tiers large enough. Then in **Settings → Secrets**:

```toml
DATABASE_URL = "postgresql://user:password@host/dbname"
EDGAR_USER_AGENT = "Your Name you@email.com"
```

Streamlit exposes secrets as environment variables, which is what
`finmodel.web.store.Database` reads — no code change. `DATABASE_URL` takes
precedence over the shipped file.

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
