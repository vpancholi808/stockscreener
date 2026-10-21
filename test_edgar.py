"""EDGAR client — the parts that can be wrong without touching the network.

The User-Agent checks here exist because of a live failure. A value pasted
into a GitHub Actions secret carried a trailing newline, so `requests`
rejected every call with `InvalidHeader` before a single byte was sent. CI
masked the secret in its own error message, so the log showed only '***',
and four separate probes failed identically — which reads exactly like SEC
blocking the runner, and is not that at all.
"""
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

from requests.exceptions import InvalidHeader
from requests.models import PreparedRequest

from finmodel.data.edgar import (EdgarClient, EdgarError, FACTS_URL,
                                 TICKERS_URL, normalize_user_agent)

fails = []


def check(name, cond, detail=""):
    print("  [%s] %s%s" % ("OK" if cond else "FAIL", name,
                           (" — " + detail) if detail else ""))
    if not cond:
        fails.append(name)


GOOD = "Vishal Pancholi you@example.com"

print("=" * 66)
print("1) USER-AGENT NORMALISATION")
print("=" * 66)

cases = [
    ("trailing newline", GOOD + "\n"),
    ("trailing CRLF", GOOD + "\r\n"),
    ("leading whitespace", "   " + GOOD),
    ("surrounding whitespace", "\t " + GOOD + " \n"),
    ("tab separated", "Vishal\tPancholi\tyou@example.com"),
    ("double spaces", "Vishal  Pancholi   you@example.com"),
    ("embedded newline", "Vishal Pancholi\nyou@example.com"),
]
for label, raw in cases:
    check("%s is repaired" % label, normalize_user_agent(raw) == GOOD,
          repr(normalize_user_agent(raw)))

check("a clean value is left alone", normalize_user_agent(GOOD) == GOOD)

for label, bad in (("empty string", ""), ("whitespace only", "  \n\t "),
                   ("None", None)):
    try:
        normalize_user_agent(bad)
        check("%s raises" % label, False)
    except ValueError as exc:
        check("%s raises with a usable message" % label,
              "EDGAR_USER_AGENT" in str(exc))

print()
print("=" * 66)
print("2) THE RESULT IS A LEGAL HTTP HEADER")
print("=" * 66)

# This is the exact validation that failed in CI. HTTP forbids CR and LF in
# a header value — permitting them would allow response splitting — so
# requests refuses to build the request at all.


def header_accepted(value):
    req = PreparedRequest()
    try:
        req.prepare_headers({"User-Agent": value})
        return True
    except InvalidHeader:
        return False


check("a raw trailing newline IS rejected by requests",
      not header_accepted(GOOD + "\n"),
      "this is the bug that broke the workflow")
check("the normalised value is accepted",
      header_accepted(normalize_user_agent(GOOD + "\n")))
for label, raw in cases:
    check("normalised %s is a legal header" % label,
          header_accepted(normalize_user_agent(raw)))

print()
print("=" * 66)
print("3) THE CLIENT NORMALISES ON THE WAY IN")
print("=" * 66)

client = EdgarClient(user_agent=GOOD + "\n")
sent = client.session.headers["User-Agent"]
check("EdgarClient strips the newline", sent == GOOD, repr(sent))
check("the session header is a legal header", header_accepted(sent))
check("the client exposes the normalised value", client.user_agent == GOOD)
check("gzip is requested",
      "gzip" in client.session.headers.get("Accept-Encoding", ""))

try:
    EdgarClient(user_agent="   ")
    check("an empty user agent raises at construction", False)
except ValueError:
    check("an empty user agent raises at construction", True)

print()
print("=" * 66)
print("4) SEC FAIR-ACCESS RATE LIMIT")
print("=" * 66)

check("default interval stays under SEC's 10 requests/second",
      EdgarClient(user_agent=GOOD).min_interval >= 0.10,
      "%.2fs between calls = %.1f req/s"
      % (client.min_interval, 1 / client.min_interval))

paced = EdgarClient(user_agent=GOOD, min_interval=0.05)
start = time.time()
for _ in range(3):
    paced._throttle()
elapsed = time.time() - start
check("the throttle actually sleeps between calls", elapsed >= 0.08,
      "3 calls took %.3fs" % elapsed)

print()
print("=" * 66)
print("5) ENDPOINTS")
print("=" * 66)

check("ticker map points at www.sec.gov",
      TICKERS_URL == "https://www.sec.gov/files/company_tickers.json")
check("company facts points at data.sec.gov",
      FACTS_URL.startswith("https://data.sec.gov/api/xbrl/companyfacts/"))
check("CIK is zero-padded to 10 digits",
      FACTS_URL.format(cik=723125).endswith("CIK0000723125.json"),
      FACTS_URL.format(cik=723125).rsplit("/", 1)[-1])
check("EdgarError is a RuntimeError", issubclass(EdgarError, RuntimeError))

print()
print("=" * 66)
print("6) THE BULK PATH NORMALISES TOO")
print("=" * 66)

from finmodel.data.bulk import BulkData

import tempfile
tmp = tempfile.mkdtemp(prefix="finmodel_bulk_")
bulk = BulkData(directory=tmp, user_agent=GOOD + "\n")
check("BulkData strips the newline",
      bulk.headers["User-Agent"] == GOOD, repr(bulk.headers["User-Agent"]))
check("the bulk header is legal",
      header_accepted(bulk.headers["User-Agent"]))
try:
    BulkData(directory=tmp, user_agent="no-email-here")
    check("BulkData still demands an email", False)
except ValueError:
    check("BulkData still demands an email", True)

print()
if fails:
    print("FAILED (%d): %s" % (len(fails), "; ".join(fails)))
    sys.exit(1)
print("test_edgar: all checks passed")
