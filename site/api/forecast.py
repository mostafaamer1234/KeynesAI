"""GET /api/forecast?ticker=AAPL -> the walk-forward model's call for the next trading day.

Prices come from Yahoo Finance's public chart endpoint (adjusted closes since 2000). The model
code is evaluation/walk_forward.py, copied here as _walk_forward.py by
`python -m evaluation.export_site_data` so this function deploys on its own.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.parse
import urllib.error
import urllib.request

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _walk_forward import forecast  # noqa: E402

TICKER = re.compile(r"^[A-Z0-9^][A-Z0-9.\-=^]{0,14}$")


def fetch(ticker: str) -> tuple[pd.DataFrame, dict]:
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(ticker)}?period1=946684800&period2={int(time.time()) + 86400}&interval=1d&events=div%2Csplit"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; keynesai-demo)"})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.load(r)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise LookupError(f"Yahoo Finance has no ticker called {ticker}.") from e
        raise
    res = (data.get("chart") or {}).get("result")
    if not res:
        raise LookupError("No price history found for that ticker.")
    res = res[0]
    q = res["indicators"]["quote"][0]
    adj = (res["indicators"].get("adjclose") or [{}])[0].get("adjclose") or q["close"]
    df = pd.DataFrame({"Open": q["open"], "High": q["high"], "Low": q["low"], "Close": adj, "Volume": q["volume"]},
                      index=pd.to_datetime(res["timestamp"], unit="s", utc=True)).dropna()
    meta = res.get("meta", {})
    regular = (meta.get("currentTradingPeriod") or {}).get("regular") or {}
    if regular and regular.get("start", 0) <= time.time() < regular.get("end", 0) and len(df) and df.index[-1].timestamp() >= regular["start"]:
        df = df.iloc[:-1]  # today's bar is still moving; use the last completed close
    return df, meta


def _reply(start_response, code: int, body: dict, cache: bool = False):
    raw = json.dumps(body).encode()
    status = {200: "200 OK", 400: "400 Bad Request", 404: "404 Not Found", 422: "422 Unprocessable Entity", 502: "502 Bad Gateway"}[code]
    start_response(status, [("Content-Type", "application/json"), ("Content-Length", str(len(raw))),
                            ("Cache-Control", "public, s-maxage=3600, stale-while-revalidate=86400" if cache else "no-store")])
    return [raw]


def app(environ, start_response):
    """WSGI entry point. Vercel keeps a WSGI app warm between requests."""
    qs = urllib.parse.parse_qs(environ.get("QUERY_STRING", ""))
    ticker = (qs.get("ticker", [""])[0]).strip().upper()
    if not TICKER.match(ticker):
        return _reply(start_response, 400, {"error": "Enter a ticker like AAPL, MSFT or ^GSPC."})
    try:
        t0 = time.time()
        prices, meta = fetch(ticker)
        t1 = time.time()
        out = forecast(prices, years=3, n_estimators=60, n_jobs=1)  # lighter than the 200-tree study so it answers in seconds
        t2 = time.time()  # one vCPU: worker processes only add start-up time
    except LookupError as e:
        return _reply(start_response, 404, {"error": str(e)})
    except ValueError as e:
        return _reply(start_response, 422, {"error": str(e)})
    except Exception:  # network or upstream format problems
        return _reply(start_response, 502, {"error": "Couldn't get prices for that ticker right now. Try again in a minute."})
    out["timing"] = {"fetch_s": round(t1 - t0, 2), "model_s": round(t2 - t1, 2)}
    out.update({"ticker": ticker, "name": meta.get("longName") or meta.get("shortName") or ticker, "currency": meta.get("currency", "")})
    return _reply(start_response, 200, out, cache=True)
