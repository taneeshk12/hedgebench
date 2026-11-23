"""Real market data, downloaded by each user from free public sources and cached
locally. HedgeBench never redistributes market data: Deribit's terms restrict
redistribution, so the benchmark ships download code, not data.

Cache folder: $HEDGEBENCH_CACHE, or ~/.cache/hedgebench
"""
import json
import os
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

CACHE = Path(os.environ.get("HEDGEBENCH_CACHE", Path.home() / ".cache" / "hedgebench"))
DERIBIT = "https://www.deribit.com/api/v2/public"
DERIBIT_HIST = "https://history.deribit.com/api/v2/public"


def _cached(name, build):
    path = CACHE / name
    if path.exists():
        return pd.read_csv(path, index_col=0, parse_dates=True)
    df = build()
    if df.empty:
        raise RuntimeError(f"Download for {name} returned no data")
    CACHE.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    return df


def spy_vix():
    """Daily SPY close and VIX close from Yahoo Finance (needs `pip install yfinance`)."""
    def build():
        import yfinance as yf
        raw = yf.download(["SPY", "^VIX"], start="2005-01-01", end="2026-01-01",
                          auto_adjust=True, progress=False)
        return raw["Close"].rename(columns={"SPY": "price", "^VIX": "vol"}).dropna()
    return _cached("spy_vix.csv", build)


def _get(base, method, **params):
    url = f"{base}/{method}?" + "&".join(f"{k}={v}" for k, v in params.items())
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                return json.load(r)["result"]
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)


def btc_dvol():
    """Daily BTC index at 08:00 UTC (Deribit settlement) and DVOL, from Deribit."""
    def build():
        rows, offset = [], 0
        while True:
            res = _get(DERIBIT, "get_delivery_prices", index_name="btc_usd", offset=offset, count=100)
            rows += res["data"]
            offset += len(res["data"])
            if not res["data"] or offset >= res["records_total"]:
                break
        price = pd.Series({pd.Timestamp(r["date"]): r["delivery_price"] for r in rows}).sort_index()
        dvol, start = {}, pd.Timestamp("2021-03-01")
        while start < price.index[-1]:
            end = start + pd.Timedelta(days=900)
            res = _get(DERIBIT, "get_volatility_index_data", currency="BTC", resolution="1D",
                       start_timestamp=int(start.timestamp() * 1000), end_timestamp=int(end.timestamp() * 1000))
            for ts, o, *_ in res["data"]:
                dvol[pd.Timestamp(ts, unit="ms").normalize()] = o
            start = end
        df = pd.DataFrame({"price": price, "vol": pd.Series(dvol)}).dropna()
        return df[df.index < "2026-01-01"]
    return _cached("btc_dvol.csv", build)


def btc_monthly_calls(days=30):
    """For each monthly BTC expiry: the listed call closest to the money `days`
    before expiry and its Deribit mark price at 08:00 UTC that day."""
    def build():
        market = btc_dvol()
        insts = _get(DERIBIT_HIST, "get_instruments", currency="BTC", kind="option", expired="true")
        by_exp = {}
        for i in insts:
            if i["settlement_period"] == "month" and i["option_type"] == "call":
                by_exp.setdefault(i["expiration_timestamp"], []).append(i)
        out = []
        for exp_ms in sorted(by_exp):
            expiry = pd.Timestamp(exp_ms, unit="ms").normalize()
            start = expiry - pd.Timedelta(days=days)
            if start not in market.index or expiry not in market.index:
                continue
            S0 = market.loc[start, "price"]
            t0 = int(datetime(start.year, start.month, start.day, 8, tzinfo=timezone.utc).timestamp() * 1000)
            listed = [i for i in by_exp[exp_ms] if i["creation_timestamp"] <= t0]
            if not listed:
                continue
            inst = min(listed, key=lambda i: abs(i["strike"] - S0))
            trades = _get(DERIBIT_HIST, "get_last_trades_by_instrument_and_time",
                          instrument_name=inst["instrument_name"], start_timestamp=t0 - 6 * 3600_000,
                          end_timestamp=t0 + 6 * 3600_000, count=1000)["trades"]
            if trades:
                tr = min(trades, key=lambda t: abs(t["timestamp"] - t0))
                out.append({"start": start, "expiry": expiry, "strike": inst["strike"], "S0": S0,
                            "premium": tr["mark_price"] * tr["index_price"], "instrument": inst["instrument_name"]})
            time.sleep(0.1)
        return pd.DataFrame(out).set_index("start")
    df = _cached(f"btc_calls_{days}d.csv", build)
    df["expiry"] = pd.to_datetime(df["expiry"])      # CSV round-trip gives strings
    return df
