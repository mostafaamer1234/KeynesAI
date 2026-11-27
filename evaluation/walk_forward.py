"""Leakage-free walk-forward evaluation of next-day direction prediction.

The original backtest applied *_pattern_model.pkl models that were trained on the
full price history (with 5-day-ahead labels) to every test window, so test-period
information leaked into the predictions. This script re-evaluates the same idea
honestly:

  * every feature at day t uses only prices up to and including day t
  * the classifier is retrained every `step` days on strictly earlier data
  * results are compared with the naive "always up" baseline and, as a trading
    strategy (long when the model predicts up, otherwise cash, with costs),
    with buy-and-hold

    python evaluation/walk_forward.py                 # all tickers in the repo
    python evaluation/walk_forward.py --tickers AAPL
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parent.parent
HORIZONS = [2, 5, 20, 60, 250]
TRADING_DAYS = 252


def load_prices(ticker: str) -> pd.DataFrame:
    df = pd.read_csv(ROOT / f"{ticker}.csv", index_col=0)
    df.index = pd.to_datetime(df.index, utc=True)
    return df[["Open", "High", "Low", "Close", "Volume"]].sort_index()


def feature_frame(prices: pd.DataFrame) -> pd.DataFrame:
    """Features for every day, including the latest one, using data up to that day only."""
    close = prices["Close"]
    ret = close.pct_change()
    feats = pd.DataFrame(index=prices.index)
    for h in HORIZONS:
        feats[f"close_ratio_{h}"] = close / close.rolling(h).mean()
        feats[f"up_days_{h}"] = (ret > 0).astype(float).rolling(h).sum() / h
    feats["vol_20"] = ret.rolling(20).std()
    feats["mom_10"] = close / close.shift(10) - 1
    feats["vol_ratio_20"] = prices["Volume"] / prices["Volume"].rolling(20).mean()
    return feats


def make_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Features for day t use data up to day t; the target is whether day t+1 closes higher."""
    close = prices["Close"]
    feats = feature_frame(prices)
    feats["next_ret"] = close.shift(-1) / close - 1
    feats["target"] = (feats["next_ret"] > 0).astype(int)
    return feats.dropna()


def walk_forward(feats: pd.DataFrame, start: str = "2000-01-01", step: int = 250, seed: int = 0, n_estimators: int = 200, n_jobs: int = -1) -> pd.DataFrame:
    cols = [c for c in feats.columns if c not in ("next_ret", "target")]
    first = feats.index.searchsorted(pd.Timestamp(start, tz="UTC"))
    out = []
    for i in range(first, len(feats), step):
        train, test = feats.iloc[:i], feats.iloc[i : i + step]
        model = RandomForestClassifier(n_estimators=n_estimators, min_samples_leaf=50, random_state=seed, n_jobs=n_jobs)
        model.fit(train[cols], train["target"])
        out.append(test[["next_ret", "target"]].assign(prob_up=model.predict_proba(test[cols])[:, 1]))
    return pd.concat(out)


def strategy_stats(daily_ret: pd.Series) -> dict[str, float]:
    equity = (1 + daily_ret).cumprod()
    years = len(daily_ret) / TRADING_DAYS
    return {
        "CAGR": equity.iloc[-1] ** (1 / years) - 1,
        "Sharpe": np.sqrt(TRADING_DAYS) * daily_ret.mean() / daily_ret.std(),
        "MaxDD": (equity / equity.cummax() - 1).min(),
    }


def evaluate(ticker: str, cost_bps: float = 5.0, threshold: float = 0.5) -> dict[str, float | str]:
    preds = walk_forward(make_features(load_prices(ticker)))
    signal = (preds["prob_up"] > threshold).astype(int)
    switches = signal.diff().abs().fillna(signal.iloc[0])
    strat = signal * preds["next_ret"] - switches * cost_bps / 1e4
    s, bh = strategy_stats(strat), strategy_stats(preds["next_ret"])
    up_calls = signal == 1
    return {
        "ticker": ticker,
        "days": len(preds),
        "accuracy": float((signal == preds["target"]).mean()),
        "always_up": float(preds["target"].mean()),
        "precision_up": float(preds.loc[up_calls, "target"].mean()) if up_calls.any() else float("nan"),
        "strat_CAGR": s["CAGR"],
        "bh_CAGR": bh["CAGR"],
        "strat_Sharpe": s["Sharpe"],
        "bh_Sharpe": bh["Sharpe"],
        "strat_MaxDD": s["MaxDD"],
        "bh_MaxDD": bh["MaxDD"],
    }


def forecast(prices: pd.DataFrame, years: int = 5, cost_bps: float = 5.0, threshold: float = 0.5, seed: int = 0, n_estimators: int = 200, n_jobs: int = -1) -> dict:
    """The model's call for the next trading day, plus its walk-forward record over recent years.

    The call comes from a classifier trained on every labelled day so far, so it uses no future
    data. The record retrains yearly on strictly earlier data, exactly as in `evaluate`.
    """
    feats = make_features(prices)
    if len(feats) < 2 * 250:
        raise ValueError("Need at least two years of daily prices after the 250-day warm-up.")
    cols = [c for c in feats.columns if c not in ("next_ret", "target")]
    latest = feature_frame(prices).dropna().iloc[[-1]]
    model = RandomForestClassifier(n_estimators=n_estimators, min_samples_leaf=50, random_state=seed, n_jobs=n_jobs)
    model.fit(feats[cols], feats["target"])
    prob_up = float(model.predict_proba(latest[cols])[0, 1])

    start = max(feats.index[-1] - pd.Timedelta(days=365 * years), feats.index[min(len(feats) - 1, 250)])
    preds = walk_forward(feats, start=str(start.date()), seed=seed, n_estimators=n_estimators, n_jobs=n_jobs)
    signal = (preds["prob_up"] > threshold).astype(int)
    switches = signal.diff().abs().fillna(signal.iloc[0])
    strat_ret = signal * preds["next_ret"] - switches * cost_bps / 1e4
    strat, hold = (1 + strat_ret).cumprod(), (1 + preds["next_ret"]).cumprod()
    weekly = strat.resample("W").last().dropna().index
    s, bh = strategy_stats(strat_ret), strategy_stats(preds["next_ret"])
    return {
        "as_of": latest.index[-1].strftime("%Y-%m-%d"),
        "last_close": float(prices["Close"].iloc[-1]),
        "prob_up": round(prob_up, 4),
        "record": {
            "from": preds.index[0].strftime("%Y-%m-%d"),
            "days": int(len(preds)),
            "accuracy": round(float((signal == preds["target"]).mean()), 4),
            "always_up": round(float(preds["target"].mean()), 4),
            "in_market": round(float(signal.mean()), 4),
            "strat_total": round(float(strat.iloc[-1] - 1), 4),
            "hold_total": round(float(hold.iloc[-1] - 1), 4),
            "strat_MaxDD": round(float(s["MaxDD"]), 4),
            "hold_MaxDD": round(float(bh["MaxDD"]), 4),
            "dates": [d.strftime("%Y-%m-%d") for d in weekly],
            "strategy": [round(float(strat.asof(d)), 4) for d in weekly],
            "hold": [round(float(hold.asof(d)), 4) for d in weekly],
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+", default=["^GSPC", "AAPL", "MSFT", "IBM"])
    args = ap.parse_args()
    rows = [evaluate(t) for t in args.tickers]
    print("Walk-forward from 2000, yearly retraining, 5 bp per position change\n")
    print("| Ticker | Test days | Accuracy | Always-up baseline | Precision of 'up' calls | Strategy CAGR / Sharpe / MaxDD | Buy & hold CAGR / Sharpe / MaxDD |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        print(
            f"| {r['ticker']} | {r['days']} | {r['accuracy']:.1%} | {r['always_up']:.1%} | {r['precision_up']:.1%} | "
            f"{r['strat_CAGR']:.1%} / {r['strat_Sharpe']:.2f} / {r['strat_MaxDD']:.0%} | "
            f"{r['bh_CAGR']:.1%} / {r['bh_Sharpe']:.2f} / {r['bh_MaxDD']:.0%} |"
        )


if __name__ == "__main__":
    main()
