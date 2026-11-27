"""Export walk-forward results and weekly equity curves for the static site (site/data.json),
and copy evaluation/walk_forward.py next to the live-signal function (site/api/_walk_forward.py).

    python -m evaluation.export_site_data
"""

from __future__ import annotations

import json
import shutil

from evaluation.walk_forward import ROOT, evaluate, load_prices, make_features, walk_forward

TICKERS = ["^GSPC", "AAPL", "MSFT", "IBM"]


def curves(ticker: str, cost_bps: float = 5.0, threshold: float = 0.5) -> dict[str, list]:
    preds = walk_forward(make_features(load_prices(ticker)))
    signal = (preds["prob_up"] > threshold).astype(int)
    switches = signal.diff().abs().fillna(signal.iloc[0])
    strat = (1 + signal * preds["next_ret"] - switches * cost_bps / 1e4).cumprod()
    hold = (1 + preds["next_ret"]).cumprod()
    weekly = strat.resample("W").last().dropna().index
    return {
        "dates": [d.strftime("%Y-%m-%d") for d in weekly],
        "strategy": [round(float(strat.asof(d)), 4) for d in weekly],
        "hold": [round(float(hold.asof(d)), 4) for d in weekly],
        "in_market": round(float(signal.mean()), 4),
    }


def main() -> None:
    out = {}
    for t in TICKERS:
        stats = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in evaluate(t).items()}
        out[t] = {**stats, **curves(t)}
        print(t, stats["accuracy"], stats["always_up"])
    (ROOT / "site").mkdir(exist_ok=True)
    (ROOT / "site" / "data.json").write_text(json.dumps(out))
    # the live-signal function deploys from site/, so it carries its own copy of the model code
    shutil.copy(ROOT / "evaluation" / "walk_forward.py", ROOT / "site" / "api" / "_walk_forward.py")


if __name__ == "__main__":
    main()
