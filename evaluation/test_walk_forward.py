from __future__ import annotations

import numpy as np
import pandas as pd

from evaluation.walk_forward import load_prices, make_features, strategy_stats


def test_features_do_not_depend_on_future_prices():
    prices = load_prices("AAPL").iloc[:2000]
    base = make_features(prices)
    tampered = prices.copy()
    tampered.iloc[1500:, :] *= 10  # change every price after day 1500
    after = make_features(tampered)
    cols = [c for c in base.columns if c not in ("next_ret", "target")]
    cutoff = prices.index[1499]
    pd.testing.assert_frame_equal(base.loc[:cutoff, cols], after.loc[:cutoff, cols])


def test_target_is_next_day_direction():
    feats = make_features(load_prices("MSFT").iloc[:500])
    assert set(feats["target"].unique()) <= {0, 1}
    assert ((feats["next_ret"] > 0).astype(int) == feats["target"]).all()


def test_strategy_stats_on_constant_growth():
    daily = pd.Series([0.001] * 252)
    stats = strategy_stats(daily)
    assert np.isclose(stats["CAGR"], 1.001**252 - 1)
    assert stats["MaxDD"] == 0
