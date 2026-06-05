import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "Machine Learning - Feature Engineering"))
from feature_engineering import (
    forward_return_label,
    make_features,
    purged_train_test_split,
    triple_barrier_labels,
)


def _prices(n=600, seed=0):
    rng = np.random.default_rng(seed)
    return pd.Series(100 * np.cumprod(1 + 0.0004 + 0.01 * rng.standard_normal(n)))


def test_make_features_no_nans_and_columns():
    feats = make_features(_prices(), windows=(5, 10, 20))
    assert not feats.isna().any().any()
    assert "ret_1" in feats.columns
    assert "rsi_14" in feats.columns
    assert "zscore_20" in feats.columns
    # RSI stays in [0, 100].
    assert feats["rsi_14"].between(0, 100).all()


def test_forward_return_label_direction():
    p = pd.Series([1.0, 2.0, 1.0, 2.0, 1.0, 3.0, 4.0, 5.0])
    fwd = forward_return_label(p, horizon=1)
    # direction is the sign of the next-bar return.
    assert set(np.unique(fwd["direction"])).issubset({-1.0, 0.0, 1.0})


def test_triple_barrier_labels_values():
    tb = triple_barrier_labels(_prices(seed=2), pt=0.02, sl=0.02, max_holding=10)
    assert set(np.unique(tb["label"])).issubset({-1, 0, 1})
    assert set(tb["hit"].unique()).issubset({"pt", "sl", "time"})
    assert len(tb) > 0


def test_triple_barrier_hits_profit_take():
    # Monotonic rise: a stop-loss can never trigger, and bars with a full
    # window ahead must hit the profit-take barrier.
    p = pd.Series(np.linspace(100, 130, 40))
    tb = triple_barrier_labels(p, pt=0.05, sl=0.05, max_holding=20)
    assert (tb["hit"] != "sl").all()
    assert (tb["label"] >= 0).all()
    early = tb.iloc[:10]  # plenty of room to reach +5%
    assert (early["hit"] == "pt").all()


def test_purged_split_is_chronological():
    feats = make_features(_prices(seed=3))
    labels = forward_return_label(_prices(seed=3), horizon=5)["direction"]
    x_tr, y_tr, x_te, y_te = purged_train_test_split(feats, labels, test_size=0.3, embargo=5)
    assert len(x_tr) > 0 and len(x_te) > 0
    # Training indices all precede test indices (embargo enforces a gap).
    assert x_tr.index.max() < x_te.index.min()
    assert list(x_tr.columns) == list(feats.columns)
