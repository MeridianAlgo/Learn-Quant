"""
Feature Engineering for Financial Machine Learning
--------------------------------------------------
In quant ML, the model is rarely the hard part — the *features* and the *labels*
are. Raw prices are non-stationary and tell a model almost nothing; engineered
features (returns, momentum, volatility, technical signals) and carefully built
labels are where the signal lives.

This module builds a tidy feature matrix from a price series and implements two
labelling schemes:

* a simple **forward-return / direction** label, and
* the **triple-barrier method** (López de Prado) — label a trade by whether it
  hits a profit-take, a stop-loss, or a time limit first.

It also provides a **purged, leak-free train/test split** so you never validate
on information from the future.

Built on pandas + NumPy.
"""

from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd

PriceLike = Union[list, np.ndarray, pd.Series]


def _as_series(prices: PriceLike) -> pd.Series:
    if isinstance(prices, pd.Series):
        return prices.astype(float)
    return pd.Series(np.asarray(prices, dtype=float))


def make_features(prices: PriceLike, windows=(5, 10, 20)) -> pd.DataFrame:
    """Construct a feature matrix from a single price series.

    Features (all stationary, all backward-looking):

    * ``ret_1`` — one-period log return
    * ``mom_w`` — w-period momentum (cumulative return)
    * ``vol_w`` — rolling return volatility
    * ``zscore_w`` — z-score of price vs. its rolling mean
    * ``rsi_14`` — Wilder's Relative Strength Index
    * ``dist_sma_w`` — distance of price from its w-period SMA (fraction)

    Returns a DataFrame aligned to ``prices`` with leading NaNs dropped.
    """
    p = _as_series(prices)
    log_ret = np.log(p / p.shift(1))
    feats = pd.DataFrame(index=p.index)
    feats["ret_1"] = log_ret

    for w in windows:
        feats[f"mom_{w}"] = p / p.shift(w) - 1.0
        feats[f"vol_{w}"] = log_ret.rolling(w).std()
        roll_mean = p.rolling(w).mean()
        roll_std = p.rolling(w).std()
        feats[f"zscore_{w}"] = (p - roll_mean) / roll_std
        feats[f"dist_sma_{w}"] = p / roll_mean - 1.0

    feats["rsi_14"] = _rsi(p, 14)
    return feats.dropna()


def _rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI."""
    delta = prices.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return (100.0 - 100.0 / (1.0 + rs)).fillna(50.0)


def forward_return_label(prices: PriceLike, horizon: int = 5) -> pd.DataFrame:
    """Label each bar by its forward return over ``horizon`` periods.

    Returns columns ``fwd_ret`` (continuous) and ``direction`` (+1/-1).
    The last ``horizon`` rows are dropped — their future is unknown.
    """
    p = _as_series(prices)
    fwd = p.shift(-horizon) / p - 1.0
    out = pd.DataFrame({"fwd_ret": fwd})
    out["direction"] = np.sign(fwd)
    return out.dropna()


def triple_barrier_labels(
    prices: PriceLike,
    pt: float = 0.02,
    sl: float = 0.02,
    max_holding: int = 10,
) -> pd.DataFrame:
    """Triple-barrier labelling (López de Prado, *Advances in Financial ML*).

    For each starting bar, walk forward until one of three barriers is hit:

    * **profit-take** at ``+pt`` → label ``+1``
    * **stop-loss** at ``-sl`` → label ``-1``
    * **time limit** after ``max_holding`` bars → label ``0`` (sign of the
      return at expiry)

    Args:
        prices: Price series.
        pt: Upper barrier as a positive return fraction.
        sl: Lower barrier as a positive return fraction (applied as ``-sl``).
        max_holding: Vertical (time) barrier in bars.

    Returns:
        DataFrame with ``label``, ``ret`` (return at the touch) and
        ``hit`` ('pt' / 'sl' / 'time').
    """
    p = _as_series(prices).reset_index(drop=True)
    n = len(p)
    labels, rets, hits = [], [], []
    idx = []
    for t in range(n - 1):
        end = min(t + max_holding, n - 1)
        entry = p.iloc[t]
        label, ret, hit = 0, p.iloc[end] / entry - 1.0, "time"
        for s in range(t + 1, end + 1):
            change = p.iloc[s] / entry - 1.0
            if change >= pt:
                label, ret, hit = 1, change, "pt"
                break
            if change <= -sl:
                label, ret, hit = -1, change, "sl"
                break
        else:
            label = int(np.sign(ret))
        idx.append(t)
        labels.append(label)
        rets.append(ret)
        hits.append(hit)
    return pd.DataFrame({"label": labels, "ret": rets, "hit": hits}, index=idx)


def purged_train_test_split(features: pd.DataFrame, labels: pd.Series, test_size: float = 0.3, embargo: int = 5):
    """Chronological split with an embargo gap to prevent leakage.

    The first ``1 - test_size`` of the (time-ordered) rows are training; an
    ``embargo`` of rows is dropped between train and test so look-back windows in
    the test set cannot peek at training data.
    """
    df = features.join(labels.rename("label"), how="inner").dropna()
    n = len(df)
    cut = int(n * (1.0 - test_size))
    train = df.iloc[: max(cut - embargo, 0)]
    test = df.iloc[cut:]
    x_cols = list(features.columns)
    return (train[x_cols], train["label"], test[x_cols], test["label"])


if __name__ == "__main__":
    rng = np.random.default_rng(3)
    n = 1000
    prices = pd.Series(100 * np.cumprod(1 + 0.0004 + 0.01 * rng.standard_normal(n)))

    feats = make_features(prices)
    print("Feature matrix")
    print("=" * 60)
    print(f"shape: {feats.shape}  columns: {list(feats.columns)}")
    print(feats.tail(3).round(4).to_string())

    fwd = forward_return_label(prices, horizon=5)
    print("\nForward-return labels (last 3):")
    print(fwd.tail(3).round(4).to_string())

    tb = triple_barrier_labels(prices, pt=0.02, sl=0.02, max_holding=10)
    counts = tb["label"].value_counts().sort_index()
    print("\nTriple-barrier label balance:")
    for lbl, c in counts.items():
        print(f"  label {int(lbl):+d}: {c:4d}  ({c / len(tb):.1%})")
    print("  hit breakdown:", tb["hit"].value_counts().to_dict())

    x_tr, y_tr, x_te, y_te = purged_train_test_split(feats, fwd["direction"], test_size=0.3, embargo=5)
    print(f"\nPurged split -> train {x_tr.shape}, test {x_te.shape}")
