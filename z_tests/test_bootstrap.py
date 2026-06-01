import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "Quantitative Methods - Bootstrap"))
from bootstrap import (
    block_bootstrap,
    confidence_interval,
    iid_bootstrap,
    stationary_bootstrap,
)


def test_iid_bootstrap_shape_and_centering():
    rng = np.random.default_rng(0)
    data = rng.normal(0.5, 1.0, 500)
    est = iid_bootstrap(data, np.mean, n_boot=1000, seed=1)
    assert est.shape == (1000,)
    # Bootstrap mean-of-means should sit near the sample mean.
    assert abs(est.mean() - data.mean()) < 0.1


def test_block_bootstrap_shape():
    rng = np.random.default_rng(1)
    data = rng.normal(0, 1, 300)
    est = block_bootstrap(data, np.mean, block_size=20, n_boot=500, seed=2)
    assert est.shape == (500,)


def test_stationary_bootstrap_shape():
    rng = np.random.default_rng(2)
    data = rng.normal(0, 1, 300)
    est = stationary_bootstrap(data, np.mean, expected_block=20, n_boot=500, seed=3)
    assert est.shape == (500,)


def test_confidence_interval_brackets_truth():
    rng = np.random.default_rng(3)
    data = rng.normal(0.5, 1.0, 1000)
    est = iid_bootstrap(data, np.mean, n_boot=2000, seed=4)
    lo, hi = confidence_interval(est, alpha=0.05)
    assert lo < data.mean() < hi
    assert lo < hi


def test_reproducible_with_seed():
    rng = np.random.default_rng(4)
    data = rng.normal(0, 1, 200)
    a = iid_bootstrap(data, np.mean, n_boot=300, seed=42)
    b = iid_bootstrap(data, np.mean, n_boot=300, seed=42)
    assert np.array_equal(a, b)
