import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "Quantitative Methods - Principal Component Analysis"))
from pca import cumulative_variance, pca, reconstruct, standardize


def test_standardize_zero_mean_unit_var():
    rng = np.random.default_rng(0)
    data = rng.normal(5, 3, (200, 4))
    z = standardize(data)
    assert np.allclose(z.mean(axis=0), 0, atol=1e-10)
    assert np.allclose(z.std(axis=0, ddof=1), 1, atol=1e-10)


def test_variance_ratio_sums_to_one():
    rng = np.random.default_rng(1)
    data = rng.normal(0, 1, (300, 5))
    result = pca(data)
    assert abs(result["explained_variance_ratio"].sum() - 1.0) < 1e-10


def test_variance_ratios_descending():
    rng = np.random.default_rng(2)
    data = rng.normal(0, 1, (300, 5))
    evr = pca(data)["explained_variance_ratio"]
    assert np.all(np.diff(evr) <= 1e-12)


def test_dominant_factor_captured():
    rng = np.random.default_rng(3)
    # One strong common factor across all columns → PC1 should dominate.
    factor = rng.normal(0, 1, 400)
    data = np.column_stack([factor + rng.normal(0, 0.1, 400) for _ in range(4)])
    evr = pca(data)["explained_variance_ratio"]
    assert evr[0] > 0.9


def test_full_reconstruction_is_exact():
    rng = np.random.default_rng(4)
    data = rng.normal(0, 1, (150, 3))
    result = pca(data)
    recon = reconstruct(result["scores"], result["components"], data.mean(axis=0))
    assert np.allclose(recon, data, atol=1e-8)


def test_cumulative_variance_monotonic():
    rng = np.random.default_rng(5)
    evr = pca(rng.normal(0, 1, (200, 4)))["explained_variance_ratio"]
    cum = cumulative_variance(evr)
    assert np.all(np.diff(cum) >= -1e-12)
    assert abs(cum[-1] - 1.0) < 1e-10
