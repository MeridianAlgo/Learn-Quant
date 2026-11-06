import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "Portfolio Management - Hierarchical Risk Parity"))
from hrp import (
    _simulate_blocks,
    cluster_variance,
    correlation_distance,
    hrp_weights,
    inverse_variance_weights,
    quasi_diagonal_order,
    recursive_bisection,
    single_linkage,
)


def test_distance_is_zero_on_diagonal_and_symmetric():
    corr = np.array([[1.0, 0.3, -0.2], [0.3, 1.0, 0.5], [-0.2, 0.5, 1.0]])
    d = correlation_distance(corr)
    assert np.allclose(np.diag(d), 0.0)
    assert np.allclose(d, d.T)


def test_distance_endpoints():
    d = correlation_distance(np.array([[1.0, -1.0], [-1.0, 1.0]]))
    assert d[0, 1] == pytest.approx(1.0)
    d = correlation_distance(np.array([[1.0, 1.0], [1.0, 1.0]]))
    assert d[0, 1] == pytest.approx(0.0)


def test_distance_rejects_non_square():
    with pytest.raises(ValueError):
        correlation_distance(np.ones((2, 3)))


def test_linkage_shape_and_final_size():
    dist = correlation_distance(np.corrcoef(_simulate_blocks(), rowvar=False))
    link = single_linkage(dist)
    assert link.shape == (6, 4)
    assert link[-1, 3] == 7


def test_linkage_distances_never_decrease():
    dist = correlation_distance(np.corrcoef(_simulate_blocks(), rowvar=False))
    link = single_linkage(dist)
    assert np.all(np.diff(link[:, 2]) >= -1e-12)


def test_linkage_merges_closest_pair_first():
    dist = np.array([[0.0, 0.1, 0.9], [0.1, 0.0, 0.8], [0.9, 0.8, 0.0]])
    link = single_linkage(dist)
    assert link[0, :2].tolist() == [0.0, 1.0]
    assert link[0, 2] == pytest.approx(0.1)
    # Single linkage: gap from {0,1} to 2 is the closer of 0.9 and 0.8.
    assert link[1, 2] == pytest.approx(0.8)


def test_linkage_validation():
    with pytest.raises(ValueError):
        single_linkage(np.zeros((1, 1)))
    with pytest.raises(ValueError):
        single_linkage(np.array([[0.0, 0.1], [0.2, 0.0]]))


def test_order_is_a_permutation():
    dist = correlation_distance(np.corrcoef(_simulate_blocks(), rowvar=False))
    order = quasi_diagonal_order(single_linkage(dist))
    assert sorted(order) == list(range(7))


def test_order_keeps_blocks_together():
    dist = correlation_distance(np.corrcoef(_simulate_blocks(), rowvar=False))
    order = quasi_diagonal_order(single_linkage(dist))
    pos = {asset: i for i, asset in enumerate(order)}
    block_a = sorted(pos[i] for i in (0, 1, 2))
    block_b = sorted(pos[i] for i in (3, 4, 5))
    assert block_a[-1] - block_a[0] == 2
    assert block_b[-1] - block_b[0] == 2


def test_inverse_variance_weights():
    w = inverse_variance_weights(np.diag([1.0, 4.0]))
    assert np.allclose(w, [0.8, 0.2])


def test_inverse_variance_rejects_zero_variance():
    with pytest.raises(ValueError):
        inverse_variance_weights(np.diag([1.0, 0.0]))


def test_cluster_variance_single_asset_is_its_variance():
    cov = np.diag([0.04, 0.09])
    assert cluster_variance(cov, [1]) == pytest.approx(0.09)


def test_bisection_equal_risk_gives_equal_weights():
    cov = np.eye(4) * 0.02
    w = recursive_bisection(cov, [0, 1, 2, 3])
    assert np.allclose(w, 0.25)


def test_bisection_odd_count_equal_risk():
    w = recursive_bisection(np.eye(3), [0, 1, 2])
    assert np.allclose(w, 1 / 3)


def test_hrp_weights_sum_to_one_and_positive():
    w = hrp_weights(_simulate_blocks())
    assert w.sum() == pytest.approx(1.0)
    assert np.all(w > 0)


def test_hrp_favours_low_vol_asset():
    w = hrp_weights(_simulate_blocks())
    assert np.argmax(w) == 6


def test_hrp_is_stable_to_small_sample_changes():
    rets = _simulate_blocks()
    assert np.abs(hrp_weights(rets) - hrp_weights(rets[25:])).max() < 0.02


def test_hrp_validation():
    with pytest.raises(ValueError):
        hrp_weights(np.ones((10, 1)))
    with pytest.raises(ValueError):
        hrp_weights(np.ones((2, 3)))
