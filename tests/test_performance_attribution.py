import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "UTILS - Finance - Performance Attribution"))
from performance_attribution import (
    brinson_attribution,
    information_ratio,
    tracking_error,
    two_factor_brinson,
)


def test_brinson_active_decomposition_sums():
    wp = [0.4, 0.2, 0.1, 0.3]
    rp = [0.12, 0.05, -0.03, 0.08]
    wb = [0.3, 0.25, 0.20, 0.25]
    rb = [0.10, 0.06, -0.02, 0.07]
    res = brinson_attribution(wp, rp, wb, rb)
    total = res["total_allocation"] + res["total_selection"] + res["total_interaction"]
    assert abs(total - res["active_return"]) < 1e-10


def test_brinson_zero_active_when_identical():
    wp = wb = [0.5, 0.5]
    rp = rb = [0.10, 0.05]
    res = brinson_attribution(wp, rp, wb, rb)
    assert abs(res["active_return"]) < 1e-12
    assert abs(res["total_allocation"]) < 1e-12
    assert abs(res["total_selection"]) < 1e-12


def test_brinson_length_mismatch():
    with pytest.raises(ValueError):
        brinson_attribution([0.5, 0.5], [0.1], [0.5, 0.5], [0.1, 0.1])


def test_two_factor_brinson_active_match():
    wp = [0.4, 0.6]
    rp = [0.10, 0.05]
    wb = [0.5, 0.5]
    rb = [0.08, 0.06]
    res = two_factor_brinson(wp, rp, wb, rb)
    total = res["total_allocation"] + res["total_selection"]
    assert abs(total - res["active_return"]) < 1e-10


def test_information_ratio_positive():
    np.random.seed(0)
    bench = np.random.normal(0.0005, 0.01, 252)
    port = bench + 0.0003
    ir = information_ratio(port, bench)
    assert ir > 0


def test_information_ratio_zero_te():
    bench = np.array([0.01, 0.02, 0.03])
    ir = information_ratio(bench, bench)
    assert ir == 0.0


def test_tracking_error_positive():
    np.random.seed(1)
    rp = np.random.normal(0, 0.012, 252)
    rb = np.random.normal(0, 0.010, 252)
    te = tracking_error(rp, rb)
    assert te > 0
