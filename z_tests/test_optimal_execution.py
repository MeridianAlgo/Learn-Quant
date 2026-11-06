import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "Strategies - Optimal Execution"))
from optimal_execution import (
    ExecutionProblem,
    cost_variance,
    efficient_frontier,
    expected_cost,
    optimal_trajectory,
    trade_list,
    twap_trajectory,
    urgency,
)


@pytest.fixture
def problem():
    return ExecutionProblem(
        shares=1_000_000,
        horizon=5,
        periods=10,
        sigma=0.95,
        eta=2.5e-6,
        gamma=2.5e-7,
        epsilon=0.0625,
    )


def test_trajectory_endpoints(problem):
    x = optimal_trajectory(problem, 1e-6)
    assert x[0] == pytest.approx(1_000_000)
    assert x[-1] == pytest.approx(0.0, abs=1e-6)
    assert len(x) == 11


def test_trajectory_monotone_decreasing(problem):
    x = optimal_trajectory(problem, 1e-6)
    assert np.all(np.diff(x) < 0)


def test_zero_risk_aversion_is_twap(problem):
    x = twap_trajectory(problem)
    assert np.allclose(trade_list(x), 100_000)


def test_tiny_risk_aversion_close_to_twap(problem):
    assert np.allclose(optimal_trajectory(problem, 1e-14), twap_trajectory(problem), rtol=1e-4)


def test_more_risk_aversion_front_loads(problem):
    low = optimal_trajectory(problem, 1e-7)
    high = optimal_trajectory(problem, 1e-5)
    assert high[1] < low[1]
    assert trade_list(high)[0] > trade_list(low)[0]


def test_trades_sum_to_position(problem):
    assert trade_list(optimal_trajectory(problem, 1e-6)).sum() == pytest.approx(1_000_000)


def test_urgency_zero_and_increasing(problem):
    assert urgency(problem, 0.0) == 0.0
    assert urgency(problem, 1e-5) > urgency(problem, 1e-6) > 0


def test_urgency_satisfies_defining_equation(problem):
    lam = 1e-6
    k = urgency(problem, lam)
    kt2 = lam * problem.sigma**2 / problem.eta_tilde
    assert math.cosh(k * problem.tau) == pytest.approx(1 + 0.5 * kt2 * problem.tau**2)


def test_urgency_rejects_negative(problem):
    with pytest.raises(ValueError):
        urgency(problem, -1.0)


def test_twap_cost_matches_formula(problem):
    x = twap_trajectory(problem)
    n = 100_000
    expected = 0.5 * problem.gamma * 1e12 + problem.epsilon * 1e6 + problem.eta_tilde / problem.tau * 10 * n * n
    assert expected_cost(problem, x) == pytest.approx(expected)


def test_variance_formula(problem):
    x = twap_trajectory(problem)
    assert cost_variance(problem, x) == pytest.approx(problem.sigma**2 * problem.tau * np.sum(x[1:] ** 2))


def test_twap_minimises_expected_cost(problem):
    twap = expected_cost(problem, twap_trajectory(problem))
    for lam in (1e-7, 1e-6, 1e-5):
        assert expected_cost(problem, optimal_trajectory(problem, lam)) > twap


def test_frontier_trades_cost_for_risk(problem):
    frontier = efficient_frontier(problem, [0.0, 1e-7, 1e-6, 1e-5])
    costs = [c for _, c, _ in frontier]
    stds = [s for _, _, s in frontier]
    assert costs == sorted(costs)
    assert stds == sorted(stds, reverse=True)


def test_optimal_beats_twap_on_its_own_objective(problem):
    lam = 1e-6

    def objective(x):
        return expected_cost(problem, x) + lam * cost_variance(problem, x)

    assert objective(optimal_trajectory(problem, lam)) < objective(twap_trajectory(problem))


def test_single_period_sells_everything():
    p = ExecutionProblem(shares=500, horizon=1, periods=1, sigma=1.0, eta=0.01)
    assert np.allclose(optimal_trajectory(p, 1.0), [500, 0])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"shares": 0},
        {"horizon": 0},
        {"periods": 0},
        {"eta": 0},
        {"sigma": -1},
        {"gamma": 1.0},
    ],
)
def test_problem_validation(kwargs):
    base = {"shares": 1000, "horizon": 1, "periods": 4, "sigma": 1.0, "eta": 0.01}
    base.update(kwargs)
    with pytest.raises(ValueError):
        ExecutionProblem(**base)


def test_trade_list_validation():
    with pytest.raises(ValueError):
        trade_list([1.0])
