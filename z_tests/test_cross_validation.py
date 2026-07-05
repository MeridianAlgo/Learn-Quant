import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "Machine Learning - Cross Validation"))
from cross_validation import (
    cross_val_score,
    kfold_indices,
    train_test_split,
    walk_forward_splits,
)


def test_split_sizes():
    x = np.arange(100).reshape(-1, 1)
    y = np.arange(100)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_fraction=0.2, seed=0)
    assert len(x_test) == 20
    assert len(x_train) == 80
    assert len(y_test) == 20


def test_split_is_a_partition():
    x = np.arange(50).reshape(-1, 1)
    y = np.arange(50)
    x_train, x_test, _, _ = train_test_split(x, y, seed=1)
    combined = sorted(np.concatenate([x_train[:, 0], x_test[:, 0]]).tolist())
    assert combined == list(range(50))


def test_unshuffled_split_takes_first_block_as_test():
    x = np.arange(10).reshape(-1, 1)
    y = np.arange(10)
    _, x_test, _, _ = train_test_split(x, y, test_fraction=0.3, shuffle=False)
    assert x_test[:, 0].tolist() == [0, 1, 2]


def test_split_rejects_bad_fraction():
    x = np.arange(10).reshape(-1, 1)
    with pytest.raises(ValueError):
        train_test_split(x, np.arange(10), test_fraction=1.5)


def test_split_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        train_test_split(np.arange(10).reshape(-1, 1), np.arange(9))


def test_kfold_tests_every_row_exactly_once():
    pairs = kfold_indices(23, k=5, seed=0)
    tested = np.concatenate([test for _, test in pairs])
    assert sorted(tested.tolist()) == list(range(23))


def test_kfold_train_and_test_are_disjoint():
    for train_idx, test_idx in kfold_indices(30, k=3, seed=2):
        assert set(train_idx.tolist()).isdisjoint(test_idx.tolist())


def test_kfold_rejects_bad_k():
    with pytest.raises(ValueError):
        kfold_indices(10, k=1)
    with pytest.raises(ValueError):
        kfold_indices(3, k=5)


def test_cross_val_score_returns_k_scores():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, size=(60, 1))
    y = 2 * x[:, 0]

    def mean_model(x_train, y_train, x_test):
        return np.full(len(x_test), y_train.mean())

    scores = cross_val_score(mean_model, x, y, k=4, seed=0)
    assert len(scores) == 4
    assert all(s >= 0 for s in scores)


def test_linear_model_beats_mean_baseline():
    rng = np.random.default_rng(7)
    x = rng.uniform(0, 10, size=(150, 1))
    y = 3 * x[:, 0] + rng.normal(0, 0.5, size=150)

    def linear(x_train, y_train, x_test):
        a = np.column_stack([x_train, np.ones(len(x_train))])
        coef, *_ = np.linalg.lstsq(a, y_train, rcond=None)
        return np.column_stack([x_test, np.ones(len(x_test))]) @ coef

    def mean_model(x_train, y_train, x_test):
        return np.full(len(x_test), y_train.mean())

    assert np.mean(cross_val_score(linear, x, y, seed=0)) < np.mean(cross_val_score(mean_model, x, y, seed=0))


def test_walk_forward_never_tests_the_past():
    for train_idx, test_idx in walk_forward_splits(100, n_splits=4, min_train=20):
        assert train_idx.max() < test_idx.min()


def test_walk_forward_training_window_expands():
    sizes = [len(train) for train, _ in walk_forward_splits(100, n_splits=4, min_train=20)]
    assert sizes == sorted(sizes)
    assert sizes[0] == 20


def test_walk_forward_rejects_impossible_request():
    with pytest.raises(ValueError):
        walk_forward_splits(5, n_splits=10, min_train=3)
