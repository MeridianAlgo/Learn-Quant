import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "Machine Learning - Support Vector Machine"))
from support_vector_machine import LinearSVM, accuracy, hinge_loss, standardize


def _separable(n=60, seed=0):
    rng = np.random.default_rng(seed)
    positive = rng.normal([2.5, 2.5], 0.4, size=(n, 2))
    negative = rng.normal([-2.5, -2.5], 0.4, size=(n, 2))
    features = np.vstack([positive, negative]).tolist()
    labels = [1] * n + [-1] * n
    return features, labels


def _overlapping(n=60, seed=1):
    rng = np.random.default_rng(seed)
    positive = rng.normal([0.8, 0.6], 1.0, size=(n, 2))
    negative = rng.normal([-0.8, -0.6], 1.0, size=(n, 2))
    features = np.vstack([positive, negative]).tolist()
    labels = [1] * n + [-1] * n
    return features, labels


def test_standardize_centers_and_scales():
    scaled, _, _ = standardize([[1.0, 10.0], [3.0, 30.0], [5.0, 50.0]])
    data = np.asarray(scaled)
    assert np.allclose(data.mean(axis=0), 0.0, atol=1e-12)
    assert np.allclose(data.std(axis=0), 1.0, atol=1e-12)


def test_standardize_returns_reusable_statistics():
    _, mean, std = standardize([[1.0], [3.0], [5.0]])
    assert mean == [3.0]
    assert abs(std[0] - np.std([1.0, 3.0, 5.0])) < 1e-12


def test_standardize_survives_a_constant_column():
    scaled, _, std = standardize([[1.0, 7.0], [2.0, 7.0], [3.0, 7.0]])
    assert std[1] == 1.0
    assert all(row[1] == 0.0 for row in scaled)


def test_standardize_rejects_a_flat_list():
    with pytest.raises(ValueError):
        standardize([1.0, 2.0, 3.0])


def test_hinge_loss_is_zero_beyond_the_margin():
    assert hinge_loss([2.0, -3.0], [1, -1]) == 0.0


def test_hinge_loss_charges_points_inside_the_margin():
    assert abs(hinge_loss([0.4], [1]) - 0.6) < 1e-12


def test_hinge_loss_charges_a_point_exactly_on_the_boundary():
    assert abs(hinge_loss([0.0], [1]) - 1.0) < 1e-12


def test_hinge_loss_grows_with_the_error():
    assert hinge_loss([-2.0], [1]) > hinge_loss([-1.0], [1])


def test_hinge_loss_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        hinge_loss([1.0, 2.0], [1])


def test_hinge_loss_rejects_no_data():
    with pytest.raises(ValueError):
        hinge_loss([], [])


def test_fit_separates_clean_classes():
    features, labels = _separable()
    scaled, _, _ = standardize(features)
    model = LinearSVM(learning_rate=0.05, lam=0.01, epochs=1500).fit(scaled, labels)
    assert accuracy(model.predict(scaled), labels) == 1.0


def test_fit_accepts_zero_one_labels():
    features, labels = _separable()
    scaled, _, _ = standardize(features)
    zero_one = [1 if v == 1 else 0 for v in labels]
    model = LinearSVM(learning_rate=0.05, lam=0.01, epochs=1500).fit(scaled, zero_one)
    assert accuracy(model.predict(scaled), zero_one) == 1.0


def test_fit_returns_itself_for_chaining():
    features, labels = _separable()
    model = LinearSVM(epochs=10)
    assert model.fit(features, labels) is model


def test_loss_falls_over_training():
    features, labels = _separable()
    scaled, _, _ = standardize(features)
    model = LinearSVM(learning_rate=0.05, lam=0.01, epochs=800).fit(scaled, labels)
    assert model.loss_history[-1] < model.loss_history[0]
    assert len(model.loss_history) == 800


def test_predictions_are_only_plus_or_minus_one():
    features, labels = _separable()
    model = LinearSVM(epochs=200).fit(features, labels)
    assert set(model.predict(features)) <= {-1, 1}


def test_decision_function_sign_matches_prediction():
    features, labels = _separable()
    model = LinearSVM(epochs=300).fit(features, labels)
    scores = model.decision_function(features)
    predictions = model.predict(features)
    assert all((s >= 0) == (p == 1) for s, p in zip(scores, predictions))


def test_margin_width_is_positive():
    features, labels = _separable()
    scaled, _, _ = standardize(features)
    model = LinearSVM(learning_rate=0.05, lam=0.01, epochs=1000).fit(scaled, labels)
    assert model.margin_width() > 0


def test_more_regularization_widens_the_margin():
    features, labels = _separable()
    scaled, _, _ = standardize(features)
    widths = [
        LinearSVM(learning_rate=0.05, lam=lam, epochs=1000).fit(scaled, labels).margin_width()
        for lam in (0.001, 0.01, 0.1, 1.0)
    ]
    assert widths == sorted(widths)


def test_support_vectors_are_a_minority_on_overlapping_data():
    features, labels = _overlapping()
    scaled, _, _ = standardize(features)
    model = LinearSVM(learning_rate=0.05, lam=0.01, epochs=1500).fit(scaled, labels)
    support = model.support_vectors(scaled, labels)
    assert 0 < len(support) < len(labels)


def test_support_vectors_grow_with_regularization():
    features, labels = _overlapping()
    scaled, _, _ = standardize(features)
    counts = [
        len(LinearSVM(learning_rate=0.05, lam=lam, epochs=1500).fit(scaled, labels).support_vectors(scaled, labels))
        for lam in (0.001, 0.1, 1.0)
    ]
    assert counts == sorted(counts)


def test_a_widely_separated_set_has_no_support_vectors_inside_the_margin():
    features, labels = _separable()
    scaled, _, _ = standardize(features)
    model = LinearSVM(learning_rate=0.05, lam=0.001, epochs=1500).fit(scaled, labels)
    assert model.support_vectors(scaled, labels) == []


def test_unfitted_model_refuses_to_predict():
    with pytest.raises(ValueError):
        LinearSVM().predict([[1.0, 2.0]])


def test_unfitted_model_has_no_margin():
    with pytest.raises(ValueError):
        LinearSVM().margin_width()


def test_wrong_column_count_is_rejected():
    features, labels = _separable()
    model = LinearSVM(epochs=50).fit(features, labels)
    with pytest.raises(ValueError):
        model.decision_function([[1.0, 2.0, 3.0]])


def test_labels_must_use_a_known_encoding():
    features, labels = _separable()
    with pytest.raises(ValueError):
        LinearSVM(epochs=10).fit(features, [2] * len(labels))


def test_one_label_per_row_is_required():
    features, labels = _separable()
    with pytest.raises(ValueError):
        LinearSVM(epochs=10).fit(features, labels[:-1])


def test_diverging_hyperparameters_are_refused():
    with pytest.raises(ValueError):
        LinearSVM(learning_rate=0.05, lam=100.0)


def test_constructor_validates_its_inputs():
    with pytest.raises(ValueError):
        LinearSVM(learning_rate=0)
    with pytest.raises(ValueError):
        LinearSVM(lam=-1)
    with pytest.raises(ValueError):
        LinearSVM(epochs=0)


def test_accuracy_handles_both_encodings():
    assert accuracy([1, -1, 1], [1, 0, 1]) == 1.0
    assert accuracy([1, -1, 1], [1, -1, 1]) == 1.0


def test_accuracy_counts_mistakes():
    assert abs(accuracy([1, 1, -1, -1], [1, -1, -1, -1]) - 0.75) < 1e-12


def test_accuracy_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        accuracy([1, 1], [1])
