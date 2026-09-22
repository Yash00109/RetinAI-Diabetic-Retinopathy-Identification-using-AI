import numpy as np

from dr_detection.metrics import (
    binary_referable_metrics,
    calibration_metrics,
    classification_metrics,
    quadratic_weighted_kappa,
)


def test_quadratic_weighted_kappa():
    y_true = [0, 1, 2, 3, 4]
    y_pred = [0, 1, 2, 3, 4]
    qwk = quadratic_weighted_kappa(y_true, y_pred)
    assert np.isclose(qwk, 1.0)


def test_classification_metrics():
    y_true = [0, 1, 2, 3, 4]
    y_pred = [0, 1, 2, 3, 3]
    res = classification_metrics(y_true, y_pred, num_classes=5)
    assert res["accuracy"] == 0.8
    assert "quadratic_weighted_kappa" in res
    assert "macro_f1" in res
    assert "confusion_matrix" in res
    assert "balanced_accuracy" in res
    assert "per_class_recall" in res
    # Class 4 has 0 correct predictions (predicted as 3), so recall should be 0
    assert res["per_class_recall"]["4"] == 0.0
    # Class 0 has perfect recall
    assert res["per_class_recall"]["0"] == 1.0


def test_binary_referable_metrics():
    y_true = np.array([0, 1, 2, 3, 4])
    # Referable DR classes are 2, 3, 4 (so last 3 should be 1)
    mock_probs = np.zeros((5, 5))
    mock_probs[0, 0] = 1.0
    mock_probs[1, 1] = 1.0
    mock_probs[2, 2] = 1.0
    mock_probs[3, 3] = 1.0
    mock_probs[4, 4] = 1.0
    res = binary_referable_metrics(y_true, mock_probs, threshold=0.5)
    assert res["sensitivity"] == 1.0
    assert res["specificity"] == 1.0


def test_calibration_metrics():
    y_true = np.array([0, 1, 2])
    mock_probs = np.eye(3)
    res = calibration_metrics(y_true, mock_probs)
    assert np.isclose(res["brier_score"], 0.0)
    assert np.isclose(res["expected_calibration_error"], 0.0)
