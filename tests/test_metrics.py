import numpy as np

from src.evaluate import compute_metrics

NAMES = ["a", "b", "c"]


def test_perfect_predictions() -> None:
    y = np.array([0, 1, 2, 0, 1, 2])
    m = compute_metrics(y, y, NAMES)
    assert m["accuracy"] == 1.0
    assert m["kappa"] == 1.0
    assert m["macro_f1"] == 1.0
    assert m["per_class_f1"] == {"a": 1.0, "b": 1.0, "c": 1.0}


def test_known_confusion_matrix() -> None:
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 0])
    m = compute_metrics(y_true, y_pred, NAMES)
    assert m["confusion_matrix"] == [[1, 1, 0], [0, 2, 0], [1, 0, 1]]
    assert abs(m["accuracy"] - 4 / 6) < 1e-9
