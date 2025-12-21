"""Binary classification evaluation utilities."""

from __future__ import annotations

import math
import time
from typing import Any, Dict, Tuple

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, recall_score


def g_mean(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return sqrt(TPR * TNR) for the binary confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    # TPR = TP/(TP+FN), TNR = TN/(TN+FP); g-mean = sqrt(TPR * TNR)
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    return math.sqrt(tpr * tnr)


def _collect_metrics(y_true: np.ndarray, y_pred: np.ndarray, elapsed: float) -> Dict[str, float]:
    recall_pos = float(recall_score(y_true, y_pred, pos_label=1))
    f1_pos = float(f1_score(y_true, y_pred, pos_label=1))
    gmean = float(g_mean(y_true, y_pred))
    return {
        "recall": recall_pos,
        "f1": f1_pos,
        "gmean": gmean,
        "pred_time_s": float(elapsed),
    }


def evaluate_predict(model: Any, X: np.ndarray, y: np.ndarray) -> Tuple[Dict[str, float], np.ndarray]:
    """Evaluate using model.predict(X); returns (metrics_dict, y_pred)."""
    start = time.perf_counter()
    y_pred = np.asarray(model.predict(X))
    elapsed = time.perf_counter() - start
    metrics = _collect_metrics(y, y_pred, elapsed)
    return metrics, y_pred


def evaluate_proba_threshold(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    threshold: float,
) -> Tuple[Dict[str, float], np.ndarray]:
    """
    Evaluate using model.predict_proba(X)[:, 1] with a chosen threshold.

    Note: threshold must be selected on validation data, never on the held-out test set,
    to avoid leakage in imbalanced classification reporting.
    """
    start = time.perf_counter()
    proba_pos = np.asarray(model.predict_proba(X))[:, 1]
    y_pred = (proba_pos >= threshold).astype(int)
    elapsed = time.perf_counter() - start  # includes predict_proba + thresholding
    metrics = _collect_metrics(y, y_pred, elapsed)
    return metrics, y_pred


# Backward compatibility alias
def evaluate(model: Any, X: np.ndarray, y: np.ndarray) -> Tuple[Dict[str, float], np.ndarray]:
    """Alias to evaluate_predict for existing callers."""
    return evaluate_predict(model, X, y)

