"""Threshold selection and prediction helpers for imbalanced binary tasks."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import f1_score, recall_score

try:
    import pandas as pd
except Exception:  # pragma: no cover - pandas is optional
    pd = None


def predict_with_threshold(proba_pos: np.ndarray, threshold: float) -> np.ndarray:
    """Convert positive-class probabilities to 0/1 labels using a threshold."""
    probs = np.asarray(proba_pos).reshape(-1)
    return (probs >= threshold).astype(int)


def select_threshold(
    y_true: np.ndarray,
    proba_pos: np.ndarray,
    objective: str = "f1",
    t_min: float = 0.05,
    t_max: float = 0.95,
    step: float = 0.01,
) -> Dict[str, object]:
    """
    Grid search a decision threshold on validation data (never on the test set!).

    Parameters
    ----------
    objective : {"f1", "recall"}
        Metric to maximize when choosing the threshold.
    t_min, t_max, step : float
        Search range and stride for thresholds in [t_min, t_max].
    """
    if objective not in {"f1", "recall"}:
        raise ValueError("objective 只支持 {'f1', 'recall'}")
    if step <= 0:
        raise ValueError("step 必须为正数")

    y_true = np.asarray(y_true).reshape(-1)
    proba_pos = np.asarray(proba_pos).reshape(-1)
    if y_true.shape[0] != proba_pos.shape[0]:
        raise ValueError("y_true 与 proba_pos 长度不一致")

    thresholds = np.arange(t_min, t_max + 1e-12, step)
    # Edge case: constant probabilities -> predictions identical across thresholds.
    constant_probs = np.allclose(proba_pos.min(), proba_pos.max())

    history: List[Tuple[float, float]] = []
    best_threshold = None
    best_score = -np.inf

    for t in thresholds:
        y_pred = predict_with_threshold(proba_pos, t) if not constant_probs else np.zeros_like(y_true)
        if objective == "f1":
            score = f1_score(y_true, y_pred, pos_label=1)
        else:
            score = recall_score(y_true, y_pred, pos_label=1)
        history.append((t, float(score)))
        if score > best_score:
            best_score = score
            best_threshold = t

    # In degenerate cases where all scores are equal (e.g., constant probs), pick midpoint.
    if best_threshold is None:
        best_threshold = float((t_min + t_max) / 2.0)
        best_score = float(history[0][1]) if history else 0.0

    result: Dict[str, object] = {
        "best_threshold": float(best_threshold),
        "best_score": float(best_score),
        "objective": objective,
    }

    # Provide history for plotting/inspection if desired.
    if pd is not None:
        result["history"] = pd.DataFrame(history, columns=["threshold", objective])
    else:
        result["history"] = history  # list of (threshold, score)

    return result

