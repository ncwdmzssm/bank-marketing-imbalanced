"""Model training helpers for effect package."""

from __future__ import annotations

import time
from typing import Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier


def train_rf(
    X: np.ndarray,
    y: np.ndarray,
    n_estimators: int = 200,
    max_depth: Optional[int] = None,
    random_state: Optional[int] = 42,
    n_jobs: int = -1,
    class_weight: Optional[dict] = None,
    sample_weight: Optional[np.ndarray] = None,
) -> Tuple[RandomForestClassifier, float]:
    """Train a RandomForestClassifier and return the model with elapsed seconds."""
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=n_jobs,
        class_weight=class_weight,
    )
    start = time.perf_counter()
    if sample_weight is not None:
        model.fit(X, y, sample_weight=sample_weight)
    else:
        model.fit(X, y)
    elapsed = time.perf_counter() - start
    return model, float(elapsed)

