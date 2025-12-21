"""KNN-based density-aware sample weighting for imbalanced binary datasets."""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.neighbors import NearestNeighbors


def density_weights_knn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    k: int = 5,
    alpha: float = 5.0,
    pos_only: bool = True,
    w_max: Optional[float] = None,
    include_self_fix: bool = True,
) -> np.ndarray:
    """
    Compute density-aware weights using k-NN neighborhood composition.

    Weight rule (only applied to selected samples, defaults to positives):
        w_i = 1 + alpha * neg_ratio
    where neg_ratio = (# negative neighbors) / k. Negatives keep weight 1.

    Parameters are exposed so you can discuss optional decision points in review/defense:
      - w_max: optional cap to avoid noisy samples getting extreme weights.
      - include_self_fix: drop the sample itself if it appears in its neighbor list.
      - Edge handling: no positives or too-few samples for requested k.
    """
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)

    n_samples = X_train.shape[0]
    weights = np.ones(n_samples, dtype=float)

    # If only weighting positives and none exist, return all ones (edge case).
    pos_mask = y_train == 1
    if pos_only and not np.any(pos_mask):
        return weights

    # With very small datasets, fall back to the maximum viable neighbor count.
    # include_self_fix=True implies we may query one extra neighbor to drop self.
    max_k = max(0, n_samples - 1) if include_self_fix else max(1, n_samples)
    effective_k = min(k, max_k)

    # Fit k-NN on the entire training set.
    neighbor_count = effective_k + 1 if include_self_fix else effective_k
    neighbor_count = max(1, min(neighbor_count, n_samples))  # stay within valid bounds
    nn = NearestNeighbors(n_neighbors=neighbor_count, algorithm="auto")
    nn.fit(X_train)

    # Decide which indices receive density-based weighting.
    target_indices = np.where(pos_mask)[0] if pos_only else np.arange(n_samples)

    for idx in target_indices:
        # Query neighbors for the current sample.
        distances, indices = nn.kneighbors(X_train[idx].reshape(1, -1), return_distance=True)
        indices = indices[0]

        # Optional decision point: remove self-neighbor if present.
        if include_self_fix and indices.size > 0 and indices[0] == idx:
            indices = indices[1:]

        # Ensure we only keep the top-k neighbors after self-removal.
        indices = indices[:effective_k]
        if indices.size == 0:
            # Edge case: only one sample in the dataset.
            weights[idx] = 1.0
            continue

        neighbor_labels = y_train[indices]
        neg_ratio = float(np.sum(neighbor_labels == 0)) / float(indices.size)

        # Weight formula: w_i = 1 + alpha * neg_ratio
        w_i = 1.0 + alpha * neg_ratio

        # Optional decision point: cap weights to avoid overemphasizing outliers.
        if w_max is not None:
            w_i = min(w_i, w_max)

        weights[idx] = w_i

    return weights
