"""Multi-seed evaluation with threshold tuning and PR curve exports."""

from __future__ import annotations

# Allow running as a script: python src/effect/run_multi_seed.py
if __name__ == "__main__" and (__package__ is None or __package__ == ""):  # pragma: no cover
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "effect"

import os
from dataclasses import dataclass, asdict
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from .eval import evaluate_proba_threshold
from .io import load_processed
from .models import train_rf
from .thresholding import select_threshold
from .weighting import density_weights_knn


@dataclass
class MultiSeedConfig:
    processed_dir: str = "data/processed"
    outputs_dir: str = "outputs"
    seeds: List[int] = None  # seeds for model randomness; data split is fixed by frozen .npy
    n_estimators: int = 200
    max_depth: int | None = None
    n_jobs: int = -1
    pos_class_weight: float = 4.0
    knn_k: int = 5
    density_alpha: float = 5.0
    # Threshold tuning settings (validation only, never test to avoid leakage)
    tune_threshold: bool = True
    threshold_objective: str = "f1"
    t_min: float = 0.05
    t_max: float = 0.95
    t_step: float = 0.01

    def __post_init__(self) -> None:
        if self.seeds is None:
            self.seeds = [0, 1, 2, 3, 4,42]


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _pr_curve_to_csv(recalls: np.ndarray, precisions: np.ndarray, thresholds: np.ndarray, path: str) -> None:
    """Save PR curve points to CSV for plotting."""
    df = pd.DataFrame(
        {
            "threshold": np.concatenate([thresholds, [thresholds[-1] if thresholds.size else 0.0]]),  # align lengths
            "recall": recalls,
            "precision": precisions,
        }
    )
    df.to_csv(path, index=False, encoding="utf-8")


def run_multi_seed(cfg: MultiSeedConfig) -> pd.DataFrame:
    X_train, X_val, X_test, y_train, y_val, y_test = load_processed(cfg.processed_dir)

    # Precompute density-based sample weights once.
    sw = density_weights_knn(
        X_train=X_train,
        y_train=y_train,
        k=cfg.knn_k,
        alpha=cfg.density_alpha,
        pos_only=True,
        w_max=None,
        include_self_fix=True,
    )
    class_weight = {0: 1.0, 1: cfg.pos_class_weight}

    rows: List[Dict[str, float | str]] = []

    pr_dir = os.path.join(cfg.outputs_dir, "pr_curves")
    _ensure_dir(pr_dir)

    for seed in cfg.seeds:
        model, _ = train_rf(
            X_train,
            y_train,
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            random_state=seed,
            n_jobs=cfg.n_jobs,
            class_weight=class_weight,
            sample_weight=sw,
        )

        # Tune threshold on validation set only.
        best_threshold = 0.5
        if cfg.tune_threshold:
            proba_val = model.predict_proba(X_val)[:, 1]
            tune_result = select_threshold(
                y_true=y_val,
                proba_pos=proba_val,
                objective=cfg.threshold_objective,
                t_min=cfg.t_min,
                t_max=cfg.t_max,
                step=cfg.t_step,
            )
            best_threshold = float(tune_result["best_threshold"])

        proba_test = model.predict_proba(X_test)[:, 1]
        metrics, _ = evaluate_proba_threshold(model, X_test, y_test, best_threshold)

        # Save PR curve on test set for analysis (no tuning).
        precisions, recalls, pr_thresholds = precision_recall_curve(y_test, proba_test, pos_label=1)
        pr_path = os.path.join(pr_dir, f"pr_curve_seed{seed}.csv")
        _pr_curve_to_csv(recalls, precisions, pr_thresholds, pr_path)

        rows.append(
            {
                "seed": seed,
                "pos_class_weight": cfg.pos_class_weight,
                "k": cfg.knn_k,
                "density_alpha": cfg.density_alpha,
                "threshold_objective": cfg.threshold_objective,
                "best_threshold (val)": best_threshold,
                "Recall (test)": metrics["recall"],
                "F1 (test)": metrics["f1"],
                "G-mean (test)": metrics["gmean"],
            }
        )

    df = pd.DataFrame(rows)
    # Aggregate summary row (mean across seeds).
    summary = {
        "seed": "mean",
        "pos_class_weight": cfg.pos_class_weight,
        "k": cfg.knn_k,
        "density_alpha": cfg.density_alpha,
        "threshold_objective": cfg.threshold_objective,
        "best_threshold (val)": df["best_threshold (val)"].mean(),
        "Recall (test)": df["Recall (test)"].mean(),
        "F1 (test)": df["F1 (test)"].mean(),
        "G-mean (test)": df["G-mean (test)"].mean(),
    }
    df = pd.concat([df, pd.DataFrame([summary])], ignore_index=True)
    return df


def main() -> None:
    cfg = MultiSeedConfig()
    df = run_multi_seed(cfg)

    _ensure_dir(cfg.outputs_dir)
    out_path = os.path.join(cfg.outputs_dir, "results_multi_seed.csv")
    df.to_csv(out_path, index=False, encoding="utf-8")

    print("Config:", asdict(cfg))
    print("\nMulti-seed results (per seed + mean):")
    print(df)
    print("\nPR curves saved under:", os.path.join(cfg.outputs_dir, "pr_curves"))


if __name__ == "__main__":
    main()
