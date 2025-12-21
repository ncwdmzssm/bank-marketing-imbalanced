"""Sweep density_alpha to study density weight strength sensitivity."""

from __future__ import annotations

# Allow running as a script: python src/effect/run_density_alpha_sweep.py
if __name__ == "__main__" and __package__ is None:  # pragma: no cover
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "effect"

import os
from dataclasses import dataclass, asdict
from typing import Dict, List

import numpy as np
import pandas as pd

from .eval import evaluate_proba_threshold
from .io import load_processed
from .models import train_rf
from .thresholding import select_threshold
from .weighting import density_weights_knn


@dataclass
class DensityAlphaSweepConfig:
    processed_dir: str = "data/processed"
    outputs_dir: str = "outputs"
    random_state: int = 42
    n_estimators: int = 200
    max_depth: int | None = None
    n_jobs: int = -1
    # Fixed settings for this sweep
    pos_class_weight: float = 4.0
    knn_k: int = 5
    # Threshold tuning settings (validation only)
    tune_threshold: bool = True
    threshold_objective: str = "f1"
    t_min: float = 0.05
    t_max: float = 0.95
    t_step: float = 0.01
    # Sweep values for density_alpha
    density_alphas: List[float] = None

    def __post_init__(self) -> None:
        if self.density_alphas is None:
            self.density_alphas = [2.0, 5.0, 8.0]


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def run_density_alpha_sweep(cfg: DensityAlphaSweepConfig) -> pd.DataFrame:
    X_train, X_val, X_test, y_train, y_val, y_test = load_processed(cfg.processed_dir)

    rows: List[Dict[str, float | str]] = []
    class_weight = {0: 1.0, 1: cfg.pos_class_weight}

    for idx, alpha in enumerate(cfg.density_alphas, start=1):
        exp_id = f"DA-{idx}"

        sw = density_weights_knn(
            X_train=X_train,
            y_train=y_train,
            k=cfg.knn_k,
            alpha=alpha,
            pos_only=True,
            w_max=None,
            include_self_fix=True,
        )

        model, _ = train_rf(
            X_train,
            y_train,
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            random_state=cfg.random_state,
            n_jobs=cfg.n_jobs,
            class_weight=class_weight,
            sample_weight=sw,
        )

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

        metrics, _ = evaluate_proba_threshold(model, X_test, y_test, best_threshold)

        rows.append(
            {
                "Exp ID": exp_id,
                "pos_class_weight": cfg.pos_class_weight,
                "k": cfg.knn_k,
                "density_alpha (β)": alpha,
                "tune_threshold": "Yes" if cfg.tune_threshold else "No",
                "objective": cfg.threshold_objective,
                "best_threshold (val)": best_threshold,
                "Recall (test)": metrics["recall"],
                "F1 (test)": metrics["f1"],
                "G-mean (test)": metrics["gmean"],
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    cfg = DensityAlphaSweepConfig()
    df = run_density_alpha_sweep(cfg)
    _ensure_dir(cfg.outputs_dir)
    out_path = os.path.join(cfg.outputs_dir, "results_density_alpha_sweep.csv")
    df.to_csv(out_path, index=False, encoding="utf-8")

    print("Config:", asdict(cfg))
    print("\nDensity alpha sweep results:")
    print(df)


if __name__ == "__main__":
    main()

