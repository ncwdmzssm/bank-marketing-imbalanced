"""Sweep pos_class_weight to study cost-sensitive strength (alpha_cw)."""

from __future__ import annotations

# Allow running as a script: python src/effect/run_cw_sweep.py
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
class CWSweepConfig:
    processed_dir: str = "data/processed"
    outputs_dir: str = "outputs"
    random_state: int = 42
    n_estimators: int = 200
    max_depth: int | None = None
    n_jobs: int = -1
    # Fixed density settings
    knn_k: int = 5
    density_alpha: float = 5.0
    use_density: bool = True
    # Threshold tuning settings (validation only, never test to avoid leakage)
    tune_threshold: bool = True
    threshold_objective: str = "f1"
    t_min: float = 0.05
    t_max: float = 0.95
    t_step: float = 0.01
    # Sweep values for pos_class_weight (alpha_cw)
    pos_class_weights: List[float] = None

    def __post_init__(self) -> None:
        if self.pos_class_weights is None:
            self.pos_class_weights = [2.0, 4.0, 6.0,8.0,10.0]


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def run_cw_sweep(cfg: CWSweepConfig) -> pd.DataFrame:
    X_train, X_val, X_test, y_train, y_val, y_test = load_processed(cfg.processed_dir)

    # Precompute density-based sample weights once (fixed k, alpha).
    sw = None
    if cfg.use_density:
        sw = density_weights_knn(
            X_train=X_train,
            y_train=y_train,
            k=cfg.knn_k,
            alpha=cfg.density_alpha,
            pos_only=True,
            w_max=None,
            include_self_fix=True,
        )

    rows: List[Dict[str, float | str]] = []

    for idx, cw_val in enumerate(cfg.pos_class_weights, start=1):
        exp_id = f"CW-{idx}"
        class_weight = {0: 1.0, 1: cw_val}

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

        # Tune threshold on validation ONLY.
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
                "pos_class_weight (α)": cw_val,
                "k": cfg.knn_k,
                "density_alpha (β)": cfg.density_alpha,
                "tune_threshold": "Yes" if cfg.tune_threshold else "No",
                "objective": cfg.threshold_objective,
                "best_threshold (val)": best_threshold,
                "Recall (test)": metrics["recall"],
                "F1 (test)": metrics["f1"],
                "G-mean (test)": metrics["gmean"],
            }
        )

    df = pd.DataFrame(rows)
    return df


def main() -> None:
    cfg = CWSweepConfig()
    df = run_cw_sweep(cfg)

    _ensure_dir(cfg.outputs_dir)
    out_path = os.path.join(cfg.outputs_dir, "results_cw_sweep.csv")
    df.to_csv(out_path, index=False, encoding="utf-8")

    print("Config:", asdict(cfg))
    print("\nCW sweep results:")
    print(df)


if __name__ == "__main__":
    main()

