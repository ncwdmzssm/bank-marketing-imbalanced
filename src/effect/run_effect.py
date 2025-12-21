"""Run effect experiments and save aggregated results to CSV."""

from __future__ import annotations

# Allow running as a script: python src/effect/run_effect.py
if __name__ == "__main__" and __package__ is None:  # pragma: no cover
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "effect"

import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .eval import evaluate_predict, evaluate_proba_threshold
from .io import load_processed
from .models import train_rf
from .thresholding import select_threshold
from .weighting import density_weights_knn


@dataclass
class EffectConfig:
    processed_dir: str = "data/processed"
    outputs_dir: str = "outputs"
    random_state: int = 42
    n_estimators: int = 200
    max_depth: int | None = None
    n_jobs: int = -1
    # Decision points exposed for discussion/experiments:
    pos_class_weight: float = 4.0  # cost-sensitive strength alpha_cw
    knn_k: int = 5  # density weighting neighbors
    density_alpha: float = 5.0  # density weighting strength alpha_dw
    use_density: bool = True
    use_class_weight: bool = True  # allow toggling cw / sw / both
    w_max: float | None = None  # cap weights to mitigate noisy outliers
    save_csv: bool = True
    # Threshold tuning on validation set (never on test to avoid leakage).
    tune_threshold: bool = True
    threshold_objective: str = "f1"  # or "recall"
    t_min: float = 0.05
    t_max: float = 0.95
    t_step: float = 0.01


def _ensure_outputs_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _threshold_output_dir(base_outputs: str) -> str:
    """Folder to store threshold search CSVs and plots."""
    return os.path.join(base_outputs, "threshold_search_csv_results")


def _safe_method_slug(method_name: str) -> str:
    """Make method name filesystem friendly for CSV outputs."""
    return (
        method_name.replace(" ", "_")
        .replace("(", "_")
        .replace(")", "")
        .replace("/", "_")
    )


def _history_to_df(history: object, objective: str) -> pd.DataFrame:
    """Normalize threshold search history to a DataFrame for CSV export."""
    if isinstance(history, pd.DataFrame):
        return history
    if isinstance(history, list):
        return pd.DataFrame(history, columns=["threshold", objective])
    return pd.DataFrame(columns=["threshold", objective])


def _tune_threshold_and_save(
    method_name: str,
    model,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: EffectConfig,
) -> float:
    """Tune threshold on validation set only; save search curve CSV."""
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

    history = tune_result.get("history")
    if history is not None:
        target_dir = _threshold_output_dir(cfg.outputs_dir)
        _ensure_outputs_dir(target_dir)
        history_df = _history_to_df(history, cfg.threshold_objective)
        out_path = os.path.join(target_dir, f"threshold_search_{_safe_method_slug(method_name)}.csv")
        # Export UTF-8 CSV for easy Excel import; validation-only to avoid test leakage.
        history_df.to_csv(out_path, index=False, encoding="utf-8")

        # Read back the CSV (explicitly using the frozen artifact) and plot threshold curves.
        reloaded = pd.read_csv(out_path)
        if "threshold" in reloaded.columns and cfg.threshold_objective in reloaded.columns:
            try:
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots()
                ax.plot(
                    reloaded["threshold"],
                    reloaded[cfg.threshold_objective],
                    label=cfg.threshold_objective,
                )
                ax.set_xlabel("threshold")
                ax.set_ylabel(cfg.threshold_objective)
                ax.set_title(f"Threshold search ({method_name})")
                ax.grid(True, linestyle="--", alpha=0.5)
                ax.legend()
                fig.tight_layout()
                png_path = os.path.join(
                    target_dir, f"threshold_search_{_safe_method_slug(method_name)}.png"
                )
                fig.savefig(png_path, dpi=200)
                plt.close(fig)
            except Exception:
                # Best-effort plotting; skip silently if backend not available.
                pass

    return best_threshold


def main() -> None:
    cfg = EffectConfig()

    # Open questions (left for your decision, noted for presentation):
    # 1) Combine class_weight and sample_weight? They target different imbalance aspects; combining can compound effects but risks over-penalizing majority.
    # 2) Should we cap weights with w_max? Helps limit noisy/outlier influence but may blunt gains near decision boundaries.
    # 3) How to pick k and density_alpha? Consider a small grid (e.g., k in {3,5,7}, alpha in {2.5,5,7.5}) validated on hold-out/val.
    # 4) Threshold tuning must use validation only—never the test set—to avoid leakage; tuning thresholds matters in imbalanced tasks because probability calibration and decision costs shift the optimal operating point away from 0.5.

    X_train, X_val, X_test, y_train, y_val, y_test = load_processed(cfg.processed_dir)

    results: List[Dict[str, float | str]] = []

    def run_with_optional_tuning(
        method_name: str,
        class_weight: Dict[int, float] | None,
        sample_weight: np.ndarray | None,
    ) -> None:
        """Train model, optionally tune threshold on val, then evaluate on test."""
        model, train_time_s = train_rf(
            X_train,
            y_train,
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            random_state=cfg.random_state,
            n_jobs=cfg.n_jobs,
            class_weight=class_weight,
            sample_weight=sample_weight,
        )

        threshold = 0.5
        label = method_name
        if cfg.tune_threshold:
            threshold = _tune_threshold_and_save(method_name, model, X_val, y_val, cfg)
            label = f"{method_name}(tuned)"
            # Note: threshold is selected on validation only; test is used purely for final reporting.
            metrics, _ = evaluate_proba_threshold(model, X_test, y_test, threshold)
        else:
            metrics, _ = evaluate_predict(model, X_test, y_test)

        results.append(
            {
                "method": label,
                "train_time_s": float(train_time_s),
                "pred_time_s": float(metrics["pred_time_s"]),
                "recall": float(metrics["recall"]),
                "f1": float(metrics["f1"]),
                "gmean": float(metrics["gmean"]),
                "threshold": float(threshold) if cfg.tune_threshold else 0.5,
            }
        )

    # Baseline: no class_weight, no sample_weight, with fair threshold tuning on validation.
    run_with_optional_tuning(method_name="RF_baseline", class_weight=None, sample_weight=None)

    # Cost-sensitive: controlled by use_class_weight flag.
    cw = {0: 1.0, 1: cfg.pos_class_weight} if cfg.use_class_weight else None
    method_cw = f"RF_cost_sensitive_cw{cfg.pos_class_weight}" if cfg.use_class_weight else "RF_cost_sensitive_disabled"
    run_with_optional_tuning(method_name=method_cw, class_weight=cw, sample_weight=None)

    # Density-aware sample weighting (optionally combined with class_weight).
    sw = None
    if cfg.use_density:
        sw = density_weights_knn(
            X_train=X_train,
            y_train=y_train,
            k=cfg.knn_k,
            alpha=cfg.density_alpha,
            pos_only=True,
            w_max=cfg.w_max,
            include_self_fix=True,
        )
    density_label = f"k{cfg.knn_k}_alpha{cfg.density_alpha}"
    if cfg.use_class_weight and cfg.use_density:
        method_dw = f"RF_cost_sensitive_plus_density({density_label}_cw{cfg.pos_class_weight})"
    elif cfg.use_density:
        method_dw = f"RF_density_only({density_label})"
    else:
        method_dw = "RF_density_disabled"

    # Density-aware variant (or disabled placeholder) with the same validation-only tuning.
    run_with_optional_tuning(
        method_name=method_dw,
        class_weight=cw if cfg.use_class_weight else None,
        sample_weight=sw,
    )

    df = pd.DataFrame(results)
    df = df.sort_values(by=["recall", "f1"], ascending=False, ignore_index=True)

    if cfg.save_csv:
        _ensure_outputs_dir(cfg.outputs_dir)
        output_path = os.path.join(cfg.outputs_dir, "results_effect.csv")
        df.to_csv(output_path, index=False, encoding="utf-8")

    print("Config:", asdict(cfg))
    print("\nResults:")
    print(df)


if __name__ == "__main__":
    main()
