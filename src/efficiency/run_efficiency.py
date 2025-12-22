"""Run efficiency optimization experiments with three-layer comparison."""

from __future__ import annotations

import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score, recall_score, roc_auc_score, roc_curve

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from effect.eval import evaluate_proba_threshold, g_mean
from effect.io import load_processed
from effect.models import train_rf
from effect.thresholding import select_threshold
from effect.weighting import density_weights_knn
from .models import TreePruner, predict_ensemble


@dataclass
class EfficiencyConfig:
    processed_dir: str = "data/processed"
    outputs_dir: str = "outputs"
    random_state: int = 42
    n_estimators: int = 200
    max_depth: int | None = None
    n_jobs: int = -1
    
    # Effect optimization params
    pos_class_weight: float = 4.0
    knn_k: int = 5
    density_alpha: float = 5.0
    
    # Efficiency optimization params - will be set dynamically
    n_clusters: int | None = None
    use_threshold_tuning: bool = True
    threshold_objective: str = "f1"
    t_min: float = 0.05
    t_max: float = 0.95
    t_step: float = 0.01


def _ensure_outputs_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _load_optimal_clusters(outputs_dir: str) -> int:
    """Load optimal cluster count from cluster selection results."""
    csv_path = os.path.join(outputs_dir, "cluster_selection_results.csv")
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            
            # Try to infer baseline recall from the data
            # The baseline recall should be the minimum threshold that makes sense
            # Based on run_cluster_selection.py logic, we're looking for:
            # Recall > baseline_recall (which is ~0.7042)
            
            # We'll use a conservative estimate: look for the highest recall that appears
            # to be at a "natural" level (not the Effect model's 0.7996)
            # For now, use 0.70 as a conservative baseline estimate
            baseline_recall_estimate = 0.70
            
            # First try: Recall > 0.70 AND F1_retention >= 95% AND speedup >= 3x
            balanced = df[
                (df["recall"] > baseline_recall_estimate) &
                (df["f1_retention"] >= 95) & 
                (df["speedup"] >= 3)
            ]
            
            if len(balanced) > 0:
                optimal_row = balanced.loc[balanced["f1"].idxmax()]
                optimal_clusters = int(optimal_row["n_clusters"])
                print(f"✓ Loaded optimal cluster count (strict criteria): k={optimal_clusters}")
                return optimal_clusters
            
            # Second try: Recall > 0.70 AND speedup >= 3x (relaxed F1 retention)
            relaxed = df[
                (df["recall"] > baseline_recall_estimate) &
                (df["speedup"] >= 3)
            ]
            
            if len(relaxed) > 0:
                optimal_row = relaxed.loc[relaxed["f1"].idxmax()]
                optimal_clusters = int(optimal_row["n_clusters"])
                print(f"✓ Loaded optimal cluster count (relaxed criteria): k={optimal_clusters}")
                return optimal_clusters
            
            # Third try: Just maximize speedup while keeping recall > 0.70
            recall_constrained = df[df["recall"] > baseline_recall_estimate]
            if len(recall_constrained) > 0:
                optimal_row = recall_constrained.loc[recall_constrained["speedup"].idxmax()]
                optimal_clusters = int(optimal_row["n_clusters"])
                print(f"✓ Loaded optimal cluster count (recall priority): k={optimal_clusters}")
                return optimal_clusters
            
            # Fallback: best F1
            optimal_row = df.loc[df["f1"].idxmax()]
            optimal_clusters = int(optimal_row["n_clusters"])
            print(f"⚠ No recall-constrained option found, using best F1: k={optimal_clusters}")
            return optimal_clusters
            
        except Exception as e:
            print(f"⚠ Could not read cluster selection results: {e}")
    
    # Default fallback
    print(f"⚠ Using default cluster count: 10")
    return 10


def _select_threshold_on_val(
    y_true: np.ndarray,
    proba_pos: np.ndarray,
    objective: str = "f1",
    t_min: float = 0.05,
    t_max: float = 0.95,
    step: float = 0.01,
) -> float:
    """
    Select optimal threshold on validation data using grid search.
    
    Returns best threshold that maximizes the specified objective.
    """
    tune_result = select_threshold(
        y_true=y_true,
        proba_pos=proba_pos,
        objective=objective,
        t_min=t_min,
        t_max=t_max,
        step=step,
    )
    return float(tune_result["best_threshold"])


def main() -> None:
    cfg = EfficiencyConfig()
    _ensure_outputs_dir(cfg.outputs_dir)
    
    # Load optimal cluster count from previous analysis
    if cfg.n_clusters is None:
        cfg.n_clusters = _load_optimal_clusters(cfg.outputs_dir)
    
    print("=" * 80)
    print("EFFICIENCY OPTIMIZATION: 三层对比实验")
    print("=" * 80)
    print(f"Using n_clusters={cfg.n_clusters} (from optimal cluster selection)")
    print(f"Config: {asdict(cfg)}\n")
    
    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_processed(cfg.processed_dir)
    
    results: List[Dict[str, float | str]] = []
    roc_curves_data: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    
    # ============================================================================
    # Layer 1: Baseline (RF without optimization)
    # ============================================================================
    print("\n[Layer 1] 基线模型 (Baseline)")
    print("-" * 80)
    
    model_baseline, train_time_baseline = train_rf(
        X_train, y_train,
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        random_state=cfg.random_state,
        n_jobs=cfg.n_jobs,
        class_weight=None,
        sample_weight=None,
    )
    
    # Threshold tuning on VALIDATION set only
    proba_val_baseline = model_baseline.predict_proba(X_val)[:, 1]
    threshold_baseline = _select_threshold_on_val(
        y_val, proba_val_baseline,
        objective=cfg.threshold_objective,
        t_min=cfg.t_min,
        t_max=cfg.t_max,
        step=cfg.t_step
    )
    
    # Evaluate on TEST set using evaluate_proba_threshold
    metrics_baseline, y_pred_baseline = evaluate_proba_threshold(
        model_baseline, X_test, y_test, threshold_baseline
    )
    
    # Store ROC curve data
    proba_test_baseline = model_baseline.predict_proba(X_test)[:, 1]
    fpr_baseline, tpr_baseline, _ = roc_curve(y_test, proba_test_baseline)
    roc_curves_data["Baseline"] = (fpr_baseline, tpr_baseline)
    
    results.append({
        "method": "Baseline",
        "trees_count": cfg.n_estimators,
        "train_time_s": float(train_time_baseline),
        "pred_time_s": float(metrics_baseline["pred_time_s"]),
        "threshold": float(threshold_baseline),
        "recall": float(metrics_baseline["recall"]),
        "f1": float(metrics_baseline["f1"]),
        "gmean": float(metrics_baseline["gmean"]),
        "auc": float(roc_auc_score(y_test, proba_test_baseline)),
    })
    
    print(f"Recall: {metrics_baseline['recall']:.4f}, F1: {metrics_baseline['f1']:.4f}, G-mean: {metrics_baseline['gmean']:.4f}")
    print(f"Prediction time: {metrics_baseline['pred_time_s']*1000:.2f}ms, Tree count: {cfg.n_estimators}")
    
    # ============================================================================
    # Intermediate Step: Train Cost-Sensitive Model (for random state alignment)
    # ============================================================================
    # 为了与 run_effect.py 保持随机数生成器状态一致，需要先训练 Cost-sensitive 模型
    # 这样可以确保 Effect 模型的训练结果完全相同
    print("\n[Intermediate] 训练 Cost-Sensitive 模型 (用于对齐随机状态)")
    print("-" * 80)
    
    class_weight_cs = {0: 1.0, 1: cfg.pos_class_weight}
    model_cost_sensitive, _ = train_rf(
        X_train, y_train,
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        random_state=cfg.random_state,
        n_jobs=cfg.n_jobs,
        class_weight=class_weight_cs,
        sample_weight=None,  # Cost-sensitive 不使用 sample_weight
    )
    print("Cost-Sensitive 模型已训练（仅用于对齐随机状态，不评估）")
    
    # ============================================================================
    # Layer 2: Effect Optimization (Cost-sensitive + Density-aware weighting)
    # ============================================================================
    print("\n[Layer 2] Effect优化 (Cost-Sensitive + Density-Aware Weighting)")
    print("-" * 80)
    
    # Prepare sample weights with density-aware strategy
    sample_weights = density_weights_knn(
        X_train=X_train,
        y_train=y_train,
        k=cfg.knn_k,
        alpha=cfg.density_alpha,
        pos_only=True,
        w_max=None,
        include_self_fix=True,
    )
    
    class_weight_effect = {0: 1.0, 1: cfg.pos_class_weight}
    
    model_effect, train_time_effect = train_rf(
        X_train, y_train,
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        random_state=cfg.random_state,
        n_jobs=cfg.n_jobs,
        class_weight=class_weight_effect,
        sample_weight=sample_weights,
    )
    
    # Threshold tuning on VALIDATION set only
    proba_val_effect = model_effect.predict_proba(X_val)[:, 1]
    threshold_effect = _select_threshold_on_val(
        y_val, proba_val_effect,
        objective=cfg.threshold_objective,
        t_min=cfg.t_min,
        t_max=cfg.t_max,
        step=cfg.t_step
    )
    
    # Evaluate on TEST set using evaluate_proba_threshold
    metrics_effect, y_pred_effect = evaluate_proba_threshold(
        model_effect, X_test, y_test, threshold_effect
    )
    
    # Store ROC curve data
    proba_test_effect = model_effect.predict_proba(X_test)[:, 1]
    fpr_effect, tpr_effect, _ = roc_curve(y_test, proba_test_effect)
    roc_curves_data["Effect Optimized"] = (fpr_effect, tpr_effect)
    
    results.append({
        "method": "Effect Optimized",
        "trees_count": cfg.n_estimators,
        "train_time_s": float(train_time_effect),
        "pred_time_s": float(metrics_effect["pred_time_s"]),
        "threshold": float(threshold_effect),
        "recall": float(metrics_effect["recall"]),
        "f1": float(metrics_effect["f1"]),
        "gmean": float(metrics_effect["gmean"]),
        "auc": float(roc_auc_score(y_test, proba_test_effect)),
    })
    
    print(f"Recall: {metrics_effect['recall']:.4f}, F1: {metrics_effect['f1']:.4f}, G-mean: {metrics_effect['gmean']:.4f}")
    print(f"Prediction time: {metrics_effect['pred_time_s']*1000:.2f}ms, Tree count: {cfg.n_estimators}")
    
    # ============================================================================
    # Layer 3: Effect + Efficiency Optimization (Pruning + Weighted Ensemble)
    # ============================================================================
    print("\n[Layer 3] Effect + Efficiency优化 (Pruning + Weighted Ensemble)")
    print("-" * 80)
    
    # Use the effect-optimized model for pruning
    pruner = TreePruner(n_clusters=cfg.n_clusters, random_state=cfg.random_state)
    pruner.fit(model_effect, X_val, y_val)
    
    selected_trees = pruner.get_selected_trees()
    tree_weights = pruner.get_tree_weights()
    
    print(f"Selected {len(selected_trees)} trees from {cfg.n_estimators} trees via clustering")
    print(f"Pruned to {len(selected_trees)/cfg.n_estimators*100:.1f}% of original size")
    
    # Threshold tuning on VALIDATION set with pruned ensemble
    proba_val_pruned = predict_ensemble(
        model_effect, X_val, selected_trees, tree_weights, use_proba=True
    )
    threshold_pruned = _select_threshold_on_val(
        y_val, proba_val_pruned,
        objective=cfg.threshold_objective,
        t_min=cfg.t_min,
        t_max=cfg.t_max,
        step=cfg.t_step
    )
    
    # Evaluate on TEST set with pruned ensemble using evaluate_proba_threshold
    # Create a wrapper class to use evaluate_proba_threshold with pruned ensemble
    class PrunedEnsembleWrapper:
        def __init__(self, model, selected_trees, tree_weights):
            self.model = model
            self.selected_trees = selected_trees
            self.tree_weights = tree_weights
        
        def predict_proba(self, X):
            """Return probabilities for both classes"""
            proba_pos = predict_ensemble(
                self.model, X, self.selected_trees, self.tree_weights, use_proba=True
            )
            proba_neg = 1.0 - proba_pos
            return np.column_stack([proba_neg, proba_pos])
    
    pruned_model_wrapper = PrunedEnsembleWrapper(model_effect, selected_trees, tree_weights)
    
    metrics_pruned, y_pred_pruned = evaluate_proba_threshold(
        pruned_model_wrapper, X_test, y_test, threshold_pruned
    )
    
    # Store ROC curve data
    proba_test_pruned = predict_ensemble(
        model_effect, X_test, selected_trees, tree_weights, use_proba=True
    )
    fpr_pruned, tpr_pruned, _ = roc_curve(y_test, proba_test_pruned)
    roc_curves_data["Effect + Efficiency"] = (fpr_pruned, tpr_pruned)
    
    results.append({
        "method": "Effect + Efficiency",
        "trees_count": len(selected_trees),
        "train_time_s": float(train_time_effect),
        "pred_time_s": float(metrics_pruned["pred_time_s"]),
        "threshold": float(threshold_pruned),
        "recall": float(metrics_pruned["recall"]),
        "f1": float(metrics_pruned["f1"]),
        "gmean": float(metrics_pruned["gmean"]),
        "auc": float(roc_auc_score(y_test, proba_test_pruned)),
    })
    
    print(f"Recall: {metrics_pruned['recall']:.4f}, F1: {metrics_pruned['f1']:.4f}, G-mean: {metrics_pruned['gmean']:.4f}")
    print(f"Prediction time: {metrics_pruned['pred_time_s']*1000:.2f}ms, Tree count: {len(selected_trees)}")
    
    # ============================================================================
    # Summary and Visualization
    # ============================================================================
    print("\n" + "=" * 80)
    print("三层对比总结")
    print("=" * 80)
    
    df_results = pd.DataFrame(results)
    print("\n完整结果表:")
    print(df_results.to_string(index=False))
    
    # Save results to CSV
    output_csv = os.path.join(cfg.outputs_dir, "results_efficiency_comparison.csv")
    df_results.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"\n结果已保存: {output_csv}")
    
    # Generate visualizations
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("三层对比: Baseline vs Effect vs Effect+Efficiency", fontsize=14, fontweight="bold")
        
        # Subplot 1: ROC Curves
        ax = axes[0, 0]
        for method, (fpr, tpr) in roc_curves_data.items():
            ax.plot(fpr, tpr, marker="o", label=method, linewidth=2)
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Subplot 2: Recall vs F1 Comparison
        ax = axes[0, 1]
        methods = df_results["method"].tolist()
        x = np.arange(len(methods))
        width = 0.35
        ax.bar(x - width/2, df_results["recall"], width, label="Recall", alpha=0.8)
        ax.bar(x + width/2, df_results["f1"], width, label="F1-Score", alpha=0.8)
        ax.set_xlabel("Method")
        ax.set_ylabel("Score")
        ax.set_title("Recall & F1-Score Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=15, ha="right")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        
        # Subplot 3: Prediction Time Comparison (ms)
        ax = axes[1, 0]
        pred_times_ms = df_results["pred_time_s"] * 1000
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
        bars = ax.bar(methods, pred_times_ms, color=colors, alpha=0.8)
        ax.set_ylabel("Prediction Time (ms)")
        ax.set_title("Prediction Time Comparison")
        ax.set_xticklabels(methods, rotation=15, ha="right")
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f"{height:.2f}ms", ha="center", va="bottom", fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")
        
        # Subplot 4: G-mean Comparison (changed from Tree Count vs Performance)
        ax = axes[1, 1]
        gmean_scores = df_results["gmean"].tolist()
        colors_gmean = ["#1f77b4", "#ff7f0e", "#2ca02c"]
        bars = ax.bar(methods, gmean_scores, color=colors_gmean, alpha=0.8)
        ax.set_ylabel("G-mean Score")
        ax.set_title("G-mean Comparison")
        ax.set_xticklabels(methods, rotation=15, ha="right")
        ax.set_ylim([0, 1.0])
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f"{height:.4f}", ha="center", va="bottom", fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")
        
        plt.tight_layout()
        
        # Save figure
        output_fig = os.path.join(cfg.outputs_dir, "efficiency_comparison.png")
        fig.savefig(output_fig, dpi=200, bbox_inches="tight")
        print(f"可视化已保存: {output_fig}")
        plt.close(fig)
        
    except Exception as e:
        print(f"可视化生成出错: {e}")
    
    # Performance improvement analysis
    print("\n" + "=" * 80)
    print("性能改进分析")
    print("=" * 80)
    
    recall_improvement_effect = (metrics_effect["recall"] - metrics_baseline["recall"]) / metrics_baseline["recall"] * 100
    f1_improvement_effect = (metrics_effect["f1"] - metrics_baseline["f1"]) / metrics_baseline["f1"] * 100
    
    recall_improvement_pruned = (metrics_pruned["recall"] - metrics_baseline["recall"]) / metrics_baseline["recall"] * 100
    f1_improvement_pruned = (metrics_pruned["f1"] - metrics_baseline["f1"]) / metrics_baseline["f1"] * 100
    
    speedup_factor = metrics_baseline["pred_time_s"] / metrics_pruned["pred_time_s"]
    
    print(f"\nEffect优化相对Baseline的改进:")
    print(f"  Recall: {recall_improvement_effect:+.2f}%")
    print(f"  F1-Score: {f1_improvement_effect:+.2f}%")
    
    print(f"\nEffect+Efficiency优化相对Baseline的改进:")
    print(f"  Recall: {recall_improvement_pruned:+.2f}%")
    print(f"  F1-Score: {f1_improvement_pruned:+.2f}%")
    print(f"  Model Size: {len(selected_trees)/cfg.n_estimators*100:.1f}% ({len(selected_trees)}/{cfg.n_estimators})")
    print(f"  Speedup: {speedup_factor:.2f}x faster")


if __name__ == "__main__":
    main()
