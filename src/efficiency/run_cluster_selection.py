"""Grid search for optimal number of clusters in tree pruning."""

from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from effect.eval import evaluate_proba_threshold
from effect.io import load_processed
from effect.models import train_rf
from effect.thresholding import select_threshold
from effect.weighting import density_weights_knn
from .models import TreePruner, predict_ensemble


@dataclass
class ClusterSelectionConfig:
    """Configuration for cluster selection."""
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
    
    # Cluster search params
    n_clusters_range: List[int] = None
    threshold_objective: str = "f1"
    t_min: float = 0.05
    t_max: float = 0.95
    t_step: float = 0.01
    
    def __post_init__(self):
        """Set default cluster range if not provided."""
        if self.n_clusters_range is None:
            self.n_clusters_range = [5, 8, 10, 12, 15, 20, 25, 30]


def _ensure_outputs_dir(path: str) -> None:
    """Create output directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def _select_threshold_on_val(
    y_true: np.ndarray,
    proba_pos: np.ndarray,
    objective: str = "f1",
    t_min: float = 0.05,
    t_max: float = 0.95,
    step: float = 0.01,
) -> float:
    """Select optimal threshold on validation data using grid search."""
    tune_result = select_threshold(
        y_true=y_true,
        proba_pos=proba_pos,
        objective=objective,
        t_min=t_min,
        t_max=t_max,
        step=step,
    )
    return float(tune_result["best_threshold"])


def evaluate_cluster_count(
    n_clusters: int,
    model_effect,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    cfg: ClusterSelectionConfig,
) -> Dict:
    """Evaluate performance for a specific number of clusters."""
    # Perform tree pruning
    pruner = TreePruner(n_clusters=n_clusters, random_state=cfg.random_state)
    pruner.fit(model_effect, X_val, y_val)
    
    selected_trees = pruner.get_selected_trees()
    tree_weights = pruner.get_tree_weights()
    
    # Threshold tuning on validation set
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
    
    # Evaluate on test set
    class PrunedEnsembleWrapper:
        def __init__(self, model, selected_trees, tree_weights):
            self.model = model
            self.selected_trees = selected_trees
            self.tree_weights = tree_weights
        
        def predict_proba(self, X):
            proba_pos = predict_ensemble(
                self.model, X, self.selected_trees, self.tree_weights, use_proba=True
            )
            proba_neg = 1.0 - proba_pos
            return np.column_stack([proba_neg, proba_pos])
    
    pruned_model_wrapper = PrunedEnsembleWrapper(model_effect, selected_trees, tree_weights)
    
    metrics_pruned, y_pred_pruned = evaluate_proba_threshold(
        pruned_model_wrapper, X_test, y_test, threshold_pruned
    )
    
    return {
        "n_clusters": n_clusters,
        "n_trees": len(selected_trees),
        "compression_ratio": len(selected_trees) / cfg.n_estimators,
        "threshold": float(threshold_pruned),
        "recall": float(metrics_pruned["recall"]),
        "f1": float(metrics_pruned["f1"]),
        "gmean": float(metrics_pruned["gmean"]),
        "auc": float(roc_auc_score(y_test, predict_ensemble(
            model_effect, X_test, selected_trees, tree_weights, use_proba=True
        ))),
        "pred_time_s": float(metrics_pruned["pred_time_s"]),
    }


def main() -> int:
    """Run cluster count selection experiment. Returns optimal cluster count."""
    cfg = ClusterSelectionConfig()
    _ensure_outputs_dir(cfg.outputs_dir)
    
    print("=" * 80)
    print("CLUSTER SELECTION: Grid Search for Optimal Number of Clusters")
    print("=" * 80)
    print(f"Testing cluster counts: {cfg.n_clusters_range}\n")
    
    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_processed(cfg.processed_dir)
    
    # Train Baseline RF model (for recall constraint)
    print("[Step 1] Training Baseline RF Model (for recall constraint)")
    print("-" * 80)
    
    model_baseline, _ = train_rf(
        X_train, y_train,
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        random_state=cfg.random_state,
        n_jobs=cfg.n_jobs,
        class_weight=None,
        sample_weight=None,
    )
    
    proba_val_baseline = model_baseline.predict_proba(X_val)[:, 1]
    threshold_baseline = _select_threshold_on_val(y_val, proba_val_baseline)
    
    metrics_baseline, _ = evaluate_proba_threshold(
        model_baseline, X_test, y_test, threshold_baseline
    )
    
    baseline_recall = metrics_baseline["recall"]
    print(f"Baseline RF Model:")
    print(f"  Recall: {baseline_recall:.4f} (target: pruned models must exceed this)\n")
    
    # Train Effect model (baseline for F1 comparison)
    print("[Step 2] Training Effect Optimized Model")
    print("-" * 80)
    
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
    
    # Get baseline Effect model metrics on test set
    proba_val_effect = model_effect.predict_proba(X_val)[:, 1]
    threshold_effect = _select_threshold_on_val(y_val, proba_val_effect)
    
    metrics_effect, _ = evaluate_proba_threshold(
        model_effect, X_test, y_test, threshold_effect
    )
    
    print(f"Effect Model Baseline:")
    print(f"  F1: {metrics_effect['f1']:.4f}")
    print(f"  Recall: {metrics_effect['recall']:.4f}")
    print(f"  Pred Time: {metrics_effect['pred_time_s']*1000:.2f}ms\n")
    
    # Grid search for optimal cluster count
    print("[Step 3] Grid Search for Optimal Cluster Count")
    print("-" * 80)
    
    results = []
    
    for n_clusters in cfg.n_clusters_range:
        print(f"Testing n_clusters={n_clusters}...", end=" ", flush=True)
        try:
            result = evaluate_cluster_count(
                n_clusters, model_effect, X_val, y_val, X_test, y_test, cfg
            )
            
            # Calculate additional metrics
            result["f1_retention"] = (result["f1"] / metrics_effect["f1"]) * 100
            result["speedup"] = metrics_effect["pred_time_s"] / result["pred_time_s"]
            result["f1_loss"] = ((metrics_effect["f1"] - result["f1"]) / metrics_effect["f1"]) * 100
            
            results.append(result)
            
            print(f"✓ F1={result['f1']:.4f}, Speedup={result['speedup']:.2f}x, Trees={result['n_trees']}")
        except Exception as e:
            print(f"✗ Error: {e}")
    
    # Convert to DataFrame
    df_results = pd.DataFrame(results)
    
    print("\n" + "=" * 80)
    print("CLUSTER SELECTION RESULTS")
    print("=" * 80)
    print("\nComplete Results Table:")
    print(df_results.to_string(index=False))
    
    # Find optimal cluster count using different criteria
    print("\n" + "=" * 80)
    print("OPTIMAL CLUSTER ANALYSIS")
    print("=" * 80)
    
    # Criterion 1: Best F1 score
    best_f1_idx = df_results["f1"].idxmax()
    best_f1_row = df_results.loc[best_f1_idx]
    print(f"\n1. Best F1 Score:")
    print(f"   n_clusters={int(best_f1_row['n_clusters'])}")
    print(f"   F1={best_f1_row['f1']:.4f}, Speedup={best_f1_row['speedup']:.2f}x")
    
    # Criterion 2: Best balance (Recall > Baseline AND F1 retention >= 95% AND speedup >= 3x)
    print(f"\n2. Best Balance (Recall > {baseline_recall:.4f} AND F1_retention >= 95% AND speedup >= 3x):")
    
    balanced = df_results[
        (df_results["recall"] > baseline_recall) &
        (df_results["f1_retention"] >= 95) & 
        (df_results["speedup"] >= 3)
    ]
    
    if len(balanced) > 0:
        # Among balanced options, prefer higher F1 score (performance > speed)
        best_balanced_idx = balanced["f1"].idxmax()
        best_balanced_row = df_results.loc[best_balanced_idx]
        print(f"   ✓ Found! n_clusters={int(best_balanced_row['n_clusters'])}")
        print(f"   Recall={best_balanced_row['recall']:.4f}, F1={best_balanced_row['f1']:.4f}")
        print(f"   F1_retention={best_balanced_row['f1_retention']:.1f}%, Speedup={best_balanced_row['speedup']:.2f}x")
        optimal_n_clusters = int(best_balanced_row["n_clusters"])
    else:
        print(f"   ⚠ No option meets all criteria!")
        print(f"   Trying: Recall > Baseline AND speedup >= 3x (relaxed F1 retention)")
        
        relaxed = df_results[
            (df_results["recall"] > baseline_recall) &
            (df_results["speedup"] >= 3)
        ]
        
        if len(relaxed) > 0:
            best_relaxed_idx = relaxed["f1"].idxmax()
            best_relaxed_row = df_results.loc[best_relaxed_idx]
            print(f"   ✓ Found relaxed option - n_clusters={int(best_relaxed_row['n_clusters'])}")
            print(f"   Recall={best_relaxed_row['recall']:.4f}, F1={best_relaxed_row['f1']:.4f}")
            print(f"   F1_retention={best_relaxed_row['f1_retention']:.1f}%, Speedup={best_relaxed_row['speedup']:.2f}x")
            optimal_n_clusters = int(best_relaxed_row["n_clusters"])
        else:
            print(f"   Trying: Just Recall > Baseline (most important constraint)")
            recall_only = df_results[df_results["recall"] > baseline_recall]
            
            if len(recall_only) > 0:
                best_recall_idx = recall_only["speedup"].idxmax()
                best_recall_row = df_results.loc[best_recall_idx]
                print(f"   ✓ Found - n_clusters={int(best_recall_row['n_clusters'])}")
                print(f"   Recall={best_recall_row['recall']:.4f}, F1={best_recall_row['f1']:.4f}")
                print(f"   Speedup={best_recall_row['speedup']:.2f}x")
                optimal_n_clusters = int(best_recall_row["n_clusters"])
            else:
                print(f"   ✗ No option can achieve recall > Baseline!")
                print(f"   Falling back to best F1 score...")
                optimal_n_clusters = int(best_f1_row["n_clusters"])
    
    # Criterion 3: Highest speedup
    best_speedup_idx = df_results["speedup"].idxmax()
    best_speedup_row = df_results.loc[best_speedup_idx]
    print(f"\n3. Highest Speedup:")
    print(f"   n_clusters={int(best_speedup_row['n_clusters'])}")
    print(f"   F1={best_speedup_row['f1']:.4f}, Speedup={best_speedup_row['speedup']:.2f}x")
    
    # Criterion 4: F1 loss < 5% with maximum speedup
    acceptable = df_results[df_results["f1_loss"] < 5]
    if len(acceptable) > 0:
        best_acceptable_idx = acceptable["speedup"].idxmax()
        best_acceptable_row = df_results.loc[best_acceptable_idx]
        print(f"\n4. F1 Loss < 5% with Max Speedup:")
        print(f"   n_clusters={int(best_acceptable_row['n_clusters'])}")
        print(f"   F1={best_acceptable_row['f1']:.4f}, F1_loss={best_acceptable_row['f1_loss']:.2f}%, Speedup={best_acceptable_row['speedup']:.2f}x")
    
    print(f"\n" + "=" * 80)
    print(f"RECOMMENDED OPTIMAL CLUSTER COUNT: {optimal_n_clusters}")
    print("=" * 80)
    
    # Save results to CSV
    output_csv = os.path.join(cfg.outputs_dir, "cluster_selection_results.csv")
    df_results.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"\nResults saved to: {output_csv}")
    
    # Generate visualization
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Cluster Count Selection: Grid Search Analysis", fontsize=14, fontweight="bold")
        
        # Subplot 1: Recall vs Cluster Count (with baseline constraint)
        ax = axes[0, 0]
        ax.plot(df_results["n_clusters"], df_results["recall"], marker="o", linewidth=2, markersize=8, color="#1f77b4", label="Pruned Models")
        ax.axhline(y=baseline_recall, color="red", linestyle="--", label=f"Baseline Recall ({baseline_recall:.4f})", linewidth=2)
        optimal_recall = df_results[df_results["n_clusters"]==optimal_n_clusters]["recall"].values[0]
        ax.scatter([optimal_n_clusters], [optimal_recall], s=300, color="green", marker="*", label=f"Optimal (k={optimal_n_clusters})", zorder=5)
        ax.set_xlabel("Number of Clusters")
        ax.set_ylabel("Recall Score")
        ax.set_title("Recall vs Cluster Count (Must Exceed Baseline)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Subplot 2: Speedup vs Cluster Count
        ax = axes[0, 1]
        ax.plot(df_results["n_clusters"], df_results["speedup"], marker="s", linewidth=2, markersize=8, color="#ff7f0e")
        ax.axhline(y=3.0, color="gray", linestyle="--", label="3x Speedup Threshold", linewidth=1)
        optimal_speedup = df_results[df_results["n_clusters"]==optimal_n_clusters]["speedup"].values[0]
        ax.scatter([optimal_n_clusters], [optimal_speedup], s=300, color="green", marker="*", label=f"Optimal (k={optimal_n_clusters})", zorder=5)
        ax.set_xlabel("Number of Clusters")
        ax.set_ylabel("Speedup Factor (x)")
        ax.set_title("Speedup vs Cluster Count")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Subplot 3: F1 Retention vs Cluster Count
        ax = axes[1, 0]
        ax.plot(df_results["n_clusters"], df_results["f1_retention"], marker="^", linewidth=2, markersize=8, color="#2ca02c")
        ax.axhline(y=95, color="gray", linestyle="--", label="95% Retention Threshold", linewidth=1)
        optimal_retention = df_results[df_results["n_clusters"]==optimal_n_clusters]["f1_retention"].values[0]
        ax.scatter([optimal_n_clusters], [optimal_retention], s=300, color="green", marker="*", label=f"Optimal (k={optimal_n_clusters})", zorder=5)
        ax.set_xlabel("Number of Clusters")
        ax.set_ylabel("F1 Retention (%)")
        ax.set_title("F1 Retention vs Cluster Count")
        ax.set_ylim([80, 105])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Subplot 4: Pareto Front (Speedup vs Recall)
        ax = axes[1, 1]
        scatter = ax.scatter(df_results["speedup"], df_results["recall"], s=200, alpha=0.6, c=df_results["n_clusters"], 
                  cmap="viridis", edgecolors="black", linewidth=1)
        ax.axhline(y=baseline_recall, color="red", linestyle="--", linewidth=2, alpha=0.7, label=f"Baseline ({baseline_recall:.4f})")
        ax.scatter([optimal_speedup], [optimal_recall], s=500, color="green", marker="*", 
                  label=f"Optimal (k={optimal_n_clusters})", zorder=5, edgecolors="black", linewidth=2)
        
        # Add cluster count labels
        for idx, row in df_results.iterrows():
            ax.annotate(f"k={int(row['n_clusters'])}", 
                       (row["speedup"], row["recall"]), 
                       textcoords="offset points", xytext=(5, 5), fontsize=9)
        
        ax.set_xlabel("Speedup Factor (x)")
        ax.set_ylabel("Recall Score")
        ax.set_title("Pareto Front: Speedup vs Recall (Recall must exceed baseline)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Number of Clusters")
        
        plt.tight_layout()
        
        # Save figure
        output_fig = os.path.join(cfg.outputs_dir, "cluster_selection_analysis.png")
        fig.savefig(output_fig, dpi=200, bbox_inches="tight")
        print(f"Visualization saved to: {output_fig}")
        plt.close(fig)
        
    except Exception as e:
        print(f"Visualization generation failed: {e}")
    
    print("\n" + "=" * 80)
    print("CLUSTER SELECTION COMPLETED")
    print("=" * 80)
    return optimal_n_clusters


if __name__ == "__main__":
    optimal_clusters = main()
    print(f"\n✓ Optimal cluster count: {optimal_clusters}")
