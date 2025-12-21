#!/usr/bin/env python3
"""
Generate all PPT-ready figures for the bank marketing imbalanced project.
"""

import glob
import os
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

try:
    import matplotlib.pyplot as plt
    import pandas as pd
except ImportError as exc:  # pragma: no cover - dependency guard
    print("Please install pandas and matplotlib before running this script.")
    print(exc)
    sys.exit(1)


# Style: readable fonts and white background
plt.rcParams.update(
    {
        "figure.figsize": (6, 4),
        "axes.titlesize": 15,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
    }
)


BASE_DIR = Path(__file__).resolve().parent.parent
FIG_DIR = BASE_DIR / "outputs" / "figures"

generated: List[str] = []
skipped: List[Tuple[str, str]] = []


def normalize_col(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def find_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    normalized = {normalize_col(col): col for col in df.columns}
    for cand in candidates:
        key = normalize_col(cand)
        if key in normalized:
            return normalized[key]
    return None


def ensure_fig_dir() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def adjust_ylim(ax: plt.Axes, data: Iterable[float]) -> None:
    data = list(data)
    if not data:
        return
    dmin, dmax = min(data), max(data)
    if dmin >= 0 and dmax <= 1.05:
        ax.set_ylim(0, 1)


def record_skip(name: str, reason: str) -> None:
    print(f"[skip] {name}: {reason}")
    skipped.append((name, reason))


def record_success(name: str, path: Path) -> None:
    print(f"[saved] {name}: {path}")
    generated.append(name)


def save_current_fig(filename: str, label: str, data_for_ylim: Iterable[float]) -> None:
    ensure_fig_dir()
    path = FIG_DIR / filename
    ax = plt.gca()
    adjust_ylim(ax, data_for_ylim)
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    record_success(label, path)


def load_dataframe(path: Path, fig_names: Iterable[str]) -> Optional[pd.DataFrame]:
    if not path.exists():
        for name in fig_names:
            record_skip(name, f"missing file {path}")
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - defensive
        for name in fig_names:
            record_skip(name, f"failed to read {path}: {exc}")
        return None


def to_numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def plot_effectiveness() -> None:
    fig_names = [
        "fig_effect_recall.png",
        "fig_effect_f1.png",
        "fig_effect_gmean.png",
        "fig_effect_threshold.png",
    ]
    df = load_dataframe(BASE_DIR / "outputs" / "results_effect.csv", fig_names)
    if df is None:
        return

    method_col = find_column(df, ["method"])
    if method_col is None:
        for name in fig_names:
            record_skip(name, "missing method column")
        return

    method_map = [
        ("Baseline", "rf_baseline", None),
        ("Cost-sensitive", "cost_sensitive", "plus_density"),
        ("Cost+Density", "plus_density", None),
    ]

    rows = []
    for label, keyword, exclude in method_map:
        mask = df[method_col].str.contains(keyword, case=False, na=False)
        if exclude:
            mask &= ~df[method_col].str.contains(exclude, case=False, na=False)
        match = df.loc[mask]
        if match.empty:
            print(f"[info] method '{label}' not found with keyword '{keyword}'")
            continue
        rows.append((label, match.iloc[0]))

    if not rows:
        for name in fig_names:
            record_skip(name, "no matching methods found")
        return

    def bar_plot(metric_candidates: List[str], filename: str, title: str, ylabel: str) -> None:
        col = find_column(df, metric_candidates)
        if col is None:
            record_skip(filename, f"missing metric columns {metric_candidates}")
            return

        labels, values = [], []
        for label, series in rows:
            val = pd.to_numeric(series.get(col, float("nan")), errors="coerce")
            if pd.isna(val):
                continue
            labels.append(label)
            values.append(val)

        if not values:
            record_skip(filename, "no numeric values to plot")
            return

        plt.figure()
        bars = plt.bar(labels, values, color=["#4c72b0", "#55a868", "#c44e52"][: len(values)], alpha=0.85)
        for bar, val in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{val:.3f}", ha="center", va="bottom")
        plt.title(title)
        plt.ylabel(ylabel)
        save_current_fig(filename, filename, values)

    bar_plot(
        ["recall(test)", "recall"],
        "fig_effect_recall.png",
        "Baseline vs Cost-sensitive vs Cost+Density – Recall (test)",
        "Recall (test)",
    )
    bar_plot(
        ["f1(test)", "f1"],
        "fig_effect_f1.png",
        "Baseline vs Cost-sensitive vs Cost+Density – F1 (test)",
        "F1 (test)",
    )
    bar_plot(
        ["gmean(test)", "gmean", "g-mean(test)"],
        "fig_effect_gmean.png",
        "Baseline vs Cost-sensitive vs Cost+Density – G-mean (test)",
        "G-mean (test)",
    )

    threshold_col = find_column(df, ["threshold", "best_threshold", "best_threshold (val)"])
    if threshold_col is None:
        record_skip("fig_effect_threshold.png", "missing threshold column")
        return

    labels, values = [], []
    for label, series in rows:
        val = pd.to_numeric(series.get(threshold_col, float("nan")), errors="coerce")
        if pd.isna(val):
            continue
        labels.append(label)
        values.append(val)

    if not values:
        record_skip("fig_effect_threshold.png", "no numeric threshold values")
        return

    plt.figure()
    bars = plt.bar(labels, values, color="#937860", alpha=0.85)
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{val:.3f}", ha="center", va="bottom")
    plt.title("Baseline vs Cost-sensitive vs Cost+Density – Final threshold")
    plt.ylabel("Threshold")
    save_current_fig("fig_effect_threshold.png", "fig_effect_threshold.png", values)


def plot_threshold_search_curves() -> None:
    patterns = [
        BASE_DIR / "outputs" / "threshold_search_*.csv",
        BASE_DIR / "outputs" / "threshold_search_csv_results" / "threshold_search_*.csv",
    ]
    csv_files: List[str] = []
    for pattern in patterns:
        csv_files.extend(glob.glob(str(pattern)))
    if not csv_files:
        print("[info] no threshold_search_*.csv files found; skipping threshold search plots")
        return

    for csv_path in csv_files:
        method_name = Path(csv_path).stem.replace("threshold_search_", "")
        fig_name = f"fig_threshold_search_{method_name}.png"
        df = load_dataframe(Path(csv_path), [fig_name])
        if df is None:
            continue

        x_col = find_column(df, ["threshold"])
        y_col = find_column(df, ["f1", "recall", "objective_score", "objective", "score"])
        if x_col is None or y_col is None:
            record_skip(fig_name, "required columns for threshold search missing")
            continue

        x = to_numeric_series(df[x_col])
        y = to_numeric_series(df[y_col])
        valid = x.notna() & y.notna()
        if not valid.any():
            record_skip(fig_name, "no numeric data in threshold search file")
            continue

        x, y = x[valid], y[valid]
        plt.figure()
        plt.plot(x, y, marker="o", color="#4c72b0")
        best_idx = y.idxmax()
        plt.scatter([x.loc[best_idx]], [y.loc[best_idx]], color="#c44e52", zorder=5)
        plt.annotate(
            f"best={x.loc[best_idx]:.3f}",
            (x.loc[best_idx], y.loc[best_idx]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
        )
        plt.title(f"Threshold search: {method_name}")
        plt.xlabel("Threshold")
        plt.ylabel(y_col)
        save_current_fig(fig_name, fig_name, y)


def annotate_peak(ax: plt.Axes, x_val: float, y_val: float, label: str) -> None:
    ax.annotate(
        f"{label}, {y_val:.3f}",
        xy=(x_val, y_val),
        xytext=(0, 10),
        textcoords="offset points",
        ha="center",
        color="#c44e52",
    )


def line_plot_with_points(
    x: List[float],
    y: List[float],
    filename: str,
    title: str,
    xlabel: str,
    ylabel: str,
    annotate_max: bool = False,
    xticks: Optional[List[float]] = None,
) -> None:
    if not x or not y:
        record_skip(filename, "no data to plot")
        return

    plt.figure()
    plt.plot(x, y, marker="o", color="#4c72b0")
    ax = plt.gca()
    if annotate_max:
        max_idx = max(range(len(y)), key=lambda i: y[i])
        annotate_peak(ax, x[max_idx], y[max_idx], f"{xlabel}={x[max_idx]}")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if xticks is not None:
        ax.set_xticks(xticks)
    save_current_fig(filename, filename, y)


def plot_cw_sensitivity() -> None:
    fig_names = ["fig_sensitivity_cw_recall.png", "fig_sensitivity_cw_f1.png"]
    df = load_dataframe(BASE_DIR / "outputs" / "results_cw_sweep.csv", fig_names)
    if df is None:
        return

    cw_col = find_column(df, ["pos_class_weight (α)", "pos_class_weight", "pos_class_weightalpha", "cw"])
    recall_col = find_column(df, ["recall(test)", "recall", "recall (test)"])
    f1_col = find_column(df, ["f1(test)", "f1", "f1 (test)"])
    if cw_col is None:
        for name in fig_names:
            record_skip(name, "missing pos_class_weight column")
        return

    cw = to_numeric_series(df[cw_col])
    recall = to_numeric_series(df[recall_col]) if recall_col else pd.Series(dtype=float)
    f1 = to_numeric_series(df[f1_col]) if f1_col else pd.Series(dtype=float)
    valid_recall = cw.notna() & recall.notna()
    valid_f1 = cw.notna() & f1.notna()

    if valid_recall.any():
        cw_vals = cw[valid_recall].tolist()
        recall_vals = recall[valid_recall].tolist()
        line_plot_with_points(
            cw_vals,
            recall_vals,
            "fig_sensitivity_cw_recall.png",
            "Sensitivity: pos_class_weight vs Recall",
            "pos_class_weight",
            "Recall (test)",
            annotate_max=True,
        )
    else:
        record_skip("fig_sensitivity_cw_recall.png", "no recall data available")

    if valid_f1.any():
        cw_vals = cw[valid_f1].tolist()
        f1_vals = f1[valid_f1].tolist()
        line_plot_with_points(
            cw_vals,
            f1_vals,
            "fig_sensitivity_cw_f1.png",
            "Sensitivity: pos_class_weight vs F1",
            "pos_class_weight",
            "F1 (test)",
            annotate_max=True,
        )
    else:
        record_skip("fig_sensitivity_cw_f1.png", "no f1 data available")


def plot_knn_sensitivity() -> None:
    fig_names = ["fig_sensitivity_k_recall.png", "fig_sensitivity_k_f1.png"]
    df = load_dataframe(BASE_DIR / "outputs" / "results_knn_sweep.csv", fig_names)
    if df is None:
        return

    k_col = find_column(df, ["knn_k", "k"])
    recall_col = find_column(df, ["recall(test)", "recall", "recall (test)"])
    f1_col = find_column(df, ["f1(test)", "f1", "f1 (test)"])
    if k_col is None:
        for name in fig_names:
            record_skip(name, "missing k column")
        return

    k_vals = to_numeric_series(df[k_col])
    recall = to_numeric_series(df[recall_col]) if recall_col else pd.Series(dtype=float)
    f1 = to_numeric_series(df[f1_col]) if f1_col else pd.Series(dtype=float)

    valid_recall = k_vals.notna() & recall.notna()
    valid_f1 = k_vals.notna() & f1.notna()

    if valid_recall.any():
        x = k_vals[valid_recall].tolist()
        y = recall[valid_recall].tolist()
        line_plot_with_points(
            x,
            y,
            "fig_sensitivity_k_recall.png",
            "Sensitivity: k vs Recall",
            "k",
            "Recall (test)",
            xticks=sorted(set(x)),
        )
    else:
        record_skip("fig_sensitivity_k_recall.png", "no recall data available")

    if valid_f1.any():
        x = k_vals[valid_f1].tolist()
        y = f1[valid_f1].tolist()
        line_plot_with_points(
            x,
            y,
            "fig_sensitivity_k_f1.png",
            "Sensitivity: k vs F1",
            "k",
            "F1 (test)",
            xticks=sorted(set(x)),
        )
    else:
        record_skip("fig_sensitivity_k_f1.png", "no f1 data available")


def plot_alpha_sensitivity() -> None:
    fig_names = ["fig_sensitivity_alpha_recall.png", "fig_sensitivity_alpha_f1.png"]
    df = load_dataframe(BASE_DIR / "outputs" / "results_density_alpha_sweep.csv", fig_names)
    if df is None:
        return

    alpha_col = find_column(df, ["density_alpha (β)", "density_alpha", "alpha"])
    recall_col = find_column(df, ["recall(test)", "recall", "recall (test)"])
    f1_col = find_column(df, ["f1(test)", "f1", "f1 (test)"])
    if alpha_col is None:
        for name in fig_names:
            record_skip(name, "missing density_alpha column")
        return

    alpha_vals = to_numeric_series(df[alpha_col])
    recall = to_numeric_series(df[recall_col]) if recall_col else pd.Series(dtype=float)
    f1 = to_numeric_series(df[f1_col]) if f1_col else pd.Series(dtype=float)

    valid_recall = alpha_vals.notna() & recall.notna()
    valid_f1 = alpha_vals.notna() & f1.notna()

    if valid_recall.any():
        x = alpha_vals[valid_recall].tolist()
        y = recall[valid_recall].tolist()
        line_plot_with_points(
            x,
            y,
            "fig_sensitivity_alpha_recall.png",
            "Sensitivity: density_alpha vs Recall",
            "density_alpha",
            "Recall (test)",
        )
    else:
        record_skip("fig_sensitivity_alpha_recall.png", "no recall data available")

    if valid_f1.any():
        x = alpha_vals[valid_f1].tolist()
        y = f1[valid_f1].tolist()
        line_plot_with_points(
            x,
            y,
            "fig_sensitivity_alpha_f1.png",
            "Sensitivity: density_alpha vs F1",
            "density_alpha",
            "F1 (test)",
        )
    else:
        record_skip("fig_sensitivity_alpha_f1.png", "no f1 data available")


def plot_multi_seed() -> None:
    fig_names = [
        "fig_seed_recall.png",
        "fig_seed_f1.png",
        "fig_seed_gmean.png",
        "fig_seed_threshold.png",
    ]
    df = load_dataframe(BASE_DIR / "outputs" / "results_multi_seed.csv", fig_names)
    if df is None:
        return

    seed_col = find_column(df, ["seed"])
    if seed_col is None:
        for name in fig_names:
            record_skip(name, "missing seed column")
        return

    seed_vals = to_numeric_series(df[seed_col])
    df_numeric = df.loc[seed_vals.notna()].copy()
    df_numeric[seed_col] = seed_vals.dropna().astype(int)
    df_numeric.sort_values(seed_col, inplace=True)

    def seed_plot(metric_candidates: List[str], filename: str, ylabel: str) -> None:
        metric_col = find_column(df_numeric, metric_candidates)
        if metric_col is None:
            record_skip(filename, f"missing metric columns {metric_candidates}")
            return

        y_vals = to_numeric_series(df_numeric[metric_col])
        seeds = df_numeric[seed_col]
        valid = y_vals.notna()
        if not valid.any():
            record_skip(filename, "no numeric seed data")
            return

        x = seeds[valid].tolist()
        y = y_vals[valid].tolist()
        mean_val = sum(y) / len(y)

        plt.figure()
        plt.plot(x, y, marker="o", color="#4c72b0", label="per-seed")
        plt.axhline(mean_val, color="#c44e52", linestyle="--", label=f"mean={mean_val:.3f}")
        plt.title(f"Robustness across random seeds – {ylabel}")
        plt.xlabel("seed")
        plt.ylabel(ylabel)
        plt.legend(loc="lower right")
        save_current_fig(filename, filename, y + [mean_val])

    seed_plot(["recall(test)", "recall", "recall (test)"], "fig_seed_recall.png", "Recall (test)")
    seed_plot(["f1(test)", "f1", "f1 (test)"], "fig_seed_f1.png", "F1 (test)")
    seed_plot(["gmean(test)", "gmean", "g-mean (test)", "g-mean(test)"], "fig_seed_gmean.png", "G-mean (test)")
    seed_plot(["best_threshold (val)", "best_threshold", "threshold"], "fig_seed_threshold.png", "best_threshold (val)")


def main() -> None:
    try:
        ensure_fig_dir()
        plot_effectiveness()
        plot_threshold_search_curves()
        plot_cw_sensitivity()
        plot_knn_sensitivity()
        plot_alpha_sensitivity()
        plot_multi_seed()
    finally:
        print("\n=== Summary ===")
        if generated:
            print("Generated:")
            for name in generated:
                print(f" - {name}")
        if skipped:
            print("Skipped:")
            for name, reason in skipped:
                print(f" - {name}: {reason}")
        if not generated and not skipped:
            print("Nothing to do.")


if __name__ == "__main__":
    main()
