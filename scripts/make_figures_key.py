#!/usr/bin/env python3
"""
Generate a compact set of key PPT figures (A–E) for the bank marketing project.
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:
    import matplotlib.pyplot as plt
    import pandas as pd
except ImportError as exc:  # pragma: no cover - dependency guard
    print("Please install pandas and matplotlib before running this script (pip install pandas matplotlib).")
    print(exc)
    sys.exit(1)


plt.rcParams.update(
    {
        "axes.titlesize": 15,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.facecolor": "white",
    }
)

BASE_DIR = Path(__file__).resolve().parent.parent
FIG_DIR = BASE_DIR / "outputs" / "figures"

generated: List[str] = []
skipped: List[Tuple[str, str]] = []


def ensure_fig_dir() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def normalize_col(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def find_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    normalized = {normalize_col(col): col for col in df.columns}
    for cand in candidates:
        key = normalize_col(cand)
        if key in normalized:
            return normalized[key]
    return None


def record_skip(name: str, reason: str) -> None:
    print(f"[skip] {name}: {reason}")
    skipped.append((name, reason))


def record_success(name: str, path: Path) -> None:
    print(f"[saved] {name}: {path}")
    generated.append(name)


def adjust_ylim(ax: plt.Axes, data: Iterable[float], emphasize: bool = False) -> None:
    data = list(data)
    if not data:
        return
    dmin, dmax = min(data), max(data)
    if dmin >= 0 and dmax <= 1.05:
        if emphasize:
            span = max(dmax - dmin, 0.02)
            lower = max(0.0, dmin - 0.25 * span - 0.02)
            upper = min(1.1, dmax + 0.35 * span + 0.02)
            if upper - lower < 0.1:
                upper = min(1.1, lower + 0.1)
            ax.set_ylim(lower, upper)
        else:
            upper = min(1.1, max(1.0, dmax + 0.1))
            ax.set_ylim(0, upper)
    else:
        padding = (dmax - dmin) * 0.1 if dmax != dmin else 0.1
        ax.set_ylim(dmin - padding, dmax + padding)


def annotate_missing(ax: plt.Axes, message: str) -> None:
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=12, color="#666666", transform=ax.transAxes)
    ax.set_axis_off()


def add_value_labels(ax: plt.Axes, bars: Iterable[plt.Artist]) -> None:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.01, f"{height:.3f}", ha="center", va="bottom", fontsize=10)


def get_method_rows(df: pd.DataFrame, method_col: str) -> Dict[str, pd.Series]:
    def first_match(include: List[str], exclude: List[str] | None = None) -> Optional[pd.Series]:
        mask = pd.Series(True, index=df.index)
        for kw in include:
            mask &= df[method_col].str.contains(kw, case=False, na=False)
        if exclude:
            for kw in exclude:
                mask &= ~df[method_col].str.contains(kw, case=False, na=False)
        match = df.loc[mask]
        return match.iloc[0] if not match.empty else None

    density_row = first_match(["density"])
    if density_row is None:
        density_row = first_match(["plus_density"])

    return {
        "Baseline": first_match(["baseline"]),
        "Cost-sensitive": first_match(["cost_sensitive"], ["density"]),
        "Cost+Density": density_row,
    }


BAR_COLORS = {
    "Baseline": "#8c8c8c",
    "Cost-sensitive": "#4c8eda",
    "Cost+Density": "#e67e22",
}


def save_fig(fig: plt.Figure, filename: str, data_for_ylim: Iterable[float] | None = None) -> None:
    ensure_fig_dir()
    path = FIG_DIR / filename
    if data_for_ylim is not None:
        adjust_ylim(plt.gca(), data_for_ylim)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    record_success(filename, path)


def plot_figure_a_effect_overview() -> None:
    filename = "fig_A_effect_overview.png"
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle("Effectiveness: proposed method improves minority performance under fair tuning", fontsize=16)

    df_path = BASE_DIR / "outputs" / "results_effect.csv"
    if not df_path.exists():
        for ax in axes.ravel():
            annotate_missing(ax, f"missing {df_path.name}")
        save_fig(fig, filename)
        return

    try:
        df = pd.read_csv(df_path)
    except Exception as exc:  # pragma: no cover - defensive
        for ax in axes.ravel():
            annotate_missing(ax, f"failed to read {df_path.name}\n{exc}")
        save_fig(fig, filename)
        return

    method_col = find_column(df, ["method"])
    if method_col is None:
        for ax in axes.ravel():
            annotate_missing(ax, "missing method column")
        save_fig(fig, filename)
        return

    rows = get_method_rows(df, method_col)

    def bar_on_axis(ax: plt.Axes, metric_candidates: List[str], title: str, ylabel: str) -> None:
        col = find_column(df, metric_candidates)
        if col is None:
            annotate_missing(ax, "missing metric column")
            ax.set_title(title)
            return

        labels, values = [], []
        for label, series in rows.items():
            if series is None:
                continue
            val = pd.to_numeric(series.get(col, float("nan")), errors="coerce")
            if pd.notna(val):
                labels.append(label)
                values.append(val)

        if not values:
            annotate_missing(ax, "no matching methods")
            ax.set_title(title)
            return

        colors = [BAR_COLORS.get(label, "#4c72b0") for label in labels]
        bars = ax.bar(labels, values, color=colors, alpha=0.9, edgecolor="white")

        # Highlight the best-performing bar
        best_idx = max(range(len(values)), key=lambda i: values[i])
        bars[best_idx].set_edgecolor("#000000")
        bars[best_idx].set_linewidth(2)

        add_value_labels(ax, bars)

        # Annotate delta vs Baseline to make contrast clearer
        if "Baseline" in labels:
            base_idx = labels.index("Baseline")
            baseline_val = values[base_idx]
            for lbl, bar, val in zip(labels, bars, values):
                if lbl == "Baseline":
                    continue
                delta = val - baseline_val
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.05,
                    f"{delta:+.3f} vs base",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color="#c44e52" if delta >= 0 else "#c44e52",
                )

        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, linestyle="--", alpha=0.6, axis="y")
        adjust_ylim(ax, values, emphasize=True)

    bar_on_axis(axes[0, 0], ["recall(test)", "recall"], "Recall (test)", "Recall")
    bar_on_axis(axes[0, 1], ["f1(test)", "f1"], "F1 (test)", "F1")
    bar_on_axis(axes[1, 0], ["gmean(test)", "gmean", "g-mean(test)"], "G-mean (test)", "G-mean")

    threshold_col = find_column(df, ["threshold", "best_threshold", "best_threshold (val)"])
    fallback_col = find_column(df, ["train_time_s", "pred_time_s"])
    def_col = threshold_col or fallback_col
    title = "Threshold (val)" if threshold_col else ("Runtime" if fallback_col else "No threshold/runtime column")
    if def_col is None:
        annotate_missing(axes[1, 1], "no threshold/train_time/pred_time")
        axes[1, 1].set_title(title)
    else:
        labels, values = [], []
        for label, series in rows.items():
            if series is None:
                continue
            val = pd.to_numeric(series.get(def_col, float("nan")), errors="coerce")
            if pd.notna(val):
                labels.append(label)
                values.append(val)
        if not values:
            annotate_missing(axes[1, 1], "no numeric values")
        else:
            bars = axes[1, 1].bar(labels, values, color="#937860", alpha=0.85)
            add_value_labels(axes[1, 1], bars)
            axes[1, 1].set_ylabel(title)
            adjust_ylim(axes[1, 1], values, emphasize=True)
        axes[1, 1].set_title(title)
        axes[1, 1].grid(True, linestyle="--", alpha=0.6, axis="y")

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_fig(fig, filename)


def method_label_from_filename(path: Path) -> str:
    name = path.stem.lower()
    if "plus" in name or "density" in name:
        return "Cost+Density"
    if "cost_sensitive" in name:
        return "Cost-sensitive"
    if "baseline" in name:
        return "Baseline"
    return path.stem


def plot_figure_b_threshold_search() -> None:
    filename = "fig_B_threshold_search_compare.png"
    patterns = [
        BASE_DIR / "outputs" / "threshold_search_*.csv",
        BASE_DIR / "outputs" / "threshold_search_csv_results" / "threshold_search_*.csv",
    ]
    csv_files: List[str] = []
    for pattern in patterns:
        csv_files.extend(glob.glob(str(pattern)))

    if not csv_files:
        record_skip(filename, "no threshold_search_*.csv files found")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    plotted = False
    offsets = [(0, 12), (0, -14), (14, 12), (-14, 10), (14, -12), (-12, -10)]
    all_y: List[float] = []

    for idx, csv_path in enumerate(csv_files):
        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[warn] failed to read {csv_path}: {exc}")
            continue

        x_col = find_column(df, ["threshold"])
        y_col = find_column(df, ["f1", "recall", "objective_score", "objective", "score"])
        if x_col is None or y_col is None:
            print(f"[warn] missing threshold or score columns in {csv_path}")
            continue

        x = pd.to_numeric(df[x_col], errors="coerce")
        y = pd.to_numeric(df[y_col], errors="coerce")
        valid = x.notna() & y.notna()
        if not valid.any():
            print(f"[warn] no numeric data in {csv_path}")
            continue

        x, y = x[valid], y[valid]
        label = method_label_from_filename(Path(csv_path))
        ax.plot(x, y, marker="o", label=label)
        all_y.extend(y.tolist())

        best_idx = y.idxmax()
        ax.scatter([x.loc[best_idx]], [y.loc[best_idx]], color="#c44e52", zorder=5)
        off = offsets[idx % len(offsets)]
        ax.annotate(
            f"best t={x.loc[best_idx]:.3f}, score={y.loc[best_idx]:.3f}",
            (x.loc[best_idx], y.loc[best_idx]),
            textcoords="offset points",
            xytext=off,
            ha="center",
            fontsize=10,
            )
        plotted = True

    if not plotted:
        record_skip(filename, "no usable threshold search data")
        plt.close(fig)
        return

    ax.set_title("Decision threshold tuning on validation set (same protocol for all methods)", fontsize=15)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score (F1 or objective)")
    ax.grid(True, linestyle="--", alpha=0.6)
    adjust_ylim(ax, all_y, emphasize=True)
    ax.legend(loc="best")
    fig.tight_layout()
    save_fig(fig, filename)


def line_with_markers(ax: plt.Axes, x: List[float], y: List[float], xlabel: str, ylabel: str, label: Optional[str] = None) -> None:
    ax.plot(x, y, marker="o", color="#4c72b0", label=label if label else None)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.6)
    adjust_ylim(ax, y)
    if label:
        ax.legend(loc="best")


def plot_sensitivity_subplot(
    ax: plt.Axes,
    csv_path: Path,
    x_candidates: List[str],
    title: str,
    baseline_recall: Optional[float],
) -> None:
    if not csv_path.exists():
        annotate_missing(ax, f"missing {csv_path.name}")
        ax.set_title(title)
        return
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:  # pragma: no cover - defensive
        annotate_missing(ax, f"failed to read {csv_path.name}\n{exc}")
        ax.set_title(title)
        return

    x_col = find_column(df, x_candidates)
    y_col = find_column(df, ["recall(test)", "recall", "recall (test)"])
    if x_col is None or y_col is None:
        annotate_missing(ax, "missing x or recall column")
        ax.set_title(title)
        return

    x = pd.to_numeric(df[x_col], errors="coerce")
    y = pd.to_numeric(df[y_col], errors="coerce")
    valid = x.notna() & y.notna()
    if not valid.any():
        annotate_missing(ax, "no numeric data")
        ax.set_title(title)
        return

    x_vals = x[valid].tolist()
    y_vals = y[valid].tolist()
    line_with_markers(ax, x_vals, y_vals, x_col, "Recall (test)")
    if baseline_recall is not None:
        ax.axhline(baseline_recall, color="#c44e52", linestyle="--", label=f"Baseline recall={baseline_recall:.3f}")
        ax.legend(loc="best")

    ax.set_title(title)
    if title.startswith("Cost") and y_vals:
        max_idx = max(range(len(y_vals)), key=lambda i: y_vals[i])
        ax.annotate(
            f"{x_col}={x_vals[max_idx]}, recall={y_vals[max_idx]:.3f}",
            xy=(x_vals[max_idx], y_vals[max_idx]),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            color="#c44e52",
            fontsize=10,
        )


def plot_figure_c_sensitivity(baseline_recall: Optional[float]) -> None:
    filename = "fig_C_sensitivity_overview.png"
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle("Robustness: improvement persists across reasonable parameter ranges", fontsize=16)

    plot_sensitivity_subplot(
        axes[0],
        BASE_DIR / "outputs" / "results_cw_sweep.csv",
        ["pos_class_weight (α)", "pos_class_weight", "cw"],
        "Cost strength (pos_class_weight)",
        baseline_recall,
    )
    plot_sensitivity_subplot(
        axes[1],
        BASE_DIR / "outputs" / "results_knn_sweep.csv",
        ["knn_k", "k"],
        "Neighborhood size (k)",
        baseline_recall,
    )
    plot_sensitivity_subplot(
        axes[2],
        BASE_DIR / "outputs" / "results_density_alpha_sweep.csv",
        ["density_alpha (β)", "density_alpha", "alpha"],
        "Density strength (alpha)",
        baseline_recall,
    )

    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    save_fig(fig, filename)


def plot_seed_metric(ax: plt.Axes, seeds: List[int], values: List[float], title: str, ylabel: str) -> None:
    ax.plot(seeds, values, marker="o", color="#4c72b0", label="per-seed")
    mean_val = sum(values) / len(values)
    ax.axhline(mean_val, color="#c44e52", linestyle="--", label=f"mean={mean_val:.3f}")
    ax.text(0.02, 0.9, f"mean={mean_val:.3f}", transform=ax.transAxes, fontsize=10, color="#c44e52")
    ax.set_title(title)
    ax.set_xlabel("seed")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="best")
    adjust_ylim(ax, values + [mean_val])


def plot_figure_d_seed_robustness() -> None:
    filename = "fig_D_seed_robustness.png"
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle("Robustness: consistent results across random seeds", fontsize=16)
    df_path = BASE_DIR / "outputs" / "results_multi_seed.csv"

    if not df_path.exists():
        for ax in axes.ravel():
            annotate_missing(ax, f"missing {df_path.name}")
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_fig(fig, filename)
        return

    try:
        df = pd.read_csv(df_path)
    except Exception as exc:  # pragma: no cover - defensive
        for ax in axes.ravel():
            annotate_missing(ax, f"failed to read {df_path.name}\n{exc}")
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_fig(fig, filename)
        return

    seed_col = find_column(df, ["seed"])
    if seed_col is None:
        for ax in axes.ravel():
            annotate_missing(ax, "missing seed column")
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_fig(fig, filename)
        return

    seed_vals = pd.to_numeric(df[seed_col], errors="coerce")
    df_num = df.loc[seed_vals.notna()].copy()
    if df_num.empty:
        for ax in axes.ravel():
            annotate_missing(ax, "no numeric seed rows")
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_fig(fig, filename)
        return

    df_num[seed_col] = seed_vals.dropna().astype(int)
    df_num.sort_values(seed_col, inplace=True)

    def plot_metric(ax: plt.Axes, candidates: List[str], title: str, ylabel: str) -> None:
        metric_col = find_column(df_num, candidates)
        if metric_col is None:
            annotate_missing(ax, "metric column missing")
            ax.set_title(title)
            return
        vals = pd.to_numeric(df_num[metric_col], errors="coerce")
        valid = vals.notna()
        if not valid.any():
            annotate_missing(ax, "no numeric values")
            ax.set_title(title)
            return
        seeds = df_num.loc[valid, seed_col].tolist()
        values = vals[valid].tolist()
        plot_seed_metric(ax, seeds, values, title, ylabel)

    plot_metric(axes[0, 0], ["recall(test)", "recall", "recall (test)"], "Recall vs seed", "Recall (test)")
    plot_metric(axes[0, 1], ["f1(test)", "f1", "f1 (test)"], "F1 vs seed", "F1 (test)")
    plot_metric(axes[1, 0], ["gmean(test)", "gmean", "g-mean (test)", "g-mean(test)"], "G-mean vs seed", "G-mean (test)")
    plot_metric(axes[1, 1], ["best_threshold (val)", "best_threshold", "threshold"], "best_threshold(val) vs seed", "best_threshold (val)")

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_fig(fig, filename)


def plot_figure_e_takeaway(baseline_recall: Optional[float], proposed_recall: Optional[float]) -> None:
    filename = "fig_E_takeaway.png"
    if baseline_recall is None or proposed_recall is None:
        record_skip(filename, "baseline or proposed recall not available")
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["Baseline tuned", "Proposed tuned"]
    values = [baseline_recall, proposed_recall]
    bars = ax.bar(labels, values, color=["#4c72b0", "#55a868"], alpha=0.85)
    add_value_labels(ax, bars)
    ax.set_ylabel("Recall (test)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)
    adjust_ylim(ax, values)
    ax.set_title(f"Key takeaway: Recall improves from {baseline_recall:.3f} to {proposed_recall:.3f}", fontsize=16)
    fig.tight_layout()
    save_fig(fig, filename)


def extract_baseline_and_proposed(df: pd.DataFrame, method_col: str) -> Tuple[Optional[float], Optional[float]]:
    rows = get_method_rows(df, method_col)
    recall_col = find_column(df, ["recall(test)", "recall"])
    baseline = proposed = None
    if recall_col:
        if rows.get("Baseline") is not None:
            baseline = pd.to_numeric(rows["Baseline"].get(recall_col, float("nan")), errors="coerce")
        if rows.get("Cost+Density") is not None:
            proposed = pd.to_numeric(rows["Cost+Density"].get(recall_col, float("nan")), errors="coerce")
        elif rows.get("Cost-sensitive") is not None:
            proposed = pd.to_numeric(rows["Cost-sensitive"].get(recall_col, float("nan")), errors="coerce")
    return (
        baseline if baseline is not None and pd.notna(baseline) else None,
        proposed if proposed is not None and pd.notna(proposed) else None,
    )


def main() -> None:
    ensure_fig_dir()

    baseline_recall = None
    proposed_recall = None
    effect_path = BASE_DIR / "outputs" / "results_effect.csv"
    if effect_path.exists():
        try:
            effect_df = pd.read_csv(effect_path)
            method_col = find_column(effect_df, ["method"])
            if method_col:
                baseline_recall, proposed_recall = extract_baseline_and_proposed(effect_df, method_col)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[warn] could not parse baseline/proposed from results_effect.csv: {exc}")

    plot_figure_a_effect_overview()
    plot_figure_b_threshold_search()
    plot_figure_c_sensitivity(baseline_recall)
    plot_figure_d_seed_robustness()
    plot_figure_e_takeaway(baseline_recall, proposed_recall)

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
        print("Nothing generated.")


if __name__ == "__main__":
    main()
