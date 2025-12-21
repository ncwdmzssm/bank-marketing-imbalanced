#!/usr/bin/env python3
"""
Plot the Precision-Recall curve for seed=42 using outputs/pr_curves/pr_curve_seed42.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, Optional

try:
    import matplotlib.pyplot as plt
    import pandas as pd
except ImportError as exc:  # pragma: no cover - dependency guard
    print("Please install pandas and matplotlib before running this script (pip install pandas matplotlib).")
    print(exc)
    sys.exit(1)


BASE_DIR = Path(__file__).resolve().parent.parent
PR_FILE = BASE_DIR / "outputs" / "pr_curves" / "pr_curve_seed42.csv"
FIG_DIR = BASE_DIR / "outputs" / "figures"


def normalize_col(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def find_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    normalized = {normalize_col(col): col for col in df.columns}
    for cand in candidates:
        key = normalize_col(cand)
        if key in normalized:
            return normalized[key]
    return None


def main() -> None:
    if not PR_FILE.exists():
        print(f"[skip] missing file: {PR_FILE}")
        return

    try:
        df = pd.read_csv(PR_FILE)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[skip] failed to read {PR_FILE}: {exc}")
        return

    recall_col = find_column(df, ["recall", "recall(test)"])
    precision_col = find_column(df, ["precision", "prec"])

    if recall_col is None or precision_col is None:
        print("[skip] required columns (recall, precision) not found in CSV")
        return

    recall = pd.to_numeric(df[recall_col], errors="coerce")
    precision = pd.to_numeric(df[precision_col], errors="coerce")
    valid = recall.notna() & precision.notna()
    if not valid.any():
        print("[skip] no numeric precision/recall values to plot")
        return

    recall, precision = recall[valid], precision[valid]

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision, marker="o", color="#4c72b0", linewidth=2)
    plt.title("Precision-Recall curve (seed=42)", fontsize=15)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid(True, linestyle="--", alpha=0.6)
    if recall.min() >= 0 and recall.max() <= 1:
        plt.xlim(0, 1)
    if precision.min() >= 0 and precision.max() <= 1:
        plt.ylim(0, 1)

    out_path = FIG_DIR / "fig_pr_curve_seed42.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[saved] PR curve for seed=42 -> {out_path}")


if __name__ == "__main__":
    main()
