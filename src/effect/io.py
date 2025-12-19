"""Frozen data interface for loading preprocessed splits."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np


def load_processed(processed_dir: str | os.PathLike = "data/processed"):
    """
    Load preprocessed datasets saved by data_prep.py.

    Returns
    -------
    tuple
        X_train, X_val, X_test, y_train, y_val, y_test as numpy arrays.
    """
    base = Path(processed_dir)
    required_files = {
        "X_train": base / "X_train.npy",
        "X_val": base / "X_val.npy",
        "X_test": base / "X_test.npy",
        "y_train": base / "y_train.npy",
        "y_val": base / "y_val.npy",
        "y_test": base / "y_test.npy",
    }

    for name, path in required_files.items():
        if not path.exists():
            raise FileNotFoundError(f"{path} 不存在，请先运行 data_prep.py")

    X_train = np.load(required_files["X_train"], allow_pickle=False)
    X_val = np.load(required_files["X_val"], allow_pickle=False)
    X_test = np.load(required_files["X_test"], allow_pickle=False)
    y_train = np.load(required_files["y_train"], allow_pickle=False)
    y_val = np.load(required_files["y_val"], allow_pickle=False)
    y_test = np.load(required_files["y_test"], allow_pickle=False)

    X_list = [("X_train", X_train), ("X_val", X_val), ("X_test", X_test)]
    y_list = [("y_train", y_train), ("y_val", y_val), ("y_test", y_test)]

    # Validate shapes: X 2D, y 1D, and matching lengths.
    for name, arr in X_list:
        if arr.ndim != 2:
            raise ValueError(f"{name} 应为二维数组，实际 ndim={arr.ndim}")
    for name, arr in y_list:
        if arr.ndim != 1:
            raise ValueError(f"{name} 应为一维数组，实际 ndim={arr.ndim}")

    if not all(X.shape[0] == y.shape[0] for (_, X), (_, y) in zip(X_list, y_list)):
        raise ValueError("X 与 y 的样本数量不匹配")

    # Validate binary labels and convert to int.
    for name, arr in y_list:
        unique = np.unique(arr)
        if not set(unique.tolist()).issubset({0, 1}):
            raise ValueError(f"{name} 仅应包含 {0,1}，实际为 {unique}")

    y_train = y_train.astype(int, copy=False)
    y_val = y_val.astype(int, copy=False)
    y_test = y_test.astype(int, copy=False)

    return X_train, X_val, X_test, y_train, y_val, y_test

