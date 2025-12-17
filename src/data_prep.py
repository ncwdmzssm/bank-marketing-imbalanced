# data_prep.py
# -*- coding: utf-8 -*-

"""
Data preparation for UCI Bank Marketing dataset.

- One-Hot Encoding for categorical features
- Standardization for numerical features
- Train / Validation / Test split: 60 / 20 / 20
- Freeze data interface as .npy files
"""

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# -----------------------
# Config
# -----------------------

RAW_DATA_PATH = "data/raw/bank-full.csv"
PROCESSED_DIR = "data/processed"

RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.25  # 0.25 * 0.8 = 0.2


# -----------------------
# Main
# -----------------------

def main():
    # 1. Load raw data
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(
            f"Cannot find {RAW_DATA_PATH}. Please put bank-full.csv under data/raw/"
        )

    df = pd.read_csv(RAW_DATA_PATH, sep=";")
    print(f"[INFO] Raw data shape: {df.shape}")

    # 2. Separate features and target
    y = df["y"].map({"yes": 1, "no": 0}).astype(int)
    X = df.drop(columns=["y"])

    # 3. Identify feature types
    categorical_features = [
        "job", "marital", "education", "default",
        "housing", "loan", "contact", "month", "poutcome"
    ]

    numerical_features = [
        "age", "balance", "day", "duration",
        "campaign", "pdays", "previous"
    ]

    # 4. Preprocessing pipeline
    categorical_transformer = OneHotEncoder(
        drop=None,
        sparse_output=False,  # 修改此处
        handle_unknown="ignore"
    )

    numerical_transformer = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop"
    )

    # 5. Train / Test split (first split)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # 6. Train / Validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_train_val
    )

    print(f"[INFO] Train size: {X_train.shape[0]}")
    print(f"[INFO] Val size:   {X_val.shape[0]}")
    print(f"[INFO] Test size:  {X_test.shape[0]}")

    # 7. Fit preprocessing ONLY on training data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)

    print(f"[INFO] Processed feature dimension: {X_train_processed.shape[1]}")

    # 8. Create output directory
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # 9. Save frozen interfaces
    np.save(os.path.join(PROCESSED_DIR, "X_train.npy"), X_train_processed)
    np.save(os.path.join(PROCESSED_DIR, "X_val.npy"), X_val_processed)
    np.save(os.path.join(PROCESSED_DIR, "X_test.npy"), X_test_processed)

    np.save(os.path.join(PROCESSED_DIR, "y_train.npy"), y_train.values)
    np.save(os.path.join(PROCESSED_DIR, "y_val.npy"), y_val.values)
    np.save(os.path.join(PROCESSED_DIR, "y_test.npy"), y_test.values)

    print("[SUCCESS] Data preprocessing completed.")
    print("[SUCCESS] Frozen data interface saved to data/processed/")

    # Optional: print class distribution
    print("\n[INFO] Class distribution:")
    print("Train:", np.bincount(y_train))
    print("Val:  ", np.bincount(y_val))
    print("Test: ", np.bincount(y_test))


if __name__ == "__main__":
    main()
