"""
Preprocessing pipeline untuk model prediksi inaktivitas.

Prinsip:
- Scaler, imputer, semua transformer HANYA difit pada training data.
- Pipeline di-persist untuk diaplikasikan ke test data identically.
- Kolom ipt_mean dan ipt_std memiliki 32% missing (pelanggan 1 transaksi).
  Diisi dengan median training set, bukan global median.
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from typing import Tuple
import joblib
from pathlib import Path


FEATURE_COLUMNS = [
    "recency_days",
    "frequency",
    "monetary_total",
    "monetary_mean",
    "monetary_std",
    "avg_qty_per_order",
    "n_unique_products",
    "n_unique_months",
    "tenure_days",
    "ipt_mean",        # 32% missing — diimpute dengan median
    "ipt_std",         # 32% missing — diimpute dengan median
    "recent_freq_ratio",
    "spending_trend",
    "cancellation_rate",
    "country_UK",
]

LOG_TRANSFORM_COLS = [
    "monetary_total",
    "monetary_mean",
    "monetary_std",
    "avg_qty_per_order",
    "n_unique_products",
    "tenure_days",
    "ipt_mean",
    "ipt_std",
]


def apply_log_transform(df: pd.DataFrame) -> pd.DataFrame:
    """Terapkan log1p pada kolom yang sangat skewed."""
    df = df.copy()
    for col in LOG_TRANSFORM_COLS:
        if col in df.columns:
            df[col] = np.log1p(df[col].clip(lower=0))
    return df


def build_preprocessing_pipeline() -> Pipeline:
    """
    Bangun sklearn Pipeline: impute missing values + standard scaling.
    Pipeline ini harus difit HANYA pada training data.
    """
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    return pipeline


def fit_transform_train(
    X_train: pd.DataFrame,
) -> Tuple[np.ndarray, Pipeline]:
    """
    Fit pipeline pada training data dan transformasi.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.

    Returns
    -------
    Tuple[np.ndarray, Pipeline]
        - X_train_transformed: array yang sudah diproses
        - fitted_pipeline: pipeline yang sudah difit (untuk transform test)
    """
    X_train = X_train[FEATURE_COLUMNS].copy()
    X_train = apply_log_transform(X_train)

    pipeline = build_preprocessing_pipeline()
    X_transformed = pipeline.fit_transform(X_train)

    return X_transformed, pipeline


def transform_test(
    X_test: pd.DataFrame,
    fitted_pipeline: Pipeline,
) -> np.ndarray:
    """
    Terapkan pipeline yang sudah difit ke test data.
    TIDAK ada fitting ulang di sini.
    """
    X_test = X_test[FEATURE_COLUMNS].copy()
    X_test = apply_log_transform(X_test)
    return fitted_pipeline.transform(X_test)


def save_pipeline(pipeline: Pipeline, path: str):
    joblib.dump(pipeline, path)
    print(f"Pipeline disimpan: {path}")


def load_pipeline(path: str) -> Pipeline:
    return joblib.load(path)
