# Data Preprocessing Module
"""
Reusable sklearn preprocessing pipeline for the Telco Customer Churn project.

Responsibilities
----------------
* OneHotEncode categorical features.
* StandardScale numeric features.
* Wrap everything in a :class:`~sklearn.compose.ColumnTransformer` so that
  a single fitted object can be serialised / deserialised with **joblib**.

**No model training happens here.**
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from logger import logger
from exception import CustomException


# ------------------------------------------------------------------ #
#  Column definitions – sensible defaults for the Telco Churn dataset
#  after DataIngestion + feature_engineering have been applied.
# ------------------------------------------------------------------ #

#: Columns to drop before any transformation (IDs, target, etc.)
DROP_COLUMNS: list[str] = ["customerID"]

#: Target column name
TARGET_COLUMN: str = "Churn"

#: Default numeric features (override via function args if needed)
DEFAULT_NUMERIC_FEATURES: list[str] = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "SeniorCitizen",
    "total_services",
    "CLV",
]

#: Default categorical features
DEFAULT_CATEGORICAL_FEATURES: list[str] = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "tenure_bucket",
]


# ------------------------------------------------------------------ #
#  Pipeline construction
# ------------------------------------------------------------------ #


def build_pipeline(
    numeric_features: Optional[list[str]] = None,
    categorical_features: Optional[list[str]] = None,
) -> ColumnTransformer:
    """
    Build (but do **not** fit) a :class:`ColumnTransformer` pipeline.

    Parameters
    ----------
    numeric_features : list[str], optional
        Columns to StandardScale.  Falls back to
        :data:`DEFAULT_NUMERIC_FEATURES`.
    categorical_features : list[str], optional
        Columns to OneHotEncode.  Falls back to
        :data:`DEFAULT_CATEGORICAL_FEATURES`.

    Returns
    -------
    ColumnTransformer
        Unfitted transformer ready for ``fit`` / ``fit_transform``.
    """
    try:
        numeric_features = numeric_features or DEFAULT_NUMERIC_FEATURES
        categorical_features = categorical_features or DEFAULT_CATEGORICAL_FEATURES

        numeric_pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
            ]
        )

        categorical_pipeline = Pipeline(
            steps=[
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False,
                        drop="first",      # avoid multicollinearity
                    ),
                ),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, numeric_features),
                ("cat", categorical_pipeline, categorical_features),
            ],
            remainder="drop",               # drop unlisted columns
            verbose_feature_names_out=True,
        )

        logger.info(
            "Preprocessing pipeline built – %d numeric, %d categorical features.",
            len(numeric_features),
            len(categorical_features),
        )
        return preprocessor

    except Exception as exc:
        raise CustomException(exc, sys) from exc


# ------------------------------------------------------------------ #
#  Helpers – prepare X / y from a raw DataFrame
# ------------------------------------------------------------------ #


def _prepare_dataframe(
    df: pd.DataFrame,
    target_col: str = TARGET_COLUMN,
    drop_cols: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    Separate features and target, dropping non-feature columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe (post feature-engineering).
    target_col : str
        Name of the target column.  If absent, *y* is ``None``.
    drop_cols : list[str], optional
        Extra columns to drop (e.g. ``customerID``).

    Returns
    -------
    tuple[pd.DataFrame, pd.Series | None]
        ``(X, y)`` – *y* is ``None`` when the target column is missing.
    """
    df = df.copy()
    drop_cols = drop_cols or DROP_COLUMNS

    # Drop ID / helper columns
    existing_drops = [c for c in drop_cols if c in df.columns]
    if existing_drops:
        df = df.drop(columns=existing_drops)

    # Separate target
    y: Optional[pd.Series] = None
    if target_col in df.columns:
        y = df[target_col].map({"Yes": 1, "No": 0}).astype(int)
        df = df.drop(columns=[target_col])

    return df, y


# ------------------------------------------------------------------ #
#  Fit / Transform public API
# ------------------------------------------------------------------ #


def fit_transform(
    df: pd.DataFrame,
    preprocessor: Optional[ColumnTransformer] = None,
    numeric_features: Optional[list[str]] = None,
    categorical_features: Optional[list[str]] = None,
    target_col: str = TARGET_COLUMN,
    drop_cols: Optional[list[str]] = None,
) -> tuple[np.ndarray, Optional[pd.Series], ColumnTransformer]:
    """
    Fit the preprocessing pipeline on *df* **and** return transformed data.

    Parameters
    ----------
    df : pd.DataFrame
        Training dataframe (post feature-engineering).
    preprocessor : ColumnTransformer, optional
        Pre-built pipeline.  A new one is created when ``None``.
    numeric_features, categorical_features : list[str], optional
        Forwarded to :func:`build_pipeline` when *preprocessor* is ``None``.
    target_col : str
        Target column name.
    drop_cols : list[str], optional
        Columns to drop before transforming.

    Returns
    -------
    tuple[np.ndarray, pd.Series | None, ColumnTransformer]
        ``(X_transformed, y, fitted_preprocessor)``
    """
    try:
        logger.info("fit_transform – preparing dataframe.")
        X, y = _prepare_dataframe(df, target_col=target_col, drop_cols=drop_cols)

        if preprocessor is None:
            # Auto-detect columns that actually exist in the dataframe
            num_feats = numeric_features or [
                c for c in DEFAULT_NUMERIC_FEATURES if c in X.columns
            ]
            cat_feats = categorical_features or [
                c for c in DEFAULT_CATEGORICAL_FEATURES if c in X.columns
            ]
            preprocessor = build_pipeline(
                numeric_features=num_feats,
                categorical_features=cat_feats,
            )

        X_transformed: np.ndarray = preprocessor.fit_transform(X)

        logger.info(
            "fit_transform complete – output shape: %s",
            X_transformed.shape,
        )
        return X_transformed, y, preprocessor

    except CustomException:
        raise
    except Exception as exc:
        raise CustomException(exc, sys) from exc


def transform(
    df: pd.DataFrame,
    preprocessor: ColumnTransformer,
    target_col: str = TARGET_COLUMN,
    drop_cols: Optional[list[str]] = None,
) -> tuple[np.ndarray, Optional[pd.Series]]:
    """
    Transform *df* using an **already fitted** preprocessor.

    Parameters
    ----------
    df : pd.DataFrame
        Unseen / test dataframe.
    preprocessor : ColumnTransformer
        A **fitted** ``ColumnTransformer`` (from :func:`fit_transform`).
    target_col : str
        Target column name.
    drop_cols : list[str], optional
        Columns to drop before transforming.

    Returns
    -------
    tuple[np.ndarray, pd.Series | None]
        ``(X_transformed, y)``
    """
    try:
        logger.info("transform – preparing dataframe.")
        X, y = _prepare_dataframe(df, target_col=target_col, drop_cols=drop_cols)

        X_transformed: np.ndarray = preprocessor.transform(X)

        logger.info(
            "transform complete – output shape: %s",
            X_transformed.shape,
        )
        return X_transformed, y

    except CustomException:
        raise
    except Exception as exc:
        raise CustomException(exc, sys) from exc


# ------------------------------------------------------------------ #
#  Serialisation helpers (joblib)
# ------------------------------------------------------------------ #


def save_pipeline(preprocessor: ColumnTransformer, path: str | Path) -> Path:
    """
    Persist a fitted ``ColumnTransformer`` to disk using joblib.

    Parameters
    ----------
    preprocessor : ColumnTransformer
        Fitted transformer.
    path : str | Path
        Destination file path (e.g. ``"artifacts/preprocessor.joblib"``).

    Returns
    -------
    Path
        Resolved path that was written.
    """
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(preprocessor, path)
        logger.info("Pipeline saved to '%s'.", path)
        return path
    except Exception as exc:
        raise CustomException(exc, sys) from exc


def load_pipeline(path: str | Path) -> ColumnTransformer:
    """
    Load a fitted ``ColumnTransformer`` from disk.

    Parameters
    ----------
    path : str | Path
        Path to a ``.joblib`` file produced by :func:`save_pipeline`.

    Returns
    -------
    ColumnTransformer
        The deserialised, fitted transformer.
    """
    try:
        path = Path(path)
        preprocessor: ColumnTransformer = joblib.load(path)
        logger.info("Pipeline loaded from '%s'.", path)
        return preprocessor
    except Exception as exc:
        raise CustomException(exc, sys) from exc


# ------------------------------------------------------------------ #
#  Feature-name introspection
# ------------------------------------------------------------------ #


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """
    Return the output feature names from a **fitted** preprocessor.

    Parameters
    ----------
    preprocessor : ColumnTransformer
        Must already be fitted.

    Returns
    -------
    list[str]
        Ordered list of transformed column names.
    """
    try:
        names = list(preprocessor.get_feature_names_out())
        logger.info("Feature names extracted – %d features.", len(names))
        return names
    except Exception as exc:
        raise CustomException(exc, sys) from exc
