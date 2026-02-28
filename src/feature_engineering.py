import sys
import pandas as pd
import numpy as np

from logger import logger
from exception import CustomException


# ------------------------------------------------------------------ #
#  Service columns used to compute total_services
# ------------------------------------------------------------------ #
SERVICE_COLUMNS: list[str] = [
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

# ------------------------------------------------------------------ #
#  Tenure bucket definitions  (label, lower_bound_inclusive, upper_bound_inclusive)
# ------------------------------------------------------------------ #
TENURE_BINS: list[int] = [0, 12, 24, 36, 48, 60, 72]
TENURE_LABELS: list[str] = [
    "0-12",
    "13-24",
    "25-36",
    "37-48",
    "49-60",
    "61-72",
]


# ------------------------------------------------------------------ #
#  Public API – pure functions, no model logic
# ------------------------------------------------------------------ #


def add_tenure_buckets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bin the *tenure* column into categorical buckets.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a numeric ``tenure`` column.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with a new ``tenure_bucket`` column.
    """
    try:
        logger.info("Creating tenure buckets.")
        df = df.copy()
        df["tenure_bucket"] = pd.cut(
            df["tenure"],
            bins=TENURE_BINS,
            labels=TENURE_LABELS,
            include_lowest=True,
        )
        logger.info("tenure_bucket distribution:\n%s", df["tenure_bucket"].value_counts().to_string())
        return df
    except Exception as exc:
        raise CustomException(exc, sys) from exc


def add_total_services(df: pd.DataFrame) -> pd.DataFrame:
    """
    Count the number of services each customer has subscribed to.

    A service is counted as subscribed when its value is **not**
    ``"No"`` and **not** ``"No internet service"`` / ``"No phone service"``.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain the columns listed in :data:`SERVICE_COLUMNS`.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with a new ``total_services`` (int) column.
    """
    try:
        logger.info("Computing total_services feature.")
        df = df.copy()

        not_subscribed = {"No", "No internet service", "No phone service"}

        present_cols: list[str] = [c for c in SERVICE_COLUMNS if c in df.columns]
        missing_cols: list[str] = [c for c in SERVICE_COLUMNS if c not in df.columns]

        if missing_cols:
            logger.warning("Service columns not found (skipped): %s", missing_cols)

        df["total_services"] = df[present_cols].apply(
            lambda row: sum(1 for val in row if val not in not_subscribed),
            axis=1,
        ).astype(int)

        logger.info(
            "total_services stats – min=%d, max=%d, mean=%.2f",
            df["total_services"].min(),
            df["total_services"].max(),
            df["total_services"].mean(),
        )
        return df
    except Exception as exc:
        raise CustomException(exc, sys) from exc


def add_clv_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a simple Customer Lifetime Value proxy.

    .. math::

        CLV = MonthlyCharges \\times tenure

    Parameters
    ----------
    df : pd.DataFrame
        Must contain numeric ``MonthlyCharges`` and ``tenure`` columns.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with a new ``CLV`` (float) column.
    """
    try:
        logger.info("Computing CLV proxy (MonthlyCharges × tenure).")
        df = df.copy()
        df["CLV"] = (df["MonthlyCharges"] * df["tenure"]).round(2)
        logger.info(
            "CLV stats – min=%.2f, max=%.2f, mean=%.2f",
            df["CLV"].min(),
            df["CLV"].max(),
            df["CLV"].mean(),
        )
        return df
    except Exception as exc:
        raise CustomException(exc, sys) from exc


# ------------------------------------------------------------------ #
#  Convenience: run all feature-engineering steps in one call
# ------------------------------------------------------------------ #


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply **all** feature-engineering transforms sequentially.

    Calls :func:`add_tenure_buckets`, :func:`add_total_services`, and
    :func:`add_clv_proxy` in order.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame (output of :class:`DataIngestion.load_data`).

    Returns
    -------
    pd.DataFrame
        DataFrame enriched with ``tenure_bucket``, ``total_services``,
        and ``CLV`` columns.
    """
    try:
        logger.info("Starting feature engineering pipeline.")
        df = add_tenure_buckets(df)
        df = add_total_services(df)
        df = add_clv_proxy(df)
        logger.info(
            "Feature engineering complete – final shape: %s", df.shape
        )
        return df
    except CustomException:
        raise
    except Exception as exc:
        raise CustomException(exc, sys) from exc
