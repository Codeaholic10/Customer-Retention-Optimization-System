import sys
import pandas as pd
import numpy as np

from logger import logger
from exception import CustomException


class DataIngestion:
    """
    Responsible solely for loading and cleaning raw data from a CSV file.

    Cleaning steps performed:
        1. Strip leading/trailing whitespace from column names and string values.
        2. Convert TotalCharges to numeric (coerce errors to NaN).
        3. Ensure tenure and MonthlyCharges are numeric.
        4. Drop rows where the target column 'Churn' is missing.
        5. Fill remaining numeric NaN values with column medians.

    No feature engineering or categorical encoding is performed.
    """

    # Expected numeric columns that must be coerced explicitly
    _NUMERIC_COLS: list[str] = ["TotalCharges", "tenure", "MonthlyCharges"]

    # Target column – rows missing this value are dropped
    _TARGET_COL: str = "Churn"

    def __init__(self, file_path: str) -> None:
        """
        Parameters
        ----------
        file_path : str
            Absolute or relative path to the raw CSV data file.
        """
        self.file_path: str = file_path
        logger.info("DataIngestion initialised with file_path='%s'", self.file_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_data(self) -> pd.DataFrame:
        """
        Load, validate, and clean the raw CSV file.

        Returns
        -------
        pd.DataFrame
            Cleaned DataFrame ready for preprocessing / exploration.

        Raises
        ------
        CustomException
            Wraps any underlying I/O or processing error with full
            traceback context.
        """
        try:
            logger.info("Starting data ingestion from '%s'", self.file_path)

            df: pd.DataFrame = self._read_csv()
            df = self._strip_whitespace(df)
            df = self._drop_missing_target(df)
            df = self._coerce_numeric_columns(df)
            df = self._fill_numeric_nulls(df)
            self._validate_dtypes(df)

            logger.info(
                "Data ingestion complete – %d rows, %d columns returned.",
                df.shape[0],
                df.shape[1],
            )
            return df

        except CustomException:
            raise
        except Exception as exc:
            raise CustomException(exc, sys) from exc

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _read_csv(self) -> pd.DataFrame:
        """Read the raw CSV into a DataFrame."""
        logger.info("Reading CSV file.")
        try:
            df = pd.read_csv(self.file_path)
        except FileNotFoundError as exc:
            raise CustomException(exc, sys) from exc
        except Exception as exc:
            raise CustomException(exc, sys) from exc

        logger.info("CSV loaded – raw shape: %s", df.shape)
        return df

    def _strip_whitespace(self, df: pd.DataFrame) -> pd.DataFrame:
        """Strip whitespace from column names and all object-dtype columns."""
        logger.info("Stripping whitespace from column names and string values.")
        df.columns = df.columns.str.strip()
        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].str.strip()
        return df

    def _drop_missing_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop rows where the target column is null or an empty string."""
        before: int = len(df)

        # Treat empty strings (after stripping) as missing
        df[self._TARGET_COL] = df[self._TARGET_COL].replace("", np.nan)
        df = df.dropna(subset=[self._TARGET_COL])

        dropped: int = before - len(df)
        if dropped:
            logger.warning(
                "Dropped %d row(s) with missing '%s' values.", dropped, self._TARGET_COL
            )
        else:
            logger.info("No missing '%s' values found.", self._TARGET_COL)
        return df

    def _coerce_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Coerce expected numeric columns to float64.
        Unparseable entries become NaN (handled in a later step).
        """
        for col in self._NUMERIC_COLS:
            if col not in df.columns:
                logger.warning("Expected numeric column '%s' not found – skipping.", col)
                continue
            before_nulls: int = df[col].isnull().sum()
            df[col] = pd.to_numeric(df[col], errors="coerce")
            after_nulls: int = df[col].isnull().sum()
            new_nulls: int = after_nulls - before_nulls
            if new_nulls:
                logger.warning(
                    "Column '%s': %d value(s) could not be parsed and were set to NaN.",
                    col,
                    new_nulls,
                )
            logger.info("Column '%s' coerced to %s.", col, df[col].dtype)
        return df

    def _fill_numeric_nulls(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill NaN values in all numeric columns with their column median."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            null_count: int = df[col].isnull().sum()
            if null_count:
                median_val: float = df[col].median()
                df[col] = df[col].fillna(median_val)
                logger.info(
                    "Filled %d NaN(s) in '%s' with median=%.4f.",
                    null_count,
                    col,
                    median_val,
                )
        return df

    def _validate_dtypes(self, df: pd.DataFrame) -> None:
        """
        Assert that explicitly required numeric columns have a numeric dtype.
        Logs a warning for any column that failed coercion.
        """
        for col in self._NUMERIC_COLS:
            if col not in df.columns:
                continue
            if not pd.api.types.is_numeric_dtype(df[col]):
                logger.warning(
                    "Validation warning: column '%s' is still non-numeric (%s).",
                    col,
                    df[col].dtype,
                )
            else:
                logger.info("Dtype check passed: '%s' → %s.", col, df[col].dtype)
