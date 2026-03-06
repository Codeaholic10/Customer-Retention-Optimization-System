# Customer Segmentation Module
"""
K-Means based customer segmentation for the Telco Customer Churn project.

Responsibilities
----------------
* Find the optimal number of clusters K in [2, 6] via silhouette score.
* Fit a KMeans model with the chosen K.
* Assign cluster labels back to a DataFrame.
* Persist / reload the fitted model with joblib.
* Predict the cluster for unseen data.

**No churn modelling and no profit logic here.**
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from logger import logger
from exception import CustomException


# ------------------------------------------------------------------ #
#  Constants
# ------------------------------------------------------------------ #

K_MIN: int = 2
K_MAX: int = 6
DEFAULT_RANDOM_STATE: int = 42
LABEL_COLUMN: str = "segment"


# ------------------------------------------------------------------ #
#  CustomerSegmentation class
# ------------------------------------------------------------------ #


class CustomerSegmentation:
    """
    End-to-end KMeans customer segmentation.

    Parameters
    ----------
    k_min : int
        Smallest K to evaluate (inclusive).  Default ``2``.
    k_max : int
        Largest K to evaluate (inclusive).  Default ``6``.
    random_state : int
        Seed for reproducibility.  Default ``42``.

    Attributes
    ----------
    optimal_k_ : int
        K chosen by silhouette search (set after :meth:`find_optimal_k`).
    model_ : KMeans
        Fitted KMeans model (set after :meth:`fit`).
    silhouette_scores_ : dict[int, float]
        Mapping of K → silhouette score evaluated during search.
    """

    def __init__(
        self,
        k_min: int = K_MIN,
        k_max: int = K_MAX,
        random_state: int = DEFAULT_RANDOM_STATE,
    ) -> None:
        self.k_min: int = k_min
        self.k_max: int = k_max
        self.random_state: int = random_state

        # Set after training
        self.optimal_k_: Optional[int] = None
        self.model_: Optional[KMeans] = None
        self.silhouette_scores_: dict[int, float] = {}

        logger.info(
            "CustomerSegmentation initialised – k_min=%d, k_max=%d, random_state=%d",
            self.k_min,
            self.k_max,
            self.random_state,
        )

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def find_optimal_k(self, X: np.ndarray) -> int:
        """
        Evaluate K in ``[k_min, k_max]`` and return the K with the highest
        silhouette score.

        Parameters
        ----------
        X : np.ndarray
            Pre-processed feature matrix (output of the preprocessing pipeline).

        Returns
        -------
        int
            Optimal K value.
        """
        try:
            logger.info(
                "Searching for optimal K in range [%d, %d] ...", self.k_min, self.k_max
            )
            self.silhouette_scores_ = {}

            for k in range(self.k_min, self.k_max + 1):
                km = KMeans(n_clusters=k, random_state=self.random_state, n_init="auto")
                labels = km.fit_predict(X)
                score = float(silhouette_score(X, labels))
                self.silhouette_scores_[k] = score
                logger.info("  k=%d -> silhouette score=%.4f", k, score)

            best_k = max(self.silhouette_scores_, key=self.silhouette_scores_.get)  # type: ignore[arg-type]
            self.optimal_k_ = best_k

            logger.info(
                "Optimal K: %d  (silhouette=%.4f)",
                best_k,
                self.silhouette_scores_[best_k],
            )
            return best_k

        except Exception as exc:
            raise CustomException(exc, sys) from exc

    def fit(self, X: np.ndarray, k: Optional[int] = None) -> "CustomerSegmentation":
        """
        Fit KMeans with the given K (or the optimal K found by
        :meth:`find_optimal_k`).

        Parameters
        ----------
        X : np.ndarray
            Pre-processed feature matrix.
        k : int, optional
            Number of clusters.  Falls back to ``optimal_k_`` if ``None``;
            runs :meth:`find_optimal_k` automatically when neither is set.

        Returns
        -------
        CustomerSegmentation
            Self – for method chaining.
        """
        try:
            if k is None:
                if self.optimal_k_ is None:
                    logger.info("No K provided – running find_optimal_k() first.")
                    self.find_optimal_k(X)
                k = self.optimal_k_

            logger.info("Fitting KMeans with k=%d ...", k)
            self.model_ = KMeans(
                n_clusters=k,
                random_state=self.random_state,
                n_init="auto",
            )
            self.model_.fit(X)
            logger.info(
                "KMeans fitted – inertia=%.4f", self.model_.inertia_
            )
            return self

        except CustomException:
            raise
        except Exception as exc:
            raise CustomException(exc, sys) from exc

    def assign_labels(
        self,
        df: pd.DataFrame,
        X: np.ndarray,
        label_column: str = LABEL_COLUMN,
    ) -> pd.DataFrame:
        """
        Predict cluster labels for *X* and attach them to *df* as a new column.

        Parameters
        ----------
        df : pd.DataFrame
            Original (or enriched) DataFrame whose rows correspond to rows in *X*.
        X : np.ndarray
            Pre-processed feature matrix (same row order as *df*).
        label_column : str
            Name of the new column.  Default ``"segment"``.

        Returns
        -------
        pd.DataFrame
            Copy of *df* with an integer ``segment`` column added.
        """
        try:
            self._assert_fitted()
            labels: np.ndarray = self.model_.predict(X)  # type: ignore[union-attr]
            df = df.copy()
            df[label_column] = labels
            counts = pd.Series(labels).value_counts().sort_index().to_string()
            logger.info("Cluster label distribution:\n%s", counts)
            return df

        except CustomException:
            raise
        except Exception as exc:
            raise CustomException(exc, sys) from exc

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the cluster for each row in *X*.

        Parameters
        ----------
        X : np.ndarray
            Pre-processed feature matrix.

        Returns
        -------
        np.ndarray
            Integer array of cluster labels, shape ``(n_samples,)``.
        """
        try:
            self._assert_fitted()
            labels: np.ndarray = self.model_.predict(X)  # type: ignore[union-attr]
            logger.info("predict() – %d samples classified.", len(labels))
            return labels

        except CustomException:
            raise
        except Exception as exc:
            raise CustomException(exc, sys) from exc

    def save(self, path: str | Path) -> Path:
        """
        Persist the fitted model to disk using joblib.

        Parameters
        ----------
        path : str | Path
            Destination file (e.g. ``"artifacts/segmentation_model.joblib"``).

        Returns
        -------
        Path
            Resolved path that was written.
        """
        try:
            self._assert_fitted()
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self, path)
            logger.info("CustomerSegmentation model saved to '%s'.", path)
            return path

        except CustomException:
            raise
        except Exception as exc:
            raise CustomException(exc, sys) from exc

    @staticmethod
    def load(path: str | Path) -> "CustomerSegmentation":
        """
        Load a previously saved :class:`CustomerSegmentation` instance.

        Parameters
        ----------
        path : str | Path
            Path to the ``.joblib`` file produced by :meth:`save`.

        Returns
        -------
        CustomerSegmentation
            The deserialised, fitted instance.
        """
        try:
            path = Path(path)
            obj: CustomerSegmentation = joblib.load(path)
            logger.info("CustomerSegmentation model loaded from '%s'.", path)
            return obj

        except Exception as exc:
            raise CustomException(exc, sys) from exc

    # ------------------------------------------------------------------ #
    #  Introspection helpers
    # ------------------------------------------------------------------ #

    def silhouette_summary(self) -> pd.DataFrame:
        """
        Return a tidy :class:`~pandas.DataFrame` of K vs. silhouette score,
        sorted by descending score.

        Returns
        -------
        pd.DataFrame
            Columns: ``k``, ``silhouette_score``.
        """
        if not self.silhouette_scores_:
            logger.warning("No silhouette scores available – run find_optimal_k() first.")
            return pd.DataFrame(columns=["k", "silhouette_score"])

        df = pd.DataFrame(
            list(self.silhouette_scores_.items()),
            columns=["k", "silhouette_score"],
        ).sort_values("silhouette_score", ascending=False).reset_index(drop=True)
        return df

    def cluster_centers(self) -> np.ndarray:
        """
        Return the cluster centroids from the fitted KMeans model.

        Returns
        -------
        np.ndarray
            Shape ``(k, n_features)``.
        """
        self._assert_fitted()
        return self.model_.cluster_centers_  # type: ignore[union-attr]

    # ------------------------------------------------------------------ #
    #  Private helpers
    # ------------------------------------------------------------------ #

    def _assert_fitted(self) -> None:
        """Raise ``RuntimeError`` when the model has not been fitted yet."""
        if self.model_ is None:
            raise RuntimeError(
                "Model is not fitted yet. Call fit() before using this method."
            )
