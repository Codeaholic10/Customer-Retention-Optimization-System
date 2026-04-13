# Profit Engine Module
"""
This module calculates profit and ROI metrics for customer retention strategies.
"""

import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any
from logger import logger


class ProfitOptimizationEngine:
    """
    Independent module to optimize customer retention strategies based on expected value.
    Computes Expected Retention Value (ERV), ranks customers, and simulates targeting under a budget constraint.
    """

    def __init__(self) -> None:
        logger.info("Initialized ProfitOptimizationEngine.")

    def optimize_targeting(
        self,
        df: pd.DataFrame,
        retention_cost: float,
        budget: float,
        id_col: str = "customerID",
        p_churn_col: str = "churn_probability",
        clv_col: str = "clv"
    ) -> Dict[str, Any]:
        """
        Simulate targeting under a budget constraint and compute expected profit and ROI.

        Args:
            df (pd.DataFrame): DataFrame containing customer data.
            retention_cost (float): The cost to target a single customer.
            budget (float): The maximum total budget for retention efforts.
            id_col (str): Column name for customer identifiers.
            p_churn_col (str): Column name for predicted churn probability.
            clv_col (str): Column name for customer lifetime value.

        Returns:
            Dict[str, Any]: Dictionary containing:
                - 'campaign_summary': Campaign-level summary metrics.
                - 'customer_details': Row-level details for selected customers.
                - 'selected_customers': List of targeted customer IDs.
                - 'expected_profit': Total expected profit from the campaign.
                - 'roi': Return on investment.
                - 'total_targeted_count': Number of customers targeted.
        """
        logger.info("Optimizing targeting strategy with budget=%.2f, retention_cost=%.2f", budget, retention_cost)

        # Validate columns
        for col in [id_col, p_churn_col, clv_col]:
            if col not in df.columns:
                error_msg = f"Missing required column: {col}"
                logger.error(error_msg)
                raise ValueError(error_msg)

        # Compute Expected Retention Value (ERV)
        # Formula: (P_churn * CLV) - retention_cost
        df_eval = df[[id_col, p_churn_col, clv_col]].copy()
        df_eval["expected_value"] = (df_eval[p_churn_col] * df_eval[clv_col]) - retention_cost

        # Filter only positive expected value
        positive_ev_df = df_eval[df_eval["expected_value"] > 0].copy()
        
        # Rank customers by expected value descending
        ranked_df = positive_ev_df.sort_values(by="expected_value", ascending=False)

        # Simulate targeting under budget constraint
        max_targets = int(budget // retention_cost) if retention_cost > 0 else len(ranked_df)
        
        selected_df = ranked_df.iloc[:max_targets]
        total_targeted_count = len(selected_df)

        # Per-customer ROI is expected value relative to per-customer retention cost.
        if retention_cost > 0:
            selected_df["per_customer_roi"] = selected_df["expected_value"] / retention_cost
        else:
            selected_df["per_customer_roi"] = 0.0
        
        expected_profit = selected_df["expected_value"].sum()
        total_cost = total_targeted_count * retention_cost
        
        roi = 0.0
        if total_cost > 0:
            roi = expected_profit / total_cost

        logger.info(
            "Optimization complete: %d customers selected, Expected Profit: %.2f, ROI: %.2f%%",
            total_targeted_count, expected_profit, roi * 100
        )

        customer_details = selected_df[
            [id_col, p_churn_col, clv_col, "expected_value", "per_customer_roi"]
        ].to_dict(orient="records")

        campaign_summary = {
            "expected_profit": float(expected_profit),
            "roi": float(roi),
            "total_targeted_count": int(total_targeted_count),
            "total_cost": float(total_cost),
            "retention_cost": float(retention_cost),
            "budget": float(budget),
        }

        return {
            "campaign_summary": campaign_summary,
            "customer_details": customer_details,
            "selected_customers": selected_df[id_col].tolist(),
            "expected_profit": float(expected_profit),
            "roi": float(roi),
            "total_targeted_count": int(total_targeted_count)
        }


def plot_profit_curve(results: pd.DataFrame) -> tuple[plt.Figure, plt.Axes, pd.DataFrame]:
    """
    Plot cumulative expected profit as the number of targeted customers increases.

    The input is sorted by expected value in descending order before cumulative
    profit is computed.

    Args:
        results (pd.DataFrame): DataFrame containing at least 'expected_value'.

    Returns:
        tuple[plt.Figure, plt.Axes, pd.DataFrame]:
            - Matplotlib figure object.
            - Matplotlib axes object.
            - DataFrame with columns:
              'customers_targeted' and 'cumulative_expected_profit'.
    """
    required_cols = ["expected_value"]
    for col in required_cols:
        if col not in results.columns:
            raise ValueError(f"Missing required column: {col}")

    ranked_df = results.sort_values(by="expected_value", ascending=False).reset_index(drop=True)
    curve_df = pd.DataFrame(
        {
            "customers_targeted": range(1, len(ranked_df) + 1),
            "cumulative_expected_profit": ranked_df["expected_value"].cumsum(),
        }
    )

    fig, ax = plt.subplots()
    ax.plot(curve_df["customers_targeted"], curve_df["cumulative_expected_profit"])
    ax.set_xlabel("Number of Customers Targeted")
    ax.set_ylabel("Cumulative Expected Profit")
    ax.set_title("Profit Curve: Customers Targeted vs Expected Profit")

    return fig, ax, curve_df


def threshold_targeting(results: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, float, int, float]:
    """
    Apply threshold-based targeting using expected value.

    Args:
        results (pd.DataFrame): DataFrame with 'expected_value' and 'per_customer_roi'.
        threshold (float): Keep customers where expected_value > threshold.

    Returns:
        tuple[pd.DataFrame, float, int, float]:
            - Filtered DataFrame.
            - Total expected profit.
            - Number of customers selected.
            - Average per-customer ROI for selected customers.
    """
    required_cols = ["expected_value", "per_customer_roi"]
    for col in required_cols:
        if col not in results.columns:
            raise ValueError(f"Missing required column: {col}")

    filtered_df = results[results["expected_value"] > threshold].copy()
    total_profit = float(filtered_df["expected_value"].sum())
    selected_count = int(len(filtered_df))
    average_roi = float(filtered_df["per_customer_roi"].mean()) if selected_count > 0 else 0.0

    return filtered_df, total_profit, selected_count, average_roi


def segment_profit_analysis(results: pd.DataFrame) -> pd.DataFrame:
    """
    Compute segment-wise profit and risk profile summary.

    Args:
        results (pd.DataFrame): DataFrame containing 'cluster', 'expected_value',
            'churn_probability', and 'clv'.

    Returns:
        pd.DataFrame: Segment summary sorted by highest average expected value,
            with columns:
            'cluster', 'avg_expected_value', 'avg_churn_probability',
            'avg_clv', and 'customer_count'.
    """
    required_cols = ["cluster", "expected_value", "churn_probability", "clv"]
    if "cluster" not in results.columns:
        return pd.DataFrame(
            columns=[
                "cluster",
                "avg_expected_value",
                "avg_churn_probability",
                "avg_clv",
                "customer_count",
            ]
        )

    for col in required_cols:
        if col not in results.columns:
            raise ValueError(f"Missing required column: {col}")

    summary_df = (
        results.groupby("cluster", as_index=False)
        .agg(
            avg_expected_value=("expected_value", "mean"),
            avg_churn_probability=("churn_probability", "mean"),
            avg_clv=("clv", "mean"),
            customer_count=("cluster", "size"),
        )
        .sort_values(by="avg_expected_value", ascending=False)
        .reset_index(drop=True)
    )

    return summary_df
