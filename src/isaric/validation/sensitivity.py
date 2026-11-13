"""
Sensitivity analyses for the RAPID methodology.

This module provides functions to determine how robust findings are to
changes in methods, assumptions, or data input (Step 5.3 of the RAPID
methodology). The goal is to verify that conclusions are not dependent
on arbitrary decisions.

Techniques:
- alternative_missing_handling: Compare MICE vs median/mode imputation.
- outlier_variation: Test results with and without outlier removal.
- outcome_variation: Re-run with alternative outcome definition.
"""

import pandas as pd
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple


def alternative_missing_handling(
    data: pd.DataFrame,
    target_col: str,
    predictors: List[str],
    primary_strategy: str = "mice",
    alternative_strategy: str = "median"
) -> Dict[str, pd.DataFrame]:
    """
    Compare results using different missing data handling strategies.

    Re-runs the analysis with the primary strategy (e.g., MICE) and an
    alternative strategy (e.g., median imputation), comparing results.

    Args:
        data: Input DataFrame with missing values.
        target_col: Outcome variable.
        predictors: Predictor variables.
        primary_strategy: Primary imputation strategy ("mice", "median", "mode").
        alternative_strategy: Alternative imputation strategy.

    Returns:
        Dictionary with 'primary_data' and 'alternative_data' (DataFrames).

    Raises:
        ValueError: If strategies are invalid.
    """
    valid_strategies = ("mice", "median", "mode", "mean")
    if primary_strategy not in valid_strategies:
        raise ValueError(
            f"primary_strategy must be one of {valid_strategies}. "
            f"Received: {primary_strategy}"
        )
    if alternative_strategy not in valid_strategies:
        raise ValueError(
            f"alternative_strategy must be one of {valid_strategies}. "
            f"Received: {alternative_strategy}"
        )

    # Apply primary strategy
    primary_data = _apply_imputation(data, target_col, predictors, primary_strategy)

    # Apply alternative strategy
    alternative_data = _apply_imputation(
        data, target_col, predictors, alternative_strategy
    )

    return {
        'primary_data': primary_data,
        'alternative_data': alternative_data
    }


def outlier_variation(
    data: pd.DataFrame,
    columns: List[str],
    method: str = "iqr",
    threshold: float = 1.5
) -> Dict[str, pd.DataFrame]:
    """
    Compare results with and without outlier removal.

    Creates two datasets: one with outliers removed and one with outliers
    retained, allowing comparison of model robustness.

    Args:
        data: Input DataFrame.
        columns: Numeric columns to check for outliers.
        method: Outlier detection method: "iqr" or "zscore".
        threshold: Threshold for outlier detection (IQR multiplier or
            Z-score cutoff).

    Returns:
        Dictionary with 'with_outliers' and 'without_outliers' DataFrames.

    Raises:
        ValueError: If method is invalid.
    """
    if method not in ("iqr", "zscore"):
        raise ValueError(
            f"method must be 'iqr' or 'zscore'. Received: {method}"
        )

    data_with_outliers = data.copy()
    data_without_outliers = _remove_outliers(data, columns, method, threshold)

    return {
        'with_outliers': data_with_outliers,
        'without_outliers': data_without_outliers
    }


def outcome_variation(
    data: pd.DataFrame,
    target_col: str,
    threshold: float,
    new_target_name: Optional[str] = None
) -> Dict[str, pd.DataFrame]:
    """
    Re-run analysis with an alternative threshold for defining the outcome.

    Creates a new outcome variable using a different threshold (e.g.,
    prolonged ICU stay defined as >14 days instead of >21 days).

    Args:
        data: Input DataFrame.
        target_col: Continuous outcome variable.
        threshold: New threshold for binary outcome definition.
        new_target_name: Name for the new binary variable (optional).

    Returns:
        Dictionary with 'original_data' and 'alternative_data'.

    Raises:
        ValueError: If target_col is not numeric.
    """
    if target_col not in data.columns:
        raise ValueError(f"Column '{target_col}' not found in DataFrame.")

    if not pd.api.types.is_numeric_dtype(data[target_col]):
        raise ValueError(f"Column '{target_col}' must be numeric.")

    alternative_data = data.copy()
    new_col = new_target_name or f'{target_col}_binary_{threshold}'
    alternative_data[new_col] = (alternative_data[target_col] > threshold).astype(int)

    return {
        'original_data': data.copy(),
        'alternative_data': alternative_data
    }


def _apply_imputation(
    data: pd.DataFrame,
    target_col: str,
    predictors: List[str],
    strategy: str
) -> pd.DataFrame:
    """
    Apply imputation strategy to a dataset.

    Args:
        data: Input DataFrame.
        target_col: Outcome variable.
        predictors: Predictor variables.
        strategy: Imputation strategy.

    Returns:
        DataFrame with imputed values.
    """
    result = data.copy()

    if strategy == "mice":
        from isaric.preprocessing.imputation import mice_imputation
        return mice_imputation(result)

    elif strategy == "median":
        numeric_cols = result[predictors].select_dtypes(include=[np.number]).columns
        result[numeric_cols] = result[numeric_cols].fillna(result[numeric_cols].median())

    elif strategy == "mean":
        numeric_cols = result[predictors].select_dtypes(include=[np.number]).columns
        result[numeric_cols] = result[numeric_cols].fillna(result[numeric_cols].mean())

    elif strategy == "mode":
        for col in predictors:
            mode_value = result[col].mode()
            if not mode_value.empty:
                result[col] = result[col].fillna(mode_value.iloc[0])

    return result


def _remove_outliers(
    data: pd.DataFrame,
    columns: List[str],
    method: str,
    threshold: float
) -> pd.DataFrame:
    """
    Remove outliers from specified columns.

    Args:
        data: Input DataFrame.
        columns: Numeric columns to check.
        method: "iqr" or "zscore".
        threshold: Detection threshold.

    Returns:
        DataFrame without outliers.
    """
    result = data.copy()

    for col in columns:
        if col not in result.columns:
            continue
        if not pd.api.types.is_numeric_dtype(result[col]):
            continue

        if method == "iqr":
            q1 = result[col].quantile(0.25)
            q3 = result[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            result = result[(result[col] >= lower_bound) & (result[col] <= upper_bound)]

        elif method == "zscore":
            from scipy import stats
            z_scores = np.abs(stats.zscore(result[col]))
            result = result[z_scores < threshold]

    return result.reset_index(drop=True)