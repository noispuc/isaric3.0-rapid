"""
Collinearity analysis for the RAPID methodology.

This module identifies and quantifies the linear relationship between
two or more predictor variables. Multicollinearity can inflate the
variance of regression coefficients, making them unstable and difficult
to interpret.

Techniques:
- vif_analysis: Compute Variance Inflation Factor for each feature.
- pearson_correlation: Compute Pearson correlation between feature pairs.
- parse_collinearity_strategy: Parse strategy string for collinearity analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from statsmodels.stats.outliers_influence import variance_inflation_factor


def vif_analysis(
    data: pd.DataFrame,
    threshold: float = 5.0
) -> pd.DataFrame:
    """
    Remove features with Variance Inflation Factor (VIF) above a threshold.

    VIF measures how much the variance of an estimated regression
    coefficient is increased due to collinearity. A high VIF
    (e.g., > 5 or 10) is a diagnostic of problematic multicollinearity.

    Args:
        data: Input DataFrame.
        threshold: VIF threshold for feature removal (default 5.0).

    Returns:
        DataFrame without features exceeding the VIF threshold.

    Raises:
        ValueError: If threshold is not positive.
    """
    if threshold <= 0:
        raise ValueError(
            f"threshold must be positive. Received: {threshold}"
        )

    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) == 0:
        return data.copy()

    result = data.copy()
    remaining_cols = numeric_cols.copy()

    while len(remaining_cols) > 1:
        try:
            vif_data = {}
            for col in remaining_cols:
                vif_data[col] = variance_inflation_factor(
                    result[remaining_cols].values,
                    remaining_cols.index(col)
                )
        except np.linalg.LinAlgError:
            # Perfect collinearity detected
            remaining_cols = remaining_cols[:-1]
            continue

        max_vif_col = max(vif_data, key=vif_data.get)
        max_vif_value = vif_data[max_vif_col]

        if max_vif_value > threshold:
            remaining_cols.remove(max_vif_col)
        else:
            break

    columns_to_drop = [c for c in numeric_cols if c not in remaining_cols]
    if columns_to_drop:
        result = result.drop(columns=columns_to_drop)

    return result


def get_vif_table(data: pd.DataFrame) -> pd.DataFrame:
    """
    Compute VIF for each numeric feature and return as a DataFrame.

    This is a diagnostic function that shows the VIF values without
    removing any features.

    Args:
        data: Input DataFrame.

    Returns:
        DataFrame with columns 'feature' and 'VIF'.

    Raises:
        ValueError: If there are fewer than 2 numeric columns.
    """
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        raise ValueError(
            "VIF requires at least 2 numeric columns."
        )

    vif_data = []
    for i, col in enumerate(numeric_cols):
        vif_value = variance_inflation_factor(
            data[numeric_cols].values,
            i
        )
        vif_data.append({"feature": col, "VIF": round(vif_value, 2)})

    return pd.DataFrame(vif_data)


def pearson_correlation(
    data: pd.DataFrame,
    threshold: float = 0.75
) -> pd.DataFrame:
    """
    Remove features with Pearson correlation above a threshold.

    Pearson correlation assesses linear relationships between pairs of
    numeric variables. A high absolute correlation (e.g., r > 0.75)
    flags problematic collinearity.

    Args:
        data: Input DataFrame.
        threshold: Correlation threshold for feature removal (default 0.75).

    Returns:
        DataFrame without highly correlated features.

    Raises:
        ValueError: If threshold is not between 0.0 and 1.0.
    """
    if not (0.0 < threshold <= 1.0):
        raise ValueError(
            f"threshold must be between 0.0 and 1.0. Received: {threshold}"
        )

    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) == 0:
        return data.copy()

    corr_matrix = data[numeric_cols].corr().abs()
    upper_tri = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    columns_to_drop = set()
    for col in upper_tri.columns:
        correlated = upper_tri.index[
            upper_tri[col] > threshold
        ].tolist()
        if correlated:
            # Keep the first, drop the correlated ones
            columns_to_drop.update(correlated)

    result = data.copy()
    if columns_to_drop:
        result = result.drop(columns=list(columns_to_drop))

    return result


def get_correlation_pairs(
    data: pd.DataFrame,
    threshold: float = 0.75
) -> List[Tuple[str, str, float]]:
    """
    Identify pairs of features with correlation above a threshold.

    Diagnostic function that returns correlated pairs without removing.

    Args:
        data: Input DataFrame.
        threshold: Correlation threshold (default 0.75).

    Returns:
        List of tuples (feature1, feature2, correlation_value).

    Raises:
        ValueError: If threshold is not between 0.0 and 1.0.
    """
    if not (0.0 < threshold <= 1.0):
        raise ValueError(
            f"threshold must be between 0.0 and 1.0. Received: {threshold}"
        )

    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        return []

    corr_matrix = data[numeric_cols].corr().abs()
    pairs = []

    for i in range(len(numeric_cols)):
        for j in range(i + 1, len(numeric_cols)):
            col1 = numeric_cols[i]
            col2 = numeric_cols[j]
            corr_value = corr_matrix.loc[col1, col2]
            if corr_value > threshold:
                pairs.append((col1, col2, round(corr_value, 3)))

    return pairs


def parse_collinearity_strategy(strategy: str) -> Tuple[str, float]:
    """
    Parse collinearity analysis strategy string.

    Examples:
        "vif(threshold=5.0)" -> ("vif", 5.0)
        "pearson(threshold=0.75)" -> ("pearson", 0.75)

    Args:
        strategy: Strategy string to parse.

    Returns:
        Tuple of (method, threshold).

    Raises:
        ValueError: If strategy format is invalid.
    """
    if not isinstance(strategy, str) or not strategy:
        raise ValueError("Strategy must be a non-empty string.")

    if strategy.startswith("vif"):
        threshold = _parse_threshold(strategy, "vif")
        return ("vif", threshold)

    elif strategy.startswith("pearson"):
        threshold = _parse_threshold(strategy, "pearson")
        return ("pearson", threshold)

    else:
        raise ValueError(
            f"Unknown collinearity strategy: {strategy}. "
            "Use 'vif(threshold=5.0)' or 'pearson(threshold=0.75)'."
        )


def _parse_threshold(strategy: str, method: str) -> float:
    """
    Extract threshold from strategy string.

    Args:
        strategy: Strategy string containing threshold parameter.
        method: Method name for error messages.

    Returns:
        Threshold as float.

    Raises:
        ValueError: If threshold is missing or malformed.
    """
    try:
        threshold_str = strategy.split("threshold=")[1].rstrip(")")
        threshold = float(threshold_str)

        if threshold <= 0:
            raise ValueError(
                f"threshold must be positive. Received: {threshold}"
            )

        return threshold

    except (IndexError, ValueError):
        raise ValueError(
            f"Invalid {method} format: {strategy}. "
            f"Expected: '{method}(threshold=value)'."
        )