"""
Missing value handling for the RAPID methodology.

This module addresses data points where a value is unavailable.
The method of choice depends on the mechanism of missingness.

Techniques:
- drop_rows: Remove rows with any missing values.
- drop_columns: Remove columns with missing percentage above a threshold.
- impute_mean: Fill missing values with the column mean.
- impute_median: Fill missing values with the column median.
- impute_mode: Fill missing values with the column mode (most frequent).
- parse_missing_strategy: Parse strategy string for missing value handling.
"""

import pandas as pd
from typing import Optional, Tuple


def drop_rows(data: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows that contain any missing values (listwise deletion).

    Use only when missing data is a very small percentage and
    completely random.

    Args:
        data: Input DataFrame.

    Returns:
        DataFrame without rows containing missing values.
    """
    result = data.copy()
    return result.dropna().reset_index(drop=True)


def drop_columns(
    data: pd.DataFrame,
    threshold: float = 0.3
) -> pd.DataFrame:
    """
    Remove columns where the percentage of missing values exceeds
    the specified threshold.

    Use when a feature has a high percentage of missing values
    (e.g., over 70%) that add no value.

    Args:
        data: Input DataFrame.
        threshold: Maximum acceptable missing percentage (0.0 to 1.0).
            Example: 0.3 = columns with >30% missing are removed.

    Returns:
        DataFrame without columns exceeding the missing threshold.

    Raises:
        ValueError: If threshold is not between 0.0 and 1.0.
    """
    if not (0.0 <= threshold <= 1.0):
        raise ValueError(
            f"Threshold must be between 0.0 and 1.0. "
            f"Received: {threshold}"
        )

    result = data.copy()
    missing_percentage = result.isnull().mean()
    columns_to_drop = missing_percentage[
        missing_percentage > threshold
    ].index.tolist()

    if columns_to_drop:
        result = result.drop(columns=columns_to_drop)

    return result


def impute_mean(data: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values in numeric columns with the column mean.

    Mean imputation is appropriate for normally distributed data.

    Args:
        data: Input DataFrame.

    Returns:
        DataFrame with missing values imputed by mean.
    """
    result = data.copy()
    numeric_cols = result.select_dtypes(include=["number"]).columns

    if len(numeric_cols) == 0:
        return result

    result[numeric_cols] = result[numeric_cols].fillna(
        result[numeric_cols].mean()
    )
    return result


def impute_median(data: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values in numeric columns with the column median.

    Median imputation is preferred for skewed distributions.

    Args:
        data: Input DataFrame.

    Returns:
        DataFrame with missing values imputed by median.
    """
    result = data.copy()
    numeric_cols = result.select_dtypes(include=["number"]).columns

    if len(numeric_cols) == 0:
        return result

    result[numeric_cols] = result[numeric_cols].fillna(
        result[numeric_cols].median()
    )
    return result


def impute_mode(data: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values with the column mode (most frequent category).

    Mode imputation is appropriate for categorical variables.

    Args:
        data: Input DataFrame.

    Returns:
        DataFrame with missing values imputed by mode.
    """
    result = data.copy()

    for col in result.columns:
        mode_value = result[col].mode()
        if not mode_value.empty:
            result[col] = result[col].fillna(mode_value.iloc[0])

    return result


def parse_missing_strategy(strategy: str) -> Tuple[str, Optional[float]]:
    """
    Parse missing value handling strategy string.

    Examples:
        "drop_rows()" -> ("drop_rows", None)
        "drop_columns(p=0.3)" -> ("drop_columns", 0.3)
        "imputation(type=mean)" -> ("imputation", "mean")
        "imputation(type=median)" -> ("imputation", "median")
        "imputation(type=mode)" -> ("imputation", "mode")

    Args:
        strategy: Strategy string to parse.

    Returns:
        Tuple containing (method, parameter).

    Raises:
        ValueError: If strategy format is invalid.
    """
    if not isinstance(strategy, str) or not strategy:
        raise ValueError("Strategy must be a non-empty string.")

    strategy_lower = strategy.lower()

    if strategy_lower == "drop_rows()":
        return ("drop_rows", None)

    elif strategy_lower.startswith("drop_columns"):
        threshold = _parse_drop_threshold(strategy_lower)
        return ("drop_columns", threshold)

    elif strategy_lower.startswith("imputation"):
        imputation_type = _parse_imputation_type(strategy_lower)
        return ("imputation", imputation_type)

    else:
        raise ValueError(
            f"Unknown missing value strategy: {strategy}. "
            "Use 'drop_rows()', 'drop_columns(p=%)', "
            "'imputation(type=mean)', 'imputation(type=median)', "
            "or 'imputation(type=mode)'."
        )


def _parse_drop_threshold(strategy: str) -> float:
    """
    Extract threshold from drop_columns strategy string.

    Example:
        "drop_columns(p=0.3)" -> 0.3

    Args:
        strategy: Strategy string containing threshold parameter.

    Returns:
        Threshold value as float.

    Raises:
        ValueError: If threshold is missing or malformed.
    """
    try:
        threshold_str = strategy.split("p=")[1].rstrip(")")
        threshold = float(threshold_str)

        if not (0.0 <= threshold <= 1.0):
            raise ValueError(
                f"Threshold must be between 0.0 and 1.0. "
                f"Received: {threshold}"
            )

        return threshold

    except (IndexError, ValueError):
        raise ValueError(
            f"Invalid drop_columns format: {strategy}. "
            "Expected: 'drop_columns(p=0.3)'."
        )


def _parse_imputation_type(strategy: str) -> str:
    """
    Extract imputation type from strategy string.

    Example:
        "imputation(type=mean)" -> "mean"

    Args:
        strategy: Strategy string containing type parameter.

    Returns:
        Imputation type ('mean', 'median', or 'mode').

    Raises:
        ValueError: If type is missing or invalid.
    """
    try:
        imputation_type = strategy.split("type=")[1].rstrip(")")

        if imputation_type not in ("mean", "median", "mode"):
            raise ValueError(
                f"Invalid imputation type: {imputation_type}"
            )

        return imputation_type

    except (IndexError, ValueError):
        raise ValueError(
            f"Invalid imputation format: {strategy}. "
            "Expected: 'imputation(type=mean)'."
        )