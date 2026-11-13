"""
Zero or near-zero variance feature removal for the RAPID methodology.

This module identifies and removes predictor variables that exhibit
little to no variability, which provide no predictive power and can
cause instability in modeling.

Techniques:
- frequency_ratio_analysis: Identify features where the most common
  value is disproportionately frequent.
- unique_value_count: Filter features with zero or very few unique values.
- parse_remove_variance_strategy: Parse strategy string for variance removal.
"""

import pandas as pd
from typing import List, Tuple


def frequency_ratio_analysis(
    data: pd.DataFrame,
    freq_ratio_threshold: float = 19.0
) -> pd.DataFrame:
    """
    Remove features where the frequency of the most common value
    divided by the frequency of the second most common value
    exceeds a threshold.

    This is a common near-zero variance flag (e.g., ratio > 19).

    Args:
        data: Input DataFrame.
        freq_ratio_threshold: Threshold for the frequency ratio.
            Features with ratio above this value are removed.

    Returns:
        DataFrame without near-zero variance features.

    Raises:
        ValueError: If threshold is negative.
    """
    if freq_ratio_threshold < 0:
        raise ValueError(
            f"freq_ratio_threshold must be non-negative. "
            f"Received: {freq_ratio_threshold}"
        )

    result = data.copy()
    columns_to_drop: List[str] = []

    for col in result.columns:
        value_counts = result[col].value_counts()

        if len(value_counts) < 2:
            # Only one unique value => zero variance
            columns_to_drop.append(col)
            continue

        freq_ratio = value_counts.iloc[0] / value_counts.iloc[1]

        if freq_ratio > freq_ratio_threshold:
            columns_to_drop.append(col)

    if columns_to_drop:
        result = result.drop(columns=columns_to_drop)

    return result


def unique_value_count(
    data: pd.DataFrame,
    min_unique_values: int = 2
) -> pd.DataFrame:
    """
    Remove features with zero or a very small number of unique values.

    A feature with only one unique value has zero variance and provides
    no predictive power.

    Args:
        data: Input DataFrame.
        min_unique_values: Minimum number of unique values required
            for a feature to be retained.

    Returns:
        DataFrame without zero or near-zero variance features.

    Raises:
        ValueError: If min_unique_values is less than 1.
    """
    if min_unique_values < 1:
        raise ValueError(
            f"min_unique_values must be at least 1. "
            f"Received: {min_unique_values}"
        )

    result = data.copy()
    columns_to_drop: List[str] = []

    for col in result.columns:
        n_unique = result[col].nunique()

        if n_unique < min_unique_values:
            columns_to_drop.append(col)

    if columns_to_drop:
        result = result.drop(columns=columns_to_drop)

    return result


def get_zero_variance_features(data: pd.DataFrame) -> List[str]:
    """
    Identify features with zero variance (single unique value).

    This is a diagnostic function that returns the list of columns
    that would be removed, without actually removing them.

    Args:
        data: Input DataFrame.

    Returns:
        List of column names with zero variance.
    """
    zero_variance_cols: List[str] = []

    for col in data.columns:
        if data[col].nunique() <= 1:
            zero_variance_cols.append(col)

    return zero_variance_cols


def get_near_zero_variance_features(
    data: pd.DataFrame,
    freq_ratio_threshold: float = 19.0,
    min_unique_values: int = 2
) -> List[str]:
    """
    Identify features with near-zero variance.

    This is a diagnostic function that returns the list of columns
    that would be removed, without actually removing them.

    Args:
        data: Input DataFrame.
        freq_ratio_threshold: Threshold for frequency ratio.
        min_unique_values: Minimum number of unique values.

    Returns:
        List of column names with near-zero variance.
    """
    near_zero_cols: List[str] = []

    for col in data.columns:
        value_counts = data[col].value_counts()

        if len(value_counts) < min_unique_values:
            near_zero_cols.append(col)
            continue

        if len(value_counts) >= 2:
            freq_ratio = value_counts.iloc[0] / value_counts.iloc[1]
            if freq_ratio > freq_ratio_threshold:
                near_zero_cols.append(col)

    return near_zero_cols


def parse_remove_variance_strategy(strategy: str) -> Tuple[str, float]:
    """
    Parse remove_zero_variance strategy string.

    Examples:
        "frequency_ratio_analysis(threshold=19.0)" -> ("frequency_ratio_analysis", 19.0)
        "unique_value_count(min=2)" -> ("unique_value_count", 2.0)

    Args:
        strategy: Strategy string to parse.

    Returns:
        Tuple containing (method, value).

    Raises:
        ValueError: If strategy format is invalid.
    """
    if not isinstance(strategy, str) or not strategy:
        raise ValueError("Strategy must be a non-empty string.")

    if strategy.startswith("frequency_ratio_analysis"):
        threshold = _parse_threshold(strategy)
        return ("frequency_ratio_analysis", threshold)

    elif strategy.startswith("unique_value_count"):
        min_values = _parse_min_values(strategy)
        return ("unique_value_count", min_values)

    else:
        raise ValueError(
            f"Unknown remove_zero_variance strategy: {strategy}. "
            "Use 'frequency_ratio_analysis(threshold=19.0)' "
            "or 'unique_value_count(min=2)'."
        )


def _parse_threshold(strategy: str) -> float:
    """
    Extract threshold from frequency_ratio_analysis strategy string.

    Example:
        "frequency_ratio_analysis(threshold=19.0)" -> 19.0

    Args:
        strategy: Strategy string containing threshold parameter.

    Returns:
        Threshold value as float.

    Raises:
        ValueError: If threshold is missing or malformed.
    """
    try:
        threshold_str = strategy.split("threshold=")[1].rstrip(")")
        threshold = float(threshold_str)

        if threshold < 0:
            raise ValueError(
                f"Threshold must be non-negative. Received: {threshold}"
            )

        return threshold

    except (IndexError, ValueError):
        raise ValueError(
            f"Invalid frequency_ratio_analysis format: {strategy}. "
            "Expected: 'frequency_ratio_analysis(threshold=19.0)'."
        )


def _parse_min_values(strategy: str) -> float:
    """
    Extract min from unique_value_count strategy string.

    Example:
        "unique_value_count(min=2)" -> 2.0

    Args:
        strategy: Strategy string containing min parameter.

    Returns:
        Minimum unique values as float.

    Raises:
        ValueError: If min is missing or malformed.
    """
    try:
        min_str = strategy.split("min=")[1].rstrip(")")
        min_values = float(min_str)

        if min_values < 1:
            raise ValueError(
                f"min_unique_values must be at least 1. Received: {min_values}"
            )

        return min_values

    except (IndexError, ValueError):
        raise ValueError(
            f"Invalid unique_value_count format: {strategy}. "
            "Expected: 'unique_value_count(min=2)'."
        )