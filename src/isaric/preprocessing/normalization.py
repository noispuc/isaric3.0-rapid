"""
Normalization and standardization for the RAPID methodology.

This module transforms numeric predictor variables to a common scale,
vital for distance-based and gradient-based algorithms (k-means, SVM,
Neural Networks) which are sensitive to feature magnitude.

Techniques:
- standardize: Transform data to mean=0 and standard deviation=1 (Z-score).
- minmax_scale: Rescale data to a fixed range, typically [0, 1].
- parse_normalization_strategy: Parse strategy string for normalization.
"""

import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def standardize(data: pd.DataFrame) -> pd.DataFrame:
    """
    Transform numeric columns to have mean=0 and standard deviation=1.

    Z-score standardization is appropriate for normally distributed data.

    Args:
        data: Input DataFrame.

    Returns:
        DataFrame with standardized numeric columns.
    """
    result = data.copy()
    numeric_cols = result.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) == 0:
        return result

    scaler = StandardScaler()
    result[numeric_cols] = scaler.fit_transform(result[numeric_cols])
    return result


def minmax_scale(
    data: pd.DataFrame,
    feature_range: Tuple[float, float] = (0, 1)
) -> pd.DataFrame:
    """
    Rescale numeric columns to a fixed range, typically [0, 1].

    Min-Max scaling is appropriate when the distribution is not Gaussian
    or when the algorithm requires bounded values.

    Args:
        data: Input DataFrame.
        feature_range: Target range (min, max) for scaled values.

    Returns:
        DataFrame with min-max scaled numeric columns.

    Raises:
        ValueError: If feature_range is invalid (min >= max).
    """
    if feature_range[0] >= feature_range[1]:
        raise ValueError(
            f"feature_range must have min < max. "
            f"Received: {feature_range}"
        )

    result = data.copy()
    numeric_cols = result.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) == 0:
        return result

    scaler = MinMaxScaler(feature_range=feature_range)
    result[numeric_cols] = scaler.fit_transform(result[numeric_cols])
    return result


def parse_normalization_strategy(strategy: str) -> str:
    """
    Parse normalization strategy string.

    Examples:
        "standardize" -> "standardize"
        "minmax" -> "minmax"

    Args:
        strategy: Strategy string to parse.

    Returns:
        Normalization method name.

    Raises:
        ValueError: If strategy is unknown.
    """
    if not isinstance(strategy, str) or not strategy:
        raise ValueError("Strategy must be a non-empty string.")

    strategy_lower = strategy.lower()

    if strategy_lower == "standardize":
        return "standardize"

    elif strategy_lower == "minmax":
        return "minmax"

    else:
        raise ValueError(
            f"Unknown normalization strategy: {strategy}. "
            "Use 'standardize' or 'minmax'."
        )