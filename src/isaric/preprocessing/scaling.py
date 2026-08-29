"""
Scaling and transformation for the RAPID methodology.

This module transforms numeric variables to reduce skewness and make
distributions more Gaussian. These transformations are useful when
features have heavy tails or non-normal distributions.

Techniques:
- log_transform: Apply natural logarithm to reduce right skewness.
- boxcox_transform: Apply Box-Cox transformation (requires positive values).
- parse_scaling_strategy: Parse strategy string for scaling.
"""

import pandas as pd
import numpy as np
from typing import Optional
from scipy import stats


def log_transform(data: pd.DataFrame) -> pd.DataFrame:
    """
    Apply natural logarithm to numeric columns to reduce skewness.

    Log transformation is useful for right-skewed distributions.
    Zero or negative values are shifted to be positive before applying log.

    Args:
        data: Input DataFrame.

    Returns:
        DataFrame with log-transformed numeric columns.

    Raises:
        ValueError: If any numeric column has all values <= 0.
    """
    result = data.copy()
    numeric_cols = result.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) == 0:
        return result

    for col in numeric_cols:
        min_val = result[col].min()

        if min_val <= 0:
            # Shift values to be positive
            shift = abs(min_val) + 1
            result[col] = np.log1p(result[col] + shift)
        else:
            result[col] = np.log1p(result[col])

    return result


def boxcox_transform(data: pd.DataFrame) -> pd.DataFrame:
    """
    Apply Box-Cox transformation to numeric columns.

    Box-Cox requires strictly positive values. The transformation
    reduces skewness and makes the distribution more Gaussian.

    Args:
        data: Input DataFrame.

    Returns:
        DataFrame with Box-Cox transformed numeric columns.

    Raises:
        ValueError: If any numeric column has values <= 0.
    """
    result = data.copy()
    numeric_cols = result.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) == 0:
        return result

    for col in numeric_cols:
        min_val = result[col].min()

        if min_val <= 0:
            raise ValueError(
                f"Box-Cox transformation requires strictly positive values. "
                f"Column '{col}' has minimum value: {min_val}. "
                "Use log_transform() for non-positive data."
            )

        fitted_data, _ = stats.boxcox(result[col])
        result[col] = fitted_data

    return result


def parse_scaling_strategy(strategy: str) -> str:
    """
    Parse scaling strategy string.

    Examples:
        "log" -> "log"
        "boxcox" -> "boxcox"

    Args:
        strategy: Strategy string to parse.

    Returns:
        Scaling method name.

    Raises:
        ValueError: If strategy is unknown.
    """
    if not isinstance(strategy, str) or not strategy:
        raise ValueError("Strategy must be a non-empty string.")

    strategy_lower = strategy.lower()

    if strategy_lower == "log":
        return "log"

    elif strategy_lower == "boxcox":
        return "boxcox"

    else:
        raise ValueError(
            f"Unknown scaling strategy: {strategy}. "
            "Use 'log' or 'boxcox'."
        )