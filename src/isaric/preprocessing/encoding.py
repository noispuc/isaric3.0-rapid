"""
Encoding of categorical variables for the RAPID methodology.

This module converts nominal categorical features into a format suitable
for modelling algorithms. Most machine learning methods cannot process
text labels directly.

Techniques:
- onehot_encode: Create binary (dummy) variables for each category.
- label_encode: Convert categories to integer labels.
- target_encode: Replace categories with the mean of the target variable.
- parse_encoding_strategy: Parse strategy string for encoding.
"""

import pandas as pd
import numpy as np
from typing import Optional
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


def onehot_encode(
    data: pd.DataFrame,
    drop_first: bool = True
) -> pd.DataFrame:
    """
    Convert categorical columns to binary (dummy) variables.

    One-hot encoding creates N new binary variables for a feature with
    N unique categories. Optionally drops the first to avoid perfect
    multicollinearity.

    Args:
        data: Input DataFrame.
        drop_first: If True, drops the first dummy column to avoid
            multicollinearity (default True).

    Returns:
        DataFrame with one-hot encoded categorical columns.
    """
    result = data.copy()
    categorical_cols = result.select_dtypes(
        include=['object', 'category']
    ).columns.tolist()

    if len(categorical_cols) == 0:
        return result

    for col in categorical_cols:
        dummies = pd.get_dummies(
            result[col],
            prefix=col,
            prefix_sep='!',
            drop_first=drop_first,
            dtype=int
        )
        result = pd.concat([result.drop(columns=[col]), dummies], axis=1)

    return result


def label_encode(data: pd.DataFrame) -> pd.DataFrame:
    """
    Convert categorical columns to integer labels.

    Label encoding assigns an integer to each unique category.
    Use with caution for models that assume ordinal relationships.

    Args:
        data: Input DataFrame.

    Returns:
        DataFrame with label-encoded categorical columns.
    """
    result = data.copy()
    categorical_cols = result.select_dtypes(
        include=['object', 'category']
    ).columns.tolist()

    if len(categorical_cols) == 0:
        return result

    encoder = LabelEncoder()
    for col in categorical_cols:
        result[col] = encoder.fit_transform(result[col])

    return result


def target_encode(
    data: pd.DataFrame,
    target_col: str
) -> pd.DataFrame:
    """
    Replace categorical values with the mean of the target variable.

    Target encoding is useful for high-cardinality features. It replaces
    each category with the mean target value for that category.

    Args:
        data: Input DataFrame.
        target_col: Column name of the target variable.

    Returns:
        DataFrame with target-encoded categorical columns.

    Raises:
        ValueError: If target_col is not found or is not numeric.
    """
    if target_col not in data.columns:
        raise ValueError(
            f"Target column '{target_col}' not found in DataFrame."
        )

    if not pd.api.types.is_numeric_dtype(data[target_col]):
        raise ValueError(
            f"Target column '{target_col}' must be numeric for target encoding."
        )

    result = data.copy()
    categorical_cols = result.select_dtypes(
        include=['object', 'category']
    ).columns.tolist()

    if len(categorical_cols) == 0:
        return result

    for col in categorical_cols:
        target_mean = result.groupby(col)[target_col].mean()
        result[col] = result[col].map(target_mean)

    return result


def parse_encoding_strategy(strategy: str) -> str:
    """
    Parse encoding strategy string.

    Examples:
        "onehot" -> "onehot"
        "label" -> "label"
        "target" -> "target"

    Args:
        strategy: Strategy string to parse.

    Returns:
        Encoding method name.

    Raises:
        ValueError: If strategy is unknown.
    """
    if not isinstance(strategy, str) or not strategy:
        raise ValueError("Strategy must be a non-empty string.")

    strategy_lower = strategy.lower()

    if strategy_lower == "onehot":
        return "onehot"

    elif strategy_lower == "label":
        return "label"

    elif strategy_lower == "target":
        return "target"

    else:
        raise ValueError(
            f"Unknown encoding strategy: {strategy}. "
            "Use 'onehot', 'label', or 'target'."
        )