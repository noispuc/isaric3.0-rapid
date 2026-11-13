"""
Normalization and standardization for the RAPID methodology.

This module transforms numeric predictor variables to a common scale,
vital for distance-based and gradient-based algorithms (k-means, SVM,
Neural Networks) which are sensitive to feature magnitude.

Techniques:
- standardize: Transform data to mean=0 and standard deviation=1 (Z-score).
- minmax_scale: Rescale data to a fixed range, typically [0, 1].
- parse_normalization_strategy: Parse strategy string for normalization.

Strategy formats:
- "standardize" → standardizes all numeric columns
- "standardize(columns=['age', 'bmi'])" → standardizes only listed columns
- "minmax" → min-max scales all numeric columns
- "minmax(columns=['age', 'bmi'])" → min-max scales only listed columns
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def standardize(
    data: pd.DataFrame,
    columns: Optional[List[str]] = None,
    fit_data: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Transform numeric columns to have mean=0 and standard deviation=1.

    Z-score standardization is appropriate for normally distributed data.

    When fit_data is provided, the scaler is fitted exclusively on
    fit_data and then applied to data. This prevents data leakage when
    data is the test set and fit_data is the training set.

    Args:
        data: Input DataFrame to transform.
        columns: List of columns to standardize.
            If None, standardizes all numeric columns.
        fit_data: Optional DataFrame to learn mean and std from.
            When None (default), learns from data itself (current
            behaviour). When provided, learns from fit_data and
            applies to data — use to avoid train→test leakage.

    Returns:
        DataFrame with standardized numeric columns.

    Example:
        # Without leakage (predictive models)
        X_train_scaled = standardize(X_train, columns=['age'])
        X_test_scaled  = standardize(X_test, columns=['age'],
                                     fit_data=X_train)
    """
    result = data.copy()

    if columns is None:
        numeric_cols = result.select_dtypes(include=[np.number]).columns.tolist()
    else:
        missing = [col for col in columns if col not in result.columns]
        if missing:
            raise ValueError(f"Columns not found in DataFrame: {missing}")
        numeric_cols = columns

    if len(numeric_cols) == 0:
        return result

    scaler = StandardScaler()

    if fit_data is not None:
        # Learn statistics from fit_data only
        missing_fit = [c for c in numeric_cols if c not in fit_data.columns]
        if missing_fit:
            raise ValueError(
                f"fit_data is missing columns: {missing_fit}"
            )
        scaler.fit(fit_data[numeric_cols])
        result[numeric_cols] = scaler.transform(result[numeric_cols])
    else:
        # Current behaviour — learn and apply to same data
        result[numeric_cols] = scaler.fit_transform(result[numeric_cols])

    return result


def minmax_scale(
    data: pd.DataFrame,
    columns: Optional[List[str]] = None,
    feature_range: Tuple[float, float] = (0, 1),
    fit_data: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Rescale numeric columns to a fixed range, typically [0, 1].

    Min-Max scaling is appropriate when the distribution is not Gaussian
    or when the algorithm requires bounded values.

    When fit_data is provided, min and max are learned exclusively from
    fit_data and applied to data, preventing train→test leakage.

    Args:
        data: Input DataFrame to transform.
        columns: List of columns to scale.
            If None, scales all numeric columns.
        feature_range: Target range (min, max) for scaled values.
        fit_data: Optional DataFrame to learn min and max from.
            When None (default), learns from data itself (current
            behaviour). When provided, learns from fit_data and
            applies to data.

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

    if columns is None:
        numeric_cols = result.select_dtypes(include=[np.number]).columns.tolist()
    else:
        missing = [col for col in columns if col not in result.columns]
        if missing:
            raise ValueError(f"Columns not found in DataFrame: {missing}")
        numeric_cols = columns

    if len(numeric_cols) == 0:
        return result

    scaler = MinMaxScaler(feature_range=feature_range)

    if fit_data is not None:
        missing_fit = [c for c in numeric_cols if c not in fit_data.columns]
        if missing_fit:
            raise ValueError(
                f"fit_data is missing columns: {missing_fit}"
            )
        scaler.fit(fit_data[numeric_cols])
        result[numeric_cols] = scaler.transform(result[numeric_cols])
    else:
        result[numeric_cols] = scaler.fit_transform(result[numeric_cols])

    return result


def parse_normalization_strategy(
    strategy: str
) -> Tuple[str, Optional[List[str]]]:
    """
    Parse normalization strategy string.

    Examples:
        "standardize" -> ("standardize", None)
        "standardize(columns=['age', 'bmi'])" -> ("standardize", ['age', 'bmi'])
        "minmax" -> ("minmax", None)
        "minmax(columns=['age'])" -> ("minmax", ['age'])

    Args:
        strategy: Strategy string to parse.

    Returns:
        Tuple of (method, columns):
            - method: "standardize" or "minmax"
            - columns: List of columns, or None if not specified

    Raises:
        ValueError: If strategy is unknown or malformed.
    """
    if not isinstance(strategy, str) or not strategy:
        raise ValueError("Strategy must be a non-empty string.")

    strategy = strategy.strip()

    # Verifica se tem parâmetros
    if "(" in strategy and ")" in strategy:
        method = strategy.split("(")[0].strip().lower()
        params_str = strategy.split("(")[1].rstrip(")")

        # Parse columns
        if "columns=" in params_str:
            columns_str = params_str.split("columns=")[1].strip()
            # Remove colchetes e aspas
            columns_str = columns_str.strip("[]")
            columns = [
                col.strip().strip("'\"")
                for col in columns_str.split(",")
                if col.strip()
            ]
            if not columns:
                raise ValueError(
                    f"columns parameter cannot be empty: {strategy}"
                )
        else:
            columns = None
    else:
        method = strategy.lower()
        columns = None

    # Valida método
    if method == "standardize":
        return "standardize", columns
    elif method == "minmax":
        return "minmax", columns
    else:
        raise ValueError(
            f"Unknown normalization strategy: {strategy}. "
            "Use 'standardize' or 'minmax'."
        )