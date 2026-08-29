"""
Feature selection and reduction for the RAPID methodology.

This module identifies a subset of the most relevant features to use
in the modelling stage. This improves model interpretability, reduces
training time, and enhances generalisation.

Techniques:
- variance_threshold: Remove features with variance below a threshold.
- lasso_selection: Select features using L1 regularization (LASSO).
- rfe_selection: Select features using Recursive Feature Elimination.
- filter_selection: Select features based on correlation with target.
- parse_selection_strategy: Parse strategy string for feature selection.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple
from sklearn.feature_selection import VarianceThreshold, RFE
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.preprocessing import StandardScaler


def variance_threshold(
    data: pd.DataFrame,
    threshold: float = 0.0
) -> pd.DataFrame:
    """
    Remove features with variance below a threshold.

    Features with low variance provide little predictive power.
    Variance threshold removes features where variance does not meet
    the specified criterion.

    Args:
        data: Input DataFrame.
        threshold: Variance threshold (features with variance < threshold
            are removed). Default 0.0 removes constant features.

    Returns:
        DataFrame without low-variance features.

    Raises:
        ValueError: If threshold is negative.
    """
    if threshold < 0:
        raise ValueError(
            f"threshold must be non-negative. Received: {threshold}"
        )

    result = data.copy()
    numeric_cols = result.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) == 0:
        return result

    selector = VarianceThreshold(threshold=threshold)
    selector.fit(result[numeric_cols])

    retained_cols = [
        col for col, keep in zip(numeric_cols, selector.get_support())
        if keep
    ]
    dropped_cols = [c for c in numeric_cols if c not in retained_cols]

    if dropped_cols:
        result = result.drop(columns=dropped_cols)

    return result


def lasso_selection(
    data: pd.DataFrame,
    target_col: str,
    n_features: int = 10,
    alpha: float = 0.01
) -> pd.DataFrame:
    """
    Select top n features using LASSO (L1 regularization).

    LASSO performs feature selection by penalizing the absolute size
    of coefficients, forcing less important coefficients to zero.

    Args:
        data: Input DataFrame.
        target_col: Column name of the target variable.
        n_features: Number of features to retain (default 10).
        alpha: Regularization strength (default 0.01).

    Returns:
        DataFrame with selected features plus the target column.

    Raises:
        ValueError: If target_col is not found or n_features is invalid.
    """
    if target_col not in data.columns:
        raise ValueError(
            f"Target column '{target_col}' not found in DataFrame."
        )

    if n_features < 1:
        raise ValueError(
            f"n_features must be at least 1. Received: {n_features}"
        )

    result = data.copy()
    feature_cols = [c for c in result.columns if c != target_col]
    numeric_cols = result[feature_cols].select_dtypes(
        include=[np.number]
    ).columns.tolist()

    if len(numeric_cols) == 0:
        return result

    X = result[numeric_cols]
    y = result[target_col]

    # Standardize features for LASSO
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Determine if classification or regression
    unique_y = np.unique(y)
    if len(unique_y) <= 10:
        model = LogisticRegression(
            penalty='l1',
            solver='liblinear',
            C=1/alpha,
            random_state=42
        )
    else:
        model = Lasso(alpha=alpha, random_state=42)

    model.fit(X_scaled, y)

    # Get feature importance (absolute coefficients)
    importance = np.abs(model.coef_).flatten()
    top_indices = np.argsort(importance)[::-1][:n_features]
    selected_cols = [numeric_cols[i] for i in top_indices]

    retained_cols = [target_col] + selected_cols
    return result[retained_cols]


def rfe_selection(
    data: pd.DataFrame,
    target_col: str,
    n_features: int = 10
) -> pd.DataFrame:
    """
    Select top n features using Recursive Feature Elimination (RFE).

    RFE iteratively builds and refits a model, selecting or deselecting
    features based on their importance.

    Args:
        data: Input DataFrame.
        target_col: Column name of the target variable.
        n_features: Number of features to retain (default 10).

    Returns:
        DataFrame with selected features plus the target column.

    Raises:
        ValueError: If target_col is not found or n_features is invalid.
    """
    if target_col not in data.columns:
        raise ValueError(
            f"Target column '{target_col}' not found in DataFrame."
        )

    if n_features < 1:
        raise ValueError(
            f"n_features must be at least 1. Received: {n_features}"
        )

    result = data.copy()
    feature_cols = [c for c in result.columns if c != target_col]
    numeric_cols = result[feature_cols].select_dtypes(
        include=[np.number]
    ).columns.tolist()

    if len(numeric_cols) == 0:
        return result

    X = result[numeric_cols]
    y = result[target_col]

    unique_y = np.unique(y)
    if len(unique_y) <= 10:
        estimator = LogisticRegression(max_iter=1000, random_state=42)
    else:
        estimator = Lasso(alpha=0.01, random_state=42)

    selector = RFE(estimator, n_features_to_select=n_features)
    selector.fit(X, y)

    selected_cols = [
        col for col, keep in zip(numeric_cols, selector.get_support())
        if keep
    ]

    retained_cols = [target_col] + selected_cols
    return result[retained_cols]


def filter_selection(
    data: pd.DataFrame,
    target_col: str,
    threshold: float = 0.1
) -> pd.DataFrame:
    """
    Select features based on correlation with the target variable.

    Filter methods select features based on individual statistics
    (e.g., correlation with target) before modelling.

    Args:
        data: Input DataFrame.
        target_col: Column name of the target variable.
        threshold: Minimum correlation to retain a feature (default 0.1).

    Returns:
        DataFrame with selected features plus the target column.

    Raises:
        ValueError: If target_col is not found or threshold is invalid.
    """
    if target_col not in data.columns:
        raise ValueError(
            f"Target column '{target_col}' not found in DataFrame."
        )

    if not (0.0 <= threshold <= 1.0):
        raise ValueError(
            f"threshold must be between 0.0 and 1.0. Received: {threshold}"
        )

    result = data.copy()
    feature_cols = [c for c in result.columns if c != target_col]
    numeric_cols = result[feature_cols].select_dtypes(
        include=[np.number]
    ).columns.tolist()

    if len(numeric_cols) == 0:
        return result

    correlations = result[numeric_cols].corrwith(result[target_col]).abs()
    selected_cols = correlations[
        correlations >= threshold
    ].index.tolist()

    retained_cols = [target_col] + selected_cols
    return result[retained_cols]


def parse_selection_strategy(strategy: str) -> Tuple[str, float]:
    """
    Parse feature selection strategy string.

    Examples:
        "variance(threshold=0.0)" -> ("variance", 0.0)
        "lasso(n=10)" -> ("lasso", 10.0)
        "rfe(n=15)" -> ("rfe", 15.0)
        "filter(threshold=0.1)" -> ("filter", 0.1)

    Args:
        strategy: Strategy string to parse.

    Returns:
        Tuple of (method, parameter).

    Raises:
        ValueError: If strategy format is invalid.
    """
    if not isinstance(strategy, str) or not strategy:
        raise ValueError("Strategy must be a non-empty string.")

    if strategy.startswith("variance"):
        threshold = _parse_threshold(strategy, "variance")
        return ("variance", threshold)

    elif strategy.startswith("lasso"):
        n_features = _parse_n_features(strategy, "lasso")
        return ("lasso", n_features)

    elif strategy.startswith("rfe"):
        n_features = _parse_n_features(strategy, "rfe")
        return ("rfe", n_features)

    elif strategy.startswith("filter"):
        threshold = _parse_threshold(strategy, "filter")
        return ("filter", threshold)

    else:
        raise ValueError(
            f"Unknown feature selection strategy: {strategy}. "
            "Use 'variance(threshold=0.0)', 'lasso(n=10)', "
            "'rfe(n=15)', or 'filter(threshold=0.1)'."
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

        if threshold < 0:
            raise ValueError(
                f"threshold must be non-negative. Received: {threshold}"
            )

        return threshold

    except (IndexError, ValueError):
        raise ValueError(
            f"Invalid {method} format: {strategy}. "
            f"Expected: '{method}(threshold=value)'."
        )


def _parse_n_features(strategy: str, method: str) -> float:
    """
    Extract n_features from strategy string.

    Args:
        strategy: Strategy string containing n parameter.
        method: Method name for error messages.

    Returns:
        n_features as float.

    Raises:
        ValueError: If n is missing or malformed.
    """
    try:
        n_str = strategy.split("n=")[1].rstrip(")")
        n_features = float(n_str)

        if n_features < 1:
            raise ValueError(
                f"n_features must be at least 1. Received: {n_features}"
            )

        return n_features

    except (IndexError, ValueError):
        raise ValueError(
            f"Invalid {method} format: {strategy}. "
            f"Expected: '{method}(n=value)'."
        )