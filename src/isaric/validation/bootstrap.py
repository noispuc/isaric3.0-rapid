"""
Bootstrapping for the RAPID methodology.

This module provides functions to assess the variability and stability
of model performance estimates (Step 5.2 of the RAPID methodology).
Bootstrapping repeatedly draws samples with replacement from the
original dataset to calculate confidence intervals.

Techniques:
- non_parametric_bootstrap: Draw samples with replacement.
- confidence_interval: Calculate confidence intervals from bootstrap.
- bootstrap_metrics: Bootstrap performance metrics.
"""

import pandas as pd
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple


def non_parametric_bootstrap(
    data: pd.DataFrame,
    n_iterations: int = 1000,
    random_state: int = 42
) -> List[pd.DataFrame]:
    """
    Draw bootstrap samples with replacement from the original data.

    Args:
        data: Input DataFrame.
        n_iterations: Number of bootstrap samples (default 1000).
        random_state: Seed for reproducibility.

    Returns:
        List of bootstrap samples (each is a DataFrame).

    Raises:
        ValueError: If n_iterations is invalid.
    """
    if n_iterations < 1:
        raise ValueError(
            f"n_iterations must be at least 1. Received: {n_iterations}"
        )

    rng = np.random.default_rng(random_state)
    n_samples = len(data)

    bootstrap_samples = []
    for _ in range(n_iterations):
        indices = rng.integers(0, n_samples, n_samples)
        bootstrap_samples.append(data.iloc[indices].reset_index(drop=True))

    return bootstrap_samples


def confidence_interval(
    values: np.ndarray,
    alpha: float = 0.05
) -> Tuple[float, float]:
    """
    Calculate confidence interval from bootstrap distribution.

    Args:
        values: Array of bootstrap estimates.
        alpha: Significance level (default 0.05 for 95% CI).

    Returns:
        Tuple of (lower_bound, upper_bound).

    Raises:
        ValueError: If alpha is invalid.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(
            f"alpha must be between 0.0 and 1.0. Received: {alpha}"
        )

    lower_percentile = alpha / 2 * 100
    upper_percentile = (1 - alpha / 2) * 100

    return (
        float(np.percentile(values, lower_percentile)),
        float(np.percentile(values, upper_percentile))
    )


def bootstrap_metrics(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    n_iterations: int = 1000,
    metric_func: Optional[Callable] = None,
    random_state: int = 42
) -> Dict[str, np.ndarray]:
    """
    Bootstrap performance metrics for a fitted model.

    Trains the model on bootstrap samples and evaluates on out-of-bag
    observations, or evaluates the fitted model on bootstrap samples.

    Args:
        model: Fitted model with predict() method.
        X: Predictor matrix.
        y: Outcome vector.
        n_iterations: Number of bootstrap samples (default 1000).
        metric_func: Function to compute metric (default: accuracy).
        random_state: Seed for reproducibility.

    Returns:
        Dictionary with 'values' (bootstrap metric values),
        'mean', 'ci_lower', and 'ci_upper'.

    Raises:
        ValueError: If n_iterations is invalid.
    """
    if n_iterations < 1:
        raise ValueError(
            f"n_iterations must be at least 1. Received: {n_iterations}"
        )

    if metric_func is None:
        from sklearn.metrics import accuracy_score
        metric_func = accuracy_score

    rng = np.random.default_rng(random_state)
    n_samples = len(X)
    bootstrap_values = []

    for _ in range(n_iterations):
        indices = rng.integers(0, n_samples, n_samples)
        X_boot = X.iloc[indices] if isinstance(X, pd.DataFrame) else X[indices]
        y_boot = y.iloc[indices] if isinstance(y, pd.Series) else y[indices]

        # Prediz
        y_pred = model.predict(X_boot)
        
        # Se y_pred é contínuo (probabilidades) e y é binário, converte
        if len(np.unique(y_boot)) == 2 and len(np.unique(np.round(y_pred))) == 2:
            y_pred = (y_pred >= 0.5).astype(int)
        
        value = metric_func(y_boot, y_pred)
        bootstrap_values.append(value)

    values = np.array(bootstrap_values)
    ci_lower, ci_upper = confidence_interval(values)

    return {
        'values': values,
        'mean': float(np.mean(values)),
        'ci_lower': ci_lower,
        'ci_upper': ci_upper
    }

def bootstrap_validate(
    fitted_model,
    data: pd.DataFrame,
    n_iter: int = 100,
    random_state: int = 42,
    **kwargs
) -> Dict[str, np.ndarray]:
    """
    Bootstrap validation for a fitted model.

    This function was previously used by the old model classes.
    It is retained for backward compatibility and wraps bootstrap_metrics.

    Args:
        fitted_model: Fitted model.
        data: DataFrame used for fitting.
        n_iter: Number of bootstrap iterations (default 100).
        random_state: Seed for reproducibility.
        **kwargs: Additional arguments (X, y, metric_func).

    Returns:
        Dictionary with bootstrap results.

    Raises:
        ValueError: If X and y are not provided or in kwargs.
    """
    if 'X' in kwargs and 'y' in kwargs:
        X = kwargs.pop('X')
        y = kwargs.pop('y')
    else:
        # Try to infer from data (requires dependent_var)
        raise ValueError(
            "X and y must be provided for bootstrap validation."
        )

    metric_func = kwargs.pop('metric_func', None)

    results = bootstrap_metrics(
        fitted_model,
        X,
        y,
        n_iterations=n_iter,
        metric_func=metric_func,
        random_state=random_state
    )

    return {
        'values': results['values'],
        'mean': results['mean'],
        'ci_lower': results['ci_lower'],
        'ci_upper': results['ci_upper']
    }