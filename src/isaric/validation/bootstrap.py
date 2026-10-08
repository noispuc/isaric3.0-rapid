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
- bootstrap_c_index: Bootstrap C-index for survival models.  # ← adicionar quando implementar B.1
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
    random_state: int = 42,
    prediction_method: Optional[str] = None
) -> Dict[str, np.ndarray]:
    """
    Bootstrap performance metrics for a fitted model.

    Trains the model on bootstrap samples and evaluates on out-of-bag
    observations, or evaluates the fitted model on bootstrap samples.

    Supports:
    - sklearn models (predict, predict_proba)
    - statsmodels models (predict)
    - lifelines CoxPHFitter (predict_partial_hazard)

    Args:
        model: Fitted model.
        X: Predictor matrix.
        y: Outcome vector.
        n_iterations: Number of bootstrap samples (default 1000).
        metric_func: Function to compute metric (default: accuracy).
        random_state: Seed for reproducibility.
        prediction_method: Override detection.
            Options: "predict", "predict_proba", "predict_partial_hazard".
            If None, auto-detects based on model type.

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

    # Detecta o método de predição
    if prediction_method is None:
        if hasattr(model, 'predict_partial_hazard'):
            # lifelines CoxPHFitter
            prediction_method = 'predict_partial_hazard'
        elif hasattr(model, 'predict_proba'):
            # sklearn classificadores
            prediction_method = 'predict_proba'
        elif hasattr(model, 'predict'):
            # sklearn regressores, statsmodels
            prediction_method = 'predict'
        else:
            raise ValueError(
                f"Model of type {type(model).__name__} has no known "
                f"prediction method. Provide 'prediction_method'."
            )

    rng = np.random.default_rng(random_state)
    n_samples = len(X)
    bootstrap_values = []

    for _ in range(n_iterations):
        indices = rng.integers(0, n_samples, n_samples)
        X_boot = X.iloc[indices].reset_index(drop=True) if isinstance(X, pd.DataFrame) else X[indices]
        y_boot = y.iloc[indices].reset_index(drop=True) if isinstance(y, pd.Series) else y[indices]

        # Aplica o método de predição detectado
        if prediction_method == 'predict_partial_hazard':
            # lifelines CoxPHFitter
            y_pred = model.predict_partial_hazard(X_boot).values
        elif prediction_method == 'predict_proba':
            # sklearn classificadores
            y_pred = model.predict_proba(X_boot)[:, 1]
        else:
            # sklearn regressores, statsmodels
            y_pred = model.predict(X_boot)

        # Detecta se o metric_func aceita probabilidades
        metric_name = getattr(metric_func, '__name__', '')

        if metric_name == 'roc_auc_score':
            # Passa probabilidades diretamente
            value = metric_func(y_boot, y_pred)
        else:
            # Converte para classes binárias (threshold 0.5)
            if len(np.unique(y_boot)) == 2:
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

def bootstrap_c_index(
    fitted_model,
    model_data: pd.DataFrame,
    duration_var: str,
    event_var: str,
    n_iterations: int = 1000,
    random_state: int = 42
) -> Dict[str, np.ndarray]:
    """
    Bootstrap C-index for survival models.

    Each bootstrap sample re-samples the rows of model_data, computes
    the partial hazard on the bootstrapped data, and calculates the
    concordance index against the bootstrapped durations and events.

    Args:
        fitted_model: Fitted lifelines CoxPHFitter.
        model_data: DataFrame used for fitting.
        duration_var: Time-to-event column.
        event_var: Event indicator column.
        n_iterations: Number of bootstrap samples (default 1000).
        random_state: Seed for reproducibility.

    Returns:
        Dictionary with 'values', 'mean', 'ci_lower', 'ci_upper'.
    """
    from lifelines.utils import concordance_index

    if n_iterations < 1:
        raise ValueError(
            f"n_iterations must be at least 1. Received: {n_iterations}"
        )

    rng = np.random.default_rng(random_state)
    n_samples = len(model_data)
    bootstrap_values = []

    for _ in range(n_iterations):
        indices = rng.integers(0, n_samples, n_samples)
        data_boot = model_data.iloc[indices].reset_index(drop=True)

        T_boot = data_boot[duration_var].values
        E_boot = data_boot[event_var].values

        # Partial hazard: risco previsto
        risk_boot = fitted_model.predict_partial_hazard(data_boot).values

        # Concordance index: risco alto → sobrevida curta
        c_idx = concordance_index(T_boot, -risk_boot, E_boot)
        bootstrap_values.append(c_idx)

    values = np.array(bootstrap_values)
    ci_lower, ci_upper = confidence_interval(values)

    return {
        'values': values,
        'mean': float(np.mean(values)),
        'ci_lower': ci_lower,
        'ci_upper': ci_upper
    }