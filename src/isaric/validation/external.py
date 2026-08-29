"""
External validation for the RAPID methodology.

This module provides functions to validate models with independent
datasets (Step 5.1 of the RAPID methodology). External validation
applies the final, fixed model to data not used during development.

Techniques:
- temporal_validation: Validate on data from a different time period.
- geographic_validation: Validate on data from a different location.
- recalibration: Adjust model intercept or slope for new population.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple


def temporal_validation(
    fitted_model,
    external_data: pd.DataFrame,
    duration_var: Optional[str] = None,
    event_var: Optional[str] = None,
    dependent_var: Optional[str] = None,
    independent_vars: Optional[list] = None
) -> Dict[str, float]:
    """
    Validate model on data from a different time period.

    Applies the trained model to an external dataset collected after
    the development data. Returns performance metrics on the new data.

    Args:
        fitted_model: Trained model.
        external_data: Independent dataset (different time period).
        duration_var: Time-to-event column (for survival models).
        event_var: Event indicator column (for survival models).
        dependent_var: Outcome variable (for regression/classification).
        independent_vars: Predictor variables.

    Returns:
        Dictionary with performance metrics on external data.

    Raises:
        ValueError: If model type parameters are missing.
    """
    if duration_var and event_var:
        # Survival model
        X = external_data[independent_vars]
        c_index = fitted_model.concordance_index_
        return {'c_index': float(c_index)}

    elif dependent_var and independent_vars:
        # Regression/classification model
        X = external_data[independent_vars]
        y_true = external_data[dependent_var]

        if hasattr(fitted_model, 'predict_proba'):
            y_prob = fitted_model.predict_proba(X)[:, 1]
            return _compute_classification_metrics(y_true, y_prob)
        else:
            y_pred = fitted_model.predict(X)
            return _compute_regression_metrics(y_true, y_pred)

    else:
        raise ValueError(
            "Must provide either (duration_var, event_var) for survival "
            "or (dependent_var, independent_vars) for regression."
        )


def geographic_validation(
    fitted_model,
    external_data: pd.DataFrame,
    dependent_var: str,
    independent_vars: list
) -> Dict[str, float]:
    """
    Validate model on data from a different clinical centre or country.

    Args:
        fitted_model: Trained model.
        external_data: Independent dataset (different location).
        dependent_var: Outcome variable.
        independent_vars: Predictor variables.

    Returns:
        Dictionary with performance metrics on external data.
    """
    return temporal_validation(
        fitted_model,
        external_data,
        dependent_var=dependent_var,
        independent_vars=independent_vars
    )


def recalibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    method: str = "platt"
) -> Dict[str, float]:
    """
    Recalibrate model predictions using Platt scaling.

    Adjusts predicted probabilities to better match observed frequencies
    in the new population.

    Args:
        y_true: True binary labels.
        y_prob: Predicted probabilities.
        method: Recalibration method: "platt" or "isotonic".

    Returns:
        Dictionary with recalibrated intercept and slope.

    Raises:
        ValueError: If method is invalid.
    """
    if method not in ("platt", "isotonic"):
        raise ValueError(
            f"method must be 'platt' or 'isotonic'. Received: {method}"
        )

    # Platt scaling: fit logistic regression on log-odds
    log_odds = np.log(y_prob / (1 - y_prob + 1e-10))

    from sklearn.linear_model import LogisticRegression
    calibrator = LogisticRegression(C=1.0, solver='lbfgs')
    calibrator.fit(log_odds.reshape(-1, 1), y_true)

    return {
        'intercept': float(calibrator.intercept_[0]),
        'slope': float(calibrator.coef_[0][0])
    }


def _compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray
) -> Dict[str, float]:
    """
    Compute classification metrics for external validation.

    Args:
        y_true: True binary labels.
        y_prob: Predicted probabilities.

    Returns:
        Dictionary with metrics.
    """
    from sklearn.metrics import roc_auc_score, brier_score_loss

    return {
        'auc': float(roc_auc_score(y_true, y_prob)),
        'brier_score': float(brier_score_loss(y_true, y_prob))
    }


def _compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Compute regression metrics for external validation.

    Args:
        y_true: True continuous values.
        y_pred: Predicted values.

    Returns:
        Dictionary with metrics.
    """
    from sklearn.metrics import mean_squared_error, r2_score

    return {
        'mse': float(mean_squared_error(y_true, y_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'r2': float(r2_score(y_true, y_pred))
    }