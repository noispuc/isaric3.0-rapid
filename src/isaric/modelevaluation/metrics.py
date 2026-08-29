"""
Performance metrics for the RAPID methodology.

This module provides functions to calculate model evaluation metrics
(Step 4 of the RAPID methodology). These functions are called by the
fit() method of concrete pipeline classes to compute default and
user-specified performance metrics.

Metrics Categories:
- Classification: AUC-ROC, Accuracy, Precision, Recall, F1, Log Loss, Brier Score.
- Regression: MSE, RMSE, MAE, R², Adjusted R².
- Survival: C-index, Brier Score.
- Information Criteria: AIC, BIC, McFadden R², Efron R².
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    log_loss,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    confusion_matrix,
    brier_score_loss
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute classification performance metrics.

    Args:
        y_true: True binary labels (0/1).
        y_pred: Predicted class labels (0/1).
        y_prob: Predicted probabilities (optional, for AUC and Log Loss).

    Returns:
        Dictionary with metric names and values.

    Raises:
        ValueError: If y_true is not binary.
    """
    _validate_binary(y_true)

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }

    if y_prob is not None:
        metrics['auc'] = roc_auc_score(y_true, y_prob)
        metrics['log_loss'] = log_loss(y_true, y_prob)
        metrics['brier_score'] = brier_score_loss(y_true, y_prob)

    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm.tolist()

    return metrics


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Compute regression performance metrics.

    Args:
        y_true: True continuous values.
        y_pred: Predicted continuous values.

    Returns:
        Dictionary with metric names and values.
    """
    metrics = {
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
    }

    n = len(y_true)
    p = 1  # Number of predictors (simplified)
    adjusted_r2 = 1 - (1 - metrics['r2']) * ((n - 1) / (n - p - 1))
    metrics['adjusted_r2'] = adjusted_r2 if n > p + 1 else np.nan

    return metrics


def compute_information_criteria(
    fitted_model,
    n_samples: int,
    n_params: int
) -> Dict[str, float]:
    """
    Compute AIC and BIC from a fitted model.

    Args:
        fitted_model: Fitted model with llf or log_likelihood attribute.
        n_samples: Number of observations.
        n_params: Number of model parameters.

    Returns:
        Dictionary with AIC and BIC.
    """
    if hasattr(fitted_model, 'llf'):
        llf = fitted_model.llf
    elif hasattr(fitted_model, 'log_likelihood_'):
        llf = fitted_model.log_likelihood_
    else:
        raise ValueError("Fitted model does not have log-likelihood attribute.")

    aic = -2 * llf + 2 * n_params
    bic = -2 * llf + n_params * np.log(n_samples)

    return {'aic': aic, 'bic': bic, 'llf': llf}


def compute_pseudo_r2(
    fitted_model,
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Compute pseudo R² metrics for GLM models.

    Args:
        fitted_model: Fitted GLM model with llf and llnull.
        y_true: True outcome values.
        y_pred: Predicted values.

    Returns:
        Dictionary with McFadden R², Efron R², and related metrics.
    """
    ll_model = fitted_model.llf
    ll_null = fitted_model.llnull
    n = len(y_true)

    mcfadden_r2 = 1 - (ll_model / ll_null)

    y_array = np.asarray(y_true).ravel()
    fitted = np.asarray(y_pred).ravel()
    y_mean = y_array.mean()

    efron_r2 = 1 - (
        np.sum((y_array - fitted) ** 2) /
        np.sum((y_array - y_mean) ** 2)
    )

    return {
        'mcfadden_r2': mcfadden_r2,
        'efron_r2': efron_r2,
    }


def compute_survival_metrics(
    fitted_model,
    model_data: pd.DataFrame,
    duration_var: str,
    event_var: str
) -> Dict[str, float]:
    """
    Compute survival model metrics (C-index, AIC, BIC).

    Args:
        fitted_model: Fitted lifelines CoxPHFitter.
        model_data: DataFrame used for fitting.
        duration_var: Time-to-event column.
        event_var: Event indicator column.

    Returns:
        Dictionary with C-index, AIC, BIC.
    """
    c_index = fitted_model.concordance_index_
    aic = fitted_model.AIC_partial_

    n = model_data.shape[0]
    k = len(fitted_model.params_)
    log_likelihood = fitted_model.log_likelihood_
    bic = -2 * log_likelihood + k * np.log(n)

    return {
        'c_index': c_index,
        'aic': aic,
        'bic': bic,
    }


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> Dict[str, float]:
    """
    Compute calibration metrics (Brier Score and calibration curve).

    Args:
        y_true: True binary labels.
        y_prob: Predicted probabilities.
        n_bins: Number of bins for calibration curve.

    Returns:
        Dictionary with Brier Score and calibration points.
    """
    brier = brier_score_loss(y_true, y_prob)

    # Calibration curve (simplified)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges[1:-1])

    calib_points = []
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            mean_pred = y_prob[mask].mean()
            mean_true = y_true[mask].mean()
            calib_points.append({'predicted': mean_pred, 'observed': mean_true})

    return {
        'brier_score': brier,
        'calibration_curve': calib_points,
    }


def compute_clustering_metrics(
    fitted_model,
    X: np.ndarray,
    y: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute clustering model metrics (AIC, BIC, Entropy).

    Args:
        fitted_model: Fitted clustering model (StepMix or similar).
        X: Data matrix used for fitting.
        y: Optional structural variable.

    Returns:
        Dictionary with AIC, BIC, Entropy.
    """
    if hasattr(fitted_model, 'aic') and hasattr(fitted_model, 'bic'):
        aic = fitted_model.aic(X, y)
        bic = fitted_model.bic(X, y)
    else:
        aic = np.nan
        bic = np.nan

    if hasattr(fitted_model, 'entropy'):
        entropy = fitted_model.entropy(X)
    else:
        entropy = np.nan

    return {
        'aic': aic,
        'bic': bic,
        'entropy': entropy,
    }


def select_classification_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    method: str = "f1"
) -> float:
    """
    Select optimal classification threshold.

    Args:
        y_true: True binary labels.
        y_prob: Predicted probabilities.
        method: Threshold selection method: "f1" or "youden".

    Returns:
        Optimal threshold value.

    Raises:
        ValueError: If method is unknown.
    """
    if method not in ("f1", "youden"):
        raise ValueError(f"method must be 'f1' or 'youden'. Received: {method}")

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)

    if method == "youden":
        # Youden's J statistic: max(tpr - fpr)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        return thresholds[best_idx] if best_idx < len(thresholds) else 0.5

    elif method == "f1":
        # Find threshold that maximizes F1 score
        best_threshold = 0.5
        best_f1 = 0.0

        for threshold in thresholds:
            y_pred = (y_prob >= threshold).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        return best_threshold


def _validate_binary(y: np.ndarray) -> None:
    """
    Validate that array is binary (0/1).

    Args:
        y: Input array.

    Raises:
        ValueError: If array is not binary.
    """
    unique_values = np.unique(y)
    if len(unique_values) != 2:
        raise ValueError(
            f"y must be binary. Found {len(unique_values)} unique values."
        )
    if not set(unique_values).issubset({0, 1}):
        raise ValueError(f"y must be coded as 0/1. Found: {unique_values}")