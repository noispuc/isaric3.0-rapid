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
    average_precision_score,
    roc_curve,
    log_loss,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    confusion_matrix,
    brier_score_loss,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute classification performance metrics.

    Includes sensitivity, specificity, PPV, NPV, AUC-ROC, AUC-PR,
    Brier score, and calibration slope/intercept when y_prob is provided.

    Calibration slope: logistic regression of y_true on logit(y_prob)
        — free slope, measures calibration accuracy.
    Calibration intercept: logistic regression with slope fixed at 1
        — calibration-in-the-large, measures overall calibration.

    Args:
        y_true: True binary labels (0/1).
        y_pred: Predicted class labels (0/1).
        y_prob: Predicted probabilities (optional).

    Returns:
        Dictionary with metric names and values.

    Raises:
        ValueError: If y_true is not binary.
    """
    _validate_binary(y_true)

    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0

    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else np.nan
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else np.nan
    ppv         = float(tp / (tp + fp)) if (tp + fp) > 0 else np.nan
    npv         = float(tn / (tn + fn)) if (tn + fn) > 0 else np.nan

    metrics = {
        'accuracy':    float(accuracy_score(y_true, y_pred)),
        'sensitivity': sensitivity,               # TP / (TP + FN)
        'recall':      sensitivity,               # alias
        'specificity': specificity,               # TN / (TN + FP)
        'ppv':         ppv,                       # TP / (TP + FP) = precision
        'precision':   ppv,                       # alias
        'npv':         npv,                       # TN / (TN + FN)
        'f1':          float(f1_score(y_true, y_pred, zero_division=0)),
        'confusion_matrix': cm.tolist(),
    }

    if y_prob is not None:
        metrics['auc']         = float(roc_auc_score(y_true, y_prob))
        metrics['auc_pr']      = float(average_precision_score(y_true, y_prob))
        metrics['log_loss']    = float(log_loss(y_true, y_prob))
        metrics['brier_score'] = float(brier_score_loss(y_true, y_prob))

        # Calibration metrics — diagnostic (not recalibration)
        try:
            from sklearn.linear_model import LogisticRegression as _LR
            y_prob_c = np.clip(y_prob, 1e-7, 1 - 1e-7)
            log_odds = np.log(y_prob_c / (1 - y_prob_c))

            # Calibration slope (free logistic regression)
            cal_free = _LR(C=1e6, solver='lbfgs', max_iter=500)
            cal_free.fit(log_odds.reshape(-1, 1), y_true)
            metrics['calibration_slope']     = float(cal_free.coef_[0][0])

            # Calibration intercept — slope fixed at 1 (calibration-in-the-large)
            log_odds_offset = log_odds - log_odds  # offset = 0
            cal_intercept   = _LR(C=1e6, solver='lbfgs', max_iter=500)
            # Fix slope at 1 using offset trick
            y_offset = y_prob_c - 0.5   # not ideal — use proper offset
            # Correct approach: logit(p) is fixed offset, intercept only
            from scipy.optimize import minimize_scalar
            def _neg_ll(intercept):
                p = 1 / (1 + np.exp(-(log_odds + intercept)))
                p = np.clip(p, 1e-10, 1 - 1e-10)
                return -np.sum(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))
            res = minimize_scalar(_neg_ll, bounds=(-10, 10), method='bounded')
            metrics['calibration_intercept'] = float(res.x)

        except Exception:
            metrics['calibration_slope']     = np.nan
            metrics['calibration_intercept'] = np.nan

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
    Compute survival model metrics (C-index, AIC, BIC, LRT).

    Args:
        fitted_model: Fitted lifelines CoxPHFitter.
        model_data: DataFrame used for fitting.
        duration_var: Time-to-event column.
        event_var: Event indicator column.

    Returns:
        Dictionary with C-index, AIC, BIC, LRT.
    """
    c_index = fitted_model.concordance_index_
    aic = fitted_model.AIC_partial_

    n = model_data.shape[0]
    k = len(fitted_model.params_)
    log_likelihood = fitted_model.log_likelihood_
    bic = -2 * log_likelihood + k * np.log(n)

    metrics = {
        'c_index': c_index,
        'aic': aic,
        'bic': bic,
    }

    # Log-Likelihood Ratio Test
    try:
        lrt = fitted_model.log_likelihood_ratio_test()
        metrics['lrt_statistic'] = float(lrt.test_statistic)
        metrics['lrt_p_value'] = float(lrt.p_value)
        metrics['lrt_dof'] = int(lrt.degrees_freedom)
    except Exception:
        pass

    return metrics


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
    method: str = "f1",
    grid_step: float = 0.005,
) -> float:
    """
    Select optimal classification threshold on training data.

    Args:
        y_true: True binary labels (training set).
        y_prob: Predicted probabilities (training set).
        method: Threshold selection method:
            - "f1": Maximises F1-score. Uses explicit grid from 0.01 to
              0.95 in steps of grid_step (default 0.005). This matches the
              paper method and is fully reproducible.
            - "youden": Maximises Youden's J statistic (sensitivity +
              specificity - 1) using ROC curve thresholds.
        grid_step: Step size for "f1" grid (default 0.005).

    Returns:
        Optimal threshold value (float between 0 and 1).

    Raises:
        ValueError: If method is unknown.
    """
    if method not in ("f1", "youden"):
        raise ValueError(
            f"method must be 'f1' or 'youden'. Received: '{method}'"
        )

    if method == "youden":
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        j_scores  = tpr - fpr
        best_idx  = int(np.argmax(j_scores))
        return float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5

    # method == "f1" — explicit grid (reproducible, matches paper)
    thresholds  = np.arange(0.01, 0.95 + grid_step, grid_step)
    best_threshold = 0.5
    best_f1        = 0.0

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        score  = f1_score(y_true, y_pred, zero_division=0)
        if score > best_f1:
            best_f1        = score
            best_threshold = float(t)

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


def bootstrap_metric_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    n_iterations: int = 1000,
    alpha: float = 0.05,
    random_state: int = 42,
) -> Dict[str, float]:
    """
    Bootstrap percentile confidence intervals for classification metrics.

    Resamples (y_true, y_prob) from the test set with replacement and
    computes metrics on each resample. CIs are percentile-based
    (2.5th and 97.5th percentiles for 95% CI).

    Computed for: AUC, AUC-PR, Brier score, F1, sensitivity,
    specificity, PPV, calibration slope, calibration intercept.

    Args:
        y_true: True binary labels (test set).
        y_prob: Predicted probabilities (test set).
        threshold: Classification threshold for binary predictions.
        n_iterations: Number of bootstrap resamples (default 1000).
        alpha: Significance level (default 0.05 → 95% CI).
        random_state: Seed for reproducibility.

    Returns:
        Dictionary with {metric}_ci_lower and {metric}_ci_upper for
        each computed metric. Also includes {metric}_mean.
    """
    rng = np.random.default_rng(random_state)
    n   = len(y_true)

    metric_keys = [
        'auc', 'auc_pr', 'brier_score', 'f1',
        'sensitivity', 'specificity', 'ppv',
        'calibration_slope', 'calibration_intercept',
    ]

    boot_values: Dict[str, List[float]] = {k: [] for k in metric_keys}

    for _ in range(n_iterations):
        idx        = rng.integers(0, n, n)
        y_true_b   = y_true[idx]
        y_prob_b   = y_prob[idx]

        # Skip iterations with only one class (invalid AUC)
        if len(np.unique(y_true_b)) < 2:
            continue

        y_pred_b = (y_prob_b >= threshold).astype(int)

        try:
            boot_values['auc'].append(float(roc_auc_score(y_true_b, y_prob_b)))
        except Exception:
            pass

        try:
            boot_values['auc_pr'].append(
                float(average_precision_score(y_true_b, y_prob_b))
            )
        except Exception:
            pass

        try:
            boot_values['brier_score'].append(
                float(brier_score_loss(y_true_b, y_prob_b))
            )
        except Exception:
            pass

        try:
            boot_values['f1'].append(
                float(f1_score(y_true_b, y_pred_b, zero_division=0))
            )
        except Exception:
            pass

        try:
            cm_b = confusion_matrix(y_true_b, y_pred_b)
            if cm_b.shape == (2, 2):
                tn_b, fp_b, fn_b, tp_b = cm_b.ravel()
                if (tp_b + fn_b) > 0:
                    boot_values['sensitivity'].append(tp_b / (tp_b + fn_b))
                if (tn_b + fp_b) > 0:
                    boot_values['specificity'].append(tn_b / (tn_b + fp_b))
                if (tp_b + fp_b) > 0:
                    boot_values['ppv'].append(tp_b / (tp_b + fp_b))
        except Exception:
            pass

        try:
            from sklearn.linear_model import LogisticRegression as _LR
            from scipy.optimize import minimize_scalar

            y_prob_c = np.clip(y_prob_b, 1e-7, 1 - 1e-7)
            log_odds = np.log(y_prob_c / (1 - y_prob_c))

            cal = _LR(C=1e6, solver='lbfgs', max_iter=200)
            cal.fit(log_odds.reshape(-1, 1), y_true_b)
            boot_values['calibration_slope'].append(float(cal.coef_[0][0]))

            def _neg_ll(intercept):
                p = 1 / (1 + np.exp(-(log_odds + intercept)))
                p = np.clip(p, 1e-10, 1 - 1e-10)
                return -np.sum(
                    y_true_b * np.log(p) + (1 - y_true_b) * np.log(1 - p)
                )
            res = minimize_scalar(_neg_ll, bounds=(-10, 10), method='bounded')
            boot_values['calibration_intercept'].append(float(res.x))
        except Exception:
            pass

    # Compute CIs
    result = {}
    lo_pct = (alpha / 2) * 100
    hi_pct = (1 - alpha / 2) * 100

    for key, values in boot_values.items():
        if len(values) < 10:
            result[f'{key}_mean']     = np.nan
            result[f'{key}_ci_lower'] = np.nan
            result[f'{key}_ci_upper'] = np.nan
        else:
            arr = np.array(values)
            result[f'{key}_mean']     = float(np.mean(arr))
            result[f'{key}_ci_lower'] = float(np.percentile(arr, lo_pct))
            result[f'{key}_ci_upper'] = float(np.percentile(arr, hi_pct))

    result['n_iterations']  = n_iterations
    result['n_valid']       = min(len(v) for v in boot_values.values() if v)
    result['bootstrap_type'] = 'percentile'

    return result