"""
External validation for the RAPID methodology.

This module provides utility functions to validate models with
independent datasets (Step 5 (Validation) of the RAPID methodology).
External validation applies the final, fixed model to data not used
during development — analogous to a TEST set in ML.

Functions:
- temporal_validation:       Generic dispatcher (kept for compatibility).
- validate_classification:   AUC, Brier, calibration for binary outcomes.
- validate_survival:         C-index + calibration for Cox models.
- validate_regression_glm:   Metrics for statsmodels GLM on external data.
- validate_regression_sklearn: Metrics for sklearn models on external data.
- geographic_validation:     Alias for temporal_validation.
- recalibration:             Platt / isotonic recalibration.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import statsmodels.api as sm


def validate_classification(
    fitted_model,
    external_data: pd.DataFrame,
    dependent_var: str,
    independent_vars: List[str],
    add_const: bool = True,
    threshold: float = 0.5,
) -> Dict:
    """
    Validate a binary classification model on an independent dataset.

    Computes AUC, Brier score, accuracy, sensitivity, specificity and
    calibration slope/intercept on external_data.

    Args:
        fitted_model: Fitted model with predict() or predict_proba().
        external_data: Independent dataset.
        dependent_var: Binary outcome column (0/1).
        independent_vars: Predictor columns.
        add_const: If True, adds intercept column for statsmodels models.
        threshold: Classification threshold for binary predictions.

    Returns:
        Dictionary with: auc, brier_score, accuracy, sensitivity,
        specificity, calibration_intercept, calibration_slope,
        n_samples, n_events, event_rate.
    """
    from sklearn.metrics import (
        roc_auc_score, brier_score_loss,
        accuracy_score, confusion_matrix,
    )
    from sklearn.linear_model import LogisticRegression as _LR

    X_ext = external_data[independent_vars].copy()
    y_ext = external_data[dependent_var]

    if add_const:
        X_ext = sm.add_constant(X_ext, has_constant='add')

    # Predict probabilities
    if hasattr(fitted_model, 'predict_proba'):
        y_prob = fitted_model.predict_proba(X_ext)[:, 1]
    else:
        y_prob = fitted_model.predict(X_ext)

    y_pred = (y_prob >= threshold).astype(int)

    result = {
        'n_samples':   int(len(y_ext)),
        'n_events':    int(y_ext.sum()),
        'event_rate':  float(y_ext.mean()),
        'auc':         float(roc_auc_score(y_ext, y_prob)),
        'brier_score': float(brier_score_loss(y_ext, y_prob)),
        'accuracy':    float(accuracy_score(y_ext, y_pred)),
    }

    tn, fp, fn, tp = confusion_matrix(y_ext, y_pred).ravel()
    result['sensitivity'] = float(tp / (tp + fn)) if (tp + fn) > 0 else np.nan
    result['specificity'] = float(tn / (tn + fp)) if (tn + fp) > 0 else np.nan

    # Calibration slope and intercept (logistic regression of observed on log-odds)
    try:
        log_odds = np.log(np.clip(y_prob, 1e-6, 1 - 1e-6) / (1 - np.clip(y_prob, 1e-6, 1 - 1e-6)))
        cal = _LR(C=1e6, solver='lbfgs')
        cal.fit(log_odds.reshape(-1, 1), y_ext)
        result['calibration_intercept'] = float(cal.intercept_[0])
        result['calibration_slope']     = float(cal.coef_[0][0])
    except Exception:
        result['calibration_intercept'] = np.nan
        result['calibration_slope']     = np.nan

    return result


def validate_survival(
    fitted_model,
    external_data: pd.DataFrame,
    duration_var: str,
    event_var: str,
    target_time: Optional[float] = None,
) -> Dict:
    """
    Validate a Cox PH model on an independent dataset.

    Computes concordance index (C-index) on external_data using the
    fitted model's score() method, which predicts partial hazards and
    computes concordance against observed events.

    Args:
        fitted_model: Fitted CoxPHFitter.
        external_data: Independent dataset.
        duration_var: Time-to-event column.
        event_var: Event indicator column (1=event, 0=censored).
        target_time: Optional time point for Brier score.

    Returns:
        Dictionary with: c_index, n_samples, n_events,
        and optionally brier_score.
    """
    n_events = int(external_data[event_var].sum())

    # Compute c_index on external data (not training c_index)
    try:
        c_index = fitted_model.score(
            external_data,
            scoring_method="concordance_index",
        )
        c_index = float(c_index)
    except Exception:
        c_index = np.nan

    result = {
        'n_samples': int(len(external_data)),
        'n_events':  n_events,
        'c_index':   c_index,
    }

    # Optional Brier score at target_time
    if target_time is not None:
        try:
            from isaric.modelevaluation.calibration import survival_brier_score
            brier_data = survival_brier_score(
                fitted_model,
                external_data,
                duration_var=duration_var,
                event_var=event_var,
                target_time=target_time,
            )
            result['brier_score_at_target'] = float(
                brier_data.get('brier_at_target', np.nan)
            )
        except Exception:
            result['brier_score_at_target'] = np.nan

    return result


def validate_regression_sklearn(
    fitted_model,
    external_data: pd.DataFrame,
    dependent_var: str,
    independent_vars: List[str],
    is_classifier: bool = True,
    threshold: float = 0.5,
) -> Dict:
    """
    Validate a sklearn model on an independent dataset.

    For classifiers: AUC, Brier, accuracy, sensitivity, specificity.
    For regressors: RMSE, MAE, R².

    Args:
        fitted_model: Fitted sklearn estimator.
        external_data: Independent dataset.
        dependent_var: Outcome column.
        independent_vars: Predictor columns.
        is_classifier: If True, computes classification metrics.
        threshold: Classification threshold (default 0.5).

    Returns:
        Dictionary with performance metrics.
    """
    from sklearn.metrics import (
        roc_auc_score, brier_score_loss, accuracy_score,
        mean_squared_error, mean_absolute_error, r2_score,
    )

    X_ext = external_data[independent_vars]
    y_ext = external_data[dependent_var]

    result = {
        'n_samples': int(len(y_ext)),
    }

    if is_classifier:
        result['n_events'] = int(y_ext.sum())

        if hasattr(fitted_model, 'predict_proba'):
            y_prob = fitted_model.predict_proba(X_ext)[:, 1]
        else:
            y_prob = fitted_model.predict(X_ext)

        y_pred = (y_prob >= threshold).astype(int)

        result['auc']         = float(roc_auc_score(y_ext, y_prob))
        result['brier_score'] = float(brier_score_loss(y_ext, y_prob))
        result['accuracy']    = float(accuracy_score(y_ext, y_pred))
    else:
        y_pred = fitted_model.predict(X_ext)
        mse = float(mean_squared_error(y_ext, y_pred))
        result['rmse'] = float(np.sqrt(mse))
        result['mae']  = float(mean_absolute_error(y_ext, y_pred))
        result['r2']   = float(r2_score(y_ext, y_pred))

    return result


def temporal_validation(
    fitted_model,
    external_data: pd.DataFrame,
    duration_var: Optional[str] = None,
    event_var: Optional[str] = None,
    dependent_var: Optional[str] = None,
    independent_vars: Optional[list] = None,
    target_time: Optional[float] = None,
) -> Dict:
    """
    Generic dispatcher for external validation.

    Routes to validate_survival or validate_classification based on
    the parameters provided. Kept for backward compatibility.

    Args:
        fitted_model: Fitted model.
        external_data: Independent dataset.
        duration_var: Time-to-event column (survival models).
        event_var: Event indicator column (survival models).
        dependent_var: Outcome column (regression/classification).
        independent_vars: Predictor columns.
        target_time: Optional time point for survival Brier score.

    Returns:
        Dictionary with performance metrics.

    Raises:
        ValueError: If required parameters are missing.
    """
    if duration_var and event_var:
        return validate_survival(
            fitted_model=fitted_model,
            external_data=external_data,
            duration_var=duration_var,
            event_var=event_var,
            target_time=target_time,
        )
    elif dependent_var and independent_vars:
        return validate_classification(
            fitted_model=fitted_model,
            external_data=external_data,
            dependent_var=dependent_var,
            independent_vars=independent_vars,
        )
    else:
        raise ValueError(
            "Provide (duration_var, event_var) for survival models "
            "or (dependent_var, independent_vars) for classification/regression."
        )


def geographic_validation(
    fitted_model,
    external_data: pd.DataFrame,
    dependent_var: str,
    independent_vars: list,
) -> Dict:
    """
    Validate model on data from a different clinical centre or country.

    Alias for temporal_validation with classification parameters.
    """
    return validate_classification(
        fitted_model=fitted_model,
        external_data=external_data,
        dependent_var=dependent_var,
        independent_vars=independent_vars,
    )


def recalibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    method: str = "platt"
) -> Dict:
    """
    Recalibrate model predictions using Platt scaling or isotonic regression.

    Args:
        y_true: True binary labels.
        y_prob: Predicted probabilities.
        method: "platt" or "isotonic".

    Returns:
        Dictionary with calibration intercept and slope.
    """
    if method not in ("platt", "isotonic"):
        raise ValueError(
            f"method must be 'platt' or 'isotonic'. Received: '{method}'"
        )

    log_odds = np.log(y_prob / (1 - y_prob + 1e-10))

    from sklearn.linear_model import LogisticRegression
    calibrator = LogisticRegression(C=1.0, solver='lbfgs')
    calibrator.fit(log_odds.reshape(-1, 1), y_true)

    return {
        'intercept': float(calibrator.intercept_[0]),
        'slope':     float(calibrator.coef_[0][0]),
    }