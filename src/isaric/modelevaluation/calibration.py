"""
Calibration curves and diagnostics for the RAPID methodology.

This module provides functions to assess the agreement between predicted
probabilities (or risks) and observed outcome frequencies (Step 4 of the
RAPID methodology). A well-calibrated model has predicted risks that
match reality.

Techniques:
- calibration_curve: Compute calibration curve for binary outcomes.
- binned_calibration: Compute binned calibration metrics.
- compute_brier_score: Compute Brier Score.
- predicted_vs_observed: Compare predicted vs observed for numeric outcomes.
- survival_calibration: Compute calibration for survival models.
- residuals_vs_fitted: Compute residuals vs fitted values.
- qq_plot: Compute Q-Q plot data for normality assessment.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import brier_score_loss
from scipy import stats
from sklearn.calibration import calibration_curve as sklearn_calibration_curve


def calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    strategy: str = "quantile"
) -> Dict[str, np.ndarray]:
    """
    Compute calibration curve for binary outcomes.

    Groups observations into bins based on predicted probabilities and
    calculates the observed frequency of positive outcomes per bin.

    Args:
        y_true: True binary labels (0/1).
        y_prob: Predicted probabilities.
        n_bins: Number of bins (default 10).
        strategy: Binning strategy: "uniform" or "quantile".

    Returns:
        Dictionary with 'fraction_positive' and 'mean_predicted'.

    Raises:
        ValueError: If strategy is invalid.
    """
    if strategy not in ("uniform", "quantile"):
        raise ValueError(
            f"strategy must be 'uniform' or 'quantile'. Received: {strategy}"
        )

    fraction_positive, mean_predicted = sklearn_calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy=strategy
    )

    return {
        'fraction_positive': fraction_positive,
        'mean_predicted': mean_predicted
    }


def binned_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    strategy: str = "quantile"
) -> pd.DataFrame:
    """
    Compute binned calibration metrics as a DataFrame.

    Args:
        y_true: True binary labels.
        y_prob: Predicted probabilities.
        n_bins: Number of bins (default 10).
        strategy: Binning strategy.

    Returns:
        DataFrame with columns: Bin, Predicted_Mean, Observed_Mean,
        Count, Difference.

    Raises:
        ValueError: If strategy is invalid.
    """
    if strategy not in ("uniform", "quantile"):
        raise ValueError(
            f"strategy must be 'uniform' or 'quantile'. Received: {strategy}"
        )

    result = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy=strategy
    )
    fraction_positive = result['fraction_positive']
    mean_predicted = result['mean_predicted']

    # Count observations per bin
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges[1:-1])
    counts = [np.sum(bin_indices == i) for i in range(n_bins)]

    rows = []
    for i in range(len(mean_predicted)):
        rows.append({
            'Bin': i + 1,
            'Predicted_Mean': round(mean_predicted[i], 4),
            'Observed_Mean': round(fraction_positive[i], 4),
            'Count': counts[i],
            'Difference': round(fraction_positive[i] - mean_predicted[i], 4)
        })

    return pd.DataFrame(rows)


def compute_brier_score(
    y_true: np.ndarray,
    y_prob: np.ndarray
) -> float:
    """
    Compute Brier Score for binary outcomes.

    Brier Score measures the mean squared difference between predicted
    probabilities and actual outcomes. Lower is better (0 = perfect).

    Args:
        y_true: True binary labels (0/1).
        y_prob: Predicted probabilities.

    Returns:
        Brier Score value.
    """
    return float(brier_score_loss(y_true, y_prob))


def predicted_vs_observed(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Compare predicted vs observed values for numeric outcomes.

    Used to assess agreement between continuous predicted values and
    true observed values.

    Args:
        y_true: True continuous values.
        y_pred: Predicted continuous values.

    Returns:
        Dictionary with 'predicted' and 'observed' arrays.
    """
    return {
        'predicted': np.asarray(y_pred),
        'observed': np.asarray(y_true)
    }


def survival_calibration(
    fitted_model,
    model_data: pd.DataFrame,
    duration_var: str,
    event_var: str,
    target_time: float
) -> Dict[str, np.ndarray]:
    """
    Compute calibration for survival models at a specific time point.

    Compares predicted survival probabilities with observed Kaplan-Meier
    estimates at the given time point.

    Args:
        fitted_model: Fitted lifelines CoxPHFitter.
        model_data: DataFrame used for fitting.
        duration_var: Time-to-event column.
        event_var: Event indicator column.
        target_time: Time point for evaluation.

    Returns:
        Dictionary with 'predicted_survival' and 'observed_survival'.
    """
    from lifelines import KaplanMeierFitter

    # Predicted survival probabilities at target time
    predicted_survival = fitted_model.predict_survival_function(
        model_data, times=[target_time]
    ).squeeze()

    # Observed survival via Kaplan-Meier
    kmf = KaplanMeierFitter()
    kmf.fit(
        model_data[duration_var],
        event_observed=model_data[event_var]
    )
    observed_survival = kmf.predict(target_time)

    return {
        'predicted_survival': predicted_survival.values,
        'observed_survival': np.repeat(observed_survival, len(predicted_survival))
    }


def survival_brier_score(
    fitted_model,
    model_data: pd.DataFrame,
    duration_var: str,
    event_var: str,
    target_time: float,
    n_points: int = 100
) -> Dict[str, np.ndarray]:
    """
    Compute IPCW Brier Score for survival models.

    Uses Inverse Probability of Censoring Weighting (IPCW) to estimate
    the time-dependent Brier Score, which measures the mean squared
    difference between predicted survival probabilities and observed
    outcomes.

    Args:
        fitted_model: Fitted lifelines CoxPHFitter.
        model_data: DataFrame used for fitting.
        duration_var: Time-to-event column.
        event_var: Event indicator column (1=event, 0=censored).
        target_time: Maximum time point for evaluation.
        n_points: Number of time points (default 100).

    Returns:
        Dictionary with 'time_points' and 'brier_scores' arrays.
    """
    from lifelines import KaplanMeierFitter

    T = model_data[duration_var]
    E = model_data[event_var]

    max_data_time = T.max()
    eval_time = min(target_time, max_data_time)

    # Time points (from first event to eval_time)
    min_time = T[E == 1].min() if (E == 1).any() else T.min()
    time_points = np.linspace(min_time, eval_time, n_points)

    # IPCW: Kaplan-Meier of censoring distribution
    kmf_censoring = KaplanMeierFitter().fit(T, 1 - E)
    G_T = kmf_censoring.predict(T, interpolate=True)

    brier_scores = []
    for t in time_points:
        predicted_probs = fitted_model.predict_survival_function(
            model_data, times=[t]
        ).squeeze()
        G_t = kmf_censoring.predict(t, interpolate=True)

        # Term 1: subjects with event before t
        is_event_before_t = (T <= t) & (E == 1)
        term1 = np.sum(
            ((predicted_probs[is_event_before_t] - 0) ** 2) / G_T[is_event_before_t]
        )

        # Term 2: subjects without event by t
        is_after_t = T > t
        term2 = np.sum(
            ((predicted_probs[is_after_t] - 1) ** 2) / G_t
        )

        score = (term1 + term2) / len(model_data)
        brier_scores.append(score)

    return {
        'time_points': time_points,
        'brier_scores': np.array(brier_scores)
    }


def survival_roc(
    fitted_model,
    model_data: pd.DataFrame,
    duration_var: str,
    event_var: str,
    target_time: float
) -> Dict[str, np.ndarray]:
    """
    Compute time-dependent ROC for survival models.

    Classifies subjects as "event before target_time" (1) or
    "no event by target_time" (0), then computes ROC using partial
    hazard as risk score.

    Args:
        fitted_model: Fitted lifelines CoxPHFitter.
        model_data: DataFrame used for fitting.
        duration_var: Time-to-event column.
        event_var: Event indicator column (1=event, 0=censored).
        target_time: Time point for evaluation.

    Returns:
        Dictionary with 'fpr', 'tpr', 'auc'.
    """
    from sklearn.metrics import roc_curve, roc_auc_score

    T = model_data[duration_var]
    E = model_data[event_var]

    # Risk scores (partial hazard)
    risk_scores = fitted_model.predict_partial_hazard(model_data).values

    # Mask and true labels
    mask = ((T <= target_time) & (E == 1)) | (T > target_time)
    y_true = ((T <= target_time) & (E == 1)).astype(int)

    fpr, tpr, _ = roc_curve(y_true[mask], risk_scores[mask])
    auc_val = roc_auc_score(y_true[mask], risk_scores[mask])

    return {
        'fpr': fpr,
        'tpr': tpr,
        'auc': float(auc_val)
    }


def residuals_vs_fitted(
    residuals: np.ndarray,
    fitted_values: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Compute residuals vs fitted values for regression diagnostics.

    Used to assess homoscedasticity (constant variance) and linearity
    assumptions. A random scatter around zero suggests assumptions hold.

    Args:
        residuals: Model residuals (observed - predicted).
        fitted_values: Fitted/predicted values.

    Returns:
        Dictionary with 'residuals' and 'fitted' arrays.
    """
    return {
        'residuals': np.asarray(residuals),
        'fitted': np.asarray(fitted_values)
    }


def qq_plot(
    residuals: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Compute Q-Q plot data for normality assessment.

    Compares the quantiles of residuals against theoretical quantiles
    of a normal distribution. Points following the diagonal line
    indicate normality.

    Args:
        residuals: Model residuals.

    Returns:
        Dictionary with 'theoretical_quantiles' and 'sample_quantiles'.
    """
    theoretical_quantiles, sample_quantiles = stats.probplot(
        residuals, dist="norm"
    )

    return {
        'theoretical_quantiles': theoretical_quantiles[0],
        'sample_quantiles': theoretical_quantiles[1]
    }