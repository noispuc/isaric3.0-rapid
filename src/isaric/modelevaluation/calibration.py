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
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from scipy import stats


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

    fraction_positive, mean_predicted = calibration_curve(
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

    fraction_positive, mean_predicted = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy=strategy
    )

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