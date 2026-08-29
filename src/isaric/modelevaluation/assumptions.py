"""
Assumption checking for the RAPID methodology.

This module provides functions to verify model assumptions during
modelling (Step 3 of the RAPID methodology). Assumption checking
includes tests for linearity, independence of errors, normality,
multicollinearity, and influential observations.

Techniques:
- test_durbin_watson: Test autocorrelation in residuals.
- test_shapiro_wilk: Test normality of residuals.
- test_vif: Compute Variance Inflation Factor.
- test_cooks_distance: Detect influential observations.
- test_epv: Calculate Events Per Variable.
- test_proportional_hazards: Test PH assumption for Cox models.
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, Optional, Tuple
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor


def test_durbin_watson(residuals: np.ndarray) -> float:
    """
    Test for autocorrelation in residuals (Durbin-Watson).

    Values close to 2 suggest independence. Values below 1.5 suggest
    positive autocorrelation; above 2.5 suggest negative autocorrelation.

    Args:
        residuals: Array of model residuals.

    Returns:
        Durbin-Watson statistic.
    """
    from statsmodels.stats.stattools import durbin_watson
    return float(durbin_watson(residuals))


def test_shapiro_wilk(residuals: np.ndarray) -> Dict[str, float]:
    """
    Test for normality of residuals (Shapiro-Wilk).

    Args:
        residuals: Array of model residuals.

    Returns:
        Dictionary with 'statistic' and 'p_value'.
    """
    n = len(residuals)

    if n > 5000:
        # Shapiro-Wilk is not valid for N > 5000
        sample = np.random.choice(residuals, 5000, replace=False)
    else:
        sample = residuals

    statistic, p_value = stats.shapiro(sample)

    return {
        'statistic': float(statistic),
        'p_value': float(p_value)
    }


def test_vif(X: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Variance Inflation Factor (VIF) for each predictor.

    VIF measures how much the variance of a coefficient is inflated
    due to collinearity with other predictors.

    Args:
        X: Predictor matrix (DataFrame).

    Returns:
        DataFrame with columns 'feature' and 'VIF'.
    """
    X_with_const = X.copy()

    # Add constant if not present
    if not any(col.lower() in ('intercept', 'const', 'constant') for col in X_with_const.columns):
        X_with_const = X_with_const.assign(constant=1.0)

    vif_data = []
    for i, col in enumerate(X_with_const.columns):
        vif_value = variance_inflation_factor(X_with_const.values, i)
        vif_data.append({
            'feature': col,
            'VIF': round(vif_value, 2)
        })

    return pd.DataFrame(vif_data)


def test_cooks_distance(
    fitted_model,
    X: pd.DataFrame,
    y: np.ndarray
) -> Dict[str, Any]:
    """
    Detect influential observations using Cook's Distance.

    Args:
        fitted_model: Fitted model with get_influence() method.
        X: Predictor matrix.
        y: Outcome vector.

    Returns:
        Dictionary with Cook's distance values and influential indices.
    """
    influence = fitted_model.get_influence()
    cooks_d, _ = influence.cooks_distance

    n = len(cooks_d)
    threshold = 4 / n

    influential_indices = [
        i for i, val in enumerate(cooks_d) if val > threshold
    ]

    return {
        'cooks_distance': cooks_d,
        'threshold': threshold,
        'influential_indices': influential_indices,
        'n_influential': len(influential_indices)
    }


def test_epv(
    y: np.ndarray,
    n_predictors: int
) -> Dict[str, Any]:
    """
    Calculate Events Per Variable (EPV).

    For binary outcomes: EPV = (number of events) / (number of predictors).
    Values below 10 may lead to unstable coefficient estimates.

    Args:
        y: Outcome vector (binary).
        n_predictors: Number of predictor variables.

    Returns:
        Dictionary with EPV value and interpretation.
    """
    unique_y = np.unique(y)

    if len(unique_y) == 2:
        n_events = min(np.sum(y == val) for val in unique_y)
        context = "Classification (Minority Class)"
    else:
        n_events = len(y)
        context = "Linear Regression (Total N)"

    epv_value = n_events / n_predictors if n_predictors > 0 else np.inf

    if epv_value < 10:
        status = "High Risk (Overfitting Likely)"
    elif epv_value < 20:
        status = "Caution (Low Power)"
    else:
        status = "Robust"

    return {
        'epv': round(float(epv_value), 2),
        'n_events': int(n_events),
        'n_predictors': int(n_predictors),
        'context': context,
        'status': status
    }


def test_proportional_hazards(
    fitted_model,
    model_data: pd.DataFrame
) -> Dict[str, Any]:
    """
    Test Proportional Hazards assumption for Cox models.

    Args:
        fitted_model: Fitted lifelines CoxPHFitter.
        model_data: DataFrame used for fitting.

    Returns:
        Dictionary with test results.
    """
    try:
        from lifelines.statistics import proportional_hazards_test

        results = proportional_hazards_test(
            fitted_model,
            model_data,
            time_transform='rank'
        )

        min_p = results.p_value.min()

        return {
            'test_statistic': results.test_statistic.tolist(),
            'p_values': results.p_value.tolist(),
            'min_p_value': float(min_p),
            'status': "Acceptable" if min_p > 0.05 else "Warning: Violation"
        }

    except Exception as e:
        return {
            'error': str(e),
            'status': "See log for p-values"
        }


def likelihood_ratio_test(
    null_ll: float,
    alt_ll: float,
    null_dof: int,
    alt_dof: int
) -> Dict[str, float]:
    """
    Likelihood Ratio Test (LRT) for comparing nested models.

    Args:
        null_ll: Log-likelihood of the simpler (null) model.
        alt_ll: Log-likelihood of the complex (alternative) model.
        null_dof: Degrees of freedom of the null model.
        alt_dof: Degrees of freedom of the alternative model.

    Returns:
        Dictionary with LR statistic, delta DOF, and p-value.
    """
    lr_stat = 2 * (alt_ll - null_ll)
    dof_diff = abs(alt_dof - null_dof)

    if lr_stat < 0:
        lr_stat = 0.0

    p_value = stats.chi2.sf(lr_stat, dof_diff)

    return {
        'log_likelihood_diff': float(lr_stat),
        'dof_diff': int(dof_diff),
        'p_value': float(p_value)
    }