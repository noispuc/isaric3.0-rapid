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

def test_binned_residuals(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> Dict[str, Any]:
    """
    Compute binned residuals for binary outcomes.

    Divides predicted probabilities into bins and computes the mean
    residual per bin. Residuals should fall within ±2 standard errors
    if the model is well-calibrated.

    Args:
        y_true: True binary labels (0/1).
        y_prob: Predicted probabilities.
        n_bins: Number of bins (default 10).

    Returns:
        Dictionary with:
        - 'bin_centers': mean predicted probability per bin.
        - 'mean_residuals': mean residual per bin.
        - 'se': standard error per bin.
        - 'n_per_bin': number of observations per bin.
        - 'lower_bound': mean_residuals - 2*se.
        - 'upper_bound': mean_residuals + 2*se.
    """
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()

    # Bin edges based on quantiles (uniform bins in probability space)
    edges = np.linspace(0, 1, n_bins + 1)
    # Evita extremos idênticos
    bin_idx = np.digitize(y_prob, edges[1:-1])

    # Residuals para classificação binária: y_true - y_prob
    residuals = y_true - y_prob

    bin_centers = []
    mean_residuals = []
    ses = []
    ns = []

    for i in range(n_bins):
        mask = bin_idx == i
        n = int(mask.sum())
        if n == 0:
            bin_centers.append(np.nan)
            mean_residuals.append(np.nan)
            ses.append(np.nan)
            ns.append(0)
            continue

        bin_centers.append(float(y_prob[mask].mean()))
        mean_residuals.append(float(residuals[mask].mean()))
        # SE do resíduo médio
        se = float(residuals[mask].std(ddof=1) / np.sqrt(n))
        ses.append(se)
        ns.append(n)

    bin_centers = np.array(bin_centers)
    mean_residuals = np.array(mean_residuals)
    ses = np.array(ses)

    return {
        'bin_centers': bin_centers,
        'mean_residuals': mean_residuals,
        'se': ses,
        'n_per_bin': np.array(ns),
        'lower_bound': mean_residuals - 2 * ses,
        'upper_bound': mean_residuals + 2 * ses,
    }

def bootstrap_lrt_lca(
    X: np.ndarray,
    n_components: int,
    n_bootstraps: int = 100,
    measurement: str = "bernoulli",
    random_state: int = 42,
    n_init: int = 5,
    max_iter: int = 500
) -> Dict[str, Any]:
    """
    Bootstrapped Likelihood Ratio Test for LCA (k vs k-1).

    Uses StepMix's bootstrapped_likelihood_ratio_test: fits k-1 as the
    null model, generates n_bootstraps samples under H0, fits k in each,
    and compares the observed LRT statistic against the bootstrap
    distribution.

    Args:
        X: Binary matrix (n_samples, n_features).
        n_components: Number of components under the alternative (k).
        n_bootstraps: Number of bootstrap samples (default 100).
        measurement: StepMix measurement type (default "bernoulli").
        random_state: Seed for reproducibility.
        n_init: Number of random initializations.
        max_iter: Maximum EM iterations.

    Returns:
        Dictionary with:
        - 'lrt_statistic': observed LRT statistic
        - 'p_value': bootstrapped p-value
        - 'n_components': k tested
        - 'n_bootstraps': number of bootstraps
    """
    try:
        from stepmix import bootstrapped_likelihood_ratio_test
    except ImportError:
        return {
            'error': 'stepmix.bootstrapped_likelihood_ratio_test not available',
            'n_components': n_components,
        }

    try:
        result = bootstrapped_likelihood_ratio_test(
            n_components=n_components,
            measurement=measurement,
            X=X,
            n_bootstraps=n_bootstraps,
            random_state=random_state,
            n_init=n_init,
            max_iter=max_iter,
        )

        # O resultado pode ser um objeto ou dict, dependendo da versão
        if hasattr(result, 'pvalue'):
            p_value = float(result.pvalue)
            lrt_stat = float(result.likelihood_ratio) if hasattr(result, 'likelihood_ratio') else None
        elif isinstance(result, dict):
            p_value = float(result.get('pvalue', result.get('p_value', np.nan)))
            lrt_stat = float(result.get('likelihood_ratio', result.get('lrt_stat', np.nan)))
        else:
            p_value = np.nan
            lrt_stat = np.nan

        return {
            'lrt_statistic': lrt_stat,
            'p_value': p_value,
            'n_components': n_components,
            'n_bootstraps': n_bootstraps,
        }

    except Exception as e:
        return {
            'error': str(e),
            'n_components': n_components,
        }