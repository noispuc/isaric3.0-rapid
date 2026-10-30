"""
Bootstrapping for the RAPID methodology.

This module provides functions for bootstrap confidence intervals and
statistical inference. Bootstrap is called during fit() when bootstrap=True,
not during validation().

Bootstrap is applicable to:
- Regression (logistic, GLM): CIs for coefficients + bootstrapped p-values.
- Survival (CoxPH): CIs for hazard ratios.
- Clustering (LCA): CIs for metrics by k (parametric bootstrap).
- Clustering (KMeans): CIs for centroid stability + silhouette.
- Tree-based / Predictive: CIs for feature importance / coefficients.

Types of bootstrap CIs:
- percentile: Simple quantile-based CI [α/2, 1-α/2] percentiles.
- bias_corrected (BC): Corrects for median bias in the bootstrap
  distribution using the z0 correction factor.
- bca: Bias-corrected and accelerated. Standard for regression
  coefficients; accounts for both bias and skewness via jackknife
  acceleration. Computationally more expensive than BC.
- parametric: Samples from the fitted model's distribution (LCA only).
  Uses bootstrap_stepmix from gridsearch module.

Functions:
- non_parametric_bootstrap: Draw samples with replacement.
- confidence_interval: Compute CI from bootstrap distribution.
- bootstrap_metrics: Bootstrap performance metrics.
- bootstrap_c_index: Bootstrap C-index for survival models.
- bootstrap_coefficients: Bootstrap CIs + p-values for coefficients.
- bootstrap_hazard_ratios: Bootstrap CIs for Cox hazard ratios.
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
    alpha: float = 0.05,
    bootstrap_type: str = "percentile",
    observed: Optional[float] = None,
    jackknife_values: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """
    Compute confidence interval from a bootstrap distribution.

    Supports three non-parametric CI types. The "percentile" type
    requires only the bootstrap values. "bias_corrected" and "bca"
    additionally require the observed estimate. "bca" further requires
    jackknife leave-one-out estimates for the acceleration factor.

    Args:
        values: Array of bootstrap estimates (1D, length B).
        alpha: Significance level (default 0.05 → 95% CI).
        bootstrap_type: CI computation method.
            - "percentile": Simple quantile-based [α/2, 1-α/2].
            - "bias_corrected": BC correction via z0.
            - "bca": BC + acceleration via jackknife (most accurate).
            Default "percentile".
        observed: Observed estimate from the original data. Required
            for "bias_corrected" and "bca". If None and type requires
            it, falls back to "percentile" with a warning.
        jackknife_values: Leave-one-out estimates for BCa acceleration
            factor (1D, length n). Required for "bca". If None and
            type is "bca", falls back to "bias_corrected".

    Returns:
        Tuple of (lower_bound, upper_bound).

    Raises:
        ValueError: If alpha is invalid or bootstrap_type is unknown.
    """
    import warnings
    from scipy.stats import norm

    if not (0.0 < alpha < 1.0):
        raise ValueError(
            f"alpha must be between 0.0 and 1.0. Received: {alpha}"
        )

    valid_types = {"percentile", "bias_corrected", "bca"}
    if bootstrap_type not in valid_types:
        raise ValueError(
            f"bootstrap_type must be one of {valid_types}. "
            f"Received: '{bootstrap_type}'"
        )

    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    B = len(values)

    if B < 2:
        return (np.nan, np.nan)

    lower_pct = alpha / 2 * 100
    upper_pct = (1 - alpha / 2) * 100

    # ── Percentile ────────────────────────────────────────────────────────
    if bootstrap_type == "percentile":
        return (
            float(np.percentile(values, lower_pct)),
            float(np.percentile(values, upper_pct)),
        )

    # ── Bias-corrected (BC) and BCa — require observed ────────────────────
    if observed is None:
        warnings.warn(
            f"bootstrap_type='{bootstrap_type}' requires observed estimate. "
            f"Falling back to 'percentile'.",
            UserWarning,
            stacklevel=2,
        )
        return (
            float(np.percentile(values, lower_pct)),
            float(np.percentile(values, upper_pct)),
        )

    z_alpha_lo = norm.ppf(alpha / 2)
    z_alpha_hi = norm.ppf(1 - alpha / 2)

    prop_below = np.mean(values < observed)
    prop_below = np.clip(prop_below, 1e-6, 1 - 1e-6)
    z0 = norm.ppf(prop_below)

    # ── Bias-corrected (BC) ───────────────────────────────────────────────
    if bootstrap_type == "bias_corrected":
        phi_lo = norm.cdf(2 * z0 + z_alpha_lo)
        phi_hi = norm.cdf(2 * z0 + z_alpha_hi)
        idx_lo = int(np.clip(np.floor(phi_lo * B), 0, B - 1))
        idx_hi = int(np.clip(np.ceil(phi_hi * B), 0, B - 1))
        sorted_vals = np.sort(values)
        return (float(sorted_vals[idx_lo]), float(sorted_vals[idx_hi]))

    # ── BCa ───────────────────────────────────────────────────────────────
    if jackknife_values is None:
        warnings.warn(
            "bootstrap_type='bca' requires jackknife_values for acceleration. "
            "Falling back to 'bias_corrected'.",
            UserWarning,
            stacklevel=2,
        )
        phi_lo = norm.cdf(2 * z0 + z_alpha_lo)
        phi_hi = norm.cdf(2 * z0 + z_alpha_hi)
        idx_lo = int(np.clip(np.floor(phi_lo * B), 0, B - 1))
        idx_hi = int(np.clip(np.ceil(phi_hi * B), 0, B - 1))
        sorted_vals = np.sort(values)
        return (float(sorted_vals[idx_lo]), float(sorted_vals[idx_hi]))

    jk = np.asarray(jackknife_values, dtype=float)
    jk = jk[~np.isnan(jk)]
    jk_mean = np.mean(jk)
    diff = jk_mean - jk
    num = np.sum(diff ** 3)
    den = np.sum(diff ** 2) ** 1.5
    a = float(num / (6 * den)) if den > 0 else 0.0

    def _bca_phi(z_alpha):
        denom = 1 - a * (z0 + z_alpha)
        if abs(denom) < 1e-10:
            return z_alpha
        return norm.cdf(z0 + (z0 + z_alpha) / denom)

    phi_lo = _bca_phi(z_alpha_lo)
    phi_hi = _bca_phi(z_alpha_hi)
    idx_lo = int(np.clip(np.floor(phi_lo * B), 0, B - 1))
    idx_hi = int(np.clip(np.ceil(phi_hi * B), 0, B - 1))
    sorted_vals = np.sort(values)
    return (float(sorted_vals[idx_lo]), float(sorted_vals[idx_hi]))

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


def bootstrap_coefficients(
    X: pd.DataFrame,
    y: pd.Series,
    fit_func: Callable,
    n_iterations: int = 1000,
    bootstrap_type: str = "bias_corrected",
    alpha: float = 0.05,
    random_state: int = 42,
    variable_names: Optional[List[str]] = None,
    collect_metrics: bool = False,
):
    """
    Bootstrap confidence intervals and p-values for model coefficients.

    Repeatedly resamples (X, y) with replacement, fits the model via
    fit_func, and collects coefficient estimates. Computes CIs of the
    requested type and bootstrapped p-values.

    fit_func can return:
    - np.ndarray of shape (p,): coefficients only (default behaviour).
    - Tuple (np.ndarray, dict): coefficients + metrics dict. When a
      tuple is returned and collect_metrics=True, metrics are collected
      across all iterations and returned as a separate DataFrame.

    Args:
        X: Predictor matrix (n_samples, n_features).
        y: Outcome vector (n_samples,).
        fit_func: Callable(X_boot, y_boot) → np.ndarray or
            Tuple[np.ndarray, dict].
        n_iterations: Number of bootstrap samples (default 1000).
        bootstrap_type: CI type: "percentile", "bias_corrected", "bca".
            Default "bias_corrected".
        alpha: Significance level (default 0.05 → 95% CI).
        random_state: Seed for reproducibility.
        variable_names: Variable names. If None, uses X.columns.
        collect_metrics: If True and fit_func returns a tuple, collects
            metrics across iterations and returns (coef_df, metrics_df).
            Default False — returns coef_df only.

    Returns:
        When collect_metrics=False (default):
            DataFrame with columns: variable, observed, bootstrap_mean,
            bootstrap_se, ci_lower, ci_upper, bootstrap_p_value,
            bootstrap_type, n_iterations.
        When collect_metrics=True and fit_func returns tuple:
            Tuple (coef_df, metrics_df) where metrics_df has one row
            per iteration with all metric keys from the fit_func dict.
    """
    if variable_names is None:
        if hasattr(X, 'columns'):
            variable_names = list(X.columns)
        else:
            n_vars = X.shape[1] if hasattr(X, 'shape') else 1
            variable_names = [f"var_{i}" for i in range(n_vars)]

    # Observed estimates
    X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
    y_arr = y.values if isinstance(y, pd.Series) else np.asarray(y)

    observed_result = fit_func(X, y)
    if isinstance(observed_result, tuple):
        observed = np.asarray(observed_result[0], dtype=float)
    else:
        observed = np.asarray(observed_result, dtype=float)
    p = len(observed)

    rng = np.random.default_rng(random_state)
    n_samples = X_arr.shape[0]
    boot_estimates  = np.full((n_iterations, p), np.nan)
    boot_metrics_list = []   # populated when fit_func returns tuple

    for i in range(n_iterations):
        idx    = rng.integers(0, n_samples, n_samples)
        X_boot = X.iloc[idx].reset_index(drop=True) if isinstance(X, pd.DataFrame) else X_arr[idx]
        y_boot = y.iloc[idx].reset_index(drop=True) if isinstance(y, pd.Series) else y_arr[idx]
        try:
            result = fit_func(X_boot, y_boot)
            if isinstance(result, tuple):
                est, iter_metrics = result[0], result[1]
                if collect_metrics and isinstance(iter_metrics, dict):
                    boot_metrics_list.append(iter_metrics)
            else:
                est = result
            boot_estimates[i] = np.asarray(est, dtype=float)
        except Exception:
            pass

    # Jackknife for BCa
    jackknife = None
    if bootstrap_type == "bca":
        jackknife = np.full((n_samples, p), np.nan)
        for i in range(n_samples):
            idx  = np.delete(np.arange(n_samples), i)
            X_jk = X.iloc[idx].reset_index(drop=True) if isinstance(X, pd.DataFrame) else X_arr[idx]
            y_jk = y.iloc[idx].reset_index(drop=True) if isinstance(y, pd.Series) else y_arr[idx]
            try:
                jk_result = fit_func(X_jk, y_jk)
                est_jk = jk_result[0] if isinstance(jk_result, tuple) else jk_result
                jackknife[i] = np.asarray(est_jk, dtype=float)
            except Exception:
                pass

    # Build coefficient DataFrame
    rows = []
    for j, var in enumerate(variable_names):
        boot_j = boot_estimates[:, j]
        boot_j = boot_j[~np.isnan(boot_j)]
        obs_j  = float(observed[j])
        jk_j   = jackknife[:, j] if jackknife is not None else None

        ci_lo, ci_hi = confidence_interval(
            values=boot_j,
            alpha=alpha,
            bootstrap_type=bootstrap_type,
            observed=obs_j,
            jackknife_values=jk_j,
        )

        if len(boot_j) > 0 and not np.isnan(obs_j):
            p_val = float(np.mean(boot_j <= 0) if obs_j >= 0 else np.mean(boot_j >= 0)) * 2
            p_val = min(p_val, 1.0)
        else:
            p_val = np.nan

        rows.append({
            'variable':          var,
            'observed':          obs_j,
            'bootstrap_mean':    float(np.mean(boot_j)) if len(boot_j) > 0 else np.nan,
            'bootstrap_se':      float(np.std(boot_j, ddof=1)) if len(boot_j) > 1 else np.nan,
            'ci_lower':          ci_lo,
            'ci_upper':          ci_hi,
            'bootstrap_p_value': p_val,
            'bootstrap_type':    bootstrap_type,
            'n_iterations':      n_iterations,
        })

    coef_df = pd.DataFrame(rows)

    # Return with metrics if requested and available
    if collect_metrics and boot_metrics_list:
        metrics_df = pd.DataFrame(boot_metrics_list)
        return coef_df, metrics_df

    return coef_df


def bootstrap_hazard_ratios(
    data: pd.DataFrame,
    duration_var: str,
    event_var: str,
    covariates: List[str],
    n_iterations: int = 1000,
    bootstrap_type: str = "bias_corrected",
    alpha: float = 0.05,
    random_state: int = 42,
    penalizer: float = 0.0,
    formula: Optional[str] = None,
) -> pd.DataFrame:
    """
    Bootstrap confidence intervals for Cox proportional hazard ratios.

    Repeatedly resamples the survival dataset with replacement, refits
    a CoxPHFitter, and collects log-HR estimates. CIs are computed on
    log-HR scale and exponentiated for the HR CI bounds.

    When formula is provided (e.g. from censoring_var conversion),
    all fitting steps use formula-based spec to handle categorical
    and string covariates correctly. Parameter names are derived
    from the fitted model (Patsy expansion) rather than from the
    covariates list.

    Args:
        data: DataFrame with duration, event, and covariate columns.
        duration_var: Time-to-event column name.
        event_var: Event indicator column (1=event, 0=censored).
        covariates: List of covariate column names.
        n_iterations: Number of bootstrap samples (default 1000).
        bootstrap_type: CI type: "percentile", "bias_corrected", "bca".
            Default "bias_corrected".
        alpha: Significance level (default 0.05 → 95% CI).
        random_state: Seed for reproducibility.
        penalizer: Ridge penalizer for CoxPHFitter (default 0.0).
        formula: Formula for the Cox model (default None).
            When provided, uses formula-based fitting in all steps.

    Returns:
        DataFrame with columns: variable, observed_hr, observed_log_hr,
        bootstrap_mean_log_hr, bootstrap_se_log_hr,
        hr_ci_lower, hr_ci_upper, bootstrap_p_value,
        bootstrap_type, n_iterations.
    """
    from lifelines import CoxPHFitter

    cols_needed = [duration_var, event_var] + covariates
    data_fit    = data[cols_needed].dropna().copy()
    n_samples   = len(data_fit)

    rng = np.random.default_rng(random_state)

    # ── Observed log-HRs ─────────────────────────────────────────────────────
    # When formula is provided, use formula-based fit to handle
    # categorical/string covariates. Parameter names come from Patsy
    # expansion and may differ from the original covariates list.
    cph_obs = CoxPHFitter(penalizer=penalizer)

    if formula is not None:
        cph_obs.fit(
            data_fit,
            duration_col=duration_var,
            event_col=event_var,
            formula=formula,
        )
        _param_names    = list(cph_obs.params_.index)
        observed_log_hr = cph_obs.params_.values.astype(float)
        p               = len(_param_names)
    else:
        cph_obs.fit(
            data_fit,
            duration_col=duration_var,
            event_col=event_var,
        )
        _param_names    = list(covariates)
        observed_log_hr = cph_obs.params_.loc[covariates].values.astype(float)
        p               = len(covariates)

    boot_log_hrs = np.full((n_iterations, p), np.nan)

    # ── Bootstrap loop ────────────────────────────────────────────────────────
    for i in range(n_iterations):
        idx       = rng.integers(0, n_samples, n_samples)
        data_boot = data_fit.iloc[idx].reset_index(drop=True)
        try:
            cph_boot = CoxPHFitter(penalizer=penalizer)
            if formula is not None:
                cph_boot.fit(
                    data_boot,
                    duration_col=duration_var,
                    event_col=event_var,
                    formula=formula,
                )
                boot_log_hrs[i] = cph_boot.params_.values.astype(float)
            else:
                cph_boot.fit(
                    data_boot,
                    duration_col=duration_var,
                    event_col=event_var,
                )
                boot_log_hrs[i] = cph_boot.params_.loc[covariates].values.astype(float)
        except Exception:
            pass

    # ── Jackknife for BCa ─────────────────────────────────────────────────────
    jackknife = None
    if bootstrap_type == "bca":
        jackknife = np.full((n_samples, p), np.nan)
        for i in range(n_samples):
            idx     = np.delete(np.arange(n_samples), i)
            data_jk = data_fit.iloc[idx].reset_index(drop=True)
            try:
                cph_jk = CoxPHFitter(penalizer=penalizer)
                if formula is not None:
                    cph_jk.fit(
                        data_jk,
                        duration_col=duration_var,
                        event_col=event_var,
                        formula=formula,
                    )
                    jackknife[i] = cph_jk.params_.values.astype(float)
                else:
                    cph_jk.fit(
                        data_jk,
                        duration_col=duration_var,
                        event_col=event_var,
                    )
                    jackknife[i] = cph_jk.params_.loc[covariates].values.astype(float)
            except Exception:
                pass

    # ── Build result DataFrame ────────────────────────────────────────────────
    rows = []
    for j, var in enumerate(_param_names):
        boot_j    = boot_log_hrs[:, j]
        boot_j    = boot_j[~np.isnan(boot_j)]
        obs_log_j = float(observed_log_hr[j])
        jk_j      = jackknife[:, j] if jackknife is not None else None

        ci_lo_log, ci_hi_log = confidence_interval(
            values=boot_j,
            alpha=alpha,
            bootstrap_type=bootstrap_type,
            observed=obs_log_j,
            jackknife_values=jk_j,
        )

        # Bootstrapped two-sided p-value on log-HR scale
        if len(boot_j) > 0 and not np.isnan(obs_log_j):
            if obs_log_j >= 0:
                p_val = float(np.mean(boot_j <= 0)) * 2
            else:
                p_val = float(np.mean(boot_j >= 0)) * 2
            p_val = min(p_val, 1.0)
        else:
            p_val = np.nan

        rows.append({
            'variable':             var,
            'observed_hr':          float(np.exp(obs_log_j)),
            'observed_log_hr':      obs_log_j,
            'bootstrap_mean_log_hr': float(np.mean(boot_j)) if len(boot_j) > 0 else np.nan,
            'bootstrap_se_log_hr':  float(np.std(boot_j, ddof=1)) if len(boot_j) > 1 else np.nan,
            'hr_ci_lower':          float(np.exp(ci_lo_log)) if not np.isnan(ci_lo_log) else np.nan,
            'hr_ci_upper':          float(np.exp(ci_hi_log)) if not np.isnan(ci_hi_log) else np.nan,
            'bootstrap_p_value':    p_val,
            'bootstrap_type':       bootstrap_type,
            'n_iterations':         n_iterations,
        })

    return pd.DataFrame(rows)