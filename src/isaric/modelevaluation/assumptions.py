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
from typing import Any, Dict, List, Optional, Tuple
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

def bootstrap_stepmix(
    fitted_estimator,
    X: np.ndarray,
    Y: Optional[np.ndarray] = None,
    n_bootstrap: int = 100,
    parametric: bool = True,
    random_state: int = 42,
    progress_bar: bool = False
) -> pd.DataFrame:
    """
    Parametric bootstrap for StepMix estimators.

    Resamples data from the fitted estimator's distribution, refits
    clones, and stores likelihood statistics for each repetition.

    Adapted from the reference implementation in the ISARIC
    phenotype-clustering branch (Adjbootstrap10.py).

    Args:
        fitted_estimator: Fitted StepMix instance.
        X: Binary matrix (n_samples, n_features).
        Y: Optional structural matrix (n_samples, n_structural).
        n_bootstrap: Number of bootstrap samples (default 100).
        parametric: If True, samples from fitted estimator; if False,
            resamples with replacement from X (non-parametric).
        random_state: Seed for reproducibility.
        progress_bar: If True, prints progress info.

    Returns:
        DataFrame with one row per repetition, columns:
        LL, AIC, BIC, CAIC, SABIC, entropy, relative_entropy,
        n_par, n, ncomp, converged.
    """
    import copy
    from sklearn.base import clone
    from sklearn.utils.validation import check_random_state, check_is_fitted

    check_is_fitted(fitted_estimator)

    n_samples = X.shape[0]
    rng = check_random_state(random_state)

    # Buffers
    ll_buffer = []
    aic_buffer = []
    bic_buffer = []
    caic_buffer = []
    sabic_buffer = []
    entropy_buffer = []
    rel_entropy_buffer = []
    conv_buffer = []

    for rep in range(n_bootstrap):
        # Resample data
        if parametric:
            X_rep, Y_rep, _ = fitted_estimator.sample(
                n_samples, random_state=rng
            )
        else:
            idx = rng.choice(n_samples, size=n_samples, replace=True)
            X_rep = X[idx]
            Y_rep = Y[idx] if Y is not None else None

        # Clone and fit
        estimator_rep = clone(fitted_estimator)
        estimator_rep.set_params(
            verbose=0, random_state=random_state
        )
        try:
            estimator_rep.fit(X_rep, Y_rep)
        except Exception:
            continue

        avg_ll = estimator_rep.score(X_rep, Y_rep)
        ll = avg_ll * n_samples
        npar = estimator_rep.n_parameters
        ncomp = estimator_rep.n_components

        aic = -2 * avg_ll * n_samples + 2 * npar
        bic = -2 * avg_ll * n_samples + npar * np.log(n_samples)
        caic = -2 * avg_ll * n_samples + npar * (np.log(n_samples) + 1)
        sabic = -2 * avg_ll * n_samples + npar * np.log(
            n_samples * ((n_samples + 2) / 24)
        )
        entropy = estimator_rep.entropy(X_rep)
        rel_entropy = (
            1 - entropy / (n_samples * np.log(ncomp))
            if ncomp > 1 else np.nan
        )

        ll_buffer.append(ll)
        aic_buffer.append(aic)
        bic_buffer.append(bic)
        caic_buffer.append(caic)
        sabic_buffer.append(sabic)
        entropy_buffer.append(entropy)
        rel_entropy_buffer.append(rel_entropy)
        conv_buffer.append(estimator_rep.converged_)

        if progress_bar and (rep + 1) % 10 == 0:
            print(f"  Bootstrap {rep + 1}/{n_bootstrap}")

    return pd.DataFrame({
        'LL': ll_buffer,
        'AIC': aic_buffer,
        'BIC': bic_buffer,
        'CAIC': caic_buffer,
        'SABIC': sabic_buffer,
        'entropy': entropy_buffer,
        'relative_entropy': rel_entropy_buffer,
        'n_par': npar,
        'n': n_samples,
        'ncomp': ncomp,
        'converged': conv_buffer,
    })

def bootstrap_lrt_stepmix(
    null_model,
    alt_model,
    stats_null: pd.DataFrame,
    stats_alt: pd.DataFrame,
    X: np.ndarray,
    Y: Optional[np.ndarray] = None,
    n_bootstrap: int = 100
) -> float:
    """
    Bootstrap Likelihood Ratio Test (BLRT) for StepMix models.

    Compares the observed LRT statistic (k vs k-1) with the bootstrap
    distribution of LRT statistics under the null (k-1 classes).

    Adapted from Adjbootstrap10.py (_blrt function).

    Args:
        null_model: Fitted StepMix with k-1 components.
        alt_model: Fitted StepMix with k components.
        stats_null: DataFrame from bootstrap_stepmix for null_model.
        stats_alt: DataFrame from bootstrap_stepmix for alt_model.
        X: Binary matrix used for fitting.
        Y: Optional structural matrix.
        n_bootstrap: Number of bootstrap samples used.

    Returns:
        Bootstrap p-value (float in [0, 1]).
    """
    n_samples = X.shape[0]

    # Observed LRT statistic on real data
    real_stat = 2 * (alt_model.score(X, Y) - null_model.score(X, Y)) * n_samples

    # Bootstrap distribution: LRT under H0
    gen_stats = 2 * (stats_alt["LL"].values - stats_null["LL"].values)

    # p-value: proportion of bootstrap LRTs >= observed
    b = np.sum(gen_stats > real_stat)
    return float(b / n_bootstrap)

def bootstrap_ci_stepmix(
    fitted_estimator,
    stats_bootstrap: pd.DataFrame,
    X: np.ndarray,
    metrics: Optional[List[str]] = None,
    alpha: float = 0.05
) -> pd.DataFrame:
    """
    Compute bias-corrected bootstrap confidence intervals (BC).

    For each metric, computes the observed value theta_hat, the bias
    correction z0, and the BC confidence interval [ci_low, ci_high].

    Adapted from Adjbootstrap10.py (_ci_boot function).

    Args:
        fitted_estimator: Fitted StepMix instance.
        stats_bootstrap: DataFrame from bootstrap_stepmix (one row per rep).
        X: Binary matrix.
        metrics: List of metric column names. If None, uses
            ['BIC', 'AIC', 'SABIC', 'CAIC', 'entropy',
             'relative_entropy'].
        alpha: Significance level (default 0.05 → 95% CI).

    Returns:
        DataFrame with one row per metric and columns:
        metric, theta_hat, z0, ci_low, ci_high.
    """
    from scipy.stats import norm

    if metrics is None:
        metrics = ['BIC', 'AIC', 'SABIC', 'CAIC', 'entropy',
                   'relative_entropy']

    n_rep = len(stats_bootstrap)
    rows = []

    for m in metrics:
        if m not in stats_bootstrap.columns:
            continue

        # Observed value
        if m == 'entropy':
            theta_hat = fitted_estimator.entropy(X)
        elif m == 'relative_entropy':
            ncomp = fitted_estimator.n_components
            n = X.shape[0]
            ent = fitted_estimator.entropy(X)
            theta_hat = (
                1 - ent / (n * np.log(ncomp)) if ncomp > 1 else np.nan
            )
        elif m == 'BIC':
            theta_hat = fitted_estimator.bic(X)
        elif m == 'AIC':
            theta_hat = fitted_estimator.aic(X)
        else:
            # Fallback: try method by name
            method = getattr(fitted_estimator, m.lower(), None)
            if method is None:
                continue
            theta_hat = method(X)

        # Bootstrap distribution
        theta_star = np.sort(stats_bootstrap[m].dropna().values)
        n_rep_valid = len(theta_star)
        if n_rep_valid < 2:
            continue

        # Bias correction
        n_below = np.sum(theta_star < theta_hat)
        bias_prop = n_below / n_rep_valid
        # Avoid extremes
        bias_prop = min(max(bias_prop, 1e-6), 1 - 1e-6)
        z0 = norm.ppf(bias_prop)

        za_low = norm.ppf(alpha / 2)
        za_high = norm.ppf(1 - alpha / 2)

        phi_low = norm.cdf(2 * z0 + za_low)
        phi_high = norm.cdf(2 * z0 + za_high)

        idx_low = max(0, min(n_rep_valid - 1, int(np.floor(phi_low * n_rep_valid))))
        idx_high = max(0, min(n_rep_valid - 1, int(np.ceil(phi_high * n_rep_valid))))

        rows.append({
            'metric': m,
            'theta_hat': float(theta_hat),
            'z0': float(z0),
            'ci_low': float(theta_star[idx_low]),
            'ci_high': float(theta_star[idx_high]),
        })

    return pd.DataFrame(rows)


def bootstrap_selection_rate(
    bootstrap_stats: pd.DataFrame,
    k_col: str = "n_clusters",
    rep_col: str = "rep",
    metrics: Optional[List[str]] = None,
    lower_is_better: Optional[List[str]] = None,
    convergence_threshold: float = 0.3
) -> pd.DataFrame:
    """
    Compute bootstrap selection rates per metric and k.

    For each bootstrap repetition, identifies the "best" k per metric
    (lowest for information criteria, highest for entropy-based metrics),
    then aggregates across repetitions as percentages.

    Adapted from the reference implementation
    (Adjbootstrap10.py::_boot_selectionRate).

    Args:
        bootstrap_stats: DataFrame from bootstrap_stepmix with columns
            k_col, rep_col, and metrics.
        k_col: Column with k values (default "n_clusters").
        rep_col: Column with repetition index (default "rep").
        metrics: List of metrics to compute selection rate.
            If None: ['AIC', 'BIC', 'CAIC', 'SABIC', 'entropy',
                      'relative_entropy'].
        lower_is_better: List of metrics where lower is better.
            If None: information criteria (AIC, BIC, CAIC, SABIC).
        convergence_threshold: Minimum convergence rate per k required
            to include it in the selection (default 0.3 = 30%).

    Returns:
        DataFrame with one row per k and one column per metric,
        containing selection rate in %.
    """
    if bootstrap_stats is None or bootstrap_stats.empty:
        return pd.DataFrame()

    if metrics is None:
        metrics = ['AIC', 'BIC', 'CAIC', 'SABIC', 'entropy', 'relative_entropy']
    if lower_is_better is None:
        lower_is_better = ['AIC', 'BIC', 'CAIC', 'SABIC']

    # Filtra só métricas existentes
    metrics = [m for m in metrics if m in bootstrap_stats.columns]
    if not metrics:
        return pd.DataFrame()

    # Adiciona índice de repetição se não existir
    df = bootstrap_stats.copy()
    if rep_col not in df.columns:
        df[rep_col] = df.groupby(k_col).cumcount()

    # === Convergência por k ===
    conv_by_k = {}
    if 'converged' in df.columns:
        for k, sub in df.groupby(k_col):
            conv_by_k[k] = sub['converged'].mean()
    else:
        # Assume 100% se não tiver coluna
        for k in df[k_col].unique():
            conv_by_k[k] = 1.0

    # Filtra só ks com convergência acima do threshold
    valid_ks = [k for k, conv in conv_by_k.items() if conv >= convergence_threshold]
    if not valid_ks:
        return pd.DataFrame()

    df = df[df[k_col].isin(valid_ks)].copy()
    n_rep = df[rep_col].nunique()
    if n_rep == 0:
        return pd.DataFrame()

    # === Para cada métrica, calcula a taxa de seleção ===
    result = {}
    for metric in metrics:
        # Pivot: linhas = repetição, colunas = k, valores = métrica
        pivot = df.pivot_table(
            index=rep_col,
            columns=k_col,
            values=metric,
            aggfunc='mean'
        )

        # Ignora repetições com NaN
        pivot = pivot.dropna(how='all')
        if pivot.empty:
            result[metric] = {k: np.nan for k in valid_ks}
            continue

        # Melhor k por repetição
        if metric in lower_is_better:
            best_k_per_rep = pivot.idxmin(axis=1)
        else:
            best_k_per_rep = pivot.idxmax(axis=1)

        # Agrega em %
        counts = best_k_per_rep.value_counts()
        total_valid = counts.sum()
        if total_valid == 0:
            result[metric] = {k: np.nan for k in valid_ks}
            continue

        result[metric] = {
            k: float(counts.get(k, 0) / total_valid * 100)
            for k in valid_ks
        }

    # Constrói DataFrame final
    df_out = pd.DataFrame(result).fillna(0.0)
    df_out.index.name = "n_clusters"
    df_out = df_out.reset_index()
    # Ordena por k
    df_out = df_out.sort_values('n_clusters').reset_index(drop=True)

    return df_out