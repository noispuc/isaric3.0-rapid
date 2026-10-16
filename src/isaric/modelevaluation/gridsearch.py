"""
Grid search for the RAPID methodology.

This module provides grid search functions and LCA-specific bootstrap
routines. The generic grid search works with any estimator following
the sklearn API. LCA bootstrap functions (BLRT, bias-corrected CI,
selection rate) are housed here because they are part of the class
enumeration process, not model assumption checks.

Generic Grid Search:
- run_grid_search: Generic grid search using sklearn's GridSearchCV.
- run_custom_grid_search: Manual grid search for non-sklearn estimators.
- run_clustering_grid_search: Grid search for unsupervised clustering.

LCA Bootstrap (StepMix):
- bootstrap_stepmix: Parametric bootstrap for a single StepMix model.
- bootstrap_lrt_stepmix: Bootstrapped LRT p-value (k vs k-1).
- bootstrap_ci_stepmix: Bias-corrected bootstrap CI per metric.
- bootstrap_selection_rate: Bootstrap selection rate per metric and k.
- bootstrap_lrt_lca: High-level BLRT using stepmix native API.
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV


def run_grid_search(
    estimator,
    X,
    y,
    param_grid: Dict[str, List[Any]],
    selection_metric: str = "auto",
    cv: int = 5,
    n_jobs: int = -1,
    refit: bool = True
) -> Tuple[Any, Dict[str, Any], pd.DataFrame]:
    """
    Generic grid search for sklearn-compatible estimators.

    Uses sklearn's GridSearchCV internally. Works with any estimator
    that implements the sklearn API (get_params, set_params, fit).

    Args:
        estimator: Unfitted sklearn-compatible estimator.
        X: Predictor matrix.
        y: Outcome vector (None for unsupervised).
        param_grid: Dictionary mapping parameter names to list of values.
        selection_metric: Scoring metric (e.g., "roc_auc", "neg_mean_squared_error").
            If "auto", uses model-specific default.
        cv: Number of cross-validation folds (default: 5).
        n_jobs: Number of parallel jobs (-1 = all cores).
        refit: Whether to refit the best estimator on full data.

    Returns:
        Tuple of (best_estimator, best_params, results_df):
            - best_estimator: Fitted estimator with best params.
            - best_params: Dictionary of best parameter values.
            - results_df: DataFrame with CV results.

    Raises:
        ValueError: If selection_metric is invalid or param_grid is empty.
    """
    if not param_grid:
        raise ValueError("param_grid cannot be empty.")

    if selection_metric == "auto":
        # Detecta automaticamente
        if hasattr(estimator, 'predict_proba'):
            selection_metric = "roc_auc"
        elif hasattr(estimator, 'score'):
            # KMeans, clustering, etc.
            selection_metric = "neg_mean_squared_error"
        else:
            selection_metric = "roc_auc"

    # GridSearchCV
    grid = GridSearchCV(
        estimator=clone(estimator),
        param_grid=param_grid,
        scoring=selection_metric if y is not None else None,
        cv=cv if y is not None else None,
        n_jobs=n_jobs,
        refit=refit,
        return_train_score=True
    )

    grid.fit(X, y)

    # Constrói DataFrame de resultados
    results_df = _build_grid_results_df(grid.cv_results_, param_grid)

    return grid.best_estimator_, grid.best_params_, results_df


def run_custom_grid_search(
    model_factory,
    fit_func,
    score_func,
    param_grid: Dict[str, List[Any]],
    selection_metric: str = "auto",
    higher_is_better: bool = False
) -> Tuple[Any, Dict[str, Any], pd.DataFrame]:
    """
    Manual grid search for non-sklearn estimators (e.g., StepMix, lifelines).

    Iterates over all parameter combinations, fits the model, computes
    the score, and selects the best.

    Args:
        model_factory: Callable that takes params and returns unfitted model.
        fit_func: Callable(model, params) → fitted_model.
        score_func: Callable(fitted_model) → score (float).
        param_grid: Dictionary mapping parameter names to list of values.
        selection_metric: Name of metric (for reporting).
        higher_is_better: If True, maximize score; else minimize.

    Returns:
        Tuple of (best_model, best_params, results_df).
    """
    from itertools import product

    # Gera todas as combinações
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(product(*param_values))

    results = []
    best_score = -np.inf if higher_is_better else np.inf
    best_model = None
    best_params = None

    for combo in combinations:
        params = dict(zip(param_names, combo))

        try:
            model = model_factory(**params)
            fitted = fit_func(model, params)
            score = score_func(fitted)

            results.append({**params, 'score': score, 'status': 'ok'})

            # Atualiza melhor
            if (higher_is_better and score > best_score) or \
               (not higher_is_better and score < best_score):
                best_score = score
                best_model = fitted
                best_params = params

        except Exception as e:
            results.append({**params, 'score': np.nan, 'status': str(e)})

    results_df = pd.DataFrame(results)

    return best_model, best_params, results_df


def _build_grid_results_df(
    cv_results: Dict,
    param_grid: Dict[str, List[Any]]
) -> pd.DataFrame:
    """
    Build a simplified DataFrame from sklearn's cv_results_.

    Args:
        cv_results: sklearn's cv_results_ dictionary.
        param_grid: Original parameter grid.

    Returns:
        Simplified DataFrame with param columns, mean_test_score, std_test_score.
    """
    param_cols = [f'param_{k}' for k in param_grid.keys()]

    cols_to_keep = param_cols + [
        'mean_test_score',
        'std_test_score',
        'mean_train_score',
        'rank_test_score'
    ]
    cols_to_keep = [c for c in cols_to_keep if c in cv_results]

    df = pd.DataFrame({col: cv_results[col] for col in cols_to_keep})
    df = df.sort_values('rank_test_score').reset_index(drop=True)

    return df


def run_clustering_grid_search(
    estimator,
    X,
    param_grid: Dict[str, List[Any]],
    selection_metric: str = "silhouette",
    cv: int = 5,
    n_jobs: int = -1
) -> Tuple[Any, Dict[str, Any], pd.DataFrame]:
    """
    Grid search for unsupervised clustering estimators.

    Uses silhouette (or Calinski-Harabasz) as the default metric, which
    is more suitable than inertia for selecting the number of clusters.

    Args:
        estimator: Unfitted clustering estimator (e.g., KMeans).
        X: Predictor matrix.
        param_grid: Dictionary mapping parameter names to list of values.
        selection_metric: "silhouette" or "calinski_harabasz".
        cv: Ignored (KMeans não usa CV na forma clássica). Mantido por compat.
        n_jobs: Número de jobs paralelos.

    Returns:
        Tuple of (best_estimator, best_params, results_df).
    """
    from itertools import product
    from sklearn.base import clone
    from sklearn.metrics import silhouette_score, calinski_harabasz_score

    if not param_grid:
        raise ValueError("param_grid cannot be empty.")

    if selection_metric == "auto":
        selection_metric = "silhouette"

    # Métricas: maior melhor
    def _score(model, X_data):
        labels = model.labels_
        if len(set(labels)) < 2:
            return -np.inf
        if selection_metric == "silhouette":
            return silhouette_score(X_data, labels)
        elif selection_metric == "calinski_harabasz":
            return calinski_harabasz_score(X_data, labels)
        else:
            raise ValueError(
                f"Unknown selection_metric: {selection_metric}. "
                f"Use 'silhouette' or 'calinski_harabasz'."
            )

    # Gera combinações
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(product(*param_values))

    results = []
    best_score = -np.inf
    best_model = None
    best_params = None

    for combo in combinations:
        params = dict(zip(param_names, combo))
        try:
            model = clone(estimator)
            model.set_params(**params)
            model.fit(X)

            score = _score(model, X)
            results.append({**params, 'score': score, 'status': 'ok'})

            if score > best_score:
                best_score = score
                best_model = model
                best_params = params
        except Exception as e:
            results.append({**params, 'score': np.nan, 'status': str(e)})

    results_df = pd.DataFrame(results)
    # Renomear coluna 'score' para o nome da métrica
    results_df = results_df.rename(columns={'score': selection_metric})

    return best_model, best_params, results_df


# ============================================================================
# LCA BOOTSTRAP FUNCTIONS (StepMix)
# Moved from modelevaluation/assumptions.py — belong to class enumeration.
# ============================================================================

def bootstrap_lrt_lca(
    X,
    n_components: int,
    n_bootstraps: int = 100,
    measurement: str = "bernoulli",
    random_state: int = 42,
    n_init: int = 5,
    max_iter: int = 500
):
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
    X,
    Y=None,
    n_bootstrap: int = 100,
    parametric: bool = True,
    random_state: int = 42,
    progress_bar: bool = False
):
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

    ll_buffer = []
    aic_buffer = []
    bic_buffer = []
    caic_buffer = []
    sabic_buffer = []
    entropy_buffer = []
    rel_entropy_buffer = []
    conv_buffer = []
    npar = None
    ncomp = None

    for rep in range(n_bootstrap):
        if parametric:
            X_rep, Y_rep, _ = fitted_estimator.sample(n_samples)
        else:
            idx = rng.choice(n_samples, size=n_samples, replace=True)
            X_rep = X[idx]
            Y_rep = Y[idx] if Y is not None else None

        estimator_rep = clone(fitted_estimator)
        estimator_rep.set_params(verbose=0, random_state=random_state)
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
    stats_null,
    stats_alt,
    X,
    Y=None,
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
    real_stat = 2 * (alt_model.score(X, Y) - null_model.score(X, Y)) * n_samples
    gen_stats = 2 * (stats_alt["LL"].values - stats_null["LL"].values)
    b = np.sum(gen_stats > real_stat)
    return float(b / n_bootstrap)


def bootstrap_ci_stepmix(
    fitted_estimator,
    stats_bootstrap,
    X,
    metrics=None,
    alpha: float = 0.05
):
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
            ['BIC', 'AIC', 'SABIC', 'CAIC', 'entropy', 'relative_entropy'].
        alpha: Significance level (default 0.05 → 95% CI).

    Returns:
        DataFrame with columns: metric, theta_hat, z0, ci_low, ci_high.
    """
    from scipy.stats import norm

    if metrics is None:
        metrics = ['BIC', 'AIC', 'SABIC', 'CAIC', 'entropy', 'relative_entropy']

    n_rep = len(stats_bootstrap)
    rows = []

    for m in metrics:
        if m not in stats_bootstrap.columns:
            continue

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
            method = getattr(fitted_estimator, m.lower(), None)
            if method is None:
                continue
            theta_hat = method(X)

        theta_star = np.sort(stats_bootstrap[m].dropna().values)
        n_rep_valid = len(theta_star)
        if n_rep_valid < 2:
            continue

        n_below = np.sum(theta_star < theta_hat)
        bias_prop = n_below / n_rep_valid
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
    bootstrap_stats,
    k_col: str = "n_clusters",
    rep_col: str = "rep",
    metrics=None,
    lower_is_better=None,
    convergence_threshold: float = 0.3
):
    """
    Compute bootstrap selection rates per metric and k.

    For each bootstrap repetition, identifies the 'best' k per metric
    (lowest for information criteria, highest for entropy-based metrics),
    then aggregates across repetitions as percentages.

    Adapted from Adjbootstrap10.py (_boot_selectionRate).

    Args:
        bootstrap_stats: DataFrame from bootstrap_stepmix with columns
            k_col, rep_col, and metrics.
        k_col: Column with k values (default "n_clusters").
        rep_col: Column with repetition index (default "rep").
        metrics: List of metrics. If None: ['AIC', 'BIC', 'CAIC', 'SABIC',
            'entropy', 'relative_entropy'].
        lower_is_better: Metrics where lower is better. If None:
            information criteria (AIC, BIC, CAIC, SABIC).
        convergence_threshold: Min convergence rate per k to include
            (default 0.3 = 30%).

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

    metrics = [m for m in metrics if m in bootstrap_stats.columns]
    if not metrics:
        return pd.DataFrame()

    df = bootstrap_stats.copy()
    if rep_col not in df.columns:
        df[rep_col] = df.groupby(k_col).cumcount()

    conv_by_k = {}
    if 'converged' in df.columns:
        for k, sub in df.groupby(k_col):
            conv_by_k[k] = sub['converged'].mean()
    else:
        for k in df[k_col].unique():
            conv_by_k[k] = 1.0

    valid_ks = [k for k, conv in conv_by_k.items() if conv >= convergence_threshold]
    if not valid_ks:
        return pd.DataFrame()

    df = df[df[k_col].isin(valid_ks)].copy()
    n_rep = df[rep_col].nunique()
    if n_rep == 0:
        return pd.DataFrame()

    result = {}
    for metric in metrics:
        pivot = df.pivot_table(
            index=rep_col,
            columns=k_col,
            values=metric,
            aggfunc='mean'
        )
        pivot = pivot.dropna(how='all')
        if pivot.empty:
            result[metric] = {k: np.nan for k in valid_ks}
            continue

        if metric in lower_is_better:
            best_k_per_rep = pivot.idxmin(axis=1)
        else:
            best_k_per_rep = pivot.idxmax(axis=1)

        counts = best_k_per_rep.value_counts()
        total_valid = counts.sum()
        if total_valid == 0:
            result[metric] = {k: np.nan for k in valid_ks}
            continue

        result[metric] = {
            k: float(counts.get(k, 0) / total_valid * 100)
            for k in valid_ks
        }

    df_out = pd.DataFrame(result).fillna(0.0)
    df_out.index.name = "n_clusters"
    df_out = df_out.reset_index()
    df_out = df_out.sort_values('n_clusters').reset_index(drop=True)

    return df_out