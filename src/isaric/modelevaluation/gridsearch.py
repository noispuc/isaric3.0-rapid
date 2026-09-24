"""
Grid search for the RAPID methodology.

This module provides a generic grid search function that works with
any estimator following the sklearn API (fit, set_params, get_params).

For models with specific grid search logic (e.g., LCA with StepMix),
the subclasses implement their own `_run_grid_search`.

Techniques:
- run_grid_search: Generic grid search using sklearn's GridSearchCV.
- run_custom_grid_search: Manual grid search for non-sklearn estimators.
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