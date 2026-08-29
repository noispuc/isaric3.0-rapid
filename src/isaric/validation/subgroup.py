"""
Subgroup analyses for the RAPID methodology.

This module provides functions to examine model performance within
specific, pre-defined subsets of the population (Step 5.4 of the RAPID
methodology). This assesses if the model performs equally well across
all relevant groups.

Techniques:
- stratified_metrics: Calculate performance metrics per subgroup.
- stratified_regression: Run separate regression per subgroup.
- interaction_test: Test for statistical interaction.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple


def stratified_metrics(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    subgroup_col: str,
    subgroup_values: Optional[List[str]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Calculate performance metrics for each subgroup.

    Applies the fitted model to each subgroup and calculates metrics
    separately. For classification models, uses AUC-ROC; for regression,
    uses RMSE.

    Args:
        model: Fitted model with predict() method.
        X: Predictor matrix.
        y: Outcome vector.
        subgroup_col: Column defining the subgroups.
        subgroup_values: List of subgroup values to analyze (None = all).

    Returns:
        Dictionary mapping subgroup name to metrics dictionary.

    Raises:
        ValueError: If subgroup_col is not found.
    """
    if subgroup_col not in X.columns:
        raise ValueError(
            f"Subgroup column '{subgroup_col}' not found in X."
        )

    if subgroup_values is None:
        subgroup_values = X[subgroup_col].unique().tolist()

    results = {}

    for subgroup in subgroup_values:
        mask = X[subgroup_col] == subgroup
        X_sub = X[mask].drop(columns=[subgroup_col])
        y_sub = y[mask]

        if len(y_sub) == 0:
            continue

        if hasattr(model, 'predict_proba'):
            # Classification
            from sklearn.metrics import roc_auc_score
            y_prob = model.predict_proba(X_sub)[:, 1]
            metric = roc_auc_score(y_sub, y_prob)
            results[str(subgroup)] = {'auc': float(metric), 'n': int(len(y_sub))}
        else:
            # Regression
            from sklearn.metrics import mean_squared_error
            y_pred = model.predict(X_sub)
            metric = mean_squared_error(y_sub, y_pred)
            results[str(subgroup)] = {'mse': float(metric), 'n': int(len(y_sub))}

    return results


def stratified_regression(
    data: pd.DataFrame,
    dependent_var: str,
    independent_vars: List[str],
    subgroup_col: str,
    model_func: Optional[callable] = None
) -> Dict[str, pd.DataFrame]:
    """
    Run separate regression models for each subgroup.

    Fits a model for each subgroup and returns coefficients for comparison.

    Args:
        data: Input DataFrame.
        dependent_var: Outcome variable.
        independent_vars: Predictor variables.
        subgroup_col: Column defining subgroups.
        model_func: Function to fit the model (default: linear regression).

    Returns:
        Dictionary mapping subgroup name to coefficients DataFrame.

    Raises:
        ValueError: If columns are not found.
    """
    for col in [dependent_var, subgroup_col] + independent_vars:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

    if model_func is None:
        from sklearn.linear_model import LinearRegression
        model_func = LinearRegression

    results = {}
    subgroups = data[subgroup_col].unique()

    for subgroup in subgroups:
        subgroup_data = data[data[subgroup_col] == subgroup]

        X = subgroup_data[independent_vars]
        y = subgroup_data[dependent_var]

        model = model_func()
        model.fit(X, y)

        coefficients = pd.DataFrame({
            'Variable': independent_vars,
            'Coefficient': model.coef_.flatten() if hasattr(model, 'coef_') else model.feature_importances_
        })
        results[str(subgroup)] = coefficients

    return results


def interaction_test(
    model,
    data: pd.DataFrame,
    dependent_var: str,
    independent_vars: List[str],
    subgroup_col: str
) -> Dict[str, float]:
    """
    Test for statistical interaction between subgroup and predictors.

    Adds interaction terms to the model and tests their significance.

    Args:
        model: Fitted model (statsmodels).
        data: Input DataFrame.
        dependent_var: Outcome variable.
        independent_vars: Predictor variables.
        subgroup_col: Subgroup variable.

    Returns:
        Dictionary with interaction p-values.

    Raises:
        ValueError: If columns are not found.
    """
    for col in [dependent_var, subgroup_col] + independent_vars:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

    # Create interaction terms
    interaction_data = data.copy()
    interaction_results = {}

    for var in independent_vars:
        interaction_col = f'{var}_x_{subgroup_col}'
        interaction_data[interaction_col] = (
            interaction_data[var] * interaction_data[subgroup_col]
        )

        # Fit model with interaction
        import statsmodels.api as sm

        X = interaction_data[[var, subgroup_col, interaction_col]]
        X = sm.add_constant(X)
        y = interaction_data[dependent_var]

        fitted = sm.GLM(y, X).fit()

        # Extract p-value for interaction term
        p_value = fitted.pvalues[interaction_col]
        interaction_results[var] = float(p_value)

    return interaction_results