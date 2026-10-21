"""
Predictive models for the RAPID methodology.

This module provides functions to configure regularized and margin-based
machine learning models and concrete pipeline classes for classification.

Functions (Configuration):
- create_lasso_model: Configure Logistic Regression with L1 penalty.
- create_ridge_model: Configure Logistic Regression with L2 penalty.
- create_elastic_net_model: Configure Logistic Regression with L1+L2.
- create_svm_model: Configure Support Vector Machine classifier.
- create_logistic_l2_model: Configure Logistic Regression with L2 (predictive).

Subclasses (Pipelines):
- Lasso: Concrete pipeline for LASSO logistic regression.
- Ridge: Concrete pipeline for Ridge logistic regression.
- ElasticNet: Concrete pipeline for Elastic Net logistic regression.
- SVM: Concrete pipeline for Support Vector Machine.
- LogisticL2: Concrete pipeline for Logistic Regression with L2.

Helper Functions:
- _prepare_data: Prepare X and y for training.
- _build_result_df: Build coefficients DataFrame.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from isaric.rapid import RAPID


# ============================================================================
# PUBLIC FUNCTIONS (CONFIGURATION)
# ============================================================================

def create_lasso_model(
    data: pd.DataFrame,
    dependent_var: str,
    independent_vars: List[str],
    C: float = 1.0,
    random_seed=42,
    class_weight=None,
) -> Tuple[LogisticRegression, pd.DataFrame, pd.Series]:
    """
    Configure a LASSO (L1) logistic regression model (not fitted).

    Returns:
        Tuple of (model, X, y).
    """
    X, y = _prepare_data(data, dependent_var, independent_vars)

    model = LogisticRegression(
        penalty='l1',
        solver='liblinear',
        C=C,
        random_state=random_seed,
        max_iter=1000,
        class_weight=class_weight,
    )

    return model, X, y


def create_ridge_model(
    data: pd.DataFrame,
    dependent_var: str,
    independent_vars: List[str],
    C: float = 1.0,
    random_seed=42,
    class_weight=None,
) -> Tuple[LogisticRegression, pd.DataFrame, pd.Series]:
    """
    Configure a Ridge (L2) logistic regression model (not fitted).

    Returns:
        Tuple of (model, X, y).
    """
    X, y = _prepare_data(data, dependent_var, independent_vars)

    model = LogisticRegression(
        penalty='l2',
        solver='lbfgs',
        C=C,
        random_state=random_seed,
        max_iter=1000,
        class_weight=class_weight,
    )

    return model, X, y


def create_elastic_net_model(
    data: pd.DataFrame,
    dependent_var: str,
    independent_vars: List[str],
    C: float = 1.0,
    l1_ratio: float = 0.5,
    random_seed=42,
    class_weight=None,
) -> Tuple[LogisticRegression, pd.DataFrame, pd.Series]:
    """
    Configure an Elastic Net logistic regression model (not fitted).

    Returns:
        Tuple of (model, X, y).
    """
    X, y = _prepare_data(data, dependent_var, independent_vars)

    model = LogisticRegression(
        penalty='elasticnet',
        solver='saga',
        C=C,
        l1_ratio=l1_ratio,
        random_state=random_seed,
        max_iter=1000,
        class_weight=class_weight,
    )

    return model, X, y


def create_svm_model(
    data: pd.DataFrame,
    dependent_var: str,
    independent_vars: List[str],
    C: float = 1.0,
    kernel: str = "rbf",
    random_seed=42,
    class_weight=None,
) -> Tuple[SVC, pd.DataFrame, pd.Series]:
    """
    Configure a Support Vector Machine (SVM) classifier (not fitted).

    Returns:
        Tuple of (model, X, y).
    """
    X, y = _prepare_data(data, dependent_var, independent_vars)

    model = SVC(
        C=C,
        kernel=kernel,
        probability=True,
        random_state=random_seed,
        class_weight=class_weight,
    )

    return model, X, y


def create_logistic_l2_model(
    data: pd.DataFrame,
    dependent_var: str,
    independent_vars: List[str],
    C: float = 1.0,
    random_seed=42,
    class_weight=None,
) -> Tuple[LogisticRegression, pd.DataFrame, pd.Series]:
    """
    Configure a Logistic Regression with L2 penalty (predictive focus).

    Returns:
        Tuple of (model, X, y).
    """
    return create_ridge_model(
        data, dependent_var, independent_vars, C=C, random_seed=random_seed,
        class_weight=class_weight,
    )


# ============================================================================
# PRIVATE HELPERS
# ============================================================================

def _prepare_data(
    data: pd.DataFrame,
    dependent_var: str,
    independent_vars: List[str]
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare X and y for training.

    Returns:
        Tuple of (X, y).
    """
    for col in [dependent_var] + independent_vars:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

    X = data[independent_vars].copy()
    y = data[dependent_var].copy()

    return X, y


def _build_result_df(
    model,
    X: pd.DataFrame
) -> pd.DataFrame:
    """
    Build coefficients DataFrame from fitted regularized model.

    Returns:
        DataFrame with Feature, Coefficient, Selected columns.
    """
    if not hasattr(model, 'coef_'):
        raise ValueError("Model does not have coef_ attribute.")

    coefficients = model.coef_.flatten()
    result_df = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': coefficients
    })
    result_df['Selected'] = result_df['Coefficient'] != 0
    result_df = result_df.sort_values(
        'Coefficient', key=abs, ascending=False
    ).reset_index(drop=True)

    return result_df


def _bootstrap_sklearn_coefficients(
    fitted_model,
    X: pd.DataFrame,
    y: pd.Series,
    n_iterations: int = 1000,
    bootstrap_type: str = "bias_corrected",
    random_state: int = 42,
) -> dict:
    """
    Bootstrap confidence intervals and p-values for sklearn coefficients.

    Uses bootstrap_coefficients() with a fit_func that clones the fitted
    model, refits on bootstrap samples, and extracts coef_.flatten().

    Args:
        fitted_model: Fitted sklearn model with coef_ attribute.
        X: Predictor matrix.
        y: Outcome vector.
        n_iterations: Number of bootstrap samples.
        bootstrap_type: CI type ('percentile', 'bias_corrected', 'bca').
            Default 'bias_corrected'.
        random_state: Seed for reproducibility.

    Returns:
        Dictionary with:
        - 'coefficients': DataFrame with Feature, observed (coefficient),
          bootstrap_mean, bootstrap_se, ci_lower, ci_upper,
          bootstrap_p_value, bootstrap_type, n_iterations.
        - 'bootstrap_type': type used.
        - 'n_iterations': iterations run.
    """
    from sklearn.base import clone
    from isaric.modelevaluation.bootstrap import bootstrap_coefficients

    feature_names = list(X.columns) if hasattr(X, 'columns') else None

    def fit_func(X_boot, y_boot):
        try:
            model_boot = clone(fitted_model)
            model_boot.fit(X_boot, y_boot)
            return model_boot.coef_.flatten()
        except Exception:
            n = X_boot.shape[1] if hasattr(X_boot, 'shape') else len(feature_names or [])
            return np.full(n, np.nan)

    df = bootstrap_coefficients(
        X=X,
        y=y,
        fit_func=fit_func,
        n_iterations=n_iterations,
        bootstrap_type=bootstrap_type,
        variable_names=feature_names,
        random_state=random_state,
    )

    return {
        'coefficients': df,
        'bootstrap_type': bootstrap_type,
        'n_iterations': n_iterations,
    }


# ============================================================================
# SUBCLASSES (INHERIT FROM RAPID)
# ============================================================================

class Lasso(RAPID):
    """Concrete pipeline for LASSO logistic regression."""

    def __init__(self, model, X, y, dependent_var, independent_vars, **kwargs):
        self._model = model
        self.X = X
        self.y = y
        self.dependent_var = dependent_var
        self.independent_vars = independent_vars
        self.model_type = "lasso"
        self.fitted_model = None
        self.result_df = None
        self.metrics = None
        self.plots_map = {}
        self._setup_plots_map()
        super().__init__()

    def _setup_plots_map(self):
        self.plots_map = {
            "confusion_matrix": self._confusion_matrix,
            "coefficient_plot": self._coefficient_plot,
            "calibration_curve": self._calibration_plot,
        }

    @classmethod
    def create(
        cls,
        data: pd.DataFrame,
        model: str = "lasso",
        dependent_var: Optional[str] = None,
        independent_vars: Optional[List[str]] = None,
        random_seed: int = 42,
        **params
    ) -> "Lasso":
        model_config, X, y = create_lasso_model(
            data=data,
            dependent_var=dependent_var,
            independent_vars=independent_vars,
            random_seed=random_seed,
            **params
        )

        instance = cls(
            model=model_config,
            X=X,
            y=y,
            dependent_var=dependent_var,
            independent_vars=independent_vars,
            **params
        )
        instance._random_seed = random_seed
        
        return instance

    # ======================================================================
    # PRIVATE METHODS (CALLED BY fit() AND validation())
    # ======================================================================

    def _train_model(self, grid_search=False, param_grid=None, selection_metric="auto"):
        """
        Train LASSO model, optionally with grid search.
        """
        if grid_search:
            from isaric.modelevaluation.gridsearch import run_grid_search

            if param_grid is None:
                param_grid = {"C": [0.01, 0.1, 1.0, 10.0]}

            if selection_metric == "auto":
                selection_metric = "roc_auc"

            best_model, best_params, results_df = run_grid_search(
                estimator=self._model,
                X=self.X,
                y=self.y,
                param_grid=param_grid,
                selection_metric=selection_metric,
                cv=5
            )

            self._model = best_model
            self.C = best_params.get("C", self.C)
            self.grid_results = results_df
            self.best_params = best_params

        return self._model.fit(self.X, self.y)
    
    def _build_result_df(self):
        return _build_result_df(self.fitted_model, self.X)

    def _calculate_metrics(self, metrics=None):
        """Calculate classification metrics."""
        from isaric.modelevaluation.metrics import compute_classification_metrics

        y_prob = self.fitted_model.predict_proba(self.X)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        result = compute_classification_metrics(
            self.y, y_pred, y_prob=y_prob
        )

        # ============ MODEL INFO ============
        info = {
            "model_type": self.model_type,
            "framework": "scikit-learn",
            "method": "Logistic Regression (L1)",
            "random_seed": self._random_seed,
            "N_observations": len(self.y),
            "N_features": self.X.shape[1],
            "N_events": int(self.y.sum()),
            "parameters": {
                "C": self.fitted_model.C,
                "penalty": "l1",
                "solver": self.fitted_model.solver,
            },
        }
        if hasattr(self.fitted_model, 'C'):
            info["C"] = float(self.fitted_model.C)
        if hasattr(self.fitted_model, 'coef_'):
            n_nonzero = int((self.fitted_model.coef_ != 0).sum())
            info["n_selected_features"] = n_nonzero

        self.model_info = info

        return result

    def _cross_validate(self, k_folds=5, repetitions=1):
        from isaric.modelevaluation.crossvalidation import kfold_cross_validation
        return kfold_cross_validation(self._model, self.X, self.y,
                                      n_splits=k_folds, scoring='roc_auc')

    def _calibration_curve(self):
        from isaric.modelevaluation.calibration import compute_brier_score, calibration_curve, binned_calibration
        y_prob = self.fitted_model.predict_proba(self.X)[:, 1]
        return {
            'brier_score': compute_brier_score(self.y, y_prob),
            'calibration_curve': calibration_curve(self.y, y_prob),
            'calibration_table': binned_calibration(self.y, y_prob)
        }

    def _check_assumptions(self):
        from isaric.modelevaluation.assumptions import test_epv
        return {'epv': test_epv(self.y, n_predictors=len(self.X.columns))}

    def _train_test_split(self, test_size=0.2):
        from isaric.modelevaluation.traintest import stratified_holdout
        data = pd.concat([self.X, self.y], axis=1)
        return stratified_holdout(data, target_col=self.dependent_var, test_size=test_size)

    def _validate_external(self, external_data):
        from isaric.validation.external import temporal_validation
        return temporal_validation(self.fitted_model, external_data,
                                   dependent_var=self.dependent_var,
                                   independent_vars=self.independent_vars)

    def _validate_bootstrap(self, n_iterations=1000):
        """
        Bootstrap confidence intervals and p-values for coefficients.

        Uses bias_corrected CI by default for regularized linear models.
        bootstrap_type is read from fit() via self._bootstrap_type.

        Returns:
            Dictionary with coefficients DataFrame (Feature, observed,
            bootstrap_mean, bootstrap_se, ci_lower, ci_upper,
            bootstrap_p_value, bootstrap_type, n_iterations).
        """
        bootstrap_type = getattr(self, '_bootstrap_type', 'auto')
        if bootstrap_type == 'auto':
            bootstrap_type = 'bias_corrected'

        return _bootstrap_sklearn_coefficients(
            fitted_model=self.fitted_model,
            X=self.X,
            y=self.y,
            n_iterations=n_iterations,
            bootstrap_type=bootstrap_type,
            random_state=getattr(self, '_random_seed', 42),
        )

    def _validate_sensitivity(self):
        from isaric.validation.sensitivity import alternative_missing_handling
        data = pd.concat([self.X, self.y], axis=1)
        return alternative_missing_handling(data, self.dependent_var, self.independent_vars)

    def _validate_subgroups(self, subgroups):
        from isaric.validation.subgroup import stratified_metrics
        return stratified_metrics(self.fitted_model, self.X, self.y, subgroups)

    def _validate_net_benefit(self):
        from isaric.validation.netprofit import decision_curve_analysis
        y_prob = self.fitted_model.predict_proba(self.X)[:, 1]
        return decision_curve_analysis(self.y, y_prob)

    # ======================================================================
    # PLOT METHODS (CALLED BY plots_map)
    # ======================================================================

    def _confusion_matrix(self, backend="plotly"):
        from isaric.visualization.heatmaps import confusion_matrix_heatmap
        y_pred = self.fitted_model.predict(self.X)
        return confusion_matrix_heatmap(
            y_true=self.y, y_pred=y_pred,
            class_names=['Negative', 'Positive'],
            title="Confusion Matrix - LASSO",
            backend=backend
        )

    def _coefficient_plot(self, backend="plotly"):
        from isaric.visualization.barplots import simple_bar_plot
        coef_df = _build_result_df(self.fitted_model, self.X)
        return simple_bar_plot(
            coef_df, x_col='Feature', y_col='Coefficient',
            title="Coefficients - LASSO",
            backend=backend
        )

    def _calibration_plot(self, backend="plotly"):
        from isaric.visualization.lineplots import line_with_ci
        from isaric.modelevaluation.calibration import calibration_curve
        y_prob = self.fitted_model.predict_proba(self.X)[:, 1]
        calib = calibration_curve(self.y, y_prob)
        data = pd.DataFrame({
            'predicted': calib['mean_predicted'],
            'observed': calib['fraction_positive'],
            'ci_lower': calib['fraction_positive'] * 0.9,
            'ci_upper': calib['fraction_positive'] * 1.1,
        })
        return line_with_ci(data=data, x_col='predicted', y_col='observed',
                           ci_lower_col='ci_lower', ci_upper_col='ci_upper',
                           title="Calibration Curve - LASSO",
                           xaxis_title="Predicted Probability",
                           yaxis_title="Observed Proportion",
                           backend=backend)


class Ridge(RAPID):
    """Concrete pipeline for Ridge logistic regression."""

    def __init__(self, model, X, y, dependent_var, independent_vars, **kwargs):
        self._model = model
        self.X = X
        self.y = y
        self.dependent_var = dependent_var
        self.independent_vars = independent_vars
        self.model_type = "ridge"
        self.fitted_model = None
        self.result_df = None
        self.metrics = None
        self.plots_map = {}
        self._setup_plots_map()
        super().__init__()

    def _setup_plots_map(self):
        self.plots_map = {
            "confusion_matrix": self._confusion_matrix,
            "coefficient_plot": self._coefficient_plot,
            "calibration_curve": self._calibration_plot,
        }

    @classmethod
    def create(
        cls,
        data: pd.DataFrame,
        model: str = "ridge",
        dependent_var: Optional[str] = None,
        independent_vars: Optional[List[str]] = None,
        random_seed: int = 42,
        **params
    ) -> "Ridge":
        model_config, X, y = create_ridge_model(
            data=data,
            dependent_var=dependent_var,
            independent_vars=independent_vars,
            random_seed=random_seed,
            **params
        )

        instance = cls(
            model=model_config,
            X=X,
            y=y,
            dependent_var=dependent_var,
            independent_vars=independent_vars,
            **params
        )
        instance._random_seed = random_seed
        
        return instance

    # ======================================================================
    # PRIVATE METHODS (CALLED BY fit() AND validation())
    # ======================================================================

    def _train_model(self, grid_search=False, param_grid=None, selection_metric="auto"):
        """
        Train Ridge model, optionally with grid search.
        """
        if grid_search:
            from isaric.modelevaluation.gridsearch import run_grid_search

            if param_grid is None:
                param_grid = {"C": [0.01, 0.1, 1.0, 10.0]}

            if selection_metric == "auto":
                selection_metric = "roc_auc"

            best_model, best_params, results_df = run_grid_search(
                estimator=self._model,
                X=self.X,
                y=self.y,
                param_grid=param_grid,
                selection_metric=selection_metric,
                cv=5
            )

            self._model = best_model
            self.C = best_params.get("C", self.C)
            self.grid_results = results_df
            self.best_params = best_params

        return self._model.fit(self.X, self.y)

    def _build_result_df(self):
        return _build_result_df(self.fitted_model, self.X)

    def _calculate_metrics(self, metrics=None):
        """Calculate classification metrics."""
        from isaric.modelevaluation.metrics import compute_classification_metrics

        y_prob = self.fitted_model.predict_proba(self.X)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        result = compute_classification_metrics(
            self.y, y_pred, y_prob=y_prob
        )

        # ============ MODEL INFO ============
        info = {
            "model_type": self.model_type,
            "framework": "scikit-learn",
            "method": "Logistic Regression (L2)",
            "random_seed": self._random_seed,
            "N_observations": len(self.y),
            "N_features": self.X.shape[1],
            "N_events": int(self.y.sum()),
            "parameters": {
                "C": self.fitted_model.C,
                "penalty": "l2",
                "solver": self.fitted_model.solver,
            },
        }

        if hasattr(self.fitted_model, 'C'):
            info["C"] = float(self.fitted_model.C)

        self.model_info = info

        return result

    def _cross_validate(self, k_folds=5, repetitions=1):
        from isaric.modelevaluation.crossvalidation import kfold_cross_validation
        return kfold_cross_validation(self._model, self.X, self.y,
                                      n_splits=k_folds, scoring='roc_auc')

    def _calibration_curve(self):
        from isaric.modelevaluation.calibration import compute_brier_score, calibration_curve, binned_calibration
        y_prob = self.fitted_model.predict_proba(self.X)[:, 1]
        return {
            'brier_score': compute_brier_score(self.y, y_prob),
            'calibration_curve': calibration_curve(self.y, y_prob),
            'calibration_table': binned_calibration(self.y, y_prob)
        }

    def _check_assumptions(self):
        from isaric.modelevaluation.assumptions import test_epv
        return {'epv': test_epv(self.y, n_predictors=len(self.X.columns))}

    def _train_test_split(self, test_size=0.2):
        from isaric.modelevaluation.traintest import stratified_holdout
        data = pd.concat([self.X, self.y], axis=1)
        return stratified_holdout(data, target_col=self.dependent_var, test_size=test_size)

    def _validate_external(self, external_data):
        from isaric.validation.external import temporal_validation
        return temporal_validation(self.fitted_model, external_data,
                                   dependent_var=self.dependent_var,
                                   independent_vars=self.independent_vars)

    def _validate_bootstrap(self, n_iterations=1000):
        """
        Bootstrap confidence intervals and p-values for coefficients.

        Uses bias_corrected CI by default for regularized linear models.
        bootstrap_type is read from fit() via self._bootstrap_type.

        Returns:
            Dictionary with coefficients DataFrame (Feature, observed,
            bootstrap_mean, bootstrap_se, ci_lower, ci_upper,
            bootstrap_p_value, bootstrap_type, n_iterations).
        """
        bootstrap_type = getattr(self, '_bootstrap_type', 'auto')
        if bootstrap_type == 'auto':
            bootstrap_type = 'bias_corrected'

        return _bootstrap_sklearn_coefficients(
            fitted_model=self.fitted_model,
            X=self.X,
            y=self.y,
            n_iterations=n_iterations,
            bootstrap_type=bootstrap_type,
            random_state=getattr(self, '_random_seed', 42),
        )

    def _validate_sensitivity(self):
        from isaric.validation.sensitivity import alternative_missing_handling
        data = pd.concat([self.X, self.y], axis=1)
        return alternative_missing_handling(data, self.dependent_var, self.independent_vars)

    def _validate_subgroups(self, subgroups):
        from isaric.validation.subgroup import stratified_metrics
        return stratified_metrics(self.fitted_model, self.X, self.y, subgroups)

    def _validate_net_benefit(self):
        from isaric.validation.netprofit import decision_curve_analysis
        y_prob = self.fitted_model.predict_proba(self.X)[:, 1]
        return decision_curve_analysis(self.y, y_prob)

    # ======================================================================
    # PLOT METHODS (CALLED BY plots_map)
    # ======================================================================

    def _confusion_matrix(self, backend="plotly"):
        from isaric.visualization.heatmaps import confusion_matrix_heatmap
        y_pred = self.fitted_model.predict(self.X)
        return confusion_matrix_heatmap(
            y_true=self.y, y_pred=y_pred,
            class_names=['Negative', 'Positive'],
            title="Confusion Matrix - Ridge",
            backend=backend
        )

    def _coefficient_plot(self, backend="plotly"):
        from isaric.visualization.barplots import simple_bar_plot
        coef_df = _build_result_df(self.fitted_model, self.X)
        return simple_bar_plot(
            coef_df, x_col='Feature', y_col='Coefficient',
            title="Coefficients - Ridge",
            backend=backend
        )

    def _calibration_plot(self, backend="plotly"):
        from isaric.visualization.lineplots import line_with_ci
        from isaric.modelevaluation.calibration import calibration_curve
        y_prob = self.fitted_model.predict_proba(self.X)[:, 1]
        calib = calibration_curve(self.y, y_prob)
        data = pd.DataFrame({
            'predicted': calib['mean_predicted'],
            'observed': calib['fraction_positive'],
            'ci_lower': calib['fraction_positive'] * 0.9,
            'ci_upper': calib['fraction_positive'] * 1.1,
        })
        return line_with_ci(data=data, x_col='predicted', y_col='observed',
                           ci_lower_col='ci_lower', ci_upper_col='ci_upper',
                           title="Calibration Curve - Ridge",
                           xaxis_title="Predicted Probability",
                           yaxis_title="Observed Proportion",
                           backend=backend)


class ElasticNet(RAPID):
    """Concrete pipeline for Elastic Net logistic regression."""

    def __init__(self, model, X, y, dependent_var, independent_vars, **kwargs):
        self._model = model
        self.X = X
        self.y = y
        self.dependent_var = dependent_var
        self.independent_vars = independent_vars
        self.model_type = "elastic_net"
        self.fitted_model = None
        self.result_df = None
        self.metrics = None
        self.plots_map = {}
        self._setup_plots_map()
        super().__init__()

    def _setup_plots_map(self):
        self.plots_map = {
            "confusion_matrix": self._confusion_matrix,
            "coefficient_plot": self._coefficient_plot,
            "calibration_curve": self._calibration_plot,
        }

    @classmethod
    def create(
        cls,
        data: pd.DataFrame,
        model: str = "elastic_net",
        dependent_var: Optional[str] = None,
        independent_vars: Optional[List[str]] = None,
        random_seed: int = 42,
        **params
    ) -> "ElasticNet":
        model_config, X, y = create_elastic_net_model(
            data=data,
            dependent_var=dependent_var,
            independent_vars=independent_vars,
            random_seed=random_seed,
            **params
        )

        instance = cls(
            model=model_config,
            X=X,
            y=y,
            dependent_var=dependent_var,
            independent_vars=independent_vars,
            **params
        )
        instance._random_seed = random_seed
        
        return instance

    # ======================================================================
    # PRIVATE METHODS (CALLED BY fit() AND validation())
    # ======================================================================

    def _train_model(self, grid_search=False, param_grid=None, selection_metric="auto"):
        """
        Train Elastic Net model, optionally with grid search.
        """
        if grid_search:
            from isaric.modelevaluation.gridsearch import run_grid_search

            if param_grid is None:
                param_grid = {"C": [0.1, 1.0], "l1_ratio": [0.3, 0.5, 0.7]}

            if selection_metric == "auto":
                selection_metric = "roc_auc"

            best_model, best_params, results_df = run_grid_search(
                estimator=self._model,
                X=self.X,
                y=self.y,
                param_grid=param_grid,
                selection_metric=selection_metric,
                cv=5
            )

            self._model = best_model
            self.C = best_params.get("C", self.C)
            self.l1_ratio = best_params.get("l1_ratio", self.l1_ratio)
            self.grid_results = results_df
            self.best_params = best_params

        return self._model.fit(self.X, self.y)

    def _build_result_df(self):
        return _build_result_df(self.fitted_model, self.X)

    def _calculate_metrics(self, metrics=None):
        """Calculate classification metrics."""
        from isaric.modelevaluation.metrics import compute_classification_metrics

        y_prob = self.fitted_model.predict_proba(self.X)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        result = compute_classification_metrics(
            self.y, y_pred, y_prob=y_prob
        )

        # ============ MODEL INFO ============
        info = {
            "model_type": self.model_type,
            "framework": "scikit-learn",
            "method": "Logistic Regression (Elastic Net)",
            "random_seed": self._random_seed,
            "N_observations": len(self.y),
            "N_features": self.X.shape[1],
            "N_events": int(self.y.sum()),
            "parameters": {
                "C": self.fitted_model.C,
                "l1_ratio": self.fitted_model.l1_ratio,
                "penalty": "elasticnet",
            },
        }

        if hasattr(self.fitted_model, 'C'):
            info["C"] = float(self.fitted_model.C)
        if hasattr(self.fitted_model, 'l1_ratio') and self.fitted_model.l1_ratio is not None:
            info["l1_ratio"] = float(self.fitted_model.l1_ratio)
        if hasattr(self.fitted_model, 'coef_'):
            n_nonzero = int((self.fitted_model.coef_ != 0).sum())
            info["n_selected_features"] = n_nonzero

        self.model_info = info

        return result

    def _cross_validate(self, k_folds=5, repetitions=1):
        from isaric.modelevaluation.crossvalidation import kfold_cross_validation
        return kfold_cross_validation(self._model, self.X, self.y,
                                      n_splits=k_folds, scoring='roc_auc')

    def _calibration_curve(self):
        from isaric.modelevaluation.calibration import compute_brier_score, calibration_curve, binned_calibration
        y_prob = self.fitted_model.predict_proba(self.X)[:, 1]
        return {
            'brier_score': compute_brier_score(self.y, y_prob),
            'calibration_curve': calibration_curve(self.y, y_prob),
            'calibration_table': binned_calibration(self.y, y_prob)
        }

    def _check_assumptions(self):
        from isaric.modelevaluation.assumptions import test_epv
        return {'epv': test_epv(self.y, n_predictors=len(self.X.columns))}

    def _train_test_split(self, test_size=0.2):
        from isaric.modelevaluation.traintest import stratified_holdout
        data = pd.concat([self.X, self.y], axis=1)
        return stratified_holdout(data, target_col=self.dependent_var, test_size=test_size)

    def _validate_external(self, external_data):
        from isaric.validation.external import temporal_validation
        return temporal_validation(self.fitted_model, external_data,
                                   dependent_var=self.dependent_var,
                                   independent_vars=self.independent_vars)

    def _validate_bootstrap(self, n_iterations=1000):
        """
        Bootstrap confidence intervals and p-values for coefficients.

        Uses bias_corrected CI by default for regularized linear models.
        bootstrap_type is read from fit() via self._bootstrap_type.

        Returns:
            Dictionary with coefficients DataFrame (Feature, observed,
            bootstrap_mean, bootstrap_se, ci_lower, ci_upper,
            bootstrap_p_value, bootstrap_type, n_iterations).
        """
        bootstrap_type = getattr(self, '_bootstrap_type', 'auto')
        if bootstrap_type == 'auto':
            bootstrap_type = 'bias_corrected'

        return _bootstrap_sklearn_coefficients(
            fitted_model=self.fitted_model,
            X=self.X,
            y=self.y,
            n_iterations=n_iterations,
            bootstrap_type=bootstrap_type,
            random_state=getattr(self, '_random_seed', 42),
        )

    def _validate_sensitivity(self):
        from isaric.validation.sensitivity import alternative_missing_handling
        data = pd.concat([self.X, self.y], axis=1)
        return alternative_missing_handling(data, self.dependent_var, self.independent_vars)

    def _validate_subgroups(self, subgroups):
        from isaric.validation.subgroup import stratified_metrics
        return stratified_metrics(self.fitted_model, self.X, self.y, subgroups)

    def _validate_net_benefit(self):
        from isaric.validation.netprofit import decision_curve_analysis
        y_prob = self.fitted_model.predict_proba(self.X)[:, 1]
        return decision_curve_analysis(self.y, y_prob)

    # ======================================================================
    # PLOT METHODS (CALLED BY plots_map)
    # ======================================================================

    def _confusion_matrix(self, backend="plotly"):
        from isaric.visualization.heatmaps import confusion_matrix_heatmap
        y_pred = self.fitted_model.predict(self.X)
        return confusion_matrix_heatmap(
            y_true=self.y, y_pred=y_pred,
            class_names=['Negative', 'Positive'],
            title="Confusion Matrix - Elastic Net",
            backend=backend
        )

    def _coefficient_plot(self, backend="plotly"):
        from isaric.visualization.barplots import simple_bar_plot
        coef_df = _build_result_df(self.fitted_model, self.X)
        return simple_bar_plot(
            coef_df, x_col='Feature', y_col='Coefficient',
            title="Coefficients - Elastic Net",
            backend=backend
        )

    def _calibration_plot(self, backend="plotly"):
        from isaric.visualization.lineplots import line_with_ci
        from isaric.modelevaluation.calibration import calibration_curve
        y_prob = self.fitted_model.predict_proba(self.X)[:, 1]
        calib = calibration_curve(self.y, y_prob)
        data = pd.DataFrame({
            'predicted': calib['mean_predicted'],
            'observed': calib['fraction_positive'],
            'ci_lower': calib['fraction_positive'] * 0.9,
            'ci_upper': calib['fraction_positive'] * 1.1,
        })
        return line_with_ci(data=data, x_col='predicted', y_col='observed',
                           ci_lower_col='ci_lower', ci_upper_col='ci_upper',
                           title="Calibration Curve - Elastic Net",
                           xaxis_title="Predicted Probability",
                           yaxis_title="Observed Proportion",
                           backend=backend)


class SVM(RAPID):
    """Concrete pipeline for Support Vector Machine."""

    def __init__(self, model, X, y, dependent_var, independent_vars, **kwargs):
        self._model = model
        self.X = X
        self.y = y
        self.dependent_var = dependent_var
        self.independent_vars = independent_vars
        self.model_type = "svm"
        self.fitted_model = None
        self.result_df = None
        self.metrics = None
        self.plots_map = {}
        self._setup_plots_map()
        super().__init__()

    def _setup_plots_map(self):
        self.plots_map = {
            "confusion_matrix": self._confusion_matrix,
            "calibration_curve": self._calibration_plot,
        }

    @classmethod
    def create(
        cls,
        data: pd.DataFrame,
        model: str = "svm",
        dependent_var: Optional[str] = None,
        independent_vars: Optional[List[str]] = None,
        random_seed: int = 42,
        **params
    ) -> "SVM":
        model_config, X, y = create_svm_model(
            data=data,
            dependent_var=dependent_var,
            independent_vars=independent_vars,
            random_seed=random_seed,
            **params
        )

        instance = cls(
            model=model_config,
            X=X,
            y=y,
            dependent_var=dependent_var,
            independent_vars=independent_vars,
            **params
        )
        instance._random_seed = random_seed
        return instance

    # ======================================================================
    # PRIVATE METHODS (CALLED BY fit() AND validation())
    # ======================================================================

    def _train_model(self, grid_search=False, param_grid=None, selection_metric="auto"):
        """
        Train SVM model, optionally with grid search.
        """
        if grid_search:
            from isaric.modelevaluation.gridsearch import run_grid_search

            if param_grid is None:
                param_grid = {"C": [0.1, 1.0, 10.0], "kernel": ["linear", "rbf"]}

            if selection_metric == "auto":
                selection_metric = "roc_auc"

            best_model, best_params, results_df = run_grid_search(
                estimator=self._model,
                X=self.X,
                y=self.y,
                param_grid=param_grid,
                selection_metric=selection_metric,
                cv=5
            )

            self._model = best_model
            self.C = best_params.get("C", self.C)
            self.kernel = best_params.get("kernel", self.kernel)
            self.grid_results = results_df
            self.best_params = best_params

        return self._model.fit(self.X, self.y)

    def _build_result_df(self):
        return pd.DataFrame({
            'Feature': self.independent_vars,
            'Support_Vectors': [self.fitted_model.n_support_[0], self.fitted_model.n_support_[1]]
        })

    def _calculate_metrics(self, metrics=None):
        """Calculate classification metrics."""
        from isaric.modelevaluation.metrics import compute_classification_metrics

        y_prob = self.fitted_model.predict_proba(self.X)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        result = compute_classification_metrics(
            self.y, y_pred, y_prob=y_prob
        )

        # ============ MODEL INFO ============
        info = {
            "model_type": self.model_type,
            "framework": "scikit-learn",
            "method": "Support Vector Machine",
            "random_seed": self._random_seed,
            "N_observations": len(self.y),
            "N_features": self.X.shape[1],
            "N_events": int(self.y.sum()),
            "parameters": {
                "C": self.fitted_model.C,
                "kernel": self.fitted_model.kernel,
                "gamma": str(self.fitted_model.gamma),
            },
        }

        if hasattr(self.fitted_model, 'C'):
            info["C"] = float(self.fitted_model.C)
        if hasattr(self.fitted_model, 'gamma'):
            info["gamma"] = str(self.fitted_model.gamma)
        if hasattr(self.fitted_model, 'n_support_'):
            info["n_support_vectors"] = int(self.fitted_model.n_support_.sum())

        self.model_info = info

        return result

    def _cross_validate(self, k_folds=5, repetitions=1):
        from isaric.modelevaluation.crossvalidation import kfold_cross_validation
        return kfold_cross_validation(self._model, self.X, self.y,
                                      n_splits=k_folds, scoring='roc_auc')

    def _calibration_curve(self):
        from isaric.modelevaluation.calibration import compute_brier_score, calibration_curve, binned_calibration
        y_prob = self.fitted_model.predict_proba(self.X)[:, 1]
        return {
            'brier_score': compute_brier_score(self.y, y_prob),
            'calibration_curve': calibration_curve(self.y, y_prob),
            'calibration_table': binned_calibration(self.y, y_prob)
        }

    def _check_assumptions(self):
        from isaric.modelevaluation.assumptions import test_epv
        return {'epv': test_epv(self.y, n_predictors=len(self.X.columns))}

    def _train_test_split(self, test_size=0.2):
        from isaric.modelevaluation.traintest import stratified_holdout
        data = pd.concat([self.X, self.y], axis=1)
        return stratified_holdout(data, target_col=self.dependent_var, test_size=test_size)

    def _validate_external(self, external_data):
        from isaric.validation.external import temporal_validation
        return temporal_validation(self.fitted_model, external_data,
                                   dependent_var=self.dependent_var,
                                   independent_vars=self.independent_vars)

    def _validate_bootstrap(self, n_iterations=1000):
        """
        Bootstrap CI for SVM.

        - Linear kernel: bootstrap coefficients (bias_corrected).
        - Non-linear kernels (rbf, poly, sigmoid): bootstrap AUC CI
          (percentile), since coef_ is not available.

        bootstrap_type is read from fit() via self._bootstrap_type.

        Returns:
            Dictionary with either 'coefficients' DataFrame (linear)
            or 'auc' CI dict (non-linear).
        """
        bootstrap_type = getattr(self, '_bootstrap_type', 'auto')
        kernel = getattr(self.fitted_model, 'kernel', 'rbf')

        # Linear SVM: has coef_
        if kernel == 'linear' and hasattr(self.fitted_model, 'coef_'):
            if bootstrap_type == 'auto':
                bootstrap_type = 'bias_corrected'
            return _bootstrap_sklearn_coefficients(
                fitted_model=self.fitted_model,
                X=self.X,
                y=self.y,
                n_iterations=n_iterations,
                bootstrap_type=bootstrap_type,
                random_state=getattr(self, '_random_seed', 42),
            )

        # Non-linear SVM: no coef_ — bootstrap AUC CI
        if bootstrap_type == 'auto':
            bootstrap_type = 'percentile'

        from sklearn.base import clone
        from sklearn.metrics import roc_auc_score
        from isaric.modelevaluation.bootstrap import confidence_interval

        rng = np.random.default_rng(getattr(self, '_random_seed', 42))
        X_arr = self.X.values if hasattr(self.X, 'values') else np.asarray(self.X)
        y_arr = self.y.values if hasattr(self.y, 'values') else np.asarray(self.y)
        n_samples = X_arr.shape[0]

        auc_values = []
        for _ in range(n_iterations):
            idx = rng.integers(0, n_samples, n_samples)
            X_boot, y_boot = X_arr[idx], y_arr[idx]
            try:
                y_prob = self.fitted_model.predict_proba(X_boot)[:, 1]
                auc_values.append(roc_auc_score(y_boot, y_prob))
            except Exception:
                pass

        auc_arr = np.array(auc_values)
        obs_auc = roc_auc_score(
            self.y,
            self.fitted_model.predict_proba(self.X)[:, 1]
        )

        ci_lo, ci_hi = confidence_interval(
            values=auc_arr,
            bootstrap_type=bootstrap_type,
            observed=float(obs_auc),
        ) if len(auc_arr) > 1 else (np.nan, np.nan)

        return {
            'auc': {
                'observed': float(obs_auc),
                'mean': float(np.mean(auc_arr)) if len(auc_arr) > 0 else np.nan,
                'ci_lower': float(ci_lo),
                'ci_upper': float(ci_hi),
            },
            'note': f'SVM kernel={kernel}: coef_ not available. AUC CI returned instead.',
            'bootstrap_type': bootstrap_type,
            'n_iterations': n_iterations,
        }
    
    def _validate_sensitivity(self):
        from isaric.validation.sensitivity import alternative_missing_handling
        data = pd.concat([self.X, self.y], axis=1)
        return alternative_missing_handling(data, self.dependent_var, self.independent_vars)

    def _validate_subgroups(self, subgroups):
        from isaric.validation.subgroup import stratified_metrics
        return stratified_metrics(self.fitted_model, self.X, self.y, subgroups)

    def _validate_net_benefit(self):
        from isaric.validation.netprofit import decision_curve_analysis
        y_prob = self.fitted_model.predict_proba(self.X)[:, 1]
        return decision_curve_analysis(self.y, y_prob)

    # ======================================================================
    # PLOT METHODS (CALLED BY plots_map)
    # ======================================================================

    def _confusion_matrix(self, backend="plotly"):
        from isaric.visualization.heatmaps import confusion_matrix_heatmap
        y_pred = self.fitted_model.predict(self.X)
        return confusion_matrix_heatmap(
            y_true=self.y, y_pred=y_pred,
            class_names=['Negative', 'Positive'],
            title="Confusion Matrix - SVM",
            backend=backend
        )

    def _calibration_plot(self, backend="plotly"):
        from isaric.visualization.lineplots import line_with_ci
        from isaric.modelevaluation.calibration import calibration_curve
        y_prob = self.fitted_model.predict_proba(self.X)[:, 1]
        calib = calibration_curve(self.y, y_prob)
        data = pd.DataFrame({
            'predicted': calib['mean_predicted'],
            'observed': calib['fraction_positive'],
            'ci_lower': calib['fraction_positive'] * 0.9,
            'ci_upper': calib['fraction_positive'] * 1.1,
        })
        return line_with_ci(data=data, x_col='predicted', y_col='observed',
                           ci_lower_col='ci_lower', ci_upper_col='ci_upper',
                           title="Calibration Curve - SVM",
                           xaxis_title="Predicted Probability",
                           yaxis_title="Observed Proportion",
                           backend=backend)


class LogisticL2(RAPID):
    """Concrete pipeline for Logistic Regression with L2 (predictive)."""

    def __init__(self, model, X, y, dependent_var, independent_vars, **kwargs):
        self._model = model
        self.X = X
        self.y = y
        self.dependent_var = dependent_var
        self.independent_vars = independent_vars
        self.model_type = "logistic_l2"
        self.fitted_model = None
        self.result_df = None
        self.metrics = None
        self.plots_map = {}
        self._setup_plots_map()
        super().__init__()

    def _setup_plots_map(self):
        self.plots_map = {
            "confusion_matrix": self._confusion_matrix,
            "coefficient_plot": self._coefficient_plot,
            "calibration_curve": self._calibration_plot,
        }

    @classmethod
    def create(
        cls,
        data: pd.DataFrame,
        model: str = "logistic_l2",
        dependent_var: Optional[str] = None,
        independent_vars: Optional[List[str]] = None,
        random_seed: int = 42,
        **params
    ) -> "LogisticL2":
        model_config, X, y = create_logistic_l2_model(
            data=data,
            dependent_var=dependent_var,
            independent_vars=independent_vars,
            random_seed=random_seed,
            **params
        )

        instance = cls(
            model=model_config,
            X=X,
            y=y,
            dependent_var=dependent_var,
            independent_vars=independent_vars,
            **params
        )
        instance._random_seed = random_seed
        
        return instance

    # ======================================================================
    # PRIVATE METHODS (CALLED BY fit() AND validation())
    # ======================================================================

    def _train_model(self, grid_search=False, param_grid=None, selection_metric="auto"):
        """
        Train Logistic L2 model, optionally with grid search.
        """
        if grid_search:
            from isaric.modelevaluation.gridsearch import run_grid_search

            if param_grid is None:
                param_grid = {"C": [0.01, 0.1, 1.0, 10.0]}

            if selection_metric == "auto":
                selection_metric = "roc_auc"

            best_model, best_params, results_df = run_grid_search(
                estimator=self._model,
                X=self.X,
                y=self.y,
                param_grid=param_grid,
                selection_metric=selection_metric,
                cv=5
            )

            self._model = best_model
            self.C = best_params.get("C", self.C)
            self.grid_results = results_df
            self.best_params = best_params

        return self._model.fit(self.X, self.y)

    def _build_result_df(self):
        return _build_result_df(self.fitted_model, self.X)

    def _calculate_metrics(self, metrics=None):
        """Calculate classification metrics."""
        from isaric.modelevaluation.metrics import compute_classification_metrics

        y_prob = self.fitted_model.predict_proba(self.X)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        result = compute_classification_metrics(
            self.y, y_pred, y_prob=y_prob
        )

        # ============ MODEL INFO ============
        info = {
            "model_type": self.model_type,
            "framework": "scikit-learn",
            "method": "Logistic Regression (L2)",
            "random_seed": self._random_seed,
            "N_observations": len(self.y),
            "N_features": self.X.shape[1],
            "N_events": int(self.y.sum()),
            "parameters": {
                "C": self.fitted_model.C,
                "penalty": "l2",
            },
        }

        if hasattr(self.fitted_model, 'C'):
            info["C"] = float(self.fitted_model.C)

        self.model_info = info

        return result

    def _cross_validate(self, k_folds=5, repetitions=1):
        from isaric.modelevaluation.crossvalidation import kfold_cross_validation
        return kfold_cross_validation(self._model, self.X, self.y,
                                      n_splits=k_folds, scoring='roc_auc')

    def _calibration_curve(self):
        from isaric.modelevaluation.calibration import compute_brier_score, calibration_curve, binned_calibration
        y_prob = self.fitted_model.predict_proba(self.X)[:, 1]
        return {
            'brier_score': compute_brier_score(self.y, y_prob),
            'calibration_curve': calibration_curve(self.y, y_prob),
            'calibration_table': binned_calibration(self.y, y_prob)
        }

    def _check_assumptions(self):
        from isaric.modelevaluation.assumptions import test_epv
        return {'epv': test_epv(self.y, n_predictors=len(self.X.columns))}

    def _train_test_split(self, test_size=0.2):
        from isaric.modelevaluation.traintest import stratified_holdout
        data = pd.concat([self.X, self.y], axis=1)
        return stratified_holdout(data, target_col=self.dependent_var, test_size=test_size)

    def _validate_external(self, external_data):
        from isaric.validation.external import temporal_validation
        return temporal_validation(self.fitted_model, external_data,
                                   dependent_var=self.dependent_var,
                                   independent_vars=self.independent_vars)

    def _validate_bootstrap(self, n_iterations=1000):
        """
        Bootstrap confidence intervals and p-values for coefficients.

        Uses bias_corrected CI by default for regularized linear models.
        bootstrap_type is read from fit() via self._bootstrap_type.

        Returns:
            Dictionary with coefficients DataFrame (Feature, observed,
            bootstrap_mean, bootstrap_se, ci_lower, ci_upper,
            bootstrap_p_value, bootstrap_type, n_iterations).
        """
        bootstrap_type = getattr(self, '_bootstrap_type', 'auto')
        if bootstrap_type == 'auto':
            bootstrap_type = 'bias_corrected'

        return _bootstrap_sklearn_coefficients(
            fitted_model=self.fitted_model,
            X=self.X,
            y=self.y,
            n_iterations=n_iterations,
            bootstrap_type=bootstrap_type,
            random_state=getattr(self, '_random_seed', 42),
        )

    def _validate_sensitivity(self):
        from isaric.validation.sensitivity import alternative_missing_handling
        data = pd.concat([self.X, self.y], axis=1)
        return alternative_missing_handling(data, self.dependent_var, self.independent_vars)

    def _validate_subgroups(self, subgroups):
        from isaric.validation.subgroup import stratified_metrics
        return stratified_metrics(self.fitted_model, self.X, self.y, subgroups)

    def _validate_net_benefit(self):
        from isaric.validation.netprofit import decision_curve_analysis
        y_prob = self.fitted_model.predict_proba(self.X)[:, 1]
        return decision_curve_analysis(self.y, y_prob)

    # ======================================================================
    # PLOT METHODS (CALLED BY plots_map)
    # ======================================================================

    def _confusion_matrix(self, backend="plotly"):
        from isaric.visualization.heatmaps import confusion_matrix_heatmap
        y_pred = self.fitted_model.predict(self.X)
        return confusion_matrix_heatmap(
            y_true=self.y, y_pred=y_pred,
            class_names=['Negative', 'Positive'],
            title="Confusion Matrix - Logistic L2",
            backend=backend
        )

    def _coefficient_plot(self, backend="plotly"):
        from isaric.visualization.barplots import simple_bar_plot
        coef_df = _build_result_df(self.fitted_model, self.X)
        return simple_bar_plot(
            coef_df, x_col='Feature', y_col='Coefficient',
            title="Coefficients - Logistic L2",
            backend=backend
        )

    def _calibration_plot(self, backend="plotly"):
        from isaric.visualization.lineplots import line_with_ci
        from isaric.modelevaluation.calibration import calibration_curve
        y_prob = self.fitted_model.predict_proba(self.X)[:, 1]
        calib = calibration_curve(self.y, y_prob)
        data = pd.DataFrame({
            'predicted': calib['mean_predicted'],
            'observed': calib['fraction_positive'],
            'ci_lower': calib['fraction_positive'] * 0.9,
            'ci_upper': calib['fraction_positive'] * 1.1,
        })
        return line_with_ci(data=data, x_col='predicted', y_col='observed',
                           ci_lower_col='ci_lower', ci_upper_col='ci_upper',
                           title="Calibration Curve - Logistic L2",
                           xaxis_title="Predicted Probability",
                           yaxis_title="Observed Proportion",
                           backend=backend)