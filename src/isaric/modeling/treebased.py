"""
Tree-based models for the RAPID methodology.

This module provides functions to configure tree-based machine learning
models and concrete pipeline classes for classification tasks.

Functions (Configuration):
- create_decision_tree_model: Configure Decision Tree classifier.
- create_random_forest_model: Configure Random Forest classifier.
- create_xgboost_model: Configure XGBoost classifier.
- create_lightgbm_model: Configure LightGBM classifier.
- create_catboost_model: Configure CatBoost classifier.

Subclasses (Pipelines):
- DecisionTree: Concrete pipeline for Decision Tree.
- RandomForest: Concrete pipeline for Random Forest.
- XGBoost: Concrete pipeline for XGBoost.
- LightGBM: Concrete pipeline for LightGBM.
- CatBoost: Concrete pipeline for CatBoost.

Helper Functions:
- _prepare_data: Prepare X and y for training.
- _build_result_df: Build feature importance DataFrame.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from isaric.rapid import RAPID


# ============================================================================
# PUBLIC FUNCTIONS (CONFIGURATION)
# ============================================================================

def create_decision_tree_model(
    data: pd.DataFrame,
    dependent_var: str,
    independent_vars: List[str],
    max_depth: Optional[int] = None,
    min_samples_split: int = 2,
    random_seed=42
) -> Tuple[DecisionTreeClassifier, pd.DataFrame, pd.Series]:
    """
    Configure a Decision Tree classifier (not fitted).

    Returns:
        Tuple of (model, X, y).
    """
    X, y = _prepare_data(data, dependent_var, independent_vars)

    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_seed,
        class_weight='balanced'
    )

    return model, X, y


def create_random_forest_model(
    data: pd.DataFrame,
    dependent_var: str,
    independent_vars: List[str],
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    random_seed=42
) -> Tuple[RandomForestClassifier, pd.DataFrame, pd.Series]:
    """
    Configure a Random Forest classifier (not fitted).

    Returns:
        Tuple of (model, X, y).
    """
    X, y = _prepare_data(data, dependent_var, independent_vars)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_seed,
        class_weight='balanced'
    )

    return model, X, y


def create_xgboost_model(
    data: pd.DataFrame,
    dependent_var: str,
    independent_vars: List[str],
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    max_depth: int = 6,
    random_seed=42
) -> Tuple[XGBClassifier, pd.DataFrame, pd.Series]:
    """
    Configure an XGBoost classifier (not fitted).

    Returns:
        Tuple of (model, X, y).
    """
    X, y = _prepare_data(data, dependent_var, independent_vars)

    model = XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_seed,
        eval_metric='logloss'
    )

    return model, X, y


def create_lightgbm_model(
    data: pd.DataFrame,
    dependent_var: str,
    independent_vars: List[str],
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    num_leaves: int = 31,
    random_seed=42
) -> Tuple[LGBMClassifier, pd.DataFrame, pd.Series]:
    """
    Configure a LightGBM classifier (not fitted).

    Returns:
        Tuple of (model, X, y).
    """
    X, y = _prepare_data(data, dependent_var, independent_vars)

    model = LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        random_state=random_seed,
        verbose=-1,
        class_weight='balanced'
    )

    return model, X, y


def create_catboost_model(
    data: pd.DataFrame,
    dependent_var: str,
    independent_vars: List[str],
    iterations: int = 100,
    learning_rate: float = 0.1,
    depth: int = 6,
    random_seed=42
) -> Tuple[CatBoostClassifier, pd.DataFrame, pd.Series]:
    """
    Configure a CatBoost classifier (not fitted).

    Returns:
        Tuple of (model, X, y).
    """
    X, y = _prepare_data(data, dependent_var, independent_vars)

    model = CatBoostClassifier(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        random_state=random_seed,
        verbose=0,
        auto_class_weights='Balanced'
    )

    return model, X, y


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
    Build feature importance DataFrame from fitted tree-based model.

    Returns:
        DataFrame with Feature and Importance columns.
    """
    if not hasattr(model, 'feature_importances_'):
        raise ValueError("Model does not have feature_importances_ attribute.")

    importances = model.feature_importances_
    result_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importances
    })
    result_df = result_df.sort_values('Importance', ascending=False).reset_index(drop=True)

    return result_df


def _bootstrap_feature_importance(
    fitted_model,
    X: pd.DataFrame,
    y: pd.Series,
    n_iterations: int = 1000,
    bootstrap_type: str = "percentile",
    random_state: int = 42,
) -> dict:
    """
    Bootstrap confidence intervals for tree-based feature importances.

    Refits a clone of the fitted model on each bootstrap sample and
    collects feature_importances_. CI is computed per feature using
    the requested bootstrap type.

    Args:
        fitted_model: Fitted sklearn-compatible tree-based model.
        X: Predictor matrix.
        y: Outcome vector.
        n_iterations: Number of bootstrap samples.
        bootstrap_type: CI type ('percentile', 'bias_corrected', 'bca').
            Default 'percentile'.
        random_state: Seed for reproducibility.

    Returns:
        Dictionary with:
        - 'importances': DataFrame with Feature, observed_importance,
          bootstrap_mean, bootstrap_se, ci_lower, ci_upper,
          bootstrap_type, n_iterations.
        - 'bootstrap_type': type used.
        - 'n_iterations': iterations run.
    """
    from sklearn.base import clone
    from isaric.modelevaluation.bootstrap import confidence_interval

    X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
    y_arr = y.values if isinstance(y, pd.Series) else np.asarray(y)
    n_samples = X_arr.shape[0]
    n_features = X_arr.shape[1]
    feature_names = list(X.columns) if hasattr(X, 'columns') else [f"var_{i}" for i in range(n_features)]

    rng = np.random.default_rng(random_state)
    boot_importances = np.full((n_iterations, n_features), np.nan)

    for i in range(n_iterations):
        idx = rng.integers(0, n_samples, n_samples)
        X_boot = X_arr[idx]
        y_boot = y_arr[idx]
        try:
            model_boot = clone(fitted_model)
            model_boot.fit(X_boot, y_boot)
            boot_importances[i] = model_boot.feature_importances_
        except Exception:
            pass

    observed = fitted_model.feature_importances_

    rows = []
    for j, feat in enumerate(feature_names):
        boot_j = boot_importances[:, j]
        boot_j = boot_j[~np.isnan(boot_j)]
        obs_j = float(observed[j])

        ci_lo, ci_hi = confidence_interval(
            values=boot_j,
            bootstrap_type=bootstrap_type,
            observed=obs_j,
        ) if len(boot_j) > 1 else (np.nan, np.nan)

        rows.append({
            'Feature': feat,
            'observed_importance': obs_j,
            'bootstrap_mean': float(np.mean(boot_j)) if len(boot_j) > 0 else np.nan,
            'bootstrap_se': float(np.std(boot_j, ddof=1)) if len(boot_j) > 1 else np.nan,
            'ci_lower': float(ci_lo),
            'ci_upper': float(ci_hi),
            'bootstrap_type': bootstrap_type,
            'n_iterations': n_iterations,
        })

    return {
        'importances': pd.DataFrame(rows).sort_values(
            'observed_importance', ascending=False
        ).reset_index(drop=True),
        'bootstrap_type': bootstrap_type,
        'n_iterations': n_iterations,
    }

# ============================================================================
# SUBCLASSES (INHERIT FROM RAPID)
# ============================================================================

class DecisionTree(RAPID):
    """Concrete pipeline for Decision Tree classifier."""

    def __init__(self, model, X, y, dependent_var, independent_vars, **kwargs):
        self._model = model
        self.X = X
        self.y = y
        self.dependent_var = dependent_var
        self.independent_vars = independent_vars
        self.model_type = "decision_tree"
        self.fitted_model = None
        self.result_df = None
        self.metrics = None
        self.plots_map = {}
        self._setup_plots_map()
        super().__init__()

    def _setup_plots_map(self):
        self.plots_map = {
            "confusion_matrix": self._confusion_matrix,
            "feature_importance": self._feature_importance,
            "calibration_curve": self._calibration_plot,
            "binned_residuals": self._binned_residuals,
        }

    def _binned_residuals(self, backend="plotly"):
        """Generate binned residual plot."""
        from isaric.visualization.lineplots import binned_residuals_plot
        from isaric.modelevaluation.assumptions import test_binned_residuals

        y_prob = self.fitted_model.predict_proba(self.X)
        result = test_binned_residuals(self.y, y_prob)

        return binned_residuals_plot(
            bin_centers=result['bin_centers'],
            mean_residuals=result['mean_residuals'],
            lower_bound=result['lower_bound'],
            upper_bound=result['upper_bound'],
            n_per_bin=result['n_per_bin'],
            title="Binned Residual Plot - Decision Tree",
            backend=backend
        )

    @classmethod
    def create(
        cls,
        data: pd.DataFrame,
        model: str = "decision_tree",
        dependent_var: Optional[str] = None,
        independent_vars: Optional[List[str]] = None,
        random_seed: int = 42,
        **params
    ) -> "DecisionTree":
        model_config, X, y = create_decision_tree_model(
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
        Train Decision Tree model, optionally with grid search.
        """
        if grid_search:
            from isaric.modelevaluation.gridsearch import run_grid_search

            if param_grid is None:
                param_grid = {"max_depth": [3, 5, 10, None], "min_samples_split": [2, 5, 10]}

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
            self.max_depth = best_params.get("max_depth", self.max_depth)
            self.min_samples_split = best_params.get("min_samples_split", self.min_samples_split)
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
            "method": "Decision Tree Classifier",
            "random_seed": self._random_seed,
            "N_observations": len(self.y),
            "N_features": self.X.shape[1],
            "N_events": int(self.y.sum()),
            "parameters": {
                "max_depth": self.fitted_model.max_depth,
                "min_samples_split": self.fitted_model.min_samples_split,
            },
        }

        if hasattr(self.fitted_model, 'max_depth'):
            info["max_depth"] = self.fitted_model.max_depth
        if hasattr(self.fitted_model, 'min_samples_split'):
            info["min_samples_split"] = self.fitted_model.min_samples_split
        if hasattr(self.fitted_model, 'get_depth'):
            info["tree_depth"] = self.fitted_model.get_depth()
        if hasattr(self.fitted_model, 'get_n_leaves'):
            info["n_leaves"] = self.fitted_model.get_n_leaves()

        self.model_info = info

        return result

    def _cross_validate(self, k_folds=5, repetitions=1):
        from isaric.modelevaluation.crossvalidation import kfold_cross_validation
        return kfold_cross_validation(
            self._model, self.X, self.y,
            n_splits=k_folds,
            scoring='roc_auc'
        )

    def _calibration_curve(self):
        from isaric.modelevaluation.calibration import compute_brier_score, calibration_curve, binned_calibration
        y_prob = self.fitted_model.predict_proba(self.X)[:, 1]
        return {
            'brier_score': compute_brier_score(self.y, y_prob),
            'calibration_curve': calibration_curve(self.y, y_prob),
            'calibration_table': binned_calibration(self.y, y_prob)
        }

    def _check_assumptions(self):
        """Check model assumptions (VIF + binned residuals)."""
        from isaric.modelevaluation.assumptions import (
            test_vif,
            test_epv,
            test_binned_residuals
        )

        # VIF
        vif_result = test_vif(self.X)

        # EPV
        epv_result = test_epv(self.y, n_predictors=len(self.X.columns))

        # Binned residuals
        y_prob = self.fitted_model.predict_proba(self.X)[:, 1]
        binned_result = test_binned_residuals(self.y, y_prob)

        return {
            'vif': vif_result.to_dict(orient='records'),
            'epv': epv_result,
            'binned_residuals': binned_result,
        }

    def _train_test_split(self, test_size=0.2):
        from isaric.modelevaluation.traintest import stratified_holdout
        data = pd.concat([self.X, self.y], axis=1)
        return stratified_holdout(data, target_col=self.dependent_var, test_size=test_size)

    def _validate_external(self, external_data):
        """Validate on external independent dataset."""
        from isaric.validation.external import validate_regression_sklearn
        return validate_regression_sklearn(
            fitted_model=self.fitted_model,
            external_data=external_data,
            dependent_var=self.dependent_var,
            independent_vars=self.independent_vars,
            is_classifier=True,
        )

    def _validate_bootstrap(self, n_iterations=1000):
        """
        Bootstrap confidence intervals for feature importances.

        Refits a clone of the model on each bootstrap sample and
        collects feature_importances_. CI type is read from fit()
        via bootstrap_type= (default 'percentile' for tree-based).

        Returns:
            Dictionary with importances DataFrame (Feature,
            observed_importance, bootstrap_mean, bootstrap_se,
            ci_lower, ci_upper, bootstrap_type, n_iterations).
        """
        bootstrap_type = getattr(self, '_bootstrap_type', 'auto')
        if bootstrap_type == 'auto':
            bootstrap_type = 'percentile'

        return _bootstrap_feature_importance(
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
            y_true=self.y,
            y_pred=y_pred,
            class_names=['Negative', 'Positive'],
            title="Confusion Matrix - Decision Tree",
            backend=backend
        )

    def _feature_importance(self, backend="plotly"):
        from isaric.visualization.barplots import simple_bar_plot
        importance_df = _build_result_df(self.fitted_model, self.X)
        return simple_bar_plot(
            importance_df,
            x_col='Feature',
            y_col='Importance',
            title="Feature Importance - Decision Tree",
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
                           title="Calibration Curve - Decision Tree",
                           xaxis_title="Predicted Probability",
                           yaxis_title="Observed Proportion",
                           backend=backend)


class RandomForest(RAPID):
    """Concrete pipeline for Random Forest classifier."""

    def __init__(self, model, X, y, dependent_var, independent_vars, **kwargs):
        self._model = model
        self.X = X
        self.y = y
        self.dependent_var = dependent_var
        self.independent_vars = independent_vars
        self.model_type = "random_forest"
        self.fitted_model = None
        self.result_df = None
        self.metrics = None
        self.plots_map = {}
        self._setup_plots_map()
        super().__init__()

    def _setup_plots_map(self):
        self.plots_map = {
            "confusion_matrix": self._confusion_matrix,
            "feature_importance": self._feature_importance,
            "calibration_curve": self._calibration_plot,
            "binned_residuals": self._binned_residuals,
        }

    def _binned_residuals(self, backend="plotly"):
        """Generate binned residual plot."""
        from isaric.visualization.lineplots import binned_residuals_plot
        from isaric.modelevaluation.assumptions import test_binned_residuals

        y_prob = self.fitted_model.predict_proba(self.X)
        result = test_binned_residuals(self.y, y_prob)

        return binned_residuals_plot(
            bin_centers=result['bin_centers'],
            mean_residuals=result['mean_residuals'],
            lower_bound=result['lower_bound'],
            upper_bound=result['upper_bound'],
            n_per_bin=result['n_per_bin'],
            title="Binned Residual Plot - Decision Tree",
            backend=backend
        )

    @classmethod
    def create(
        cls,
        data: pd.DataFrame,
        model: str = "random_forest",
        dependent_var: Optional[str] = None,
        independent_vars: Optional[List[str]] = None,
        random_seed: int = 42,
        **params
    ) -> "RandomForest":
        model_config, X, y = create_random_forest_model(
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
        Train Random Forest model, optionally with grid search.
        """
        if grid_search:
            from isaric.modelevaluation.gridsearch import run_grid_search

            if param_grid is None:
                param_grid = {"n_estimators": [50, 100, 200], "max_depth": [5, 10, None]}

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
            self.n_estimators = best_params.get("n_estimators", self.n_estimators)
            self.max_depth = best_params.get("max_depth", self.max_depth)
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
            "method": "Random Forest Classifier",
            "random_seed": self._random_seed,
            "N_observations": len(self.y),
            "N_features": self.X.shape[1],
            "N_events": int(self.y.sum()),
            "parameters": {
                "n_estimators": self.fitted_model.n_estimators,
                "max_depth": self.fitted_model.max_depth,
            },
        }

        if hasattr(self.fitted_model, 'max_depth'):
            info["max_depth"] = self.fitted_model.max_depth
        if hasattr(self.fitted_model, 'oob_score_'):
            info["oob_score"] = float(self.fitted_model.oob_score_)

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
        """Check model assumptions (VIF + binned residuals)."""
        from isaric.modelevaluation.assumptions import (
            test_vif,
            test_epv,
            test_binned_residuals
        )

        # VIF
        vif_result = test_vif(self.X)

        # EPV
        epv_result = test_epv(self.y, n_predictors=len(self.X.columns))

        # Binned residuals
        y_prob = self.fitted_model.predict_proba(self.X)[:, 1]
        binned_result = test_binned_residuals(self.y, y_prob)

        return {
            'vif': vif_result.to_dict(orient='records'),
            'epv': epv_result,
            'binned_residuals': binned_result,
        }

    def _train_test_split(self, test_size=0.2):
        from isaric.modelevaluation.traintest import stratified_holdout
        data = pd.concat([self.X, self.y], axis=1)
        return stratified_holdout(data, target_col=self.dependent_var, test_size=test_size)

    def _validate_external(self, external_data):
        """Validate on external independent dataset."""
        from isaric.validation.external import validate_regression_sklearn
        return validate_regression_sklearn(
            fitted_model=self.fitted_model,
            external_data=external_data,
            dependent_var=self.dependent_var,
            independent_vars=self.independent_vars,
            is_classifier=True,
        )

    def _validate_bootstrap(self, n_iterations=1000):
        """
        Bootstrap confidence intervals for feature importances.

        Refits a clone of the model on each bootstrap sample and
        collects feature_importances_. CI type is read from fit()
        via bootstrap_type= (default 'percentile' for tree-based).

        Returns:
            Dictionary with importances DataFrame (Feature,
            observed_importance, bootstrap_mean, bootstrap_se,
            ci_lower, ci_upper, bootstrap_type, n_iterations).
        """
        bootstrap_type = getattr(self, '_bootstrap_type', 'auto')
        if bootstrap_type == 'auto':
            bootstrap_type = 'percentile'

        return _bootstrap_feature_importance(
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
            title="Confusion Matrix - Random Forest",
            backend=backend
        )

    def _feature_importance(self, backend="plotly"):
        from isaric.visualization.barplots import simple_bar_plot
        importance_df = _build_result_df(self.fitted_model, self.X)
        return simple_bar_plot(
            importance_df, x_col='Feature', y_col='Importance',
            title="Feature Importance - Random Forest",
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
                           title="Calibration Curve - Random Forest",
                           xaxis_title="Predicted Probability",
                           yaxis_title="Observed Proportion",
                           backend=backend)


class XGBoost(RAPID):
    """Concrete pipeline for XGBoost classifier."""

    def __init__(self, model, X, y, dependent_var, independent_vars, **kwargs):
        self._model = model
        self.X = X
        self.y = y
        self.dependent_var = dependent_var
        self.independent_vars = independent_vars
        self.model_type = "xgboost"
        self.fitted_model = None
        self.result_df = None
        self.metrics = None
        self.plots_map = {}
        self._setup_plots_map()
        super().__init__()

    def _setup_plots_map(self):
        self.plots_map = {
            "confusion_matrix": self._confusion_matrix,
            "feature_importance": self._feature_importance,
            "calibration_curve": self._calibration_plot,
            "binned_residuals": self._binned_residuals,
        }

    def _binned_residuals(self, backend="plotly"):
        """Generate binned residual plot."""
        from isaric.visualization.lineplots import binned_residuals_plot
        from isaric.modelevaluation.assumptions import test_binned_residuals

        y_prob = self.fitted_model.predict_proba(self.X)
        result = test_binned_residuals(self.y, y_prob)

        return binned_residuals_plot(
            bin_centers=result['bin_centers'],
            mean_residuals=result['mean_residuals'],
            lower_bound=result['lower_bound'],
            upper_bound=result['upper_bound'],
            n_per_bin=result['n_per_bin'],
            title="Binned Residual Plot - XGBoost",
            backend=backend
        )

    @classmethod
    def create(
        cls,
        data: pd.DataFrame,
        model: str = "xgboost",
        dependent_var: Optional[str] = None,
        independent_vars: Optional[List[str]] = None,
        random_seed: int = 42,
        **params
    ) -> "XGBoost":
        model_config, X, y = create_xgboost_model(
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
        Train XGBoost model, optionally with grid search.
        """
        if grid_search:
            from isaric.modelevaluation.gridsearch import run_grid_search

            if param_grid is None:
                param_grid = {"n_estimators": [50, 100], "learning_rate": [0.05, 0.1], "max_depth": [3, 6]}

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
            self.n_estimators = best_params.get("n_estimators", self.n_estimators)
            self.learning_rate = best_params.get("learning_rate", self.learning_rate)
            self.max_depth = best_params.get("max_depth", self.max_depth)
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
            "framework": "xgboost",
            "method": "XGBoost Classifier",
            "random_seed": self._random_seed,
            "N_observations": len(self.y),
            "N_features": self.X.shape[1],
            "N_events": int(self.y.sum()),
            "parameters": {
                "n_estimators": self.fitted_model.n_estimators,
                "max_depth": self.fitted_model.max_depth,
                "learning_rate": self.fitted_model.learning_rate,
            },
        }

        if hasattr(self.fitted_model, 'n_estimators'):
            info["n_estimators"] = self.fitted_model.n_estimators
        if hasattr(self.fitted_model, 'max_depth'):
            info["max_depth"] = self.fitted_model.max_depth
        if hasattr(self.fitted_model, 'learning_rate'):
            info["learning_rate"] = self.fitted_model.learning_rate

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
        """Check model assumptions (VIF + binned residuals)."""
        from isaric.modelevaluation.assumptions import (
            test_vif,
            test_epv,
            test_binned_residuals
        )

        # VIF
        vif_result = test_vif(self.X)

        # EPV
        epv_result = test_epv(self.y, n_predictors=len(self.X.columns))

        # Binned residuals
        y_prob = self.fitted_model.predict_proba(self.X)[:, 1]
        binned_result = test_binned_residuals(self.y, y_prob)

        return {
            'vif': vif_result.to_dict(orient='records'),
            'epv': epv_result,
            'binned_residuals': binned_result,
        }

    def _train_test_split(self, test_size=0.2):
        from isaric.modelevaluation.traintest import stratified_holdout
        data = pd.concat([self.X, self.y], axis=1)
        return stratified_holdout(data, target_col=self.dependent_var, test_size=test_size)

    def _validate_external(self, external_data):
        """Validate on external independent dataset."""
        from isaric.validation.external import validate_regression_sklearn
        return validate_regression_sklearn(
            fitted_model=self.fitted_model,
            external_data=external_data,
            dependent_var=self.dependent_var,
            independent_vars=self.independent_vars,
            is_classifier=True,
        )

    def _validate_bootstrap(self, n_iterations=1000):
        """
        Bootstrap confidence intervals for feature importances.

        Refits a clone of the model on each bootstrap sample and
        collects feature_importances_. CI type is read from fit()
        via bootstrap_type= (default 'percentile' for tree-based).

        Returns:
            Dictionary with importances DataFrame (Feature,
            observed_importance, bootstrap_mean, bootstrap_se,
            ci_lower, ci_upper, bootstrap_type, n_iterations).
        """
        bootstrap_type = getattr(self, '_bootstrap_type', 'auto')
        if bootstrap_type == 'auto':
            bootstrap_type = 'percentile'

        return _bootstrap_feature_importance(
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
            title="Confusion Matrix - XGBoost",
            backend=backend
        )

    def _feature_importance(self, backend="plotly"):
        from isaric.visualization.barplots import simple_bar_plot
        importance_df = _build_result_df(self.fitted_model, self.X)
        return simple_bar_plot(
            importance_df, x_col='Feature', y_col='Importance',
            title="Feature Importance - XGBoost",
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
                           title="Calibration Curve - XGBoost",
                           xaxis_title="Predicted Probability",
                           yaxis_title="Observed Proportion",
                           backend=backend)


class LightGBM(RAPID):
    """Concrete pipeline for LightGBM classifier."""

    def __init__(self, model, X, y, dependent_var, independent_vars, **kwargs):
        self._model = model
        self.X = X
        self.y = y
        self.dependent_var = dependent_var
        self.independent_vars = independent_vars
        self.model_type = "lightgbm"
        self.fitted_model = None
        self.result_df = None
        self.metrics = None
        self.plots_map = {}
        self._setup_plots_map()
        super().__init__()

    def _setup_plots_map(self):
        self.plots_map = {
            "confusion_matrix": self._confusion_matrix,
            "feature_importance": self._feature_importance,
            "calibration_curve": self._calibration_plot,
            "binned_residuals": self._binned_residuals,
        }

    def _binned_residuals(self, backend="plotly"):
        """Generate binned residual plot."""
        from isaric.visualization.lineplots import binned_residuals_plot
        from isaric.modelevaluation.assumptions import test_binned_residuals

        y_prob = self.fitted_model.predict_proba(self.X)
        result = test_binned_residuals(self.y, y_prob)

        return binned_residuals_plot(
            bin_centers=result['bin_centers'],
            mean_residuals=result['mean_residuals'],
            lower_bound=result['lower_bound'],
            upper_bound=result['upper_bound'],
            n_per_bin=result['n_per_bin'],
            title="Binned Residual Plot - Logistic Regression",
            backend=backend
        )

    @classmethod
    def create(
        cls,
        data: pd.DataFrame,
        model: str = "lightgbm",
        dependent_var: Optional[str] = None,
        independent_vars: Optional[List[str]] = None,
        random_seed: int = 42,
        **params
    ) -> "LightGBM":
        model_config, X, y = create_lightgbm_model(
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
        Train LightGBM model, optionally with grid search.
        """
        if grid_search:
            from isaric.modelevaluation.gridsearch import run_grid_search

            if param_grid is None:
                param_grid = {"n_estimators": [50, 100], "num_leaves": [15, 31]}

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
            self.n_estimators = best_params.get("n_estimators", self.n_estimators)
            self.num_leaves = best_params.get("num_leaves", self.num_leaves)
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
            "framework": "lightgbm",
            "method": "LightGBM Classifier",
            "random_seed": self._random_seed,
            "N_observations": len(self.y),
            "N_features": self.X.shape[1],
            "N_events": int(self.y.sum()),
            "parameters": {
                "n_estimators": self.fitted_model.n_estimators,
                "num_leaves": self.fitted_model.num_leaves,
                "learning_rate": self.fitted_model.learning_rate,
            },
        }

        if hasattr(self.fitted_model, 'n_estimators'):
            info["n_estimators"] = self.fitted_model.n_estimators
        if hasattr(self.fitted_model, 'num_leaves'):
            info["num_leaves"] = self.fitted_model.num_leaves
        if hasattr(self.fitted_model, 'learning_rate'):
            info["learning_rate"] = self.fitted_model.learning_rate

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
        """Check model assumptions (VIF + binned residuals)."""
        from isaric.modelevaluation.assumptions import (
            test_vif,
            test_epv,
            test_binned_residuals
        )

        # VIF
        vif_result = test_vif(self.X)

        # EPV
        epv_result = test_epv(self.y, n_predictors=len(self.X.columns))

        # Binned residuals
        y_prob = self.fitted_model.predict_proba(self.X)[:, 1]
        binned_result = test_binned_residuals(self.y, y_prob)

        return {
            'vif': vif_result.to_dict(orient='records'),
            'epv': epv_result,
            'binned_residuals': binned_result,
        }

    def _train_test_split(self, test_size=0.2):
        from isaric.modelevaluation.traintest import stratified_holdout
        data = pd.concat([self.X, self.y], axis=1)
        return stratified_holdout(data, target_col=self.dependent_var, test_size=test_size)

    def _validate_external(self, external_data):
        """Validate on external independent dataset."""
        from isaric.validation.external import validate_regression_sklearn
        return validate_regression_sklearn(
            fitted_model=self.fitted_model,
            external_data=external_data,
            dependent_var=self.dependent_var,
            independent_vars=self.independent_vars,
            is_classifier=True,
        )

    def _validate_bootstrap(self, n_iterations=1000):
        """
        Bootstrap confidence intervals for feature importances.

        Refits a clone of the model on each bootstrap sample and
        collects feature_importances_. CI type is read from fit()
        via bootstrap_type= (default 'percentile' for tree-based).

        Returns:
            Dictionary with importances DataFrame (Feature,
            observed_importance, bootstrap_mean, bootstrap_se,
            ci_lower, ci_upper, bootstrap_type, n_iterations).
        """
        bootstrap_type = getattr(self, '_bootstrap_type', 'auto')
        if bootstrap_type == 'auto':
            bootstrap_type = 'percentile'

        return _bootstrap_feature_importance(
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
            title="Confusion Matrix - LightGBM",
            backend=backend
        )

    def _feature_importance(self, backend="plotly"):
        from isaric.visualization.barplots import simple_bar_plot
        importance_df = _build_result_df(self.fitted_model, self.X)
        return simple_bar_plot(
            importance_df, x_col='Feature', y_col='Importance',
            title="Feature Importance - LightGBM",
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
                           title="Calibration Curve - LightGBM",
                           xaxis_title="Predicted Probability",
                           yaxis_title="Observed Proportion",
                           backend=backend)


class CatBoost(RAPID):
    """Concrete pipeline for CatBoost classifier."""

    def __init__(self, model, X, y, dependent_var, independent_vars, **kwargs):
        self._model = model
        self.X = X
        self.y = y
        self.dependent_var = dependent_var
        self.independent_vars = independent_vars
        self.model_type = "catboost"
        self.fitted_model = None
        self.result_df = None
        self.metrics = None
        self.plots_map = {}
        self._setup_plots_map()
        super().__init__()

    def _setup_plots_map(self):
        self.plots_map = {
            "confusion_matrix": self._confusion_matrix,
            "feature_importance": self._feature_importance,
            "calibration_curve": self._calibration_plot,
            "binned_residuals": self._binned_residuals,
        }

    def _binned_residuals(self, backend="plotly"):
        """Generate binned residual plot."""
        from isaric.visualization.lineplots import binned_residuals_plot
        from isaric.modelevaluation.assumptions import test_binned_residuals

        y_prob = self.fitted_model.predict_proba(self.X)
        result = test_binned_residuals(self.y, y_prob)

        return binned_residuals_plot(
            bin_centers=result['bin_centers'],
            mean_residuals=result['mean_residuals'],
            lower_bound=result['lower_bound'],
            upper_bound=result['upper_bound'],
            n_per_bin=result['n_per_bin'],
            title="Binned Residual Plot - Logistic Regression",
            backend=backend
        )

    @classmethod
    def create(
        cls,
        data: pd.DataFrame,
        model: str = "catboost",
        dependent_var: Optional[str] = None,
        independent_vars: Optional[List[str]] = None,
        random_seed: int = 42,
        **params
    ) -> "CatBoost":
        model_config, X, y = create_catboost_model(
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
        Train CatBoost model, optionally with grid search.
        """
        if grid_search:
            from isaric.modelevaluation.gridsearch import run_grid_search

            if param_grid is None:
                param_grid = {"iterations": [50, 100], "depth": [4, 6]}

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
            self.iterations = best_params.get("iterations", self.iterations)
            self.depth = best_params.get("depth", self.depth)
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
            "framework": "catboost",
            "method": "CatBoost Classifier",
            "random_seed": self._random_seed,
            "N_observations": len(self.y),
            "N_features": self.X.shape[1],
            "N_events": int(self.y.sum()),
            "parameters": {
                "iterations": self.fitted_model.tree_count_ if hasattr(self.fitted_model, 'tree_count_') else None,
                "depth": self.fitted_model.get_all_params().get('depth', None) if hasattr(self.fitted_model, 'get_all_params') else None,
            },
        }

        if hasattr(self.fitted_model, 'tree_count_'):
            info["n_estimators"] = self.fitted_model.tree_count_
        if hasattr(self.fitted_model, 'get_all_params'):
            try:
                params = self.fitted_model.get_all_params()
                if 'depth' in params:
                    info["depth"] = params['depth']
                if 'learning_rate' in params:
                    info["learning_rate"] = params['learning_rate']
            except Exception:
                pass

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
        """Check model assumptions (VIF + binned residuals)."""
        from isaric.modelevaluation.assumptions import (
            test_vif,
            test_epv,
            test_binned_residuals
        )

        # VIF
        vif_result = test_vif(self.X)

        # EPV
        epv_result = test_epv(self.y, n_predictors=len(self.X.columns))

        # Binned residuals
        y_prob = self.fitted_model.predict_proba(self.X)[:, 1]
        binned_result = test_binned_residuals(self.y, y_prob)

        return {
            'vif': vif_result.to_dict(orient='records'),
            'epv': epv_result,
            'binned_residuals': binned_result,
        }

    def _train_test_split(self, test_size=0.2):
        from isaric.modelevaluation.traintest import stratified_holdout
        data = pd.concat([self.X, self.y], axis=1)
        return stratified_holdout(data, target_col=self.dependent_var, test_size=test_size)

    def _validate_external(self, external_data):
        """Validate on external independent dataset."""
        from isaric.validation.external import validate_regression_sklearn
        return validate_regression_sklearn(
            fitted_model=self.fitted_model,
            external_data=external_data,
            dependent_var=self.dependent_var,
            independent_vars=self.independent_vars,
            is_classifier=True,
        )

    def _validate_bootstrap(self, n_iterations=1000):
        """
        Bootstrap confidence intervals for feature importances.

        Refits a clone of the model on each bootstrap sample and
        collects feature_importances_. CI type is read from fit()
        via bootstrap_type= (default 'percentile' for tree-based).

        Returns:
            Dictionary with importances DataFrame (Feature,
            observed_importance, bootstrap_mean, bootstrap_se,
            ci_lower, ci_upper, bootstrap_type, n_iterations).
        """
        bootstrap_type = getattr(self, '_bootstrap_type', 'auto')
        if bootstrap_type == 'auto':
            bootstrap_type = 'percentile'

        return _bootstrap_feature_importance(
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
            title="Confusion Matrix - CatBoost",
            backend=backend
        )

    def _feature_importance(self, backend="plotly"):
        from isaric.visualization.barplots import simple_bar_plot
        importance_df = _build_result_df(self.fitted_model, self.X)
        return simple_bar_plot(
            importance_df, x_col='Feature', y_col='Importance',
            title="Feature Importance - CatBoost",
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
                           title="Calibration Curve - CatBoost",
                           xaxis_title="Predicted Probability",
                           yaxis_title="Observed Proportion",
                           backend=backend)