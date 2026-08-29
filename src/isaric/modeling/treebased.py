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
    random_state: int = 42
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
        random_state=random_state,
        class_weight='balanced'
    )

    return model, X, y


def create_random_forest_model(
    data: pd.DataFrame,
    dependent_var: str,
    independent_vars: List[str],
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    random_state: int = 42
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
        random_state=random_state,
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
    random_state: int = 42
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
        random_state=random_state,
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
    random_state: int = 42
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
        random_state=random_state,
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
    random_state: int = 42
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
        random_state=random_state,
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
        from isaric.visualization.heatmaps import confusion_matrix_heatmap
        self.plots_map = {
            "confusion_matrix": self._confusion_matrix,
        }

    @classmethod
    def create(cls, data, model="decision_tree", dependent_var=None,
               independent_vars=None, **params):
        model_config, X, y = create_decision_tree_model(
            data, dependent_var, independent_vars, **params
        )
        return cls(model_config, X, y, dependent_var, independent_vars, **params)

    def _confusion_matrix(self):
        from isaric.visualization.heatmaps import confusion_matrix_heatmap
        from sklearn.metrics import confusion_matrix as cm

        y_pred = self.fitted_model.predict(self.X)
        cm_array = cm(self.y, y_pred)

        return confusion_matrix_heatmap(
            cm_array,
            class_names=['Negative', 'Positive'],
            title="Confusion Matrix - Decision Tree"
        )


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
        }

    @classmethod
    def create(cls, data, model="random_forest", dependent_var=None,
               independent_vars=None, **params):
        model_config, X, y = create_random_forest_model(
            data, dependent_var, independent_vars, **params
        )
        return cls(model_config, X, y, dependent_var, independent_vars, **params)

    def _confusion_matrix(self):
        from isaric.visualization.heatmaps import confusion_matrix_heatmap
        from sklearn.metrics import confusion_matrix as cm

        y_pred = self.fitted_model.predict(self.X)
        cm_array = cm(self.y, y_pred)

        return confusion_matrix_heatmap(
            cm_array,
            class_names=['Negative', 'Positive'],
            title="Confusion Matrix - Random Forest"
        )

    def _feature_importance(self):
        from isaric.visualization.barplots import simple_bar_plot

        importance_df = _build_result_df(self.fitted_model, self.X)

        return simple_bar_plot(
            importance_df,
            x_col='Feature',
            y_col='Importance',
            title="Feature Importance - Random Forest"
        )


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
        }

    @classmethod
    def create(cls, data, model="xgboost", dependent_var=None,
               independent_vars=None, **params):
        model_config, X, y = create_xgboost_model(
            data, dependent_var, independent_vars, **params
        )
        return cls(model_config, X, y, dependent_var, independent_vars, **params)

    def _confusion_matrix(self):
        from isaric.visualization.heatmaps import confusion_matrix_heatmap
        from sklearn.metrics import confusion_matrix as cm

        y_pred = self.fitted_model.predict(self.X)
        cm_array = cm(self.y, y_pred)

        return confusion_matrix_heatmap(
            cm_array,
            class_names=['Negative', 'Positive'],
            title="Confusion Matrix - XGBoost"
        )

    def _feature_importance(self):
        from isaric.visualization.barplots import simple_bar_plot

        importance_df = _build_result_df(self.fitted_model, self.X)

        return simple_bar_plot(
            importance_df,
            x_col='Feature',
            y_col='Importance',
            title="Feature Importance - XGBoost"
        )


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
        }

    @classmethod
    def create(cls, data, model="lightgbm", dependent_var=None,
               independent_vars=None, **params):
        model_config, X, y = create_lightgbm_model(
            data, dependent_var, independent_vars, **params
        )
        return cls(model_config, X, y, dependent_var, independent_vars, **params)

    def _confusion_matrix(self):
        from isaric.visualization.heatmaps import confusion_matrix_heatmap
        from sklearn.metrics import confusion_matrix as cm

        y_pred = self.fitted_model.predict(self.X)
        cm_array = cm(self.y, y_pred)

        return confusion_matrix_heatmap(
            cm_array,
            class_names=['Negative', 'Positive'],
            title="Confusion Matrix - LightGBM"
        )

    def _feature_importance(self):
        from isaric.visualization.barplots import simple_bar_plot

        importance_df = _build_result_df(self.fitted_model, self.X)

        return simple_bar_plot(
            importance_df,
            x_col='Feature',
            y_col='Importance',
            title="Feature Importance - LightGBM"
        )


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
        }

    @classmethod
    def create(cls, data, model="catboost", dependent_var=None,
               independent_vars=None, **params):
        model_config, X, y = create_catboost_model(
            data, dependent_var, independent_vars, **params
        )
        return cls(model_config, X, y, dependent_var, independent_vars, **params)

    def _confusion_matrix(self):
        from isaric.visualization.heatmaps import confusion_matrix_heatmap
        from sklearn.metrics import confusion_matrix as cm

        y_pred = self.fitted_model.predict(self.X)
        cm_array = cm(self.y, y_pred)

        return confusion_matrix_heatmap(
            cm_array,
            class_names=['Negative', 'Positive'],
            title="Confusion Matrix - CatBoost"
        )

    def _feature_importance(self):
        from isaric.visualization.barplots import simple_bar_plot

        importance_df = _build_result_df(self.fitted_model, self.X)

        return simple_bar_plot(
            importance_df,
            x_col='Feature',
            y_col='Importance',
            title="Feature Importance - CatBoost"
        )