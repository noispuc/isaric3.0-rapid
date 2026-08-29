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
    random_state: int = 42
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
        random_state=random_state,
        max_iter=1000,
        class_weight='balanced'
    )

    return model, X, y


def create_ridge_model(
    data: pd.DataFrame,
    dependent_var: str,
    independent_vars: List[str],
    C: float = 1.0,
    random_state: int = 42
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
        random_state=random_state,
        max_iter=1000,
        class_weight='balanced'
    )

    return model, X, y


def create_elastic_net_model(
    data: pd.DataFrame,
    dependent_var: str,
    independent_vars: List[str],
    C: float = 1.0,
    l1_ratio: float = 0.5,
    random_state: int = 42
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
        random_state=random_state,
        max_iter=1000,
        class_weight='balanced'
    )

    return model, X, y


def create_svm_model(
    data: pd.DataFrame,
    dependent_var: str,
    independent_vars: List[str],
    C: float = 1.0,
    kernel: str = "rbf",
    random_state: int = 42
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
        random_state=random_state,
        class_weight='balanced'
    )

    return model, X, y


def create_logistic_l2_model(
    data: pd.DataFrame,
    dependent_var: str,
    independent_vars: List[str],
    C: float = 1.0,
    random_state: int = 42
) -> Tuple[LogisticRegression, pd.DataFrame, pd.Series]:
    """
    Configure a Logistic Regression with L2 penalty (predictive focus).

    Returns:
        Tuple of (model, X, y).
    """
    return create_ridge_model(
        data, dependent_var, independent_vars, C=C, random_state=random_state
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
        }

    @classmethod
    def create(cls, data, model="lasso", dependent_var=None,
               independent_vars=None, **params):
        model_config, X, y = create_lasso_model(
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
            title="Confusion Matrix - LASSO"
        )

    def _coefficient_plot(self):
        from isaric.visualization.barplots import simple_bar_plot

        coef_df = _build_result_df(self.fitted_model, self.X)

        return simple_bar_plot(
            coef_df,
            x_col='Feature',
            y_col='Coefficient',
            title="Coefficients - LASSO"
        )


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
        }

    @classmethod
    def create(cls, data, model="ridge", dependent_var=None,
               independent_vars=None, **params):
        model_config, X, y = create_ridge_model(
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
            title="Confusion Matrix - Ridge"
        )


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
        }

    @classmethod
    def create(cls, data, model="elastic_net", dependent_var=None,
               independent_vars=None, **params):
        model_config, X, y = create_elastic_net_model(
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
            title="Confusion Matrix - Elastic Net"
        )

    def _coefficient_plot(self):
        from isaric.visualization.barplots import simple_bar_plot

        coef_df = _build_result_df(self.fitted_model, self.X)

        return simple_bar_plot(
            coef_df,
            x_col='Feature',
            y_col='Coefficient',
            title="Coefficients - Elastic Net"
        )


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
        }

    @classmethod
    def create(cls, data, model="svm", dependent_var=None,
               independent_vars=None, **params):
        model_config, X, y = create_svm_model(
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
            title="Confusion Matrix - SVM"
        )


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
        }

    @classmethod
    def create(cls, data, model="logistic_l2", dependent_var=None,
               independent_vars=None, **params):
        model_config, X, y = create_logistic_l2_model(
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
            title="Confusion Matrix - Logistic L2"
        )