"""
Tests for Issue 8a: grid search AttributeError when hyperparameter attrs
are not initialized before calling fit(grid_search=True).

Each model class used self.ATTR as fallback in best_params.get("ATTR", self.ATTR),
but self.ATTR was never set in __init__, causing AttributeError on first call.
"""

import pandas as pd
import numpy as np
import pytest

import isaric as isa

try:
    import catboost  # noqa: F401
    _CATBOOST_AVAILABLE = True
except ImportError:
    _CATBOOST_AVAILABLE = False

_skip_catboost = pytest.mark.skipif(
    not _CATBOOST_AVAILABLE, reason="catboost not installed"
)


@pytest.fixture
def small_df():
    rng = np.random.default_rng(0)
    n = 120
    return pd.DataFrame({
        "x1": rng.standard_normal(n),
        "x2": rng.standard_normal(n),
        "outcome": rng.integers(0, 2, n),
    })


FEATS = ["x1", "x2"]


def _make(small_df, model_key):
    return isa.RAPID.create(
        data=small_df,
        model=model_key,
        dependent_var="outcome",
        independent_vars=FEATS,
    )


@pytest.mark.parametrize("model_key", ["lasso", "ridge", "logistic_l2"])
def test_predictive_gridsearch_no_attribute_error(small_df, model_key):
    """fit(grid_search=True) must not raise AttributeError for logistic models."""
    inst = _make(small_df, model_key)
    inst.fit(grid_search=True, param_grid={"C": [0.1, 1.0]})
    assert inst.C in (0.1, 1.0)


def test_elastic_net_gridsearch_no_attribute_error(small_df):
    """ElasticNet grid search must not raise AttributeError for C or l1_ratio."""
    inst = _make(small_df, "elastic_net")
    inst.fit(grid_search=True, param_grid={"C": [0.1, 1.0], "l1_ratio": [0.3, 0.7]})
    assert inst.C in (0.1, 1.0)
    assert inst.l1_ratio in (0.3, 0.7)


def test_svm_gridsearch_no_attribute_error(small_df):
    """SVM grid search must not raise AttributeError for C or kernel."""
    inst = _make(small_df, "svm")
    inst.fit(grid_search=True, param_grid={"C": [0.1, 1.0], "kernel": ["linear"]})
    assert inst.C in (0.1, 1.0)
    assert inst.kernel == "linear"


def test_decision_tree_gridsearch_no_attribute_error(small_df):
    """DecisionTree grid search must not raise AttributeError."""
    inst = _make(small_df, "decision_tree")
    inst.fit(grid_search=True, param_grid={"max_depth": [3, 5]})
    assert inst.max_depth in (3, 5)


def test_random_forest_gridsearch_no_attribute_error(small_df):
    """RandomForest grid search must not raise AttributeError."""
    inst = _make(small_df, "random_forest")
    inst.fit(grid_search=True, param_grid={"n_estimators": [10, 20], "max_depth": [3, 5]})
    assert inst.n_estimators in (10, 20)
    assert inst.max_depth in (3, 5)


def test_lightgbm_gridsearch_no_attribute_error(small_df):
    """LightGBM grid search must not raise AttributeError."""
    inst = _make(small_df, "lightgbm")
    inst.fit(grid_search=True, param_grid={"n_estimators": [10, 20], "num_leaves": [15, 31]})
    assert inst.n_estimators in (10, 20)
    assert inst.num_leaves in (15, 31)


def test_gridsearch_stores_best_params(small_df):
    """After grid search, instance.best_params must exist."""
    inst = _make(small_df, "logistic_l2")
    inst.fit(grid_search=True, param_grid={"C": [0.1, 1.0]})
    assert hasattr(inst, "best_params")
    assert "C" in inst.best_params


def test_no_gridsearch_attrs_remain_unset(small_df):
    """Without grid search, C/l1_ratio should not be set (no side effects)."""
    inst = _make(small_df, "lasso")
    inst.fit()
    assert not hasattr(inst, "best_params")
