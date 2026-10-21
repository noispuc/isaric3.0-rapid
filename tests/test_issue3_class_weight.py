"""
Tests for Issue 3: class_weight configurable in all sklearn classifiers.

Verifies:
- Default is None (no class weighting)
- 'balanced' can be passed and is forwarded to the underlying model
- The parameter flows through RAPID.create() → factory → sklearn model
"""

import pandas as pd
import numpy as np
import pytest

from isaric.modeling.predictive import (
    create_lasso_model,
    create_ridge_model,
    create_elastic_net_model,
    create_svm_model,
    create_logistic_l2_model,
)
from isaric.modeling.treebased import (
    create_decision_tree_model,
    create_random_forest_model,
    create_lightgbm_model,
    create_catboost_model,
)
import isaric as isa


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_df():
    rng = np.random.default_rng(42)
    n = 100
    return pd.DataFrame({
        "x1": rng.standard_normal(n),
        "x2": rng.standard_normal(n),
        "outcome": rng.integers(0, 2, n),
    })


FEATS = ["x1", "x2"]


# ---------------------------------------------------------------------------
# Factory function defaults (class_weight=None)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("factory,kwargs", [
    (create_lasso_model,       {}),
    (create_ridge_model,       {}),
    (create_elastic_net_model, {}),
    (create_svm_model,         {}),
    (create_logistic_l2_model, {}),
])
def test_predictive_default_class_weight_is_none(small_df, factory, kwargs):
    model, _, _ = factory(small_df, "outcome", FEATS, **kwargs)
    assert model.class_weight is None


@pytest.mark.parametrize("factory,kwargs", [
    (create_decision_tree_model, {}),
    (create_random_forest_model, {}),
    (create_lightgbm_model,      {}),
])
def test_treebased_default_class_weight_is_none(small_df, factory, kwargs):
    model, _, _ = factory(small_df, "outcome", FEATS, **kwargs)
    assert model.class_weight is None


def test_catboost_default_auto_class_weights_is_none(small_df):
    model, _, _ = create_catboost_model(small_df, "outcome", FEATS)
    assert model.get_param("auto_class_weights") is None


# ---------------------------------------------------------------------------
# Passing 'balanced' is forwarded correctly
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("factory", [
    create_lasso_model,
    create_ridge_model,
    create_elastic_net_model,
    create_svm_model,
    create_logistic_l2_model,
])
def test_predictive_balanced_forwarded(small_df, factory):
    model, _, _ = factory(small_df, "outcome", FEATS, class_weight="balanced")
    assert model.class_weight == "balanced"


@pytest.mark.parametrize("factory", [
    create_decision_tree_model,
    create_random_forest_model,
    create_lightgbm_model,
])
def test_treebased_balanced_forwarded(small_df, factory):
    model, _, _ = factory(small_df, "outcome", FEATS, class_weight="balanced")
    assert model.class_weight == "balanced"


def test_catboost_balanced_forwarded(small_df):
    model, _, _ = create_catboost_model(
        small_df, "outcome", FEATS, auto_class_weights="Balanced"
    )
    assert model.get_param("auto_class_weights") == "Balanced"


# ---------------------------------------------------------------------------
# Flows through RAPID.create() public API
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model_key", [
    "lasso", "ridge", "elastic_net", "logistic_l2",
])
def test_rapid_create_passes_class_weight_predictive(small_df, model_key):
    instance = isa.RAPID.create(
        data=small_df,
        model=model_key,
        dependent_var="outcome",
        independent_vars=FEATS,
        class_weight="balanced",
    )
    assert instance._model.class_weight == "balanced"


@pytest.mark.parametrize("model_key", [
    "decision_tree", "random_forest", "lightgbm",
])
def test_rapid_create_passes_class_weight_treebased(small_df, model_key):
    instance = isa.RAPID.create(
        data=small_df,
        model=model_key,
        dependent_var="outcome",
        independent_vars=FEATS,
        class_weight="balanced",
    )
    assert instance._model.class_weight == "balanced"


@pytest.mark.parametrize("model_key", [
    "lasso", "ridge", "elastic_net", "logistic_l2",
    "decision_tree", "random_forest", "lightgbm",
])
def test_rapid_create_default_is_none(small_df, model_key):
    instance = isa.RAPID.create(
        data=small_df,
        model=model_key,
        dependent_var="outcome",
        independent_vars=FEATS,
    )
    assert instance._model.class_weight is None
