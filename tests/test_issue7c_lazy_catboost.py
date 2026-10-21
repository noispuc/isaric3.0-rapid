"""
Tests for Issue 7c: import isaric must not fail when catboost is absent.

Verifies:
- treebased.py can be imported without importing CatBoostClassifier at module level.
- create_catboost_model and CatBoost.create() still work when catboost IS installed.
- A clear ImportError is raised (not AttributeError or NameError) when catboost
  is unavailable (simulated by patching the import machinery).
"""

import sys
import importlib
import types
import numpy as np
import pandas as pd
import pytest


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
# CatBoostClassifier is NOT imported at module level
# ---------------------------------------------------------------------------

def test_catboostclassifier_not_at_module_level():
    """treebased module must not hold CatBoostClassifier as a module attribute."""
    from isaric.modeling import treebased
    assert not hasattr(treebased, "CatBoostClassifier"), (
        "CatBoostClassifier is still a module-level name in treebased.py — "
        "the lazy import fix was not applied."
    )


def test_import_isaric_succeeds_without_catboost(monkeypatch):
    """Simulate catboost absence: import isaric must not raise ImportError."""
    # Block catboost import
    monkeypatch.setitem(sys.modules, "catboost", None)
    # Reload treebased to re-run module-level code under the block
    import isaric.modeling.treebased as tb
    importlib.reload(tb)
    # Must not raise; catboost class simply inaccessible until create() is called


def test_create_catboost_raises_clear_error_when_absent(monkeypatch, small_df):
    """create_catboost_model must raise ImportError (not crash) when catboost absent."""
    monkeypatch.setitem(sys.modules, "catboost", None)
    from isaric.modeling import treebased
    importlib.reload(treebased)
    with pytest.raises(ImportError, match="catboost"):
        treebased.create_catboost_model(small_df, "outcome", FEATS)


# ---------------------------------------------------------------------------
# CatBoost still works when catboost IS installed
# ---------------------------------------------------------------------------

def test_catboost_create_works(small_df):
    pytest.importorskip("catboost", reason="catboost not installed")
    from isaric.modeling.treebased import create_catboost_model
    model, X, y = create_catboost_model(small_df, "outcome", FEATS)
    assert X.shape == (100, 2)
    assert len(y) == 100


def test_rapid_catboost_create_works(small_df):
    pytest.importorskip("catboost", reason="catboost not installed")
    import isaric as isa
    instance = isa.RAPID.create(
        data=small_df,
        model="catboost",
        dependent_var="outcome",
        independent_vars=FEATS,
    )
    assert instance is not None
