"""
Tests for Issue 7b: summary(table_format="short") works for predictive models.

Verifies:
- 'Feature' and 'Coefficient' columns are shown for predictive result_df.
- Other models (logistic regression with 'Variable') still work.
- table_format="full" is unaffected.
"""

import io
import numpy as np
import pandas as pd
import pytest

import isaric as isa


@pytest.fixture
def small_df():
    rng = np.random.default_rng(42)
    n = 120
    return pd.DataFrame({
        "x1": rng.standard_normal(n),
        "x2": rng.standard_normal(n),
        "outcome": rng.integers(0, 2, n),
    })


FEATS = ["x1", "x2"]


# ---------------------------------------------------------------------------
# Predictive models: 'Feature' must appear in short summary
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model_key", ["lasso", "ridge", "elastic_net", "logistic_l2"])
def test_short_summary_shows_feature_col(small_df, model_key, capsys):
    instance = isa.RAPID.create(
        data=small_df,
        model=model_key,
        dependent_var="outcome",
        independent_vars=FEATS,
    )
    instance.fit()
    instance.summary(table_format="short")
    captured = capsys.readouterr().out
    assert "Feature" in captured, (
        f"'Feature' column missing from short summary for model '{model_key}'"
    )
    assert "Coefficient" in captured, (
        f"'Coefficient' column missing from short summary for model '{model_key}'"
    )


@pytest.mark.parametrize("model_key", ["lasso", "ridge", "elastic_net", "logistic_l2"])
def test_short_summary_contains_var_names(small_df, model_key, capsys):
    instance = isa.RAPID.create(
        data=small_df,
        model=model_key,
        dependent_var="outcome",
        independent_vars=FEATS,
    )
    instance.fit()
    instance.summary(table_format="short")
    captured = capsys.readouterr().out
    # Feature names themselves must appear
    assert "x1" in captured
    assert "x2" in captured


# ---------------------------------------------------------------------------
# full format is unaffected
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model_key", ["lasso", "ridge"])
def test_full_summary_still_works(small_df, model_key, capsys):
    instance = isa.RAPID.create(
        data=small_df,
        model=model_key,
        dependent_var="outcome",
        independent_vars=FEATS,
    )
    instance.fit()
    instance.summary(table_format="full")
    captured = capsys.readouterr().out
    assert "x1" in captured


# ---------------------------------------------------------------------------
# result_df shape sanity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model_key", ["lasso", "ridge", "elastic_net", "logistic_l2"])
def test_result_df_has_feature_and_coefficient(small_df, model_key):
    instance = isa.RAPID.create(
        data=small_df,
        model=model_key,
        dependent_var="outcome",
        independent_vars=FEATS,
    )
    instance.fit()
    instance.summary()
    assert "Feature" in instance.result_df.columns
    assert "Coefficient" in instance.result_df.columns
    assert len(instance.result_df) == len(FEATS)
