"""
Tests for Issue 5:
  - fit(balancing=) applies undersampling/SMOTE to X_train only before training.
  - fit(recalibrate=True) adds calibration_slope and calibration_intercept to metrics.
  - compute_calibration_metrics returns slope and intercept.
"""

import numpy as np
import pandas as pd
import pytest

import isaric as isa
from isaric.modelevaluation.metrics import compute_calibration_metrics


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def imbalanced_df():
    """Highly imbalanced dataset (5% positive) to test balancing."""
    rng = np.random.default_rng(0)
    n = 400
    x1 = rng.standard_normal(n)
    # ~5% positive
    y = (x1 > 1.65).astype(int)
    return pd.DataFrame({"x1": x1, "x2": rng.standard_normal(n), "outcome": y})


@pytest.fixture
def balanced_df():
    """Balanced dataset for recalibration tests."""
    rng = np.random.default_rng(7)
    n = 300
    return pd.DataFrame({
        "x1": rng.standard_normal(n),
        "x2": rng.standard_normal(n),
        "outcome": rng.integers(0, 2, n),
    })


FEATS = ["x1", "x2"]


# ---------------------------------------------------------------------------
# Balancing: undersample
# ---------------------------------------------------------------------------

def test_undersample_reduces_majority_class(imbalanced_df):
    """After undersampling, minority and majority class sizes are equal in X_train."""
    inst = isa.RAPID.create(
        data=imbalanced_df, model="lasso",
        dependent_var="outcome", independent_vars=FEATS,
        test_size=0.3, random_seed=42,
    )
    original_train_size = len(inst.y_train)
    inst.fit(balancing="undersample")

    # After fit, X_train should be smaller (only equal-sized classes)
    assert len(inst.X_train) < original_train_size
    assert inst.y_train.sum() == (len(inst.y_train) - inst.y_train.sum()), (
        "After undersampling, positive and negative counts should be equal"
    )


def test_undersample_does_not_touch_test_set(imbalanced_df):
    """Undersampling must NOT affect X_test/y_test."""
    inst = isa.RAPID.create(
        data=imbalanced_df, model="lasso",
        dependent_var="outcome", independent_vars=FEATS,
        test_size=0.3, random_seed=42,
    )
    original_test_size = len(inst.y_test)
    inst.fit(balancing="undersample")
    assert len(inst.y_test) == original_test_size


# ---------------------------------------------------------------------------
# Balancing: SMOTE
# ---------------------------------------------------------------------------

def test_smote_increases_minority_class(imbalanced_df):
    """After SMOTE, X_train is larger (minority class oversampled)."""
    inst = isa.RAPID.create(
        data=imbalanced_df, model="lasso",
        dependent_var="outcome", independent_vars=FEATS,
        test_size=0.3, random_seed=42,
    )
    original_train_size = len(inst.y_train)
    inst.fit(balancing="smote")

    assert len(inst.X_train) > original_train_size
    assert inst.y_train.sum() == (len(inst.y_train) - inst.y_train.sum()), (
        "After SMOTE, positive and negative counts should be equal"
    )


def test_smote_does_not_touch_test_set(imbalanced_df):
    """SMOTE must NOT affect X_test/y_test."""
    inst = isa.RAPID.create(
        data=imbalanced_df, model="lasso",
        dependent_var="outcome", independent_vars=FEATS,
        test_size=0.3, random_seed=42,
    )
    original_test_size = len(inst.y_test)
    inst.fit(balancing="smote")
    assert len(inst.y_test) == original_test_size


def test_invalid_balancing_raises(balanced_df):
    """Invalid balancing value raises ValueError."""
    inst = isa.RAPID.create(
        data=balanced_df, model="lasso",
        dependent_var="outcome", independent_vars=FEATS,
    )
    with pytest.raises(ValueError, match="balancing"):
        inst.fit(balancing="invalid_method")


def test_balancing_none_does_not_change_train_size(balanced_df):
    """balancing=None (default) leaves X_train unchanged."""
    inst = isa.RAPID.create(
        data=balanced_df, model="lasso",
        dependent_var="outcome", independent_vars=FEATS,
        test_size=0.3,
    )
    original_size = len(inst.y_train)
    inst.fit(balancing=None)
    assert len(inst.y_train) == original_size


# ---------------------------------------------------------------------------
# Recalibration
# ---------------------------------------------------------------------------

def test_recalibrate_adds_slope_and_intercept(balanced_df):
    """fit(recalibrate=True) adds calibration_slope and calibration_intercept to metrics."""
    inst = isa.RAPID.create(
        data=balanced_df, model="lasso",
        dependent_var="outcome", independent_vars=FEATS,
    )
    inst.fit(recalibrate=True)

    assert "calibration_slope"     in inst.metrics, "calibration_slope missing from metrics"
    assert "calibration_intercept" in inst.metrics, "calibration_intercept missing from metrics"


def test_recalibrate_false_does_not_add_keys(balanced_df):
    """fit(recalibrate=False) (default) must NOT add calibration_slope."""
    inst = isa.RAPID.create(
        data=balanced_df, model="lasso",
        dependent_var="outcome", independent_vars=FEATS,
    )
    inst.fit()  # default recalibrate=False
    assert "calibration_slope"     not in inst.metrics
    assert "calibration_intercept" not in inst.metrics


def test_recalibrate_slope_is_float(balanced_df):
    inst = isa.RAPID.create(
        data=balanced_df, model="lasso",
        dependent_var="outcome", independent_vars=FEATS,
    )
    inst.fit(recalibrate=True)
    assert isinstance(inst.metrics["calibration_slope"], float)
    assert isinstance(inst.metrics["calibration_intercept"], float)


def test_recalibrate_slope_from_test_set(balanced_df):
    """
    Recalibration must be computed on X_test/y_test.
    Independently reproduce it and compare.
    """
    from isaric.validation.external import recalibration as _recal

    inst = isa.RAPID.create(
        data=balanced_df, model="lasso",
        dependent_var="outcome", independent_vars=FEATS,
        test_size=0.3, random_seed=42,
    )
    inst.fit(recalibrate=True)

    y_prob_test = inst.fitted_model.predict_proba(inst.X_test)[:, 1]
    expected = _recal(inst.y_test.values, y_prob_test, method="platt")

    assert abs(inst.metrics["calibration_slope"]     - expected["slope"])     < 1e-9
    assert abs(inst.metrics["calibration_intercept"] - expected["intercept"]) < 1e-9


# ---------------------------------------------------------------------------
# compute_calibration_metrics: slope and intercept
# ---------------------------------------------------------------------------

def test_compute_calibration_metrics_has_slope_intercept():
    rng = np.random.default_rng(5)
    y_true = rng.integers(0, 2, 200)
    y_prob = np.clip(rng.uniform(0, 1, 200), 1e-6, 1 - 1e-6)
    result = compute_calibration_metrics(y_true, y_prob)

    assert "slope"     in result
    assert "intercept" in result
    assert isinstance(result["slope"],     float)
    assert isinstance(result["intercept"], float)


def test_compute_calibration_metrics_perfect_slope():
    """A model with perfectly calibrated probabilities has slope ≈ 1, intercept ≈ 0."""
    rng = np.random.default_rng(11)
    n = 2000
    y_prob = rng.uniform(0.05, 0.95, n)
    # Generate y_true such that P(y=1|x) = y_prob exactly
    y_true = (rng.uniform(size=n) < y_prob).astype(int)

    result = compute_calibration_metrics(y_true, y_prob)
    # Loose tolerance because of finite-sample noise
    assert abs(result["slope"]     - 1.0) < 0.3, f"slope={result['slope']:.3f}"
    assert abs(result["intercept"] - 0.0) < 0.3, f"intercept={result['intercept']:.3f}"


def test_compute_calibration_metrics_still_has_brier():
    """Existing keys must still be present after the slope/intercept addition."""
    rng = np.random.default_rng(3)
    y_true = rng.integers(0, 2, 100)
    y_prob = np.clip(rng.uniform(0, 1, 100), 1e-6, 1 - 1e-6)
    result = compute_calibration_metrics(y_true, y_prob)
    assert "brier_score"       in result
    assert "calibration_curve" in result
