"""
Tests for Issue 8c: calibration plots must use bootstrap confidence intervals
instead of fixed ±10% bands around fraction_positive.
"""

import numpy as np
import pytest
from isaric.modelevaluation.calibration import calibration_curve_with_ci


@pytest.fixture
def binary_data():
    rng = np.random.default_rng(99)
    n = 500
    y_true = rng.integers(0, 2, n)
    y_prob = np.clip(rng.beta(2, 5, n), 0.01, 0.99)
    return y_true, y_prob


def test_returns_required_keys(binary_data):
    y_true, y_prob = binary_data
    result = calibration_curve_with_ci(y_true, y_prob)
    for key in ("fraction_positive", "mean_predicted", "ci_lower", "ci_upper"):
        assert key in result, f"missing key: {key}"


def test_arrays_same_length(binary_data):
    y_true, y_prob = binary_data
    result = calibration_curve_with_ci(y_true, y_prob)
    n = len(result["fraction_positive"])
    assert len(result["mean_predicted"]) == n
    assert len(result["ci_lower"]) == n
    assert len(result["ci_upper"]) == n


def test_ci_bounds_valid(binary_data):
    """ci_lower <= fraction_positive <= ci_upper, all in [0, 1]."""
    y_true, y_prob = binary_data
    result = calibration_curve_with_ci(y_true, y_prob)
    fp = result["fraction_positive"]
    lo = result["ci_lower"]
    hi = result["ci_upper"]

    assert np.all(lo >= 0), "ci_lower must be >= 0"
    assert np.all(hi <= 1), "ci_upper must be <= 1"
    assert np.all(lo <= fp + 1e-6), "ci_lower must be <= fraction_positive"
    assert np.all(hi >= fp - 1e-6), "ci_upper must be >= fraction_positive"


def test_ci_width_is_not_fixed_percentage(binary_data):
    """Bootstrap CIs must NOT equal fraction_positive * {0.9, 1.1} for all bins."""
    y_true, y_prob = binary_data
    result = calibration_curve_with_ci(y_true, y_prob)
    fp = result["fraction_positive"]
    lo = result["ci_lower"]
    hi = result["ci_upper"]

    fixed_lower = fp * 0.9
    fixed_upper = fp * 1.1

    # At least one bin must differ from the old fixed pattern
    assert not np.allclose(lo, fixed_lower, atol=1e-6), \
        "ci_lower looks like fixed 10% band"
    assert not np.allclose(hi, fixed_upper, atol=1e-6), \
        "ci_upper looks like fixed 10% band"


def test_reproducible_with_same_seed(binary_data):
    y_true, y_prob = binary_data
    r1 = calibration_curve_with_ci(y_true, y_prob, random_state=0)
    r2 = calibration_curve_with_ci(y_true, y_prob, random_state=0)
    np.testing.assert_array_equal(r1["ci_lower"], r2["ci_lower"])
    np.testing.assert_array_equal(r1["ci_upper"], r2["ci_upper"])


def test_different_seed_gives_similar_but_not_identical(binary_data):
    y_true, y_prob = binary_data
    r1 = calibration_curve_with_ci(y_true, y_prob, random_state=1)
    r2 = calibration_curve_with_ci(y_true, y_prob, random_state=2)
    # Should be close but not identical (bootstrap randomness)
    assert not np.allclose(r1["ci_lower"], r2["ci_lower"]), \
        "different seeds should not produce identical CIs"
