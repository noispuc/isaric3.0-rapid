"""
Tests for Issue 8b: recalibration(method='isotonic') must use
sklearn.isotonic.IsotonicRegression instead of defaulting to Platt scaling.
"""

import numpy as np
import pytest
from isaric.validation.external import recalibration


@pytest.fixture
def calibration_data():
    rng = np.random.default_rng(7)
    n = 300
    y_true = rng.integers(0, 2, n)
    y_prob = np.clip(rng.beta(2, 5, n), 0.01, 0.99)
    return y_true.astype(float), y_prob


def test_platt_returns_slope_intercept(calibration_data):
    y_true, y_prob = calibration_data
    result = recalibration(y_true, y_prob, method="platt")
    assert "slope" in result
    assert "intercept" in result
    assert np.isfinite(result["slope"])
    assert np.isfinite(result["intercept"])


def test_isotonic_returns_slope_intercept(calibration_data):
    """isotonic branch must return dict with finite slope and intercept."""
    y_true, y_prob = calibration_data
    result = recalibration(y_true, y_prob, method="isotonic")
    assert "slope" in result
    assert "intercept" in result
    assert np.isfinite(result["slope"])
    assert np.isfinite(result["intercept"])


def test_isotonic_differs_from_platt(calibration_data):
    """Isotonic and Platt should produce different slopes for miscalibrated data."""
    y_true, y_prob = calibration_data
    platt = recalibration(y_true, y_prob, method="platt")
    iso = recalibration(y_true, y_prob, method="isotonic")
    # They can only be equal by coincidence; for generic data they differ
    assert not (
        abs(platt["slope"] - iso["slope"]) < 1e-9
        and abs(platt["intercept"] - iso["intercept"]) < 1e-9
    ), "isotonic and platt should produce different results"


def test_isotonic_slope_near_one_on_well_calibrated(calibration_data):
    """After isotonic calibration of its own predictions, slope should be near 1."""
    y_true, y_prob = calibration_data
    from sklearn.isotonic import IsotonicRegression
    ir = IsotonicRegression(out_of_bounds='clip')
    calibrated = ir.fit_transform(y_prob, y_true)
    result = recalibration(y_true, calibrated, method="isotonic")
    # Slope from perfectly-isotonically-calibrated probs ≈ 1
    assert 0.5 < result["slope"] < 2.0


def test_invalid_method_raises(calibration_data):
    y_true, y_prob = calibration_data
    with pytest.raises(ValueError, match="method must be"):
        recalibration(y_true, y_prob, method="sigmoid")
