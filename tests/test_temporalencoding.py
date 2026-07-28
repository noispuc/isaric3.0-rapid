import numpy as np
import pandas as pd
import pytest

from isaric.preprocessing.temporalencoding import CyclicalFeatureEncoder


def _distance(a, b):
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def test_sin_cos_unit_circle_property():
    encoder = CyclicalFeatureEncoder(period=12, source="raw")
    X = pd.DataFrame({"month": [1, 3, 6, 9, 12]})
    out = encoder.fit_transform(X)

    norms = out[:, 0] ** 2 + out[:, 1] ** 2
    np.testing.assert_allclose(norms, 1.0, atol=1e-10)


def test_periodicity_month_1_equals_month_13():
    encoder = CyclicalFeatureEncoder(period=12, source="raw")
    out = encoder.fit_transform(pd.DataFrame({"month": [1, 13]}))

    np.testing.assert_allclose(out[0], out[1], atol=1e-10)


def test_december_january_are_close_but_june_is_far():
    encoder = CyclicalFeatureEncoder(period=12, source="raw")
    out = encoder.fit_transform(pd.DataFrame({"month": [1, 12, 6]}))
    jan, dec, jun = out[0], out[1], out[2]

    assert _distance(jan, dec) < _distance(jan, jun)


def test_epiweek_source_parses_aaaass_format():
    encoder = CyclicalFeatureEncoder(period=52, source="epiweek")
    X = pd.DataFrame({"sem_pri": ["201901", "201952"]})
    out = encoder.fit_transform(X)

    expected_week1 = np.array([np.sin(2 * np.pi * 1 / 52), np.cos(2 * np.pi * 1 / 52)])
    expected_week52 = np.array([np.sin(2 * np.pi * 52 / 52), np.cos(2 * np.pi * 52 / 52)])
    np.testing.assert_allclose(out[0], expected_week1, atol=1e-10)
    np.testing.assert_allclose(out[1], expected_week52, atol=1e-10)


def test_date_source_extracts_month():
    encoder = CyclicalFeatureEncoder(period=12, source="date")
    X = pd.DataFrame({"dt_sin_pri": pd.to_datetime(["2019-01-15", "2019-12-20"])})
    out = encoder.fit_transform(X)

    expected_jan = np.array([np.sin(2 * np.pi * 1 / 12), np.cos(2 * np.pi * 1 / 12)])
    expected_dec = np.array([np.sin(2 * np.pi * 12 / 12), np.cos(2 * np.pi * 12 / 12)])
    np.testing.assert_allclose(out[0], expected_jan, atol=1e-10)
    np.testing.assert_allclose(out[1], expected_dec, atol=1e-10)


def test_missing_values_propagate_as_nan():
    encoder = CyclicalFeatureEncoder(period=12, source="raw")
    out = encoder.fit_transform(pd.DataFrame({"month": [1, np.nan]}))

    assert not np.isnan(out[0]).any()
    assert np.isnan(out[1]).all()


def test_invalid_source_raises():
    encoder = CyclicalFeatureEncoder(period=12, source="invalid")
    with pytest.raises(ValueError):
        encoder.fit_transform(pd.DataFrame({"month": [1, 2]}))
