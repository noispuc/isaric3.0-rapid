import pandas as pd
import pytest

from isaric.preprocessing.datasplitting import temporal_train_test_split


def _make_df():
    return pd.DataFrame({
        "ano_sin_pri": [2017, 2017, 2018, 2018, 2019, 2019, 2020, 2020],
        "feature": range(8),
    })


def test_no_temporal_leakage():
    df = _make_df()
    df_train, df_test = temporal_train_test_split(
        df, year_column="ano_sin_pri", train_end_year=2018, test_start_year=2019
    )

    assert df_train["ano_sin_pri"].max() <= 2018
    assert df_test["ano_sin_pri"].min() >= 2019
    assert set(df_train.index).isdisjoint(set(df_test.index))


def test_train_and_test_cover_all_rows_when_contiguous():
    df = _make_df()
    df_train, df_test = temporal_train_test_split(
        df, year_column="ano_sin_pri", train_end_year=2018, test_start_year=2019
    )

    assert len(df_train) + len(df_test) == len(df)


def test_gap_years_excluded_from_both_sets():
    df = _make_df()
    df_train, df_test = temporal_train_test_split(
        df, year_column="ano_sin_pri", train_end_year=2017, test_start_year=2020
    )

    assert 2018 not in df_train["ano_sin_pri"].values
    assert 2018 not in df_test["ano_sin_pri"].values
    assert 2019 not in df_train["ano_sin_pri"].values
    assert 2019 not in df_test["ano_sin_pri"].values


def test_raises_when_test_start_year_not_after_train_end_year():
    df = _make_df()
    with pytest.raises(ValueError):
        temporal_train_test_split(
            df, year_column="ano_sin_pri", train_end_year=2019, test_start_year=2019
        )
    with pytest.raises(ValueError):
        temporal_train_test_split(
            df, year_column="ano_sin_pri", train_end_year=2019, test_start_year=2018
        )


def test_raises_when_year_column_missing():
    df = _make_df().drop(columns=["ano_sin_pri"])
    with pytest.raises(ValueError):
        temporal_train_test_split(
            df, year_column="ano_sin_pri", train_end_year=2018, test_start_year=2019
        )


def test_raises_when_train_or_test_set_is_empty():
    df = _make_df()
    with pytest.raises(ValueError):
        temporal_train_test_split(
            df, year_column="ano_sin_pri", train_end_year=2010, test_start_year=2011
        )
    with pytest.raises(ValueError):
        temporal_train_test_split(
            df, year_column="ano_sin_pri", train_end_year=2019, test_start_year=2030
        )
