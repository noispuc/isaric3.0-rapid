"""
test_mice_imputer.py
====================
unittest test suite for MICEImputer.

Coverage
--------
- Fabricated datasets: continuous-only, categorical-only, mixed
- Missing value patterns: MCAR (random), MAR (structured), single column,
  high-missingness (>50 %), no missing values
- All enumerated parameter values:
    initial_strategy  : mean | median | most_frequent | constant
    tie_strategy      : highest_probability | first | nan | force_class
- Boolean parameters: force_binary, show_results
- Scalar parameters: n, max_iter, random_state, cutoff, prefix_sep
- Dict parameters: cutoff_map, tie_force (str and dict forms)
- fit() arguments: interest_vars (subset, full, None), show_results
- get_results() contract and pre-fit guard
- Output shape and type contracts
- Reproducibility (same random_state → identical results)
- Statistical analysis table (stats_df) content
- ZIP file creation

Run
---
    python -m unittest test_mice_imputer.py -v
"""

import os
import re
import sys
import zipfile
import tempfile
import unittest
from io import StringIO
from unittest.mock import patch

import numpy as np
import pandas as pd

from mice import MICEImputer


# ======================================================================
# Shared dataset builders
# ======================================================================

def make_continuous_df():
    """100-row dataset with three continuous columns, ~20 % MCAR missings."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        'age':    rng.integers(20, 80, size=100).astype(float),
        'weight': rng.normal(70, 15, size=100),
        'score':  rng.normal(50, 10, size=100),
    })
    df.loc[rng.choice(100, size=20, replace=False), 'age']   = np.nan
    df.loc[rng.choice(100, size=20, replace=False), 'score'] = np.nan
    return df


def make_categorical_df():
    """100-row dataset with two multi-class and one binary categorical column."""
    rng = np.random.default_rng(1)
    df = pd.DataFrame({
        'city':      rng.choice(['SP', 'RJ', 'BH', 'POA'], size=100).tolist(),
        'plan':      rng.choice(['basic', 'standard', 'premium'], size=100).tolist(),
        'is_active': rng.choice([0, 1], size=100).astype(float),
    })
    df.loc[rng.choice(100, size=15, replace=False), 'city'] = np.nan
    df.loc[rng.choice(100, size=10, replace=False), 'plan'] = np.nan
    return df


def make_mixed_df():
    """100-row mixed dataset: continuous + multi-class categorical + binary."""
    rng = np.random.default_rng(2)
    df = pd.DataFrame({
        'age':    rng.integers(18, 90, size=100).astype(float),
        'bmi':    rng.normal(25, 5, size=100),
        'sex':    rng.choice(['M', 'F'], size=100).tolist(),
        'region': rng.choice(['North', 'South', 'East', 'West'], size=100).tolist(),
        'died':   rng.choice([0, 1], size=100).astype(float),
    })
    for col, n_miss in [('age', 18), ('bmi', 12), ('sex', 8), ('region', 10)]:
        df.loc[rng.choice(100, size=n_miss, replace=False), col] = np.nan
    return df


def make_no_missing_df():
    """Dataset with no missing values at all."""
    rng = np.random.default_rng(3)
    return pd.DataFrame({
        'x': rng.normal(0, 1, size=50),
        'y': rng.integers(0, 5, size=50).astype(float),
    })


def make_high_missing_df():
    """Dataset where one column has >50 % missing values."""
    rng = np.random.default_rng(4)
    df = pd.DataFrame({
        'a': rng.normal(0, 1, size=80),
        'b': rng.normal(5, 2, size=80),
    })
    df.loc[rng.choice(80, size=55, replace=False), 'a'] = np.nan
    return df


def default_fit(df, **kwargs):
    """Instantiate with small n for speed and fit without printing."""
    imp = MICEImputer(n=2, max_iter=3, random_state=42, **kwargs)
    imp.fit(df, show_results=False)
    return imp


# ======================================================================
# 1. Basic output contracts
# ======================================================================

class TestOutputContracts(unittest.TestCase):

    def setUp(self):
        self.df = make_continuous_df()

    def test_fit_returns_self(self):
        imp = MICEImputer(n=2, max_iter=3, random_state=42)
        result = imp.fit(self.df, show_results=False)
        self.assertIs(result, imp)

    def test_get_results_returns_three_items(self):
        imp = default_fit(self.df)
        out = imp.get_results()
        self.assertEqual(len(out), 3)

    def test_imputed_datasets_count(self):
        imp = MICEImputer(n=4, max_iter=3, random_state=42)
        imp.fit(self.df, show_results=False)
        datasets, _, _ = imp.get_results()
        self.assertEqual(len(datasets), 4)

    def test_imputed_datasets_are_dataframes(self):
        imp = default_fit(self.df)
        datasets, pooled, stats = imp.get_results()
        for ds in datasets:
            self.assertIsInstance(ds, pd.DataFrame)
        self.assertIsInstance(pooled, pd.DataFrame)
        self.assertIsInstance(stats, pd.DataFrame)

    def test_output_shape_matches_input(self):
        df = make_mixed_df()
        imp = default_fit(df)
        datasets, pooled, _ = imp.get_results()
        for ds in datasets:
            self.assertEqual(ds.shape, df.shape)
        self.assertEqual(pooled.shape, df.shape)

    def test_output_columns_match_input(self):
        df = make_mixed_df()
        imp = default_fit(df)
        datasets, pooled, _ = imp.get_results()
        for ds in datasets:
            self.assertEqual(list(ds.columns), list(df.columns))
        self.assertEqual(list(pooled.columns), list(df.columns))

    def test_no_missing_in_imputed_datasets_continuous(self):
        imp = default_fit(self.df)
        datasets, pooled, _ = imp.get_results()
        for ds in datasets:
            self.assertEqual(ds.isnull().sum().sum(), 0)
        self.assertEqual(pooled.isnull().sum().sum(), 0)

    def test_no_missing_in_imputed_datasets_categorical(self):
        df = make_categorical_df()
        imp = default_fit(df)
        datasets, _, _ = imp.get_results()
        for ds in datasets:
            self.assertEqual(ds[['city', 'plan']].isnull().sum().sum(), 0)

    def test_non_missing_values_unchanged(self):
        """Values that were not missing must remain identical after imputation."""
        imp = default_fit(self.df)
        datasets, _, _ = imp.get_results()
        mask = ~self.df['weight'].isna()
        for ds in datasets:
            pd.testing.assert_series_equal(
                ds.loc[mask, 'weight'].reset_index(drop=True),
                self.df.loc[mask, 'weight'].reset_index(drop=True),
                check_names=False,
            )


# ======================================================================
# 2. Dataset variants
# ======================================================================

class TestDatasetVariants(unittest.TestCase):

    def test_continuous_only(self):
        df = make_continuous_df()
        imp = default_fit(df)
        _, pooled, stats = imp.get_results()
        self.assertEqual(pooled.shape, df.shape)
        self.assertTrue(set(stats['VARIABLE']).issubset(set(df.columns)))

    def test_categorical_only(self):
        df = make_categorical_df()
        imp = default_fit(df)
        datasets, pooled, _ = imp.get_results()
        self.assertEqual(pooled.shape, df.shape)
        valid_cities = {'SP', 'RJ', 'BH', 'POA'}
        for ds in datasets:
            actual = set(ds['city'].dropna().unique())
            self.assertTrue(
                actual.issubset(valid_cities),
                f"Unexpected city values: {actual - valid_cities}",
            )

    def test_mixed_dataset(self):
        df = make_mixed_df()
        imp = default_fit(df)
        _, pooled, _ = imp.get_results()
        self.assertEqual(pooled.shape, df.shape)

    def test_no_missing_dataset(self):
        df = make_no_missing_df()
        imp = default_fit(df)
        _, _, stats = imp.get_results()
        self.assertEqual(len(stats), 0, "stats_df should be empty when there are no missings")

    def test_high_missing_column(self):
        df = make_high_missing_df()
        imp = default_fit(df)
        _, pooled, _ = imp.get_results()
        self.assertEqual(pooled.isnull().sum().sum(), 0)

    def test_single_missing_column(self):
        rng = np.random.default_rng(10)
        df = pd.DataFrame({
            'predictor': rng.normal(0, 1, size=60),
            'target':    rng.normal(5, 2, size=60),
        })
        df.loc[rng.choice(60, size=10, replace=False), 'target'] = np.nan
        imp = default_fit(df)
        _, pooled, stats = imp.get_results()
        self.assertEqual(pooled['target'].isnull().sum(), 0)
        self.assertEqual(list(stats['VARIABLE']), ['target'])

    def test_multiclass_categorical_reconstruction(self):
        rng = np.random.default_rng(11)
        classes = ['cat', 'dog', 'bird', 'fish']
        df = pd.DataFrame({
            'num':    rng.normal(0, 1, size=80),
            'animal': rng.choice(classes, size=80).tolist(),
        })
        df.loc[rng.choice(80, size=15, replace=False), 'animal'] = np.nan
        imp = default_fit(df)
        datasets, _, _ = imp.get_results()
        for ds in datasets:
            self.assertTrue(set(ds['animal'].dropna().unique()).issubset(set(classes)))


# ======================================================================
# 3. Parameter: n
# ======================================================================

class TestParamN(unittest.TestCase):

    def setUp(self):
        self.df = make_continuous_df()

    def _assert_n_datasets(self, n):
        imp = MICEImputer(n=n, max_iter=3, random_state=42)
        imp.fit(self.df, show_results=False)
        datasets, _, _ = imp.get_results()
        self.assertEqual(len(datasets), n)

    def test_n_1(self): self._assert_n_datasets(1)
    def test_n_3(self): self._assert_n_datasets(3)
    def test_n_7(self): self._assert_n_datasets(7)


# ======================================================================
# 4. Parameter: initial_strategy
# ======================================================================

class TestParamInitialStrategy(unittest.TestCase):

    def setUp(self):
        self.df = make_continuous_df()

    def _assert_no_missing_after_fit(self, strategy, extra=None):
        imp = MICEImputer(
            n=2, max_iter=3, random_state=42,
            initial_strategy=strategy,
            imputer_kwargs_extra=extra or {},
        )
        imp.fit(self.df, show_results=False)
        datasets, _, _ = imp.get_results()
        self.assertEqual(datasets[0].isnull().sum().sum(), 0)

    def test_mean(self):          self._assert_no_missing_after_fit('mean')
    def test_median(self):        self._assert_no_missing_after_fit('median')
    def test_most_frequent(self): self._assert_no_missing_after_fit('most_frequent')
    def test_constant(self):      self._assert_no_missing_after_fit('constant', extra={'fill_value': 0})


# ======================================================================
# 5. Parameter: random_state & reproducibility
# ======================================================================

class TestParamRandomState(unittest.TestCase):

    def setUp(self):
        self.df = make_continuous_df()

    def test_same_seed_produces_identical_results(self):
        imp1 = MICEImputer(n=2, max_iter=3, random_state=99)
        imp1.fit(self.df, show_results=False)
        imp2 = MICEImputer(n=2, max_iter=3, random_state=99)
        imp2.fit(self.df, show_results=False)
        pd.testing.assert_frame_equal(imp1.pooled_df_, imp2.pooled_df_)

    def test_different_seeds_produce_different_seed_lists(self):
        imp1 = MICEImputer(n=3, random_state=1)
        imp1._seeds = imp1._generate_seeds()
        imp2 = MICEImputer(n=3, random_state=2)
        imp2._seeds = imp2._generate_seeds()
        self.assertNotEqual(imp1._seeds, imp2._seeds)

    def test_random_state_none_runs_without_error(self):
        imp = MICEImputer(n=2, max_iter=3, random_state=None)
        imp.fit(self.df, show_results=False)
        self.assertIsNotNone(imp.pooled_df_)


# ======================================================================
# 6. Parameter: max_iter
# ======================================================================

class TestParamMaxIter(unittest.TestCase):

    def setUp(self):
        self.df = make_continuous_df()

    def _assert_complete_after_fit(self, iters):
        imp = MICEImputer(n=2, max_iter=iters, random_state=42)
        imp.fit(self.df, show_results=False)
        _, pooled, _ = imp.get_results()
        self.assertEqual(pooled.isnull().sum().sum(), 0)

    def test_max_iter_1(self):  self._assert_complete_after_fit(1)
    def test_max_iter_5(self):  self._assert_complete_after_fit(5)
    def test_max_iter_15(self): self._assert_complete_after_fit(15)


# ======================================================================
# 7. Parameter: prefix_sep
# ======================================================================

class TestParamPrefixSep(unittest.TestCase):

    def setUp(self):
        self.df = make_categorical_df()

    def _assert_shape_preserved(self, sep):
        imp = MICEImputer(n=2, max_iter=3, random_state=42, prefix_sep=sep)
        imp.fit(self.df, show_results=False)
        datasets, _, _ = imp.get_results()
        self.assertEqual(datasets[0].shape, self.df.shape)

    def test_sep_pipe(self):          self._assert_shape_preserved('|')
    def test_sep_double_under(self):  self._assert_shape_preserved('__')
    def test_sep_dash(self):          self._assert_shape_preserved('-')
    def test_sep_hash(self):          self._assert_shape_preserved('#')


# ======================================================================
# 8. Parameter: force_binary
# ======================================================================

class TestParamForceBinary(unittest.TestCase):

    def setUp(self):
        self.df = make_categorical_df()

    def test_force_binary_false_default(self):
        imp = MICEImputer(n=2, max_iter=3, random_state=42, force_binary=False)
        imp.fit(self.df, show_results=False)
        datasets, _, _ = imp.get_results()
        for ds in datasets:
            vals = set(ds['is_active'].dropna().unique())
            self.assertTrue(vals.issubset({0, 1, 0.0, 1.0}))

    def test_force_binary_true_binary_column(self):
        imp = MICEImputer(n=2, max_iter=3, random_state=42, force_binary=True)
        imp.fit(self.df, show_results=False)
        datasets, _, _ = imp.get_results()
        for ds in datasets:
            vals = set(ds['is_active'].dropna().astype(int).unique())
            self.assertTrue(vals.issubset({0, 1}))

    def test_force_binary_true_multiclass(self):
        df = make_mixed_df()
        valid_regions = {'North', 'South', 'East', 'West'}
        imp = MICEImputer(n=2, max_iter=3, random_state=42, force_binary=True)
        imp.fit(df, show_results=False)
        datasets, _, _ = imp.get_results()
        for ds in datasets:
            actual = set(ds['region'].dropna().unique())
            self.assertTrue(actual.issubset(valid_regions))


# ======================================================================
# 9. Parameter: cutoff
# ======================================================================

class TestParamCutoff(unittest.TestCase):

    def setUp(self):
        self.df = make_mixed_df()

    def _assert_runs_clean(self, cutoff):
        imp = MICEImputer(
            n=2, max_iter=3, random_state=42,
            force_binary=True, cutoff=cutoff,
        )
        imp.fit(self.df, show_results=False)
        datasets, _, _ = imp.get_results()
        self.assertEqual(len(datasets), 2)
        self.assertEqual(datasets[0].isnull().sum().sum(), 0)

    def test_cutoff_03(self): self._assert_runs_clean(0.3)
    def test_cutoff_05(self): self._assert_runs_clean(0.5)
    def test_cutoff_07(self): self._assert_runs_clean(0.7)


# ======================================================================
# 10. Parameter: cutoff_map
# ======================================================================

class TestParamCutoffMap(unittest.TestCase):

    def setUp(self):
        self.df = make_mixed_df()

    def test_cutoff_map_per_variable(self):
        imp = MICEImputer(
            n=2, max_iter=3, random_state=42,
            force_binary=True,
            cutoff=0.5,
            cutoff_map={'sex': 0.3, 'region': 0.6},
        )
        imp.fit(self.df, show_results=False)
        datasets, _, _ = imp.get_results()
        valid_regions = {'North', 'South', 'East', 'West'}
        for ds in datasets:
            self.assertTrue(set(ds['region'].dropna().unique()).issubset(valid_regions))

    def test_cutoff_map_partial_override(self):
        """Variables not in cutoff_map fall back to global cutoff."""
        imp = MICEImputer(
            n=2, max_iter=3, random_state=42,
            force_binary=True,
            cutoff=0.5,
            cutoff_map={'sex': 0.4},
        )
        imp.fit(self.df, show_results=False)
        _, pooled, _ = imp.get_results()
        self.assertEqual(pooled.isnull().sum().sum(), 0)


# ======================================================================
# 11. Parameter: tie_strategy
# ======================================================================

class TestParamTieStrategy(unittest.TestCase):

    def setUp(self):
        self.df = make_mixed_df()

    def _assert_runs(self, strategy, **kwargs):
        imp = MICEImputer(
            n=2, max_iter=3, random_state=42,
            force_binary=True, tie_strategy=strategy,
            **kwargs,
        )
        imp.fit(self.df, show_results=False)
        datasets, _, _ = imp.get_results()
        self.assertEqual(len(datasets), 2)

    def test_highest_probability(self):
        self._assert_runs('highest_probability')

    def test_first(self):
        self._assert_runs('first')

    def test_nan(self):
        self._assert_runs('nan')

    def test_force_class_with_string_tie_force(self):
        self._assert_runs('force_class', tie_force='North')

    def test_force_class_with_dict_tie_force(self):
        self._assert_runs('force_class', tie_force={'region': 'South', 'sex': 'F'})

    def test_tie_strategy_nan_does_not_raise(self):
        """tie_strategy='nan' is allowed to leave NaN where ties are unresolvable."""
        imp = MICEImputer(
            n=2, max_iter=3, random_state=42,
            force_binary=True, tie_strategy='nan',
        )
        imp.fit(self.df, show_results=False)
        self.assertIsNotNone(imp.pooled_df_)


# ======================================================================
# 12. Parameter: tie_force
# ======================================================================

class TestParamTieForce(unittest.TestCase):

    def setUp(self):
        self.df = make_mixed_df()

    def test_tie_force_string(self):
        imp = MICEImputer(
            n=2, max_iter=3, random_state=42,
            force_binary=True,
            tie_strategy='force_class',
            tie_force='East',
        )
        imp.fit(self.df, show_results=False)
        self.assertIsNotNone(imp.pooled_df_)

    def test_tie_force_dict(self):
        imp = MICEImputer(
            n=2, max_iter=3, random_state=42,
            force_binary=True,
            tie_strategy='force_class',
            tie_force={'region': 'West'},
        )
        imp.fit(self.df, show_results=False)
        self.assertIsNotNone(imp.pooled_df_)


# ======================================================================
# 13. Parameter: imputer_kwargs_extra
# ======================================================================

class TestParamImputerKwargsExtra(unittest.TestCase):

    def setUp(self):
        self.df = make_continuous_df()

    def test_valid_extra_kwarg_forwarded(self):
        imp = MICEImputer(
            n=2, max_iter=3, random_state=42,
            imputer_kwargs_extra={'n_nearest_features': 2},
        )
        imp.fit(self.df, show_results=False)
        datasets, _, _ = imp.get_results()
        self.assertEqual(datasets[0].isnull().sum().sum(), 0)

    def test_none_imputer_kwargs_extra(self):
        imp = MICEImputer(n=2, max_iter=3, random_state=42, imputer_kwargs_extra=None)
        imp.fit(self.df, show_results=False)
        self.assertIsNotNone(imp.pooled_df_)


# ======================================================================
# 14. fit() argument: interest_vars
# ======================================================================

class TestFitInterestVars(unittest.TestCase):

    def setUp(self):
        self.df = make_mixed_df()

    def test_interest_vars_none_uses_all_columns(self):
        imp = default_fit(self.df)
        _, _, stats = imp.get_results()
        missing_cols = [c for c in self.df.columns if self.df[c].isnull().any()]
        self.assertTrue(set(stats['VARIABLE']).issubset(set(missing_cols)))

    def test_interest_vars_subset(self):
        imp = MICEImputer(n=2, max_iter=3, random_state=42)
        imp.fit(self.df, interest_vars=['age', 'sex'], show_results=False)
        _, _, stats = imp.get_results()
        self.assertTrue(set(stats['VARIABLE']).issubset({'age', 'sex'}))

    def test_interest_vars_no_missing_column_excluded_from_stats(self):
        """'died' has no missing values and must not appear in stats."""
        imp = MICEImputer(n=2, max_iter=3, random_state=42)
        imp.fit(self.df, interest_vars=['died', 'age'], show_results=False)
        _, _, stats = imp.get_results()
        self.assertNotIn('died', stats['VARIABLE'].values)

    def test_interest_vars_invalid_raises_value_error(self):
        imp = MICEImputer(n=2, max_iter=3, random_state=42)
        with self.assertRaises(ValueError):
            imp.fit(self.df, interest_vars=['nonexistent_col'], show_results=False)


# ======================================================================
# 15. fit() argument: show_results
# ======================================================================

class TestFitShowResults(unittest.TestCase):

    def setUp(self):
        self.df = make_continuous_df()

    def test_show_results_false_suppresses_stats_output(self):
        imp = MICEImputer(n=2, max_iter=3, random_state=42)
        with patch('sys.stdout', new_callable=StringIO) as mock_out:
            imp.fit(self.df, show_results=False)
            output = mock_out.getvalue()
        self.assertNotIn('Statistical analysis', output)

    def test_show_results_true_prints_stats_output(self):
        imp = MICEImputer(n=2, max_iter=3, random_state=42)
        with patch('sys.stdout', new_callable=StringIO) as mock_out:
            imp.fit(self.df, show_results=True)
            output = mock_out.getvalue()
        self.assertIn('Statistical analysis', output)


# ======================================================================
# 16. get_results() pre-fit guard
# ======================================================================

class TestGetResultsGuard(unittest.TestCase):

    def test_get_results_before_fit_raises_runtime_error(self):
        imp = MICEImputer()
        with self.assertRaises(RuntimeError):
            imp.get_results()

    def test_get_results_after_fit_does_not_raise(self):
        imp = default_fit(make_continuous_df())
        try:
            imp.get_results()
        except RuntimeError:
            self.fail("get_results() raised RuntimeError unexpectedly after fit()")


# ======================================================================
# 17. stats_df content
# ======================================================================

class TestStatsDf(unittest.TestCase):

    def setUp(self):
        self.df = make_continuous_df()
        self.imp = default_fit(self.df)
        _, _, self.stats = self.imp.get_results()

    def test_required_columns_present(self):
        expected = {
            'VARIABLE', 'MISSINGS', 'BEFORE IMPUTATION',
            'AFTER IMPUTATION', 'P-VALUE', 'IMPUTATION VARIANCE',
        }
        self.assertTrue(expected.issubset(set(self.stats.columns)))

    def test_only_missing_variables_appear(self):
        """'weight' has no missings and must not appear in stats."""
        self.assertNotIn('weight', self.stats['VARIABLE'].values)

    def test_missings_column_format(self):
        pattern = re.compile(r'^\d+ \(\d+\.\d{2}%\)$')
        for val in self.stats['MISSINGS']:
            self.assertRegex(val, pattern)

    def test_p_value_is_float_or_nan(self):
        for p in self.stats['P-VALUE']:
            self.assertTrue(
                isinstance(p, float) or np.isnan(p),
                f"Unexpected P-VALUE type: {type(p)}",
            )

    def test_imputation_variance_nonnegative(self):
        for v in self.stats['IMPUTATION VARIANCE']:
            if not np.isnan(v):
                self.assertGreaterEqual(v, 0, f"Negative imputation variance: {v}")

    def test_stats_empty_when_no_missings(self):
        imp = default_fit(make_no_missing_df())
        _, _, stats = imp.get_results()
        self.assertTrue(stats.empty)


# ======================================================================
# 18. ZIP file creation
# ======================================================================

class TestZipFile(unittest.TestCase):

    def setUp(self):
        self.original_dir = os.getcwd()
        self.tmp_dir = tempfile.mkdtemp()
        os.chdir(self.tmp_dir)

    def tearDown(self):
        os.chdir(self.original_dir)

    def test_zip_file_created(self):
        default_fit(make_continuous_df())
        self.assertTrue(os.path.exists('imputed_datasets.zip'))

    def test_zip_contains_correct_number_of_files(self):
        n = 3
        imp = MICEImputer(n=n, max_iter=3, random_state=42)
        imp.fit(make_continuous_df(), show_results=False)
        with zipfile.ZipFile('imputed_datasets.zip') as zf:
            self.assertEqual(len(zf.namelist()), n)

    def test_zip_files_are_xlsx(self):
        default_fit(make_continuous_df())
        with zipfile.ZipFile('imputed_datasets.zip') as zf:
            for name in zf.namelist():
                self.assertTrue(
                    name.endswith('.xlsx'),
                    f"Unexpected file in zip: {name}",
                )


# ======================================================================
# Entry point
# ======================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)