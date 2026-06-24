import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, SimpleImputer
from collections import defaultdict
import os
import zipfile
from scipy.stats import mannwhitneyu, chi2_contingency


class MICEImputer:
    """
    MICE (Multiple Imputation by Chained Equations) imputer for datasets
    containing both numerical and categorical variables.

    Wraps sklearn's IterativeImputer to support categorical variables via
    one-hot encoding/decoding, runs n independent imputations with distinct
    seeds, pools results, and produces a comparative statistical analysis table.

    Parameters
    ----------
    n : int, default=5
        Number of independent imputations (imputed datasets) to produce.
    max_iter : int, default=10
        Maximum number of imputation rounds per dataset (passed to
        IterativeImputer).
    initial_strategy : {'mean', 'median', 'most_frequent', 'constant'},
        default='most_frequent'
        Strategy used to initialize missing values before iterating.
    random_state : int or None, default=None
        Base seed for reproducibility. n unique seeds are derived from it.
        If None, a random base seed is drawn.
    prefix_sep : str, default='!'
        Separator used when one-hot encoding categorical columns
        (e.g. 'sex!Male'). Must not appear in any column name.
    force_binary : bool, default=False
        If True, thresholds imputed dummy values to hard 0/1 using cutoff
        (global) or cutoff_map (per-variable).
    cutoff : float, default=0.5
        Global decision threshold for binarising dummies when
        force_binary=True. Values >= cutoff → 1, otherwise → 0.
    cutoff_map : dict[str, float] or None, default=None
        Per-variable threshold overrides when force_binary=True.
        Variables absent from this mapping fall back to cutoff.
    tie_strategy : {'highest_probability', 'first', 'nan', 'force_class'},
        default='highest_probability'
        Tie-breaking strategy for multi-class blocks when more than one
        dummy is >= cutoff in the same row.
    tie_force : str or dict[str, str] or None, default=None
        Class to assign when tie_strategy='force_class'. Accepts a string
        (same class for all variables) or a dict mapping variable → class.
    imputer_kwargs_extra : dict or None, default=None
        Additional keyword arguments forwarded to IterativeImputer.

    Attributes (available after fit)
    ----------------------------------
    imputed_datasets_ : list of pd.DataFrame
        The n individually imputed datasets.
    pooled_df_ : pd.DataFrame
        Pooled dataset (element-wise mean across n imputations, then
        categorical reconstruction).
    stats_df_ : pd.DataFrame
        Comparative statistical analysis table (only variables that had
        at least one missing value are included).
    """

    _RNG = np.random.default_rng()

    def __init__(
        self,
        n=5,
        max_iter=10,
        initial_strategy='most_frequent',
        random_state=None,
        prefix_sep='!',
        force_binary=False,
        cutoff=0.5,
        cutoff_map=None,
        tie_strategy='highest_probability',
        tie_force=None,
        imputer_kwargs_extra=None,
    ):
        self.n = n
        self.max_iter = max_iter
        self.initial_strategy = initial_strategy
        self.random_state = random_state
        self.prefix_sep = prefix_sep
        self.force_binary = force_binary
        self.cutoff = cutoff
        self.cutoff_map = cutoff_map if cutoff_map is not None else {}
        self.tie_strategy = tie_strategy
        self.tie_force = tie_force
        self.imputer_kwargs_extra = imputer_kwargs_extra if imputer_kwargs_extra is not None else {}

        self.imputed_datasets_ = None
        self.pooled_df_ = None
        self.stats_df_ = None

        self._df_orig = None
        self._df_for_impute = None
        self._cat_levels = None
        self._missing_mask = None
        self._stacked_arrays = None
        self._seeds = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, df, interest_vars=None, show_results=True):
        """
        Run MICE imputation on df.

        Parameters
        ----------
        df : pd.DataFrame
        interest_vars : list of str or None
            Columns to include in the statistical analysis table.
        show_results : bool, default=True

        Returns
        -------
        self
        """
        self._validate_inputs(df, interest_vars)

        interest_vars = list(df.columns) if interest_vars is None else interest_vars

        self._df_orig = df.copy(deep=True)
        self._missing_mask = self._df_orig.isna()
        self._seeds = self._generate_seeds()

        self._df_for_impute, self._cat_levels = self._encode_categoricals(self._df_orig.copy(deep=True))
        self._df_for_impute = self._coerce_to_numeric(self._df_for_impute)

        self.imputed_datasets_, self._stacked_arrays = self._run_imputations(df)
        self.pooled_df_ = self._build_pooled(df)
        self.stats_df_ = self._build_stats(df, interest_vars)

        self._save_zip()

        if show_results:
            print(
                "Statistical analysis of data imputation with MICE method "
                "to numerical and categorical variables"
            )
            with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
                print(self.stats_df_)
            print(
                "\nALERT: The table will only show a variable if there is some "
                "missing register in its column. Otherwise, it will not be shown in the table."
            )

        return self

    def get_results(self):
        """
        Return (imputed_datasets, pooled_df, stats_df) produced by fit().

        Raises
        ------
        RuntimeError
            If called before fit().
        """
        self._check_is_fitted()
        return self.imputed_datasets_, self.pooled_df_, self.stats_df_

    # ------------------------------------------------------------------
    # Private: validation & seed generation
    # ------------------------------------------------------------------

    def _validate_inputs(self, df, interest_vars):
        if interest_vars is not None:
            missing = [
                v for v in interest_vars
                if v not in df.columns
                and not any(v + self.prefix_sep in c for c in df.columns)
            ]
            if missing:
                raise ValueError(f"Interest variables not found in dataframe columns: {missing}")

    def _check_is_fitted(self):
        if self.imputed_datasets_ is None:
            raise RuntimeError("MICEImputer has not been fitted yet. Call fit() first.")

    def _generate_seeds(self):
        if self.random_state is None:
            base_seed = int(self._RNG.integers(0, 2 ** 31 - 1))
        else:
            base_seed = int(self.random_state)
        rng_local = np.random.default_rng(base_seed)
        return list(rng_local.integers(0, 2 ** 31 - 1, size=self.n))

    # ------------------------------------------------------------------
    # Private: encoding / decoding categoricals
    # ------------------------------------------------------------------

    def _encode_categoricals(self, df):
        cat_levels = {}
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        for col in categorical_cols:
            dummies = pd.get_dummies(df[col], prefix=col, prefix_sep=self.prefix_sep, dummy_na=False)
            cat_levels[col] = list(dummies.columns)
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)

        detected_blocks, single_binary = self._get_categorical_blocks(df)

        for k, v in detected_blocks.items():
            if k in cat_levels:
                for col in v:
                    if col not in cat_levels[k]:
                        cat_levels[k].append(col)
            else:
                cat_levels[k] = v

        for col in single_binary:
            if col not in df.columns:
                continue
            already_tracked = any(col in v for v in cat_levels.values())
            if not already_tracked:
                cat_levels[col] = [col]

        return df, cat_levels

    def _coerce_to_numeric(self, df):
        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df

    def _reconstruct_from_expanded(self, expanded_df, original_df):
        reconstructed = expanded_df.copy(deep=True)

        for var, dummy_cols in self._cat_levels.items():
            present = [c for c in dummy_cols if c in expanded_df.columns]
            if not present:
                continue

            this_cutoff = self.cutoff_map.get(var, self.cutoff)

            if len(present) == 1:
                col_name = present[0]
                if self.force_binary:
                    reconstructed[var] = (expanded_df[col_name] >= this_cutoff).astype('Int64')
                else:
                    reconstructed[var] = expanded_df[col_name].round().astype('Int64')
                if col_name != var:
                    reconstructed = reconstructed.drop(columns=present)
            else:
                categories = [
                    c.split(self.prefix_sep, 1)[1] if self.prefix_sep in c else c
                    for c in present
                ]
                selected = []
                for idx in range(expanded_df.shape[0]):
                    row_vals = expanded_df.loc[expanded_df.index[idx], present].values
                    winner_idx, _ = self._reconstruct_categorical_from_dummies(row_vals, present, this_cutoff, var)
                    selected.append(np.nan if winner_idx is None else categories[winner_idx])
                reconstructed[var] = pd.Series(selected, index=expanded_df.index, dtype='object')
                reconstructed = reconstructed.drop(columns=present)

        final_cols = [c for c in original_df.columns if c in reconstructed.columns]
        for c in reconstructed.columns:
            if c not in final_cols:
                final_cols.append(c)

        return reconstructed[final_cols].copy(deep=True)

    # ------------------------------------------------------------------
    # Private: imputation loop
    # ------------------------------------------------------------------

    def _run_imputations(self, original_df):
        imputed_datasets = []
        stacked_arrays = []

        for seed in self._seeds:
            kwargs = dict(max_iter=self.max_iter, initial_strategy=self.initial_strategy, random_state=int(seed))
            kwargs.update(self.imputer_kwargs_extra)

            imputer = IterativeImputer(**kwargs)
            arr = imputer.fit_transform(self._df_for_impute.values)
            stacked_arrays.append(arr.copy())

            expanded_df = pd.DataFrame(arr, index=self._df_for_impute.index, columns=self._df_for_impute.columns)
            imputed_final = self._reconstruct_from_expanded(expanded_df, original_df)
            imputed_datasets.append(imputed_final)

        return imputed_datasets, stacked_arrays

    def _build_pooled(self, original_df):
        stacked = np.stack(self._stacked_arrays, axis=0)
        pooled_array = np.mean(stacked, axis=0)
        pooled_expanded = pd.DataFrame(
            pooled_array, index=self._df_for_impute.index, columns=self._df_for_impute.columns
        )
        return self._reconstruct_from_expanded(pooled_expanded, original_df)

    # ------------------------------------------------------------------
    # Private: statistical analysis
    # ------------------------------------------------------------------

    def _build_stats(self, df, interest_vars):
        stats = []

        for var in interest_vars:
            if var not in df.columns:
                continue

            count_missing = int(self._missing_mask[var].sum()) if var in self._missing_mask.columns else 0
            if count_missing == 0:
                continue

            pct_missing = float(count_missing / len(df) * 100)
            is_categorical = var in self._cat_levels
            series_before = self._df_orig[var]
            series_after = self.pooled_df_[var]

            if not is_categorical:
                stats_before = self._calculate_stats_continuous(series_before)
                stats_after = self._calculate_stats_continuous(series_after)
                p_value = self._calculate_p_value_continuous(series_before, series_after)
                imputation_variance = self._compute_variance_continuous(var, count_missing)
            else:
                stats_before = self._calculate_stats_binary(series_before)
                stats_after = self._calculate_stats_binary(series_after)
                p_value = self._calculate_p_value_categorical(series_before, series_after)
                imputation_variance = self._compute_variance_categorical(var, count_missing)

            stats.append({
                'VARIABLE': var,
                'MISSINGS': f"{count_missing} ({pct_missing:.2f}%)",
                'BEFORE IMPUTATION': stats_before,
                'AFTER IMPUTATION': stats_after,
                'P-VALUE': p_value,
                'IMPUTATION VARIANCE': imputation_variance,
            })

        return pd.DataFrame(stats)

    def _compute_variance_continuous(self, var, count_missing):
        if var not in self._df_for_impute.columns:
            return np.nan
        if count_missing == 0:
            return 0.0
        col_idx = list(self._df_for_impute.columns).index(var)
        stacked = np.stack(self._stacked_arrays, axis=0)
        vals_across = stacked[:, self._missing_mask[var].values, col_idx]
        var_per_pos = np.var(vals_across, axis=0, ddof=1)
        return float(np.nanmean(var_per_pos))

    def _compute_variance_categorical(self, var, count_missing):
        dummy_cols = [c for c in self._cat_levels.get(var, []) if c in self._df_for_impute.columns]
        if count_missing == 0:
            return 0.0
        if not dummy_cols:
            return np.nan
        stacked = np.stack(self._stacked_arrays, axis=0)
        col_indices = [list(self._df_for_impute.columns).index(c) for c in dummy_cols]
        var_per_pos_list = []
        for pos_idx, row_bool in enumerate(self._missing_mask[var].values):
            if not row_bool:
                continue
            vals = stacked[:, pos_idx, :][:, col_indices]
            var_per_dummy = np.var(vals, axis=0, ddof=1)
            var_per_pos_list.append(np.nanmean(var_per_dummy))
        if not var_per_pos_list:
            return np.nan
        return float(np.nanmean(var_per_pos_list))

    def _save_zip(self):
        zip_filename = 'imputed_datasets.zip'
        print(f"Saving {self.n} imputed datasets in '{zip_filename}'...")
        try:
            with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zf:
                for i, imp_df in enumerate(self.imputed_datasets_):
                    temp_filename = f'imputed_dataset_{i + 1}.xlsx'
                    imp_df.to_excel(temp_filename, index=True)
                    zf.write(temp_filename)
                    os.remove(temp_filename)
            print(f"Success: '{zip_filename}' saved.")
        except Exception as e:
            print(f"Error saving zip file: {e}")

    # ------------------------------------------------------------------
    # Private: statistical helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _calculate_stats_continuous(series):
        if series.count() == 0:
            return "N/A (N/A, N/A)"
        med = series.median()
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        try:
            return f"{med:.2f} ({q1:.2f}, {q3:.2f})"
        except (ValueError, TypeError):
            return "N/A (N/A, N/A)"

    @staticmethod
    def _calculate_stats_binary(series):
        total_non_null = series.count()
        if total_non_null == 0:
            return "0 (0.00%)"
        positive_count = (pd.to_numeric(series, errors='coerce') == 1).sum()
        pct = (positive_count / total_non_null) * 100 if total_non_null > 0 else 0
        return f"{positive_count} ({pct:.2f}%)"

    @staticmethod
    def _calculate_p_value_continuous(series_before, series_after):
        s_before = series_before.dropna()
        s_after = series_after.dropna()
        if len(s_before) == 0 or len(s_after) == 0:
            return np.nan
        try:
            _, p = mannwhitneyu(s_before, s_after, alternative='two-sided')
            return p
        except ValueError:
            return 1.0 if np.array_equal(s_before.values, s_after.values) else np.nan

    @staticmethod
    def _calculate_p_value_categorical(series_before, series_after):
        s_before = series_before.dropna()
        s_after = series_after.dropna()
        if len(s_before) == 0 or len(s_after) == 0:
            return np.nan

        counts_before = s_before.value_counts()
        counts_after = s_after.value_counts()
        all_categories = sorted(list(set(counts_before.index).union(set(counts_after.index))))
        all_categories = [cat for cat in all_categories if pd.notna(cat)]

        if not all_categories:
            return np.nan

        row_before = [counts_before.get(cat, 0) for cat in all_categories]
        row_after = [counts_after.get(cat, 0) for cat in all_categories]
        table = [row_before, row_after]

        if np.sum(table) == 0:
            return np.nan

        try:
            _, p, _, _ = chi2_contingency(table)
            return p
        except ValueError:
            return np.nan

    # ------------------------------------------------------------------
    # Private: categorical block detection & reconstruction helpers
    # ------------------------------------------------------------------

    def _get_categorical_blocks(self, df):
        blocks = defaultdict(list)
        single_binary = []

        for col in df.columns:
            if self.prefix_sep in col:
                var, _ = col.split(self.prefix_sep, 1)
                blocks[var].append(col)
            else:
                unique_nonnull = pd.Series(df[col].dropna().unique())
                if len(unique_nonnull) > 0 and all(x in {0, 1, 0.0, 1.0} for x in unique_nonnull):
                    single_binary.append(col)
                elif pd.api.types.is_numeric_dtype(df[col]):
                    pass
                elif len(unique_nonnull) > 0 and set(unique_nonnull).issubset({0, 1}):
                    single_binary.append(col)

        return dict(blocks), single_binary

    def _reconstruct_categorical_from_dummies(self, df_row_values, dummy_cols, cutoff, var_name):
        vals = np.array(df_row_values, dtype=float)
        ge_cutoff = vals >= cutoff
        num_ge = ge_cutoff.sum()

        if num_ge == 1:
            return int(np.where(ge_cutoff)[0][0]), vals
        elif num_ge > 1:
            if self.tie_strategy == 'highest_probability':
                return int(np.argmax(vals)), vals
            elif self.tie_strategy == 'first':
                return int(np.where(ge_cutoff)[0][0]), vals
            elif self.tie_strategy == 'nan':
                return None, vals
            elif self.tie_strategy == 'force_class':
                return self._resolve_forced_class(dummy_cols, var_name, vals)
            else:
                return None, vals
        else:
            if self.tie_strategy == 'highest_probability':
                return int(np.argmax(vals)), vals
            elif self.tie_strategy == 'first':
                return 0, vals
            elif self.tie_strategy == 'nan':
                return None, vals
            elif self.tie_strategy == 'force_class':
                return self._resolve_forced_class(dummy_cols, var_name, vals)
            else:
                return None, vals

    def _resolve_forced_class(self, dummy_cols, var_name, vals):
        if self.tie_force is None:
            return None, vals
        if isinstance(self.tie_force, dict):
            if var_name not in self.tie_force:
                return None, vals
            forced = self.tie_force[var_name]
        else:
            forced = self.tie_force
        for idx, col in enumerate(dummy_cols):
            if forced in col:
                return idx, vals
        return None, vals


# ------------------------------------------------------------------
# Compatibility wrapper
# ------------------------------------------------------------------

def impute_missing(df, strategy="mean"):
    """
    Imputes missing values using a simple strategy (mean/median/most_frequent).
    For multiple imputation, use MICEImputer instead.
    """
    imputer = SimpleImputer(strategy=strategy)
    return pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
