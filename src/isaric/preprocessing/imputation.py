"""
Imputation for the RAPID methodology.

This module provides MICE (Multiple Imputation by Chained Equations)
for handling missing data in preprocessing. MICE generates multiple
plausible completed datasets and pools the results to preserve
uncertainty about missing values.

Techniques:
- mice_imputation: Run MICE imputation and return pooled DataFrame.
- parse_imputation_strategy: Parse strategy string for imputation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from collections import defaultdict


def mice_imputation(
    data: pd.DataFrame,
    n: int = 5,
    max_iter: int = 10,
    initial_strategy: str = "most_frequent",
    random_state: int = 42
) -> pd.DataFrame:
    """
    Run MICE imputation and return a single pooled DataFrame.

    MICE operates iteratively: for each variable with missing data,
    it fits a regression model using all other variables as predictors,
    then imputes missing values by drawing from the predictive distribution.

    Args:
        data: Input DataFrame with missing values.
        n: Number of independent imputations (default 5).
        max_iter: Maximum imputation rounds per dataset (default 10).
        initial_strategy: Strategy to initialize missing values
            ('mean', 'median', 'most_frequent', 'constant').
        random_state: Base seed for reproducibility.

    Returns:
        DataFrame with missing values imputed (pooled across n imputations).
    """
    if n < 1:
        raise ValueError(f"n must be at least 1. Received: {n}")

    if max_iter < 1:
        raise ValueError(f"max_iter must be at least 1. Received: {max_iter}")

    # Copy original and track missing mask
    df_orig = data.copy(deep=True)
    missing_mask = df_orig.isna()

    # Encode categoricals
    df_for_impute, cat_levels = _encode_categoricals(df_orig.copy(deep=True))
    df_for_impute = _coerce_to_numeric(df_for_impute)

    # Generate seeds
    seeds = _generate_seeds(random_state, n)

    # Run imputations
    stacked_arrays = []
    for seed in seeds:
        imputer = IterativeImputer(
            max_iter=max_iter,
            initial_strategy=initial_strategy,
            random_state=int(seed)
        )
        arr = imputer.fit_transform(df_for_impute.values)
        stacked_arrays.append(arr.copy())

    # Pool results (element-wise mean across imputations)
    stacked = np.stack(stacked_arrays, axis=0)
    pooled_array = np.mean(stacked, axis=0)
    pooled_expanded = pd.DataFrame(
        pooled_array,
        index=df_for_impute.index,
        columns=df_for_impute.columns
    )

    # Reconstruct categorical variables
    result = _reconstruct_from_expanded(pooled_expanded, df_orig, cat_levels)

    return result


def parse_imputation_strategy(strategy: str) -> Dict:
    """
    Parse imputation strategy string.

    Examples:
        "imputation(type=mice)" -> {"type": "mice", "n": 5, "max_iter": 10}

    Args:
        strategy: Strategy string to parse.

    Returns:
        Dictionary with imputation configuration.

    Raises:
        ValueError: If strategy format is invalid.
    """
    if not isinstance(strategy, str) or not strategy:
        raise ValueError("Strategy must be a non-empty string.")

    if strategy == "imputation(type=mice)":
        return {"type": "mice", "n": 5, "max_iter": 10}

    else:
        raise ValueError(
            f"Unknown imputation strategy: {strategy}. "
            "Expected: 'imputation(type=mice)'."
        )


# ============================================================================
# PRIVATE HELPERS
# ============================================================================

def _generate_seeds(random_state: int, n: int) -> List[int]:
    """
    Generate n unique seeds from a base random_state.

    Args:
        random_state: Base seed.
        n: Number of seeds to generate.

    Returns:
        List of n integer seeds.
    """
    rng_local = np.random.default_rng(int(random_state))
    return list(rng_local.integers(0, 2 ** 31 - 1, size=n))


def _encode_categoricals(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    One-hot encode categorical variables and return expanded DataFrame
    plus mapping of original column to dummy columns.

    Args:
        df: Input DataFrame.

    Returns:
        Tuple of (expanded_df, cat_levels).
    """
    cat_levels: Dict[str, List[str]] = {}
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    for col in categorical_cols:
        dummies = pd.get_dummies(
            df[col],
            prefix=col,
            prefix_sep='!',
            dummy_na=False
        )
        cat_levels[col] = list(dummies.columns)
        df = pd.concat([df.drop(columns=[col]), dummies], axis=1)

    return df, cat_levels


def _coerce_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert all columns to numeric, coercing errors to NaN.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with numeric columns.
    """
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def _reconstruct_from_expanded(
    expanded_df: pd.DataFrame,
    original_df: pd.DataFrame,
    cat_levels: Dict[str, List[str]]
) -> pd.DataFrame:
    """
    Reconstruct original categorical variables from one-hot encoded dummies.

    Args:
        expanded_df: DataFrame with imputed dummy columns.
        original_df: Original DataFrame (to get column order).
        cat_levels: Mapping of categorical variable to dummy columns.

    Returns:
        DataFrame with categorical variables reconstructed.
    """
    reconstructed = expanded_df.copy(deep=True)

    for var, dummy_cols in cat_levels.items():
        present = [c for c in dummy_cols if c in expanded_df.columns]
        if not present:
            continue

        if len(present) == 1:
            # Binary variable
            reconstructed[var] = expanded_df[present[0]].round().astype('Int64')
            if present[0] != var:
                reconstructed = reconstructed.drop(columns=present)
        else:
            # Multi-class variable
            categories = [
                c.split('!', 1)[1] if '!' in c else c
                for c in present
            ]
            selected = []
            for idx in range(expanded_df.shape[0]):
                row_vals = expanded_df.loc[expanded_df.index[idx], present].values
                winner_idx = int(np.argmax(row_vals))
                selected.append(categories[winner_idx])
            reconstructed[var] = pd.Series(selected, index=expanded_df.index, dtype='object')
            reconstructed = reconstructed.drop(columns=present)

    # Restore original column order
    final_cols = [c for c in original_df.columns if c in reconstructed.columns]
    for c in reconstructed.columns:
        if c not in final_cols:
            final_cols.append(c)

    return reconstructed[final_cols].copy(deep=True)