"""
Train/test validation for the RAPID methodology.

This module provides functions to partition data into training and
testing sets (Step 4 of the RAPID methodology). Train/Test validation
uses a single hold-out test set for final assessment.

Techniques:
- holdout_validation: Simple random split into train/test.
- stratified_holdout: Split preserving class proportions.
- temporal_holdout: Split chronologically (train on older, test on newer).
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from sklearn.model_selection import train_test_split


def holdout_validation(
    data: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into training and testing sets using simple random sampling.

    Args:
        data: Input DataFrame.
        test_size: Proportion of data for testing (0.0 to 1.0).
        random_state: Seed for reproducibility.

    Returns:
        Tuple of (train_df, test_df).

    Raises:
        ValueError: If test_size is invalid.
    """
    if not (0.0 < test_size < 1.0):
        raise ValueError(
            f"test_size must be between 0.0 and 1.0. Received: {test_size}"
        )

    train_df, test_df = train_test_split(
        data,
        test_size=test_size,
        random_state=random_state
    )

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def stratified_holdout(
    data: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data preserving the proportion of the target variable.

    Important for imbalanced or small datasets.

    Args:
        data: Input DataFrame.
        target_col: Column name of the target/outcome variable.
        test_size: Proportion of data for testing (0.0 to 1.0).
        random_state: Seed for reproducibility.

    Returns:
        Tuple of (train_df, test_df).

    Raises:
        ValueError: If target_col is not found or test_size is invalid.
    """
    if target_col not in data.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    if not (0.0 < test_size < 1.0):
        raise ValueError(
            f"test_size must be between 0.0 and 1.0. Received: {test_size}"
        )

    train_df, test_df = train_test_split(
        data,
        test_size=test_size,
        random_state=random_state,
        stratify=data[target_col]
    )

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def temporal_holdout(
    data: pd.DataFrame,
    date_col: str,
    test_size: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data chronologically (train on older, test on newer).

    Useful for time-series data where temporal order matters.
    The split point is determined by the test_size proportion.

    Args:
        data: Input DataFrame.
        date_col: Column name containing dates/timestamps.
        test_size: Proportion of data for testing (0.0 to 1.0).

    Returns:
        Tuple of (train_df, test_df).

    Raises:
        ValueError: If date_col is not found or test_size is invalid.
    """
    if date_col not in data.columns:
        raise ValueError(f"Date column '{date_col}' not found in DataFrame.")

    if not (0.0 < test_size < 1.0):
        raise ValueError(
            f"test_size must be between 0.0 and 1.0. Received: {test_size}"
        )

    sorted_data = data.sort_values(by=date_col).reset_index(drop=True)
    split_idx = int(len(sorted_data) * (1 - test_size))

    train_df = sorted_data.iloc[:split_idx].reset_index(drop=True)
    test_df = sorted_data.iloc[split_idx:].reset_index(drop=True)

    return train_df, test_df


def temporal_train_test_split(
    data: pd.DataFrame,
    year_column: str,
    train_end_year: int,
    test_start_year: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data chronologically using explicit year boundaries.

    Training includes years up to train_end_year (inclusive).
    Testing includes years from test_start_year (inclusive).

    Args:
        data: Input DataFrame.
        year_column: Column name containing year values.
        train_end_year: Last year (inclusive) included in training.
        test_start_year: First year (inclusive) included in testing.

    Returns:
        Tuple of (train_df, test_df).

    Raises:
        ValueError: If year_column is not found or year boundaries are invalid.
    """
    if year_column not in data.columns:
        raise ValueError(f"Year column '{year_column}' not found in DataFrame.")

    if train_end_year >= test_start_year:
        raise ValueError(
            f"train_end_year ({train_end_year}) must be less than "
            f"test_start_year ({test_start_year})."
        )

    train_df = data[data[year_column] <= train_end_year].reset_index(drop=True)
    test_df = data[data[year_column] >= test_start_year].reset_index(drop=True)

    if len(train_df) == 0:
        raise ValueError("Training set is empty. Check year boundaries.")

    if len(test_df) == 0:
        raise ValueError("Testing set is empty. Check year boundaries.")

    return train_df, test_df