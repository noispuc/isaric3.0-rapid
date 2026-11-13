"""
Data splitting for the RAPID methodology.

This module partitions data into training and testing sets to prevent
overfitting. The training set is used to fit model parameters, and
the test set is reserved for unbiased assessment of generalization.

Techniques:
- simple_random_split: Random assignment to train/test sets.
- stratified_split: Preserve class proportions across sets.
- temporal_split: Split chronologically (train on older, test on newer).
- parse_split_strategy: Parse strategy string for data splitting.
"""

import pandas as pd
from typing import Optional, Tuple
from sklearn.model_selection import train_test_split


def simple_random_split(
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
        ValueError: If test_size is not between 0.0 and 1.0.
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


def stratified_split(
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


def temporal_split(
    data: pd.DataFrame,
    date_col: str,
    test_size: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data chronologically (train on older, test on newer).

    Useful for time-series data where temporal order matters.

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


def parse_split_strategy(strategy: str) -> Tuple[str, float, Optional[str]]:
    """
    Parse data splitting strategy string.

    Examples:
        "split(test=0.2)" -> ("random", 0.2, None)
        "split(test=0.2,stratify=outcome)" -> ("stratified", 0.2, "outcome")
        "split(test=0.2,method=temporal,date_col=admission_date)"
        -> ("temporal", 0.2, "admission_date")

    Args:
        strategy: Strategy string to parse.

    Returns:
        Tuple of (method, test_size, target_or_date_col).

    Raises:
        ValueError: If strategy format is invalid.
    """
    if not isinstance(strategy, str) or not strategy:
        raise ValueError("Strategy must be a non-empty string.")

    if not strategy.startswith("split("):
        raise ValueError(
            f"Invalid split strategy: {strategy}. "
            "Expected: 'split(test=0.2)'."
        )

    params_str = strategy[6:-1]  # Remove "split(" and ")"
    params = params_str.split(",")

    test_size = 0.2
    method = "random"
    target_or_date_col = None

    for param in params:
        param = param.strip()

        if "test=" in param:
            test_size = _parse_test_size(param)
        elif "stratify=" in param:
            method = "stratified"
            target_or_date_col = param.replace("stratify=", "")
        elif "method=temporal" in param:
            method = "temporal"
        elif "date_col=" in param:
            target_or_date_col = param.replace("date_col=", "")

    return (method, test_size, target_or_date_col)


def _parse_test_size(param: str) -> float:
    """
    Extract test_size from parameter string.

    Example:
        "test=0.2" -> 0.2

    Args:
        param: Parameter string containing test value.

    Returns:
        Test size as float.

    Raises:
        ValueError: If test value is missing or invalid.
    """
    try:
        test_size = float(param.split("test=")[1])

        if not (0.0 < test_size < 1.0):
            raise ValueError(
                f"test_size must be between 0.0 and 1.0. Received: {test_size}"
            )

        return test_size
    except (IndexError, ValueError):
        raise ValueError(
            f"Invalid test parameter: {param}. Expected: 'test=0.2'."
        )