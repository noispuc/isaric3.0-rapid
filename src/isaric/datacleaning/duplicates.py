"""
Duplicate handling for the RAPID methodology.

This module provides functions to identify and remove redundant
observations (rows) in a dataset, essential in clinical research
to avoid bias from repeated patient records.

Techniques:
- exact_match_removal: Remove identical rows (keeping first or last).
- key_based_deduplication: Remove duplicates based on a subset of key identifiers.
- parse_duplicate_strategy: Parse strategy string for duplicate handling.
"""

import pandas as pd
from typing import List, Tuple


def exact_match_removal(
    data: pd.DataFrame,
    keep: str = "first"
) -> pd.DataFrame:
    """
    Remove rows that are identical across all columns.

    Args:
        data: Input DataFrame.
        keep: Which occurrence to keep. Either 'first' or 'last'.

    Returns:
        DataFrame without exact duplicate rows.

    Raises:
        ValueError: If keep is not 'first' or 'last'.
    """
    if keep not in ("first", "last"):
        raise ValueError(f"Invalid keep value: {keep}. Use 'first' or 'last'.")

    return data.drop_duplicates(keep=keep).reset_index(drop=True)


def key_based_deduplication(
    data: pd.DataFrame,
    subset: List[str],
    keep: str = "first"
) -> pd.DataFrame:
    """
    Remove duplicates based on a subset of key identifier variables.

    This is useful when a patient has multiple records (e.g., repeat
    admissions) and only one observation is relevant for the analysis.

    Args:
        data: Input DataFrame.
        subset: List of column names that define a duplicate record.
        keep: Which occurrence to keep. Either 'first' or 'last'.

    Returns:
        DataFrame with duplicates removed based on key columns.

    Raises:
        ValueError: If subset is empty, keep is invalid, or columns not found.
    """
    if not subset:
        raise ValueError("Subset must contain at least one column name.")

    if keep not in ("first", "last"):
        raise ValueError(f"Invalid keep value: {keep}. Use 'first' or 'last'.")

    missing_cols = [col for col in subset if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Columns not found in DataFrame: {missing_cols}")

    return data.drop_duplicates(subset=subset, keep=keep).reset_index(drop=True)


def parse_duplicate_strategy(strategy: str) -> Tuple[str, List[str], str]:
    """
    Parse duplicate handling strategy string.

    Examples:
        "exact(keep=first)" -> ("exact", [], "first")
        "exact(keep=last)" -> ("exact", [], "last")
        "key_based(subset=id,date,keep=first)" -> ("key_based", ["id", "date"], "first")
        "key_based(subset=id,keep=last)" -> ("key_based", ["id"], "last")

    Args:
        strategy: Strategy string to parse.

    Returns:
        Tuple containing (method, subset, keep).

    Raises:
        ValueError: If strategy format is invalid.
    """
    if not isinstance(strategy, str) or not strategy:
        raise ValueError("Strategy must be a non-empty string.")

    if strategy.startswith("exact"):
        keep = _parse_keep_value(strategy)
        return ("exact", [], keep)

    elif strategy.startswith("key_based"):
        subset, keep = _parse_key_based_params(strategy)
        return ("key_based", subset, keep)

    else:
        raise ValueError(
            f"Unknown duplicate strategy: {strategy}. "
            "Use 'exact(keep=first)' or 'key_based(subset=col1,keep=first)'."
        )


def _parse_keep_value(strategy: str) -> str:
    """
    Extract keep value from strategy string.

    Example:
        "exact(keep=first)" -> "first"

    Args:
        strategy: Strategy string containing keep parameter.

    Returns:
        Keep value ('first' or 'last').

    Raises:
        ValueError: If keep parameter is missing or malformed.
    """
    try:
        keep = strategy.split("keep=")[1].rstrip(")")
        if keep not in ("first", "last"):
            raise ValueError(f"Invalid keep value: {keep}")
        return keep
    except (IndexError, ValueError):
        raise ValueError(
            f"Invalid keep format in strategy: {strategy}. "
            "Expected: 'exact(keep=first)' or 'exact(keep=last)'."
        )


def _parse_key_based_params(strategy: str) -> Tuple[List[str], str]:
    """
    Extract subset and keep from key_based strategy string.

    Example:
        "key_based(subset=id,date,keep=first)" -> (["id", "date"], "first")
        "key_based(subset=id,keep=last)" -> (["id"], "last")

    Args:
        strategy: Strategy string containing subset and keep parameters.

    Returns:
        Tuple containing (subset, keep).

    Raises:
        ValueError: If subset or keep parameters are missing or malformed.
    """
    try:
        params_str = strategy.split("(")[1].rstrip(")")
        params = params_str.split(",")

        subset: List[str] = []
        keep: str = "first"

        for param in params:
            if "subset=" in param:
                subset_str = param.replace("subset=", "")
                subset = subset_str.split("|") if "|" in subset_str else [subset_str]
                subset = [col.strip() for col in subset if col.strip()]
            elif "keep=" in param:
                keep = param.replace("keep=", "").strip()

        if not subset:
            raise ValueError("Subset cannot be empty.")

        if keep not in ("first", "last"):
            raise ValueError(f"Invalid keep value: {keep}")

        return subset, keep

    except (IndexError, ValueError):
        raise ValueError(
            f"Invalid key_based format in strategy: {strategy}. "
            "Expected: 'key_based(subset=col1,col2,keep=first)'."
        )