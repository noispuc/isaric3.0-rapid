"""
Encoding of temporal variables for the RAPID methodology.

This module transforms time-related data (e.g., dates, timestamps,
durations) into meaningful numeric features that modelling algorithms
can use effectively.

Techniques:
- duration_encode: Calculate interval between two time points.
- cyclical_encode: Use sine and cosine transformations for periodic features.
- parse_temporal_strategy: Parse strategy string for temporal encoding.
"""

import pandas as pd
import numpy as np
from typing import Tuple


def duration_encode(
    data: pd.DataFrame,
    start_col: str,
    end_col: str,
    unit: str = "days"
) -> pd.DataFrame:
    """
    Calculate the interval between two time points as a numeric feature.

    This is useful for computing time from symptom onset to admission,
    length of stay, or other durations.

    Args:
        data: Input DataFrame.
        start_col: Column name of the start time.
        end_col: Column name of the end time.
        unit: Time unit for the result ('days', 'hours', 'minutes').

    Returns:
        DataFrame with a new duration column.

    Raises:
        ValueError: If columns are not found or unit is invalid.
    """
    for col in [start_col, end_col]:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

    valid_units = ("days", "hours", "minutes", "seconds")
    if unit not in valid_units:
        raise ValueError(
            f"unit must be one of {valid_units}. Received: {unit}"
        )

    result = data.copy()
    result['duration'] = (
        pd.to_datetime(result[end_col]) - pd.to_datetime(result[start_col])
    ) / pd.Timedelta(1, unit=unit)

    return result


def cyclical_encode(
    data: pd.DataFrame,
    column: str,
    period: int = 7
) -> pd.DataFrame:
    """
    Apply sine and cosine transformations to periodic features.

    This preserves the cyclical nature of time features (e.g., day of
    week, month). For example, day 6 and day 0 are adjacent in a week,
    but a linear encoding would treat them as far apart.

    Args:
        data: Input DataFrame.
        column: Column name of the periodic feature.
        period: Period of the cycle (e.g., 7 for days of week,
            12 for months, 24 for hours).

    Returns:
        DataFrame with sin and cos columns for the periodic feature.

    Raises:
        ValueError: If column is not found or period is invalid.
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")

    if period < 2:
        raise ValueError(
            f"period must be at least 2. Received: {period}"
        )

    result = data.copy()
    values = result[column].astype(float)
    result[f'{column}_sin'] = np.sin(2 * np.pi * values / period)
    result[f'{column}_cos'] = np.cos(2 * np.pi * values / period)
    result = result.drop(columns=[column])

    return result


def parse_temporal_strategy(strategy: str) -> Tuple[str, dict]:
    """
    Parse temporal encoding strategy string.

    Examples:
        "duration(start=onset,end=admission,unit=days)"
        -> ("duration", {"start": "onset", "end": "admission", "unit": "days"})
        
        "cyclical(column=day_of_week,period=7)"
        -> ("cyclical", {"column": "day_of_week", "period": 7})

    Args:
        strategy: Strategy string to parse.

    Returns:
        Tuple of (method, parameters_dict).

    Raises:
        ValueError: If strategy format is invalid.
    """
    if not isinstance(strategy, str) or not strategy:
        raise ValueError("Strategy must be a non-empty string.")

    if strategy.startswith("duration"):
        params = _parse_duration_params(strategy)
        return ("duration", params)

    elif strategy.startswith("cyclical"):
        params = _parse_cyclical_params(strategy)
        return ("cyclical", params)

    else:
        raise ValueError(
            f"Unknown temporal encoding strategy: {strategy}. "
            "Use 'duration(start=...,end=...,unit=days)' "
            "or 'cyclical(column=...,period=7)'."
        )


def _parse_duration_params(strategy: str) -> dict:
    """
    Parse duration parameters from strategy string.

    Args:
        strategy: Strategy string containing duration parameters.

    Returns:
        Dictionary with duration parameters.

    Raises:
        ValueError: If parameters are missing or malformed.
    """
    try:
        params_str = strategy[9:-1]  # Remove "duration(" and ")"
        params = params_str.split(",")

        result = {}
        for param in params:
            if "start=" in param:
                result["start"] = param.replace("start=", "").strip()
            elif "end=" in param:
                result["end"] = param.replace("end=", "").strip()
            elif "unit=" in param:
                result["unit"] = param.replace("unit=", "").strip()

        if "start" not in result or "end" not in result:
            raise ValueError("Duration requires 'start' and 'end' parameters.")

        if "unit" not in result:
            result["unit"] = "days"

        return result

    except (IndexError, ValueError):
        raise ValueError(
            f"Invalid duration format: {strategy}. "
            "Expected: 'duration(start=onset,end=admission,unit=days)'."
        )


def _parse_cyclical_params(strategy: str) -> dict:
    """
    Parse cyclical parameters from strategy string.

    Args:
        strategy: Strategy string containing cyclical parameters.

    Returns:
        Dictionary with cyclical parameters.

    Raises:
        ValueError: If parameters are missing or malformed.
    """
    try:
        params_str = strategy[9:-1]  # Remove "cyclical(" and ")"
        params = params_str.split(",")

        result = {}
        for param in params:
            if "column=" in param:
                result["column"] = param.replace("column=", "").strip()
            elif "period=" in param:
                result["period"] = int(param.replace("period=", "").strip())

        if "column" not in result:
            raise ValueError("Cyclical requires 'column' parameter.")

        if "period" not in result:
            result["period"] = 7

        return result

    except (IndexError, ValueError):
        raise ValueError(
            f"Invalid cyclical format: {strategy}. "
            "Expected: 'cyclical(column=day_of_week,period=7)'."
        )