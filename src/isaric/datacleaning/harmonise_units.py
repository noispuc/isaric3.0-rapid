"""
Unit harmonization for the RAPID methodology.

This module standardizes variables that measure the same concept
but use different units or scales, ensuring direct comparability.
This is particularly common when combining data from multiple
clinical units or sources.

Techniques:
- linear_conversion: Apply arithmetic transformation to convert units.
- lookup_tables: Use pre-defined conversion factors for categorical codes.
- convert_temperature_celsius_to_fahrenheit: Convert temperature from Celsius to Fahrenheit.
- convert_temperature_fahrenheit_to_celsius: Convert temperature from Fahrenheit to Celsius.
- convert_weight_kg_to_lbs: Convert weight from kilograms to pounds.
- convert_weight_lbs_to_kg: Convert weight from pounds to kilograms.
- parse_harmonise_strategy: Parse strategy string for unit harmonization.
"""

import pandas as pd
from typing import Dict, Tuple, Union


def linear_conversion(
    data: pd.DataFrame,
    column: str,
    multiplier: float = 1.0,
    addend: float = 0.0
) -> pd.DataFrame:
    """
    Apply a simple linear conversion formula to a numeric column.

    Formula: new_value = (value * multiplier) + addend

    Example:
        To convert mg/dL to mmol/L for creatinine:
        multiplier = 0.0884 (1 mg/dL = 0.0884 mmol/L)

    Args:
        data: Input DataFrame.
        column: Column name to convert.
        multiplier: Multiplication factor.
        addend: Addition factor (default 0).

    Returns:
        DataFrame with converted column values.

    Raises:
        ValueError: If column is not found or is not numeric.
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")

    if not pd.api.types.is_numeric_dtype(data[column]):
        raise ValueError(
            f"Column '{column}' must be numeric for linear conversion."
        )

    result = data.copy()
    result[column] = (result[column] * multiplier) + addend
    return result


def lookup_tables(
    data: pd.DataFrame,
    column: str,
    conversion_map: Dict[Union[str, int, float], Union[str, int, float]]
) -> pd.DataFrame:
    """
    Convert values in a column using a pre-defined lookup table.

    This is useful for categorical codes, textual variable names,
    or when different sources use different labels for the same concept.

    Example:
        conversion_map = {
            "M": "Male",
            "F": "Female",
            1: "Yes",
            0: "No"
        }

    Args:
        data: Input DataFrame.
        column: Column name to convert.
        conversion_map: Dictionary mapping original values to target values.

    Returns:
        DataFrame with converted column values.

    Raises:
        ValueError: If column is not found.
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")

    result = data.copy()
    result[column] = result[column].map(conversion_map).fillna(result[column])
    return result


def convert_temperature_celsius_to_fahrenheit(
    data: pd.DataFrame,
    column: str
) -> pd.DataFrame:
    """
    Convert temperature values from Celsius to Fahrenheit.

    Formula: °F = (°C × 9/5) + 32

    Args:
        data: Input DataFrame.
        column: Column containing Celsius values.

    Returns:
        DataFrame with Fahrenheit values.

    Raises:
        ValueError: If column is not found or is not numeric.
    """
    return linear_conversion(data, column, multiplier=9/5, addend=32)


def convert_temperature_fahrenheit_to_celsius(
    data: pd.DataFrame,
    column: str
) -> pd.DataFrame:
    """
    Convert temperature values from Fahrenheit to Celsius.

    Formula: °C = (°F - 32) × 5/9

    Args:
        data: Input DataFrame.
        column: Column containing Fahrenheit values.

    Returns:
        DataFrame with Celsius values.

    Raises:
        ValueError: If column is not found or is not numeric.
    """
    return linear_conversion(data, column, multiplier=5/9, addend=-32*5/9)


def convert_weight_kg_to_lbs(
    data: pd.DataFrame,
    column: str
) -> pd.DataFrame:
    """
    Convert weight values from kilograms to pounds.

    Formula: lbs = kg × 2.20462

    Args:
        data: Input DataFrame.
        column: Column containing kg values.

    Returns:
        DataFrame with lbs values.

    Raises:
        ValueError: If column is not found or is not numeric.
    """
    return linear_conversion(data, column, multiplier=2.20462)


def convert_weight_lbs_to_kg(
    data: pd.DataFrame,
    column: str
) -> pd.DataFrame:
    """
    Convert weight values from pounds to kilograms.

    Formula: kg = lbs × 0.453592

    Args:
        data: Input DataFrame.
        column: Column containing lbs values.

    Returns:
        DataFrame with kg values.

    Raises:
        ValueError: If column is not found or is not numeric.
    """
    return linear_conversion(data, column, multiplier=0.453592)


def parse_harmonise_strategy(strategy: str) -> Tuple[str, str, str]:
    """
    Parse harmonise units strategy string.

    Examples:
        "temperature:celsius_to_fahrenheit:body_temp"
        -> ("temperature", "celsius_to_fahrenheit", "body_temp")
        
        "weight:kg_to_lbs:weight_kg"
        -> ("weight", "kg_to_lbs", "weight_kg")

    Args:
        strategy: Strategy string to parse.

    Returns:
        Tuple containing (unit_type, conversion, column).

    Raises:
        ValueError: If strategy format is invalid.
    """
    if not isinstance(strategy, str) or not strategy:
        raise ValueError("Strategy must be a non-empty string.")

    parts = strategy.split(":")

    if len(parts) != 3:
        raise ValueError(
            f"Invalid harmonise_units format: {strategy}. "
            "Expected: 'type:conversion:column_name'"
        )

    unit_type, conversion, column = parts

    if not unit_type or not conversion or not column:
        raise ValueError(
            f"Invalid harmonise_units format: {strategy}. "
            "All parts must be non-empty."
        )

    return (unit_type, conversion, column)