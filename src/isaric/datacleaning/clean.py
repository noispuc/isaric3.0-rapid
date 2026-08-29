"""
Step 1: Data Cleaning - Remove errors and inconsistencies from raw data.

This module provides the Clean class, which orchestrates all data
cleaning techniques of the RAPID methodology.

The Clean class follows the same pattern as all RAPID classes:
- Configuration is set through class attributes
- Public execute() method orchestrates the configured techniques
- Private methods call functions from technique-specific modules
"""

import pandas as pd
from typing import Optional

from isaric.datacleaning.duplicates import (
    parse_duplicate_strategy,
    exact_match_removal,
    key_based_deduplication
)
from isaric.datacleaning.harmonise_units import (
    parse_harmonise_strategy,
    convert_temperature_celsius_to_fahrenheit,
    convert_temperature_fahrenheit_to_celsius,
    convert_weight_kg_to_lbs,
    convert_weight_lbs_to_kg
)
from isaric.datacleaning.remove_zero_variance import (
    parse_remove_variance_strategy,
    frequency_ratio_analysis,
    unique_value_count
)
from isaric.datacleaning.handle_missing import (
    parse_missing_strategy,
    drop_rows,
    drop_columns,
    impute_mean,
    impute_median,
    impute_mode
)


class Clean:
    """
    Step 1: Data Cleaning.

    Orchestrates data cleaning techniques based on configured attributes.

    Attributes:
        duplicate_handling (str): Duplicate removal strategy.
            Options:
                - None (skip)
                - "exact(keep=first)" or "exact(keep=last)"
                - "key_based(subset=col1,keep=first)"
        harmonise_units (str): Unit harmonization strategy.
            Format: "type:conversion:column_name"
            Options:
                - None (skip)
                - "temperature:celsius_to_fahrenheit:col"
                - "temperature:fahrenheit_to_celsius:col"
                - "weight:kg_to_lbs:col"
                - "weight:lbs_to_kg:col"
        remove_zero_variance (str): Zero-variance feature removal strategy.
            Options:
                - None (skip)
                - "frequency_ratio_analysis(threshold=19.0)"
                - "unique_value_count(min=2)"
        handle_missing (str): Missing value handling strategy.
            Options:
                - None (skip)
                - "drop_rows()"
                - "drop_columns(p=0.3)"
                - "imputation(type=mean)"
                - "imputation(type=median)"
                - "imputation(type=mode)"
    """

    def __init__(
        self,
        duplicate_handling: Optional[str] = None,
        harmonise_units: Optional[str] = None,
        remove_zero_variance: Optional[str] = None,
        handle_missing: Optional[str] = None,
        drop_threshold: float = 0.3
    ):
        """
        Initialize the Clean class with configured techniques.

        Args:
            duplicate_handling: Strategy for duplicate removal.
                Options: None, "exact(keep=first)", "exact(keep=last)",
                        "key_based(subset=col1,keep=first)".
            harmonise_units: Strategy for unit harmonization.
                Format: "type:conversion:column_name"
                Options: None, "temperature:celsius_to_fahrenheit:col".
            remove_zero_variance: Strategy for zero-variance feature removal.
                Options:
                    - None (skip)
                    - "frequency_ratio_analysis(threshold=19.0)"
                    - "unique_value_count(min=2)"
            handle_missing: Strategy for missing value handling.
                Options: None, "drop_rows()", "drop_columns(p=0.3)",
                        "imputation(type=mean)", "imputation(type=median)",
                        "imputation(type=mode)".
            drop_threshold: Threshold for drop_columns (default 0.3).
        """
        self.duplicate_handling = duplicate_handling
        self.harmonise_units = harmonise_units
        self.remove_zero_variance = remove_zero_variance
        self.handle_missing = handle_missing
        self.drop_threshold = drop_threshold

    # ======================================================================
    # PUBLIC METHOD
    # ======================================================================

    def execute(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Execute all configured cleaning techniques.

        Args:
            data: Input DataFrame.

        Returns:
            DataFrame with configured cleaning applied.
        """
        result = data.copy()

        if self.duplicate_handling:
            result = self._handle_duplicates(result)

        if self.harmonise_units:
            result = self._harmonise_units(result)

        if self.remove_zero_variance:
            result = self._remove_zero_variance(result)

        if self.handle_missing:
            result = self._handle_missing(result)

        return result

    # ======================================================================
    # PRIVATE METHODS - TECHNIQUES
    # ======================================================================

    def _handle_duplicates(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply duplicate handling based on configured strategy.

        Args:
            data: Input DataFrame.

        Returns:
            DataFrame with duplicates removed.

        Raises:
            ValueError: If strategy is unknown.
        """
        method, subset, keep = parse_duplicate_strategy(self.duplicate_handling)

        if method == "exact":
            return exact_match_removal(data, keep=keep)

        elif method == "key_based":
            return key_based_deduplication(data, subset=subset, keep=keep)

        else:
            raise ValueError(f"Unknown duplicate method: {method}")

    def _harmonise_units(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply unit harmonization based on configured strategy.

        Args:
            data: Input DataFrame.

        Returns:
            DataFrame with harmonized units.

        Raises:
            ValueError: If strategy is unknown.
        """
        unit_type, conversion, column = parse_harmonise_strategy(
            self.harmonise_units
        )

        if unit_type == "temperature":
            if conversion == "celsius_to_fahrenheit":
                return convert_temperature_celsius_to_fahrenheit(data, column=column)
            elif conversion == "fahrenheit_to_celsius":
                return convert_temperature_fahrenheit_to_celsius(data, column=column)

        elif unit_type == "weight":
            if conversion == "kg_to_lbs":
                return convert_weight_kg_to_lbs(data, column=column)
            elif conversion == "lbs_to_kg":
                return convert_weight_lbs_to_kg(data, column=column)

        raise ValueError(
            f"Unknown harmonise_units strategy: {self.harmonise_units}"
        )

    def _remove_zero_variance(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Remove zero or near-zero variance features based on configured strategy.

        Strategies:
            - "frequency_ratio_analysis(threshold=19.0)"
            - "unique_value_count(min=2)"

        Args:
            data: Input DataFrame.

        Returns:
            DataFrame without zero or near-zero variance features.

        Raises:
            ValueError: If strategy is unknown.
        """
        method, value = parse_remove_variance_strategy(self.remove_zero_variance)

        if method == "frequency_ratio_analysis":
            return frequency_ratio_analysis(data, freq_ratio_threshold=value)

        elif method == "unique_value_count":
            return unique_value_count(data, min_unique_values=int(value))

        else:
            raise ValueError(
                f"Unknown remove_zero_variance method: {method}"
            )

    def _handle_missing(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply missing value handling based on configured strategy.

        Args:
            data: Input DataFrame.

        Returns:
            DataFrame with missing values handled.

        Raises:
            ValueError: If strategy is unknown.
        """
        method, parameter = parse_missing_strategy(self.handle_missing)

        if method == "drop_rows":
            return drop_rows(data)

        elif method == "drop_columns":
            if parameter is None:
                parameter = self.drop_threshold
            return drop_columns(data, threshold=parameter)

        elif method == "imputation":
            if parameter == "mean":
                return impute_mean(data)
            elif parameter == "median":
                return impute_median(data)
            elif parameter == "mode":
                return impute_mode(data)

        raise ValueError(f"Unknown missing value method: {method}")