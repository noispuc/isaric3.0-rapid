"""
ARC data parser for the RAPID methodology.

This module provides functions to validate and transform input data
into the ISARIC ARC-compliant dataclass format. The parser acts as
a gatekeeper, checking whether data is in ARC format and converting
it if necessary.

Note:
    The Parser development is a responsibility of ISARIC team based
    in Oxford and their partners. It is included here to demonstrate
    the architectural dependencies that RAPID requires for data
    conversion. If the Parser component is unavailable, input data
    must already be in ARC format; otherwise, the data must be rejected.
"""

import pandas as pd
from typing import Dict, Optional, Tuple


def validate_arc_format(data: pd.DataFrame) -> bool:
    """
    Check if the input DataFrame is already in ARC format.

    This function performs a basic validation to determine whether
    the data conforms to the ISARIC ARC-compliant dataclass format.

    Args:
        data: Input DataFrame.

    Returns:
        True if data appears to be in ARC format, False otherwise.
    """
    # Basic validation - this is a placeholder for the Oxford Parser
    # The real validation will be implemented by the ISARIC Oxford team

    if data is None or data.empty:
        return False

    # Check for common ARC columns (example indicators)
    arc_indicators = [
        'subjid', 'patient_id', 'record_id',
        'redcap_repeat_instrument', 'redcap_repeat_instance'
    ]

    # Data may be in ARC format if it has any of these indicators
    has_arc_column = any(col.lower() in data.columns.str.lower() for col in arc_indicators)

    if has_arc_column:
        return True

    # If no ARC indicators, check for structured clinical columns
    clinical_indicators = [
        'age', 'sex', 'outcome', 'admission_date',
        'discharge_date', 'duration', 'event'
    ]

    has_clinical = any(col.lower() in data.columns.str.lower() for col in clinical_indicators)

    return has_clinical


def parse_to_arc_format(data: pd.DataFrame) -> pd.DataFrame:
    """
    Transform input data into ARC-compliant dataclass format.

    This is a placeholder for the Oxford Parser. If the Parser is
    unavailable, the data must already be in ARC format.

    Args:
        data: Input DataFrame.

    Returns:
        DataFrame in ARC format.

    Raises:
        NotImplementedError: If the Parser is not available.
        ValueError: If data cannot be transformed.
    """
    # Placeholder for the Oxford Parser
    raise NotImplementedError(
        "The ARC Parser is developed by the ISARIC team based in Oxford "
        "and their partners. This component is not available in the "
        "current version. Input data must already be in ARC format."
    )


def prepare_data_for_rapid(data: pd.DataFrame) -> Tuple[bool, pd.DataFrame]:
    """
    Validate and prepare data for the RAPID pipeline.

    This function checks if data is in ARC format. If it is, returns
    the data as-is. If not, attempts to parse it (if Parser is available).

    Args:
        data: Input DataFrame.

    Returns:
        Tuple of (is_valid, data_ready):
            - is_valid: True if data is in ARC format.
            - data_ready: DataFrame ready for RAPID.

    Raises:
        ValueError: If data is not in ARC format and Parser is unavailable.
    """
    if data is None or data.empty:
        raise ValueError("data cannot be None or empty.")

    if validate_arc_format(data):
        return True, data.copy()

    # Data is not in ARC format - try to parse
    try:
        parsed_data = parse_to_arc_format(data)
        return True, parsed_data
    except NotImplementedError:
        raise ValueError(
            "Data is not in ARC format and the Parser is unavailable. "
            "Please ensure your data conforms to the ISARIC ARC standard."
        )