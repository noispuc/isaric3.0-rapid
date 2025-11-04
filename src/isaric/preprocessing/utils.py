import pandas as pd
import numpy as np

def check_dataframe(df):
    """
    Description:
        Validates that the input is a pandas DataFrame.

    Args:
        df (any): Input object.

    Returns:
        bool: True if valid, raises TypeError otherwise.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    return True
