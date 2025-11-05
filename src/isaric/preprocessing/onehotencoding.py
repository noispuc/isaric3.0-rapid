import pandas as pd

def encode_categoricals(df):
    """
    Description:
        Converts categorical variables into one-hot encoded format.

    Args:
        df (pandas.DataFrame): Input dataset.

    Returns:
        pandas.DataFrame: Dataset with one-hot encoded categorical features.
    """
    return pd.get_dummies(df, drop_first=True)
