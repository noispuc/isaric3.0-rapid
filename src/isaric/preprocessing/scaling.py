from sklearn.preprocessing import StandardScaler

def standardize(df):
    """
    Description:
        Applies standard scaling (z-score normalization) to features.

    Args:
        df (pandas.DataFrame): Input dataset.

    Returns:
        pandas.DataFrame: Standardized dataset.
    """
    scaler = StandardScaler()
    return pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
