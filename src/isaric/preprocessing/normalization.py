from sklearn.preprocessing import MinMaxScaler

def normalize(df):
    """
    Description:
        Applies Min-Max normalization to scale features between 0 and 1.

    Args:
        df (pandas.DataFrame): Input dataset.

    Returns:
        pandas.DataFrame: Normalized dataset.
    """
    scaler = MinMaxScaler()
    return pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
