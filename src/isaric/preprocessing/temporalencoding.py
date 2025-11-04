def encode_temporal(df, date_column, reference_date=None):
    """
    Description:
        Encodes temporal variables as time differences from a reference date.

    Args:
        df (pandas.DataFrame): Input dataset.
        date_column (str): Name of the date column.
        reference_date (str or datetime): Reference date for encoding.

    Returns:
        pandas.DataFrame: Dataset with encoded temporal feature.
    """
    df[date_column] = pd.to_datetime(df[date_column])
    ref = pd.to_datetime(reference_date) if reference_date else df[date_column].min()
    df[date_column + "_days"] = (df[date_column] - ref).dt.days
    return df.drop(columns=[date_column])
