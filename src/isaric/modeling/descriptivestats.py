def generate_summary_statistics(df):
    """
    Description:
        Generates descriptive statistics for numeric columns in the dataset.

    Args:
        df (pandas.DataFrame): Input dataset.

    Returns:
        pandas.DataFrame: Summary statistics including mean, std, min, max, and quartiles.
    """
    return df.describe()
