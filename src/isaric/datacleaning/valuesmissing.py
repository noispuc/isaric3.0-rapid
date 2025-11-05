def report_missingness(df):
    """
    Description:
        Generates a summary report of missing values per column.

    Args:
        df (pandas.DataFrame): Input dataset.

    Returns:
        pandas.Series: Sorted count of missing values per column.
    """
    return df.isnull().sum().sort_values(ascending=False)


def drop_missing_rows(df, threshold=0.5):
    """
    Description:
        Drops rows with missing values exceeding a specified threshold.

    Args:
        df (pandas.DataFrame): Input dataset.
        threshold (float): Proportion of missing columns per row allowed. Rows above this threshold are dropped.

    Returns:
        pandas.DataFrame: Dataset with high-missingness rows removed.
    """
    return df[df.isnull().mean(axis=1) < threshold]


def remove_duplicates(df):
    """
    Description:
        Removes duplicate rows from the dataset.

    Args:
        df (pandas.DataFrame): Input dataset.

    Returns:
        pandas.DataFrame: Dataset with duplicates removed.
    """
    return df.drop_duplicates()
