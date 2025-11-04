def remove_duplicates(df):
    """
    Description:
        Removes duplicated rows from the dataset to ensure data integrity.

    Args:
        df (pandas.DataFrame): Input dataset.

    Returns:
        pandas.DataFrame: Dataset with duplicates removed.
    """
    return df.drop_duplicates()
