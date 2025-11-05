def remove_duplicates(df, subset=None):
    """
    Description:
        Removes duplicated rows from the dataset to ensure data integrity.

    Args:
        df (pandas.DataFrame): Input dataset.
        subset (list or str, optional): Columns to consider when identifying duplicates.

    Returns:
        pandas.DataFrame: Dataset with duplicates removed.
    """
    return df.drop_duplicates(subset=subset)

