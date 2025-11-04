def remove_zero_variance_features(df, threshold=0.01):
    """
    Description:
        Removes features with zero or near-zero variance, which contribute little to model performance.

    Args:
        df (pandas.DataFrame): Input dataset.
        threshold (float): Variance threshold below which features are removed. Default is 0.01.

    Returns:
        pandas.DataFrame: Dataset with low-variance features removed.
    """
    from sklearn.feature_selection import VarianceThreshold
    selector = VarianceThreshold(threshold=threshold)
    filtered = selector.fit_transform(df)
    selected_columns = df.columns[selector.get_support()]
    return df[selected_columns]
