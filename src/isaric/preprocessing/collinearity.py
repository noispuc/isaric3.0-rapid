def detect_collinearity(df, threshold=0.9):
    """
    Description:
        Detects highly correlated features based on a threshold.

    Args:
        df (pandas.DataFrame): Input dataset.
        threshold (float): Correlation coefficient threshold.

    Returns:
        list: List of column names to consider for removal.
    """
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    return [column for column in upper.columns if any(upper[column] > threshold)]
