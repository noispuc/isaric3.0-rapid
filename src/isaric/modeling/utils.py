def separate_features_and_target(df, target):
    """
    Description:
        Separates features and target column from the dataset.

    Args:
        df (pandas.DataFrame): Input dataset.
        target (str): Name of the target column.

    Returns:
        tuple: (X, y) where X is the feature matrix and y is the target vector.
    """
    X = df.drop(columns=[target])
    y = df[target]
    return X, y
